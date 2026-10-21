# mypy: disable-error-code="assignment"
"""
Layer effects rendering.

This module implements rendering for Photoshop layer effects (also known as
layer styles). Effects are non-destructive visual enhancements applied to layers
such as strokes, shadows, glows, and overlays.

**Note**: Effects rendering requires scipy. Install it with::

    pip install 'psd-tools[composite]'

Currently supported effects:

- **Stroke**: Outline around layer shape or pixels
  - Supports solid color, gradient, and pattern fills
  - Position: inside, outside, or centered
  - Limited compared to Photoshop's full implementation
- **Drop shadow**: The layer's silhouette, spread, blurred and offset, painted
  behind the layer
  - Solid color; honors distance, angle (including global light), size,
    spread, contour and its anti-aliasing, opacity, blend mode and "layer
    knocks out drop shadow"
  - The blur is a gaussian approximation of Photoshop's own, so a soft
    shadow's falloff can differ; noise is read but not applied

- **Satin**: Two offset, blurred copies of the silhouette and their difference,
  painted inside the layer
  - Honors distance, angle, size, contour, invert, opacity and blend mode
- **Bevel and emboss**: The silhouette as a height field, lit from the angle
  and altitude (including global light)
  - Inner, outer, emboss and pillow emboss styles; smooth, chisel hard and
    chisel soft techniques; depth, direction, size, soften, gloss contour and
    the highlight and shadow colors, modes and opacities
  - Stroke emboss and the texture are not drawn, and the blurs are gaussian
    approximations of Photoshop's own

Not drawn: inner shadow, outer glow, inner glow.

The main function :py:func:`draw_stroke_effect` handles stroke rendering by:

1. Extracting the layer's alpha channel or shape mask
2. Placing the stroke relative to that shape according to its size and position
3. Filling the stroke region with the specified paint (solid color, gradient, pattern)
4. Returning the rendered stroke as a NumPy array

:py:func:`draw_stroke_effect_split` is the same stroke with its coverage
divided at the layer's boundary, which is what the compositor takes: the two
sides of that boundary are composited differently. Compositing is the only
caller that needs the division, so the undivided spelling above stays.

Implementation notes:

- Effects are image-based rather than vector-based, which may differ from Photoshop
- A shape layer's stroke follows its path interior but traces the stored coverage
  at the path's edge
- Some effect parameters may not be fully supported
- Complex effect combinations may not render identically to Photoshop

Example usage (internal)::

    from psd_tools.composite.effects import draw_stroke_effect

    # Called during layer compositing
    viewport = (0, 0, 100, 100)  # Region to render
    shape = layer_alpha_channel    # NumPy array
    desc = stroke_descriptor       # Effect parameters

    color, alpha = draw_stroke_effect(viewport, shape, desc, psd)

The effects system integrates with the main compositing pipeline and is
automatically applied when rendering layers that have effects enabled.
"""

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from psd_tools.api.utils import AllocBudget, check_pixel_size
from psd_tools.composite import paint
from psd_tools.composite._compat import HAS_SCIPY
from psd_tools.composite.utils import divide
from psd_tools.constants import BevelStyle, BevelTechnique
from psd_tools.psd.descriptor import Descriptor
from psd_tools.terminology import Enum, Key

if TYPE_CHECKING:
    from psd_tools.api.protocols import PSDProtocol

logger = logging.getLogger(__name__)

# Where each style puts its band, in multiples of the stroke's nominal size,
# measured from the layer's edge and positive outward. One table because the
# outer limit is also how far the stroke reaches past the layer, which is the
# canvas :py:func:`stroke_bbox` has to reserve for it: stating the two apart
# would let a change to one shave the outside of every stroke of that style,
# or reserve canvas nobody draws on.
#
# Only the inset style stays within the layer. It still gets the fixed pixel
# stroke_bbox() adds on top, which is not room for the stroke but room for the
# edge it is measured from.
_BANDS: dict[bytes, tuple[float, float]] = {
    Enum.OutsetFrame: (0.0, 1.0),
    Enum.CenteredFrame: (-0.5, 0.5),
    Enum.InsetFrame: (-1.0, 0.0),
}
# A style no descriptor Photoshop wrote can hold is drawn, and measured, as an
# outset one -- the widest of the three, so nothing it draws is clipped by the
# canvas reserved for it, and the position Photoshop itself defaults to.
_UNRECOGNISED = _BANDS[Enum.OutsetFrame]

# The widest stroke Photoshop writes; a larger size can only be a forged one.
_MAX_STROKE_SIZE = 250.0

# What a stroke canvas costs per pixel at its peak: the signed-distance
# transients (an int64 offset grid pair, float64 planes) dwarf the float32
# canvas and colour planes, so a budget is checked against this, not 4 bytes.
_STROKE_BYTES = 64

# What each offset of the distance search's grid costs at its peak.
_GRID_BYTES = 48

# Below this fraction of active pixels within the region a bounding-box crop
# would cover, :py:func:`_nearest_boundary` skips the erosion and falls back
# to its per-offset loop (#895): a thin outline on a much bigger canvas has
# active pixels too sparse within that box for the erosion to pay for itself,
# while a filled shape's boundary band fills enough of it that it does.
_DENSE_EROSION_DENSITY = 0.15


def _enum(desc: Descriptor, key: bytes) -> bytes:
    """The enum ``key`` names, or ``b""`` if the descriptor does not carry one.

    A missing or non-``Enumerated`` value joins the unrecognised enums the
    callers already handle, instead of raising ``AttributeError`` (#826).
    ``b""`` rather than None so it takes that fallback without a branch; no real
    enum collides with it, being four bytes.
    """
    return getattr(desc.get(key), "enum", b"")


def _require_scipy(what: str) -> None:
    """Raise :class:`ImportError` naming ``what`` and the package it needs."""
    if not HAS_SCIPY:
        raise ImportError(
            "%s require: scipy\n\n"
            "Install with:\n"
            "    pip install 'psd-tools[composite]'\n"
            "Or:\n"
            "    pip install scipy" % what
        )


def _stroke_size(desc: Descriptor) -> float:
    """The descriptor's stroke size, held to what Photoshop can write."""
    size = float(desc.get(Key.SizeKey, 1.0))
    if not math.isfinite(size):
        raise ValueError("Stroke size is not finite.")
    return min(size, _MAX_STROKE_SIZE)


def shadow_geometry(desc: Descriptor) -> tuple[float, float, float]:
    """A drop shadow's ``(distance, size, spread)``, as the file states them.

    Read off the descriptor rather than through :py:class:`~psd_tools.api.effects.DropShadow`,
    whose properties read a value they cannot parse as its default: to the
    compositor that value is an unreadable effect, as a stroke's size is to
    :py:func:`stroke_bbox`, and the effect is dropped rather than drawn at
    some invented size. Raises :class:`ValueError` or :class:`TypeError` for
    a value that will not parse, and :class:`ValueError` for one that is not
    finite. The size is held to what Photoshop can write, as a stroke's is.
    """
    distance = float(desc.get(Key.Distance, 0.0))
    size = float(desc.get(Key.Blur, 0.0))
    spread = float(desc.get(Key.ChokeMatte, 0.0))
    if not all(math.isfinite(value) for value in (distance, size, spread)):
        raise ValueError(
            "Drop shadow parameters must be finite: distance=%r size=%r spread=%r"
            % (distance, size, spread)
        )
    return distance, min(size, _MAX_STROKE_SIZE), spread


def stroke_bbox(
    bbox: tuple[int, int, int, int],
    desc: Descriptor,
    max_alloc_bytes: AllocBudget | None = None,
) -> tuple[int, int, int, int]:
    """The canvas :py:func:`draw_stroke_effect` needs to draw a stroke on.

    The stroke is drawn outward from the layer's edge, so an outset or
    centered one lands partly outside ``bbox``. Drawing it on ``bbox`` itself
    has no room for that part and silently discards it -- for a layer whose
    pixels fill its bounding box, that is the whole stroke (#792). Growing the
    box by the stroke's outward reach gives it somewhere to land.

    An inset stroke lands wholly inside the layer and needs no room, but it is
    still measured from the layer's edge, and on that same layer the edge is
    not in the picture either: every pixel of ``bbox`` is covered, so there is
    nothing to locate a boundary against and no stroke is drawn at all. The
    fixed pixel every style gets on top of its reach is what puts the edge back
    in view (#799).

    An empty ``bbox`` is returned untouched: there is no edge to trace, and
    growing it would place a stroke around the origin.

    A style the descriptor does not state, or states as something other than
    an enum, is measured as the unrecognised style it cannot be told apart
    from -- an outset one, which :py:func:`draw_stroke_effect` then draws it
    as -- rather than raising out of a composite (#826).

    A size above what Photoshop writes is drawn at that limit. Raises
    :class:`ValueError` when the size is not finite or the grown box is over
    the per-axis limit or ``max_alloc_bytes``, which the callers read as an
    unreadable stroke.
    """
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        return bbox
    limits = _BANDS.get(_enum(desc, Key.Style), _UNRECOGNISED)
    reach = limits[1]
    # ceil() because a fractional stroke still covers the pixel it falls in.
    # The +1 is the uncovered pixel the edge is measured against.
    size = _stroke_size(desc)
    margin = math.ceil(size * reach) + 1
    grown = (bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin)
    # The size is the file's, so the box it grows is checked before anything is
    # allocated on it. A box that fails is an unreadable stroke to both callers.
    if grown[0] < grown[2] and grown[1] < grown[3]:
        # A gradient paint peaks above the distance field it is drawn under.
        per_pixel = _STROKE_BYTES
        if _enum(desc, Key.PaintType) == Enum.GradientFill:
            per_pixel = max(per_pixel, paint.GRADIENT_FILL_BYTES)
        width, height = grown[2] - grown[0], grown[3] - grown[1]
        # The distance search's offset grid follows the stroke's size, not the
        # box: an inset one reserves a pixel however wide it is.
        radius = min(
            max(abs(limits[0]), abs(limits[1])) * size + 1.0, max(width, height)
        )
        grid = (2 * math.ceil(radius) + 1) ** 2 * _GRID_BYTES
        check_pixel_size(
            width,
            height,
            max_alloc_bytes=max_alloc_bytes,
            estimated_bytes=max(width * height * per_pixel, grid),
            warn=False,
        )
    return grown


def _grow(mask: np.ndarray) -> np.ndarray:
    """``mask`` widened by one pixel along each axis.

    Sliced rather than rolled, so the first row does not count the last one as
    its neighbour: a layer flush against one side of its viewport and
    transparent against the other would otherwise read as a boundary between
    them. No stroke effect can reach that today -- :py:func:`stroke_bbox`
    grants a pixel of clear border on every side -- so this is stated here
    rather than relied on there, because widening that margin away is the kind
    of change nothing else would notice.
    """
    grown = mask.copy()
    for axis in range(mask.ndim):
        lead: list[slice] = [slice(None)] * mask.ndim
        trail: list[slice] = [slice(None)] * mask.ndim
        lead[axis], trail[axis] = slice(1, None), slice(None, -1)
        grown[tuple(lead)] |= mask[tuple(trail)]
        grown[tuple(trail)] |= mask[tuple(lead)]
    return grown


def _prune_offsets(
    rows: np.ndarray,
    cols: np.ndarray,
    span: int,
    offsets: np.ndarray,
    lengths: np.ndarray,
    partial_pad: np.ndarray,
    near_pad: np.ndarray,
    outward: bool,
) -> np.ndarray:
    """``min``/``max`` of ``near[p] +/- |x - p|`` over offsets, pruning early.

    ``near`` never leaves ``[-0.5, 0.5]``, so once a pixel's running best is
    at least as good as ``length - 0.5`` (``outward``) or ``0.5 - length``
    (inward), no offset from here on -- ``offsets`` sorted by ascending
    ``length`` -- can improve it: the best any of them could still state is
    exactly that bound. A pixel this true of drops out of the working set,
    so offsets past it never revisit it. This is exact, not a bound in place
    of one: what it prunes could not have changed the answer.

    Most pixels resolve within a few offsets of their own unweighted nearest
    partial pixel, so the working set thins fast on a smooth boundary --
    which is the common case, a circle or a diagonal edge on a canvas much
    bigger than the stroke. It still runs every offset within ``span`` for a
    pixel that never resolves, so this does not change the loop's worst case.
    """
    prows, pcols = rows + span, cols + span
    result = np.empty(rows.shape, dtype=np.float32)
    order = np.arange(rows.size)
    best = np.full(rows.shape, np.inf if outward else -np.inf, dtype=np.float32)

    for (down, across), length in zip(offsets, lengths, strict=False):
        if order.size == 0:
            break
        row, col = prows + down, pcols + across
        states = partial_pad[row, col]
        stated = near_pad[row, col]
        if outward:
            best = np.minimum(best, np.where(states, stated + length, np.inf))
            resolved = best <= (length - 0.5)
        else:
            best = np.maximum(best, np.where(states, stated - length, -np.inf))
            resolved = best >= (0.5 - length)
        if resolved.any():
            result[order[resolved]] = best[resolved]
            keep = ~resolved
            prows, pcols, best, order = (
                prows[keep],
                pcols[keep],
                best[keep],
                order[keep],
            )

    if order.size:
        result[order] = best
    return result


def _nearest_boundary(
    alpha: np.ndarray, partial: np.ndarray, near: np.ndarray, radius: float
) -> np.ndarray:
    """The nearest boundary within ``radius``, as distance, negative inside.

    Every partial pixel states one boundary, ``near`` from its own centre and
    so up to half a pixel either side of it. A pixel with no coverage of its
    own is ``min`` over those of ``|x - p| + near[p]`` -- the nearest
    *boundary*, which is not the nearest pixel stating one: the offsets span a
    pixel, so a pixel one further away can win. Taking the nearest pixel's
    word for it leaves the answer to whichever of two equidistant pixels a
    distance transform happens to return, which is not a choice the mask
    makes; Photoshop's render of such a mask is mirror-exact, and this is the
    value it renders.

    That minimum is a grey erosion of ``near`` by a cone -- ``scipy``'s
    ``grey_erosion`` with a disc footprint and ``-length`` as the structuring
    function, which computes it without a Python-level pass per offset. The
    dual, the nearest boundary from *inside*, is the same erosion of
    ``-near``, negated.

    An erosion runs the disc footprint over every pixel of whatever it is
    given, so it only pays for itself where ``active`` -- the pixels an
    answer is wanted for -- is a healthy share of the region a crop bounding
    the partial pixels would cover: a filled shape's boundary, where the band
    the footprint runs over is close to that box already. Where the boundary
    is a thin outline on a much bigger canvas -- a circle, an ellipse, a
    diagonal edge -- that box is close to the shape's own bounding box while
    ``active`` is a sliver of it, and running the erosion there would cost
    more than it saves.

    Below :py:data:`_DENSE_EROSION_DENSITY`, :py:func:`_prune_offsets` runs
    the same minimum as a loop instead, one offset at a time in ascending
    length, dropping each pixel out as soon as no offset still to come could
    improve its answer -- exactly, per the bound in its own docstring. A
    smooth boundary resolves most pixels within a handful of offsets of their
    own nearest partial pixel, which is the common shape of this branch's
    input, but a pixel that never resolves still costs every offset within
    ``radius``: pruning cuts the constant this branch runs at, not the shape
    of its worst case, which stays the one the erosion branch exists to
    avoid.

    Both branches read and write through the same crop the density above is
    measured on, padded arrays included, rather than the whole of ``alpha``:
    everything either one needs already sits inside it, so this is memory
    tied to the shape's own extent even when the canvas around it is much
    bigger.
    """
    from scipy.ndimage import (  # type: ignore[import-untyped]  # noqa: PLC0415
        distance_transform_edt,
        grey_erosion,
    )

    # Beyond the reach of any boundary, and so of any band: infinitely far
    # out from a clear pixel and infinitely far in from an opaque one.
    field = np.where(partial, near, np.where(alpha >= 1, -np.inf, np.inf)).astype(
        np.float32
    )
    active = (distance_transform_edt(~partial) <= radius) & ~partial
    if not active.any():
        return field

    height, width = alpha.shape
    # An offset past the canvas's longest side reaches no pixel.
    span = math.ceil(min(radius, max(height, width) - 1))
    rows_p, cols_p = np.nonzero(partial)
    r0, r1 = max(rows_p.min() - span, 0), min(rows_p.max() + span + 1, height)
    c0, c1 = max(cols_p.min() - span, 0), min(cols_p.max() + span + 1, width)
    crop = (slice(r0, r1), slice(c0, c1))
    density = active.sum() / ((r1 - r0) * (c1 - c0))

    if density >= _DENSE_EROSION_DENSITY:
        sub_partial, sub_near, sub_alpha, sub_active = (
            partial[crop],
            near[crop],
            alpha[crop],
            active[crop],
        )
        grid = np.mgrid[-span : span + 1, -span : span + 1]
        length = np.hypot(*grid).astype(np.float32)
        kwargs = {
            "footprint": length <= radius,
            "structure": -length,
            "mode": "constant",
        }
        outward = grey_erosion(
            np.where(sub_partial, sub_near, np.inf).astype(np.float32),
            cval=np.inf,
            **kwargs,
        )
        inward = -grey_erosion(
            np.where(sub_partial, -sub_near, np.inf).astype(np.float32),
            cval=np.inf,
            **kwargs,
        )
        stated = np.where(sub_alpha >= 1, inward, outward)
        field[crop] = np.where(sub_active, stated, field[crop])
        return field

    offsets = np.mgrid[-span : span + 1, -span : span + 1].reshape(2, -1).T
    lengths = np.hypot(offsets[:, 0], offsets[:, 1]).astype(np.float32)
    within = lengths <= radius
    offsets, lengths = offsets[within], lengths[within]
    ascending = np.argsort(lengths, kind="stable")
    offsets, lengths = offsets[ascending], lengths[ascending]

    # partial's own pixels sit inside `crop` by construction (it is padded out
    # to `span` around their bounding box), and so does every active pixel --
    # nothing an offset lookup needs is outside it. Padding that crop, rather
    # than the whole canvas, is what keeps this branch's own memory use tied
    # to the shape's extent instead of the much bigger canvas around it.
    sub_partial, sub_near, sub_alpha, sub_active = (
        partial[crop],
        near[crop],
        alpha[crop],
        active[crop],
    )
    flat = np.flatnonzero(sub_active)
    rows, cols = np.unravel_index(flat, sub_active.shape)
    partial_pad = np.pad(sub_partial, span, constant_values=False)
    near_pad = np.pad(sub_near, span, constant_values=0.0)
    inside = (sub_alpha >= 1).ravel()[flat]

    outward = _prune_offsets(
        rows[~inside],
        cols[~inside],
        span,
        offsets,
        lengths,
        partial_pad,
        near_pad,
        outward=True,
    )
    inward = _prune_offsets(
        rows[inside],
        cols[inside],
        span,
        offsets,
        lengths,
        partial_pad,
        near_pad,
        outward=False,
    )
    field[r0 + rows[~inside], c0 + cols[~inside]] = outward
    field[r0 + rows[inside], c0 + cols[inside]] = inward
    return field


def _signed_distance(alpha: np.ndarray, reach: float | None = None) -> np.ndarray:
    """Distance from each pixel to the mask boundary, negative inside.

    Photoshop reads that boundary off the mask's **coverage**, not off its 0.5
    iso-contour: a pixel at ``alpha`` states that the boundary runs
    ``0.5 - alpha`` from its own centre, and its neighbours measure outward
    from there. On a hard-edged mask the two readings name the same line --
    one partial pixel, and the boundary is where its coverage puts it. On a
    *feathered* one they diverge without limit: a 16 px alpha ramp runs 8 px
    from the iso-contour at either end and half a pixel from the boundary
    everywhere, so a stroke of any size covers all of it.
    ``effects/feathered-stroke.psd`` and
    ``effects/antialiased-stroke-edge.psd`` are the fixtures this is measured
    against (#799).

    Coverage says nothing about an edge that has none, so a mask stepping
    0 -> 1 keeps the exact Euclidean distance to the iso-contour, ``base``.
    One mask can carry both -- a shape antialiased along one side and butted
    against its own bounding box along another -- so the two are chosen
    between per pixel, by which boundary is nearer. Seeding a single transform
    from both instead costs the corners, where a feathered edge meets a hard
    one.

    A pixel outside the partial band takes the nearest of the boundaries the
    band states, which :py:func:`_nearest_boundary` solves for outright. What
    stays approximate is that a partial pixel's ``near`` is applied along the
    line to that pixel rather than along the boundary's own normal, which
    parts company with the truth around a curve the way
    :py:func:`_distance_band` does when it paints the arc.

    A mask that is flat at 0 or 1 has no boundary at all. That is not a case
    to fall through: ``distance_transform_edt`` is undefined on an input with
    no zeros and reports distances to a phantom feature off the array corner,
    which a band paints as a wedge in the corner of the canvas. Being
    uniformly infinitely far from the boundary leaves every band empty, which
    is right. A mask flat at a *partial* value is not one of these -- every
    pixel of it states a boundary, so it takes a stroke over all of it.
    """
    if reach is None:
        # No band named, so no bound: far enough that nothing is out of range.
        reach = float(math.hypot(*alpha.shape))
    _require_scipy("Stroke effects")
    from scipy.ndimage import (  # type: ignore[import-untyped]  # noqa: PLC0415
        distance_transform_edt,
    )

    partial = (alpha > 0) & (alpha < 1)
    # The boundary segments no coverage describes: a step from opaque straight
    # to clear. Every pixel of a smoothly antialiased shape has coverage, so
    # this is usually empty and the iso-contour below is never computed --
    # which is what keeps the field to one transform rather than three.
    hard = (alpha >= 1) & _grow(alpha <= 0)
    hard |= (alpha <= 0) & _grow(alpha >= 1)

    if not partial.any() or hard.any():
        # ``distance_transform_edt`` measures to the nearest pixel of the other
        # class, so the two pixels straddling the boundary both come back 1.0
        # rather than the 0.5 their centres really sit at. The magnitude is
        # shrunk to put the boundary back between them; subtracting 0.5
        # outright would be right outside and a full pixel wrong inside, where
        # the sign is negative.
        inside = alpha >= 0.5
        if inside.all():
            base = np.full(alpha.shape, -np.inf, dtype=np.float32)
        elif not inside.any():
            base = np.full(alpha.shape, np.inf, dtype=np.float32)
        else:
            base = distance_transform_edt(~inside).astype(np.float32)
            base -= distance_transform_edt(inside).astype(np.float32)
            base = np.sign(base) * np.maximum(np.abs(base) - 0.5, 0.0)
        if not partial.any():
            return base

    # What a partial pixel states, and what a pixel with none of its own takes
    # from the nearest boundary any of them states.
    near = (0.5 - alpha).astype(np.float32)
    stated = _nearest_boundary(alpha, partial, near, reach)
    if not hard.any():
        return stated

    # ``base`` is exact along a hard step, and is the better answer for any
    # pixel nearer one of them than it is to the partial band.
    distance = distance_transform_edt(~partial)
    to_hard = distance_transform_edt(~hard)
    field = np.where(partial, near, np.where(distance <= to_hard, stated, base))
    return field.astype(np.float32)


def _distance_band(distance: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Coverage of the band between ``lo`` and ``hi``, antialiased.

    A distance field has unit gradient, so a linear ramp across one pixel is the
    exact area of a pixel cut by a straight edge, and a fractional width is not
    quantized. Where the band's edge curves, the ramp only approximates it.
    """
    return np.clip(hi - distance + 0.5, 0, 1) * np.clip(distance - lo + 0.5, 0, 1)


def draw_stroke_effect(
    viewport: tuple[int, int, int, int],
    shape: np.ndarray,
    desc: Descriptor,
    psd: "PSDProtocol",
) -> tuple[np.ndarray, np.ndarray]:
    """The stroke's paint and the coverage it puts it on."""
    color, coverage, _ = draw_stroke_effect_split(viewport, shape, desc, psd)
    return color, coverage


def draw_stroke_effect_split(
    viewport: tuple[int, int, int, int],
    shape: np.ndarray,
    desc: Descriptor,
    psd: "PSDProtocol",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:py:func:`draw_stroke_effect`, with the coverage split at the boundary.

    The third array is the part of the band that falls *inside* the layer,
    which is what the band's two halves are composited differently: the inside
    paints over the layer and knocks out what it covers, the rest paints beside
    it (#846). An inset band is wholly inside and an outset one wholly outside;
    only a centered stroke has both.

    The split is a share of the band rather than two bands drawn separately, so
    the two always sum to what :py:func:`draw_stroke_effect` returns. Drawn
    separately they would not: :py:func:`_distance_band` multiplies the two
    edges' ramps, which is the area of a pixel cut by a straight edge only
    while the band is at least a pixel wide, and halving a band halves both
    halves' widths.

    The two can be views of one array, since an inset band is the whole of the
    coverage; both are read and neither is written to.
    """
    logger.debug("Stroke effect has limited support")
    height, width = viewport[3] - viewport[1], viewport[2] - viewport[0]

    paint_type = _enum(desc, Key.PaintType)
    if paint_type == Enum.SolidColor:
        color, _ = paint.draw_solid_color_fill(viewport, psd.color_mode, desc)
        if color is None:
            color = np.ones((height, width, 1))
    elif paint_type == Enum.Pattern:
        color, _ = paint.draw_pattern_fill(viewport, psd, desc)
        if color is None:
            color = np.ones((height, width, 1))
    elif paint_type == Enum.GradientFill:
        color, _ = paint.draw_gradient_fill(viewport, psd.color_mode, desc)
        if color is None:
            color = np.ones((height, width, 1))
    else:
        logger.warning("No fill specification found.")
        color = np.ones((height, width, 1))

    coverage, inside = stroke_band(viewport, shape, desc)
    return color, coverage, inside


def stroke_band(
    viewport: tuple[int, int, int, int],
    shape: float | np.ndarray,
    desc: Descriptor,
) -> tuple[np.ndarray, np.ndarray]:
    """The coverage of a stroke effect, split at the boundary, without its paint.

    What :py:func:`draw_stroke_effect_split` returns after the colour, drawn
    on its own for a caller that only needs where the stroke lies: the
    drop shadow's knockout cuts under the band, whatever it is painted with.
    """
    height, width = viewport[3] - viewport[1], viewport[2] - viewport[0]
    if not isinstance(shape, np.ndarray):
        shape = np.full((height, width, 1), shape, dtype=np.float32)

    # Note: current implementation is purely image-based.
    # For layers with path objects, this should be based on drawing.

    style = _enum(desc, Key.Style)
    size = _stroke_size(desc)

    # A stroke of no width draws nothing, and the band has to be told so: it
    # would otherwise paint a quarter of a pixel wherever the boundary falls
    # exactly on a pixel centre. Photoshop will not author a 0 px stroke, but
    # a descriptor can carry one.
    if size <= 0.0:
        return (
            np.zeros((height, width, 1), dtype=np.float32),
            np.zeros((height, width, 1), dtype=np.float32),
        )

    # A stroke is a band in the layer's signed distance field, which is exact
    # on a hard-edged mask and needs no pen to quantize the radius to a whole
    # pixel. All three positions are the same band read off a different
    # anchor, inset included: Photoshop measures it from the same boundary as
    # the other two, and the pixel of offset that looked like a different
    # anchor was the layer's edge being clipped away before the stroke saw it
    # (#799). Nor does an inset band need clamping to the layer to stay inside
    # it: on a pixel the boundary cuts, the band already comes out equal to
    # that pixel's own coverage, feathered masks included.
    #
    # A position the descriptor states as something else, or does not state at
    # all, takes the outset band (#799). That leaves it indistinguishable from
    # a stated outset stroke, which is why it says so in the log rather than
    # only in the canvas stroke_bbox() reserved for it.
    limits = _BANDS.get(style)
    if limits is None:
        logger.debug("Unrecognised stroke position %r; drawing it as outset", style)
        limits = _UNRECOGNISED
    lo, hi = limits[0] * size, limits[1] * size
    # A boundary further out than the band's widest limit, plus the half
    # pixel a partial pixel can state either side of itself, cannot show.
    distance = _signed_distance(shape[:, :, 0], max(abs(lo), abs(hi)) + 1.0)
    coverage = _distance_band(distance, lo, hi)
    if hi <= 0.0:
        inside = coverage
    elif lo >= 0.0:
        inside = np.zeros_like(coverage)
    else:
        within = _distance_band(distance, lo, 0.0)
        beyond = _distance_band(distance, 0.0, hi)
        inside = coverage * divide(within, within + beyond, fill=0.0)
    return np.expand_dims(coverage, 2), np.expand_dims(inside, 2)


# How wide a gaussian stands in for the blur Photoshop gives a drop shadow of a
# given size. Photoshop's blur is its own, so this is fitted against its
# previews rather than derived; a third of the size, the usual rule of thumb,
# leaves the falloff too steep. The fit is asserted in the tests.
_SHADOW_SIGMA_PER_SIZE = 0.45

# What one pixel of a shadow's canvas costs at the peak of
# ``draw_drop_shadow()``: the silhouette it is cast from, the matte spread
# grows and the distance transform and masks that grow it, the blurred matte
# and the shifted one, with the group composite the silhouette of a group is
# read from.
_SHADOW_BYTES = 64

# Entries in a contour's transfer table. Photoshop states a contour's points
# on a 0-255 grid along both axes.
_CONTOUR_STEPS = 256

# How many coverage levels spread grows a silhouette at, at most. A silhouette
# with no more distinct coverages than this is grown at each of them, so none
# is rounded to another; one with more, an antialiased or feathered one, is
# grown at this many of its own, spaced evenly between its lowest and highest.
_SPREAD_LEVELS = 16


def _shadow_offset(distance: float, angle: float) -> tuple[float, float]:
    """How far a shadow is cast, as ``(dx, dy)`` in image coordinates.

    ``angle`` is where the light comes *from*, counter-clockwise from the
    positive x axis, so the shadow falls the opposite way; image rows run
    down, which is why a light at 90 degrees casts the shadow down.

    Rounded to a millionth of a pixel so that a light straight overhead or
    level casts exactly along the axis: ``cos(90)`` is 6e-17 rather than 0,
    and a canvas measured off that would take an extra column on one side
    at some positions and not at others.
    """
    theta = math.radians(angle)
    return (
        round(-distance * math.cos(theta), 6),
        round(distance * math.sin(theta), 6),
    )


def drop_shadow_bbox(
    bbox: tuple[int, int, int, int],
    distance: float,
    angle: float,
    size: float,
    max_alloc_bytes: AllocBudget | None = None,
) -> tuple[int, int, int, int]:
    """The canvas :py:func:`draw_drop_shadow` needs to draw a shadow on.

    ``bbox`` grown by the shadow's size on every side and by its offset on
    the side it is cast to. It is the box the silhouette is read on, before
    it is moved, so it has to hold that too. The two pixels added on top are
    the antialiased fringe spread leaves, and the one the offset's
    interpolation can spill into.

    An empty ``bbox`` is returned untouched, as :py:func:`stroke_bbox` does:
    there is no silhouette to cast.

    The distance and size are the file's, so the box they grow is checked
    before anything is allocated on it, as a stroke's is. Raises
    :class:`ValueError` when the grown box is over the per-axis limit or
    ``max_alloc_bytes``, which the callers read as an unreadable shadow.
    """
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        return bbox
    dx, dy = _shadow_offset(distance, angle)
    margin = math.ceil(max(size, 0.0)) + 2
    grown = (
        math.floor(bbox[0] + min(dx, 0.0)) - margin,
        math.floor(bbox[1] + min(dy, 0.0)) - margin,
        math.ceil(bbox[2] + max(dx, 0.0)) + margin,
        math.ceil(bbox[3] + max(dy, 0.0)) + margin,
    )
    width, height = grown[2] - grown[0], grown[3] - grown[1]
    check_pixel_size(
        width,
        height,
        max_alloc_bytes=max_alloc_bytes,
        estimated_bytes=width * height * _SHADOW_BYTES,
        warn=False,
    )
    return grown


def contour_table(desc: Descriptor | None) -> np.ndarray | None:
    """The transfer table of a contour descriptor, or ``None`` for the identity.

    A contour maps the coverage of a blurred matte to the coverage painted,
    along a curve through the points the descriptor lists. Photoshop draws a
    smooth curve through them; this is a monotone cubic through the same
    points, which neither overshoots a point nor turns back between two, and
    a straight line between two points is reproduced exactly. Past the first
    and last point the table holds their values.

    The default contour, Photoshop's "Linear", runs from corner to corner and
    is the identity; it comes back as ``None`` so the caller can skip the
    mapping. So does a descriptor with fewer than two points to draw a curve
    through. A point that will not read raises as the descriptor's other
    values do.
    """
    if desc is None:
        return None
    points = desc.get(Key.Curve)
    if points is None or len(points) < 2:
        return None
    xs = np.array([float(point[Key.Horizontal]) for point in points]) / 255.0
    ys = np.array([float(point[Key.Vertical]) for point in points]) / 255.0
    if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
        raise ValueError("Contour points must be finite.")
    order = np.argsort(xs, kind="stable")
    xs, ys = xs[order], ys[order]
    if len(xs) == 2 and np.array_equal(xs, ys) and xs[0] == 0.0 and xs[1] == 1.0:
        return None
    grid = np.linspace(0.0, 1.0, _CONTOUR_STEPS)
    if len(xs) > 2 and np.all(np.diff(xs) > 0.0):
        _require_scipy("Drop shadow effects")
        from scipy.interpolate import PchipInterpolator  # type: ignore[import-untyped]  # noqa: PLC0415

        table = PchipInterpolator(xs, ys)(np.clip(grid, xs[0], xs[-1]))
    else:
        # Two points, or a step the spline cannot pass through: straight
        # segments, which hold the end values past the ends on their own.
        table = np.interp(grid, xs, ys)
    return np.clip(table, 0.0, 1.0).astype(np.float32)


def _contoured(alpha: np.ndarray, table: np.ndarray, anti_aliased: bool) -> np.ndarray:
    """``alpha`` mapped through ``table``.

    Anti-aliased, the table is read between its entries, so a coverage that
    falls between two levels is painted between their two values. Otherwise
    it is read at the nearest entry, the way an 8-bit matte is: a table
    that steps paints the step.
    """
    last = len(table) - 1
    if anti_aliased:
        grid = np.linspace(0.0, 1.0, len(table))
        return np.interp(alpha, grid, table).astype(np.float32)
    return table[np.rint(alpha * last).astype(np.intp)]


def _spread(alpha: np.ndarray, grow: float) -> np.ndarray:
    """``alpha`` grown by ``grow`` pixels, keeping the coverage it grows.

    A grey dilation by a disc: every pixel takes the highest coverage within
    ``grow`` of it, so a half-covered silhouette grows into a half-covered
    matte rather than an opaque one. It is taken a coverage level at a time,
    each level a binary silhouette grown along its distance transform with
    the edge antialiased, and the levels summed, each weighted by the step
    in coverage it stands for.

    The levels are the silhouette's own coverages: all of them when it has
    few, which rounds none and costs a binary silhouette a single transform.
    With more, they are the lowest of its coverages at or above each of that
    many evenly spaced values between its lowest and highest, so a coverage
    grows at a level less than one space below it, and the grown fringe is
    never more than a space under a full grey dilation's. The spacing
    follows wherever the coverage lies: a feather that never passes 5% is
    grown in steps of a fifteenth of 5%.
    """
    from scipy.ndimage import distance_transform_edt  # type: ignore[import-untyped]  # noqa: PLC0415

    levels = np.unique(alpha[alpha > 0.0])
    if levels.size > _SPREAD_LEVELS:
        spaced = np.linspace(levels[0], levels[-1], _SPREAD_LEVELS)
        picks = np.searchsorted(levels, spaced, side="left")
        levels = np.unique(levels[np.minimum(picks, levels.size - 1)])
    grown = np.zeros(alpha.shape, dtype=np.float32)
    reach = grown
    previous: np.ndarray | None = None
    below = 0.0
    for level in levels:
        inside = alpha >= level
        if not inside.any():
            break  # The levels only shrink from here.
        # Consecutive levels no pixel falls between are one silhouette, and
        # share its transform.
        if previous is None or not np.array_equal(inside, previous):
            if inside.all():
                reach = np.ones(alpha.shape, dtype=np.float32)
            else:
                # Half a pixel past the centre of the nearest pixel inside,
                # the way ``_signed_distance()`` measures a hard edge.
                outside = distance_transform_edt(~inside).astype(np.float32)
                reach = np.clip(grow + 1.0 - outside, 0.0, 1.0)
            previous = inside
        grown += (float(level) - below) * reach
        below = float(level)
    return np.maximum(alpha, grown)


def draw_drop_shadow(
    viewport: tuple[int, int, int, int],
    shape: np.ndarray,
    *,
    distance: float,
    angle: float,
    size: float,
    spread: float,
    contour: np.ndarray | None = None,
    anti_aliased: bool = False,
) -> np.ndarray:
    """The coverage a drop shadow casts from ``shape``, on ``viewport``.

    ``shape`` is the layer's silhouette on the same canvas, which
    :py:func:`drop_shadow_bbox` sizes. The result is ``(height, width, 1)``;
    the caller tints it, knocks the layer out of it and composites it.

    ``contour`` is a transfer table from :py:func:`contour_table`, applied to
    the cast matte last, as Photoshop's contour panel shows it: coverage in,
    coverage out. It is applied on this canvas alone, so a contour that
    lifts zero coverage paints the canvas and not the document. The table
    is read between its entries when ``anti_aliased`` and at the nearest
    one otherwise. Noise is not applied.

    ``spread`` is a percentage of ``size``, which is how Photoshop's dialog
    states it, even though the descriptor labels the value in pixels. It
    *grows* the silhouette before the blur, and the blur takes what is left
    of ``size``, so a shadow turns hard-edged at 100%. (The same key shrinks
    the silhouette of an inner shadow, where the dialog calls it choke.)

    The blur is cut off at that remaining radius, rounded to a whole pixel,
    so the shadow reaches at most a pixel or two past ``size`` from the
    silhouette, and the canvas :py:func:`drop_shadow_bbox` asks for holds all
    of it; a gaussian left to its default reach runs on past both. Both
    filters treat what lies past the canvas as empty, since nothing outside
    it casts a shadow. The default for a blur is to reflect the edge instead,
    which doubles back any coverage that reaches it.

    A parameter that is not finite raises ``ValueError``: NaN passes through
    every step here without complaint, and one NaN in the coverage spreads to
    the whole canvas it is composited on.
    """
    _require_scipy("Drop shadow effects")
    from scipy import ndimage  # type: ignore[import-untyped]  # noqa: PLC0415

    if not all(math.isfinite(value) for value in (distance, angle, size, spread)):
        raise ValueError(
            "Drop shadow parameters must be finite: distance=%r angle=%r "
            "size=%r spread=%r" % (distance, angle, size, spread)
        )
    alpha = shape[:, :, 0].astype(np.float32)
    size = max(float(size), 0.0)
    grow = size * min(max(float(spread), 0.0), 100.0) / 100.0
    if grow > 0.0 and alpha.any():
        alpha = _spread(alpha, grow)
    radius = size - grow
    sigma = radius * _SHADOW_SIGMA_PER_SIZE
    if sigma > 1e-3:
        alpha = ndimage.gaussian_filter(
            alpha, sigma, mode="constant", cval=0.0, truncate=radius / sigma
        )
    dx, dy = _shadow_offset(distance, angle)
    if dx or dy:
        alpha = ndimage.shift(alpha, (dy, dx), order=1, mode="constant", cval=0.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    if contour is not None:
        alpha = _contoured(alpha, contour, anti_aliased)
    return alpha[:, :, None]


# How wide a gaussian stands in for the blur of a satin, and for the blur
# that rounds a smooth bevel, per pixel of the effect's size; how far a
# bevel's soften blurs its shading per pixel of soften; and how steep a
# bevel's faces are at a depth of 100%. Fitted against Photoshop's previews,
# and asserted in the tests.
_SATIN_SIGMA_PER_SIZE = 0.45
_BEVEL_SIGMA_PER_SIZE = 0.45
_SOFTEN_SIGMA_PER_SIZE = 0.45
_BEVEL_SLOPE_PER_DEPTH = 1.0

# How far a chisel soft bevel rounds its faces, per pixel of size.
_SLOPE_LIMIT_SIGMA_PER_SIZE = 0.125

# What a pixel of a satin's or a bevel's canvas costs at its peak: the
# silhouette, two shifted and blurred mattes or a height field and its two
# gradients, and the float64 transients of the filters.
_INNER_BYTES = 64

# Photoshop's bevel defaults, for a descriptor that leaves a value out.
_BEVEL_DEPTH = 100.0
_BEVEL_SIZE = 5.0

# The styles whose surface stands beside the layer.
_BEVEL_BESIDE = (BevelStyle.OUTER_BEVEL, BevelStyle.EMBOSS, BevelStyle.PILLOW_EMBOSS)


def _finite(what: str, **values: float) -> None:
    """Raise ``ValueError`` if any of ``values`` is not finite."""
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError(
            "%s parameters must be finite: %s"
            % (what, " ".join("%s=%r" % item for item in values.items()))
        )


def _blur(alpha: np.ndarray, sigma: float, radius: float) -> np.ndarray:
    """``alpha`` blurred by a gaussian of ``sigma`` cut off at ``radius``.

    What lies past the canvas is empty, as for a drop shadow's blur.
    """
    from scipy import ndimage  # type: ignore[import-untyped]  # noqa: PLC0415

    if sigma <= 1e-3 or radius < 1.0:
        return alpha
    return ndimage.gaussian_filter(
        alpha, sigma, mode="constant", cval=0.0, truncate=radius / sigma
    )


def inner_bbox(
    bbox: tuple[int, int, int, int],
    reach: float,
    max_alloc_bytes: AllocBudget | None = None,
) -> tuple[int, int, int, int]:
    """The canvas a satin or a bevel is drawn on: ``bbox`` grown by ``reach``.

    The silhouette is read on this box, so that a blur or an offset near the
    viewport's edge sees the layer past it rather than an edge that is not
    there (#804). The two pixels added on top are the blur's antialiased
    fringe and the offset's interpolation. Checked against the per-axis limit
    and ``max_alloc_bytes`` as :py:func:`drop_shadow_bbox` is, raising
    :class:`ValueError`. An empty ``bbox`` is returned untouched.
    """
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        return bbox
    margin = math.ceil(max(reach, 0.0)) + 2
    grown = (bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin)
    width, height = grown[2] - grown[0], grown[3] - grown[1]
    check_pixel_size(
        width,
        height,
        max_alloc_bytes=max_alloc_bytes,
        estimated_bytes=width * height * _INNER_BYTES,
        warn=False,
    )
    return grown


def satin_geometry(desc: Descriptor) -> tuple[float, float]:
    """A satin's ``(distance, size)``, read as :py:func:`shadow_geometry` reads."""
    distance = float(desc.get(Key.Distance, 0.0))
    size = float(desc.get(Key.Blur, 0.0))
    _finite("Satin", distance=distance, size=size)
    return distance, min(max(size, 0.0), _MAX_STROKE_SIZE)


def draw_satin(
    viewport: tuple[int, int, int, int],
    shape: np.ndarray,
    *,
    distance: float,
    angle: float,
    size: float,
    inverted: bool,
    contour: np.ndarray | None = None,
    anti_aliased: bool = False,
) -> np.ndarray:
    """The coverage a satin paints inside ``shape``, on ``viewport``.

    Two copies of the silhouette, moved ``distance`` either way along
    ``angle`` and each blurred by ``size``, and the absolute difference of the
    two: none where they agree, up to full where one edge passes the other. Photoshop
    states it on the inverted silhouette, which comes to the same difference.
    Then ``contour``, read as :py:func:`draw_drop_shadow` reads it, then the
    inversion, and the result masked by ``shape``. ``(height, width, 1)``.

    :py:func:`inner_bbox` grown by ``distance + size`` gives the canvas, which
    keeps the edges the blur and the offsets bring in from past it away from
    the layer.
    """
    _require_scipy("Satin effects")
    from scipy import ndimage  # type: ignore[import-untyped]  # noqa: PLC0415

    _finite("Satin", distance=distance, angle=angle, size=size)
    alpha = shape[:, :, 0].astype(np.float32)
    dx, dy = _shadow_offset(distance, angle)
    size = max(float(size), 0.0)
    copies = []
    for sign in (1.0, -1.0):
        moved = alpha
        if dx or dy:
            moved = ndimage.shift(
                alpha, (sign * dy, sign * dx), order=1, mode="constant", cval=0.0
            )
        copies.append(_blur(moved, size * _SATIN_SIGMA_PER_SIZE, size))
    satin = np.clip(np.abs(copies[0] - copies[1]), 0.0, 1.0)
    if contour is not None:
        satin = _contoured(satin, contour, anti_aliased)
    if inverted:
        satin = 1.0 - satin
    return (satin * alpha)[:, :, None].astype(np.float32)


def bevel_geometry(desc: Descriptor) -> tuple[float, float, float]:
    """A bevel's ``(depth, size, soften)``, read as :py:func:`shadow_geometry` reads.

    Depth is in percent, size and soften in pixels; each is held to zero or
    more, and size and soften to what Photoshop can write.
    """
    depth = float(desc.get(Key.StrengthRatio, _BEVEL_DEPTH))
    size = float(desc.get(Key.Blur, _BEVEL_SIZE))
    soften = float(desc.get(Key.Softness, 0.0))
    _finite("Bevel", depth=depth, size=size, soften=soften)
    return (
        max(depth, 0.0),
        min(max(size, 0.0), _MAX_STROKE_SIZE),
        min(max(soften, 0.0), _MAX_STROKE_SIZE),
    )


def bevel_reach(size: float, soften: float) -> float:
    """How far a bevel's canvas has to reach past the layer.

    The widest band beside it, and the blurs that read across the edge.
    """
    return size + soften


def bevel_beside(style: bytes) -> bool:
    """Whether a bevel style paints beside the layer, outside its silhouette."""
    return style in _BEVEL_BESIDE


def bevel_inside(style: bytes) -> bool:
    """Whether a bevel style paints inside the layer's silhouette."""
    return style != BevelStyle.OUTER_BEVEL


def _bevel_height(
    alpha: np.ndarray, style: bytes, technique: bytes, size: float
) -> np.ndarray:
    """The bevel's surface over the canvas, in pixels of height.

    Each style is a band ``size`` pixels wide at the silhouette's edge,
    rising toward the layer: inside it for an inner bevel, outside it for an
    outer one, half on each side for an emboss. A pillow emboss is split as
    an emboss is but rises away from the edge on both sides, which sinks the
    edge into what lies under it.

    A chisel hard bevel climbs that band at a constant slope along the
    signed distance to the edge; a chisel soft one is the same surface
    rounded off. A smooth bevel is the silhouette blurred across the band,
    so it is steepest at the edge and flattens away from it.
    """
    half = style in (BevelStyle.EMBOSS, BevelStyle.PILLOW_EMBOSS)
    width = size / 2.0 if half else size
    if width <= 0.0:
        return np.zeros(alpha.shape, dtype=np.float32)
    if technique not in (
        BevelTechnique.PRECISE_MATTE,
        BevelTechnique.SLOPE_LIMIT_MATTE,
    ):
        # The blur reaches a pixel short of the band, so that the band's
        # width counts the edge pixel.
        height = _blur(alpha, width * _BEVEL_SIGMA_PER_SIZE, width - 1.0)
        if style == BevelStyle.PILLOW_EMBOSS:
            height = np.where(alpha >= 0.5, height, 1.0 - height)
        return (height * width).astype(np.float32)

    distance = _signed_distance(alpha, size + 1.0)
    if style == BevelStyle.OUTER_BEVEL:
        height = np.clip(width - distance, 0.0, width)
    elif style == BevelStyle.EMBOSS:
        height = np.clip(width - distance, 0.0, 2.0 * width)
    elif style == BevelStyle.PILLOW_EMBOSS:
        height = np.clip(np.abs(distance), 0.0, width)
    else:
        height = np.clip(-distance, 0.0, width)
    height = height.astype(np.float32)
    if technique == BevelTechnique.SLOPE_LIMIT_MATTE:
        from scipy import ndimage  # type: ignore[import-untyped]  # noqa: PLC0415

        sigma = size * _SLOPE_LIMIT_SIGMA_PER_SIZE
        if sigma > 1e-3:
            height = ndimage.gaussian_filter(height, sigma, mode="nearest")
    return height


def draw_bevel(
    viewport: tuple[int, int, int, int],
    shape: np.ndarray,
    *,
    style: bytes,
    technique: bytes,
    up: bool,
    depth: float,
    size: float,
    soften: float,
    angle: float,
    altitude: float,
    contour: np.ndarray | None = None,
    anti_aliased: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """How much highlight and how much shadow a bevel paints, on ``viewport``.

    The surface from :py:func:`_bevel_height`, its slope scaled by ``depth``
    and turned over when not ``up``, lit from ``angle`` (where the light comes
    from, as a drop shadow's) and ``altitude`` above the horizon. A face lit
    more than the flat layer is takes the highlight, in proportion to how far
    it is from the flat toward full light, and one lit less the shadow, in
    proportion to how far toward none; the flat layer takes neither.
    ``soften`` blurs the lighting before that split, and ``contour`` (the
    gloss contour, read as :py:func:`draw_drop_shadow` reads one) maps the
    lighting first.

    Both results are ``(height, width, 1)`` over the whole canvas; the caller
    keeps the parts the style paints, inside the silhouette and beside it,
    which :py:func:`bevel_inside` and :py:func:`bevel_beside` say.
    """
    _require_scipy("Bevel effects")
    from scipy import ndimage  # type: ignore[import-untyped]  # noqa: PLC0415

    _finite(
        "Bevel",
        depth=depth,
        size=size,
        soften=soften,
        angle=angle,
        altitude=altitude,
    )
    alpha = shape[:, :, 0].astype(np.float32)
    height = _bevel_height(alpha, style, technique, max(float(size), 0.0))
    slope = max(float(depth), 0.0) / 100.0 * _BEVEL_SLOPE_PER_DEPTH
    if not up:
        slope = -slope
    gy, gx = np.gradient(height)
    nx, ny = -slope * gx, -slope * gy
    norm = np.sqrt(nx * nx + ny * ny + 1.0)
    theta, phi = math.radians(angle), math.radians(min(max(altitude, 0.0), 90.0))
    lx = math.cos(phi) * math.cos(theta)
    ly = -math.cos(phi) * math.sin(theta)
    lz = math.sin(phi)
    light = ((nx * lx + ny * ly + lz) / norm).astype(np.float32)
    soften = max(float(soften), 0.0)
    sigma = soften * _SOFTEN_SIGMA_PER_SIZE
    if sigma > 1e-3 and soften >= 1.0:
        light = ndimage.gaussian_filter(
            light, sigma, mode="nearest", truncate=soften / sigma
        )
    light = np.clip(light, 0.0, 1.0)
    if contour is not None:
        light = _contoured(light, contour, anti_aliased)
    highlight = (
        np.clip((light - lz) / (1.0 - lz), 0.0, 1.0)
        if lz < 1.0
        else np.zeros_like(light)
    )
    shadow = np.clip((lz - light) / lz, 0.0, 1.0) if lz > 0.0 else np.zeros_like(light)
    return (
        highlight[:, :, None].astype(np.float32),
        shadow[:, :, None].astype(np.float32),
    )
