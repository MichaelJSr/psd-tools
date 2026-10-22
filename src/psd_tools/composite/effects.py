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
- **Outer glow**: The layer's silhouette, spread and blurred (softer) or
  ramped by distance (precise), painted behind the layer and never under it
  - Solid color or a gradient along the falloff; honors size, spread,
    technique, range, contour and its anti-aliasing, opacity and blend mode
  - Noise and jitter are read but not applied
- **Inner shadow** and **inner glow**: the same blur, cast from the outside of
  the silhouette and painted inside the layer
  - Solid color; honors distance, angle, size, choke, contour, opacity and
    blend mode, and an inner glow's source, technique and range; noise is
    read but not applied, and a gradient glow is not drawn

- **Satin**: Two offset, blurred copies of the silhouette and their difference,
  painted inside the layer
  - Honors distance, angle, size, contour, invert, opacity and blend mode
- **Bevel and emboss**: The silhouette as a height field, lit from the angle
  and altitude (including global light)
  - Inner, outer, emboss and pillow emboss styles; smooth, chisel hard and
    chisel soft techniques; depth, direction, size, soften, gloss contour and
    the highlight and shadow colors, modes and opacities
  - Stroke emboss, the shape contour and the texture are not drawn, chisel
    soft rounds its faces by a guess, and the blurs are gaussian
    approximations of Photoshop's own

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
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from psd_tools.api.utils import AllocBudget, check_pixel_size
from psd_tools.composite import paint
from psd_tools.composite._compat import HAS_SCIPY
from psd_tools.composite.utils import divide, intersect
from psd_tools.constants import BevelStyle, BevelTechnique, ColorMode
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

# The widest stroke Photoshop writes, and the largest size, soften or satin
# distance of any effect; a larger one can only be a forged one.
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


class ShadowGeometry(NamedTuple):
    """Where a shadow falls and how far it reaches, in pixels and degrees."""

    distance: float
    angle: float
    size: float
    spread: float


def shadow_geometry(
    desc: Descriptor, angle: float, what: str = "Drop shadow"
) -> ShadowGeometry:
    """A shadow's or inner glow's geometry, as the file states it, checked once for all.

    The distance, size and spread are read off the descriptor rather than
    through :py:class:`~psd_tools.api.effects.DropShadow`, whose properties
    read a value they cannot parse as its default: to the compositor that
    value is an unreadable effect, as a stroke's size is to
    :py:func:`stroke_bbox`, and the effect is dropped rather than drawn at
    some invented size. ``angle`` is passed in, since the global light it can
    follow lives in the document and not in the descriptor. The spread is an
    inner effect's choke, which shares its key, and ``what`` names the effect
    in the error raised for a value that is not finite.

    Raises :class:`ValueError` or :class:`TypeError` for a value that will not
    parse, and :class:`ValueError` for one that is not finite: NaN passes
    through every step of :py:func:`draw_drop_shadow` without complaint, and
    one NaN in the coverage spreads to the whole canvas it is composited on,
    so this is the one place the values are checked. The size is held to what
    Photoshop can write, as a stroke's is.
    """
    distance = float(desc.get(Key.Distance, 0.0))
    size = float(desc.get(Key.Blur, 0.0))
    spread = float(desc.get(Key.ChokeMatte, 0.0))
    angle = float(angle)
    if not all(math.isfinite(value) for value in (distance, angle, size, spread)):
        raise ValueError(
            "%s parameters must be finite: distance=%r angle=%r size=%r spread=%r"
            % (what, distance, angle, size, spread)
        )
    return ShadowGeometry(distance, angle, min(size, _MAX_STROKE_SIZE), spread)


def glow_geometry(desc: Descriptor) -> tuple[float, float, float]:
    """An outer glow's ``(size, spread, range)``, as the file states them.

    Read as :py:func:`shadow_geometry` reads a shadow's, and for the same
    reason; raises the same way. The range defaults to Photoshop's when the
    descriptor leaves it out, and is held to the percentages the dialog
    offers, since the glow divides by it.
    """
    size = float(desc.get(Key.Blur, 0.0))
    spread = float(desc.get(Key.ChokeMatte, 0.0))
    range_ = float(desc.get(Key.InputRange, _DEFAULT_GLOW_RANGE))
    if not all(math.isfinite(value) for value in (size, spread, range_)):
        raise ValueError(
            "Outer glow parameters must be finite: size=%r spread=%r range=%r"
            % (size, spread, range_)
        )
    return min(size, _MAX_STROKE_SIZE), spread, min(max(range_, 1.0), 100.0)


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


def stroke_reach(desc: Descriptor) -> int | None:
    """How far a stroke's band reads from each pixel it covers, in pixels.

    The band is a function of the distance to the nearest boundary within
    its widest limit, and a partial pixel states its boundary up to half a
    pixel from its centre: coverage further away than that moves no pixel
    of the band, so the band on a box is the band on any larger box, cut.
    The two pixels on top round the half pixel out and keep clear of the
    neighbourhood a hard edge is read from.

    None for a band painted with a gradient or a pattern, which is laid out
    across the box it is drawn on and so reads all of it. Raises as
    :py:func:`stroke_bbox` does for a size it cannot read.
    """
    if _enum(desc, Key.PaintType) != Enum.SolidColor:
        return None
    limits = _BANDS.get(_enum(desc, Key.Style), _UNRECOGNISED)
    return math.ceil(max(abs(limits[0]), abs(limits[1])) * _stroke_size(desc)) + 2


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
    stating_pad: np.ndarray,
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
    stating pixel, so the working set thins fast on a smooth boundary --
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
        states = stating_pad[row, col]
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
    alpha: np.ndarray, stating: np.ndarray, near: np.ndarray, radius: float
) -> np.ndarray:
    """The nearest boundary within ``radius``, as distance, negative inside.

    Every pixel in ``stating`` states one boundary, ``near`` from its own
    centre and so up to half a pixel either side of it. A pixel that states
    none is ``min`` over those of ``|x - p| + near[p]`` -- the nearest
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
    the stating pixels would cover: a filled shape's boundary, where the band
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
    own nearest stating pixel, which is the common shape of this branch's
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
    field = np.where(stating, near, np.where(alpha >= 1, -np.inf, np.inf)).astype(
        np.float32
    )
    active = (distance_transform_edt(~stating) <= radius) & ~stating
    if not active.any():
        return field

    height, width = alpha.shape
    # An offset past the canvas's longest side reaches no pixel.
    span = math.ceil(min(radius, max(height, width) - 1))
    rows_p, cols_p = np.nonzero(stating)
    r0, r1 = max(rows_p.min() - span, 0), min(rows_p.max() + span + 1, height)
    c0, c1 = max(cols_p.min() - span, 0), min(cols_p.max() + span + 1, width)
    crop = (slice(r0, r1), slice(c0, c1))
    density = active.sum() / ((r1 - r0) * (c1 - c0))

    if density >= _DENSE_EROSION_DENSITY:
        sub_stating, sub_near, sub_alpha, sub_active = (
            stating[crop],
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
            np.where(sub_stating, sub_near, np.inf).astype(np.float32),
            cval=np.inf,
            **kwargs,
        )
        inward = -grey_erosion(
            np.where(sub_stating, -sub_near, np.inf).astype(np.float32),
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

    # stating's own pixels sit inside `crop` by construction (it is padded out
    # to `span` around their bounding box), and so does every active pixel --
    # nothing an offset lookup needs is outside it. Padding that crop, rather
    # than the whole canvas, is what keeps this branch's own memory use tied
    # to the shape's extent instead of the much bigger canvas around it.
    sub_stating, sub_near, sub_alpha, sub_active = (
        stating[crop],
        near[crop],
        alpha[crop],
        active[crop],
    )
    flat = np.flatnonzero(sub_active)
    rows, cols = np.unravel_index(flat, sub_active.shape)
    stating_pad = np.pad(sub_stating, span, constant_values=False)
    near_pad = np.pad(sub_near, span, constant_values=0.0)
    inside = (sub_alpha >= 1).ravel()[flat]

    outward = _prune_offsets(
        rows[~inside],
        cols[~inside],
        span,
        offsets,
        lengths,
        stating_pad,
        near_pad,
        outward=True,
    )
    inward = _prune_offsets(
        rows[inside],
        cols[inside],
        span,
        offsets,
        lengths,
        stating_pad,
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

    A step from opaque straight to clear states a boundary too, half a pixel
    from the centre of the pixel on either side of it, where the exact
    Euclidean distance to the iso-contour puts it. One mask can carry both --
    a glyph antialiased round its curves and stepping hard along its stems --
    and every pixel takes the nearest boundary any pixel states, of either
    kind, which :py:func:`_nearest_boundary` solves for outright
    (``effects/antialiased-stroke-edge.psd``, where a partial edge meets a
    hard one). What stays approximate is that a pixel's ``near`` is applied
    along the line to that pixel rather than along the boundary's own normal,
    which parts company with the truth around a curve the way
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
    if not partial.any():
        # ``distance_transform_edt`` measures to the nearest pixel of the other
        # class, so the two pixels straddling the boundary both come back 1.0
        # rather than the 0.5 their centres really sit at. The magnitude is
        # shrunk to put the boundary back between them; subtracting 0.5
        # outright would be right outside and a full pixel wrong inside, where
        # the sign is negative.
        inside = alpha >= 0.5
        if inside.all():
            return np.full(alpha.shape, -np.inf, dtype=np.float32)
        if not inside.any():
            return np.full(alpha.shape, np.inf, dtype=np.float32)
        base = distance_transform_edt(~inside).astype(np.float32)
        base -= distance_transform_edt(inside).astype(np.float32)
        return np.sign(base) * np.maximum(np.abs(base) - 0.5, 0.0)

    # The pixels on either side of a step from opaque straight to clear.
    hard = (alpha >= 1) & _grow(alpha <= 0)
    hard |= (alpha <= 0) & _grow(alpha >= 1)
    # What each states: a partial pixel by its coverage, a hard one at the
    # edge it shares with the other side.
    near = (0.5 - np.clip(alpha, 0.0, 1.0)).astype(np.float32)
    return _nearest_boundary(alpha, partial | hard, near, reach)


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
        color, _ = paint.draw_gradient_fill(
            viewport, psd.color_mode, desc, paint.gradient_anchor(psd, desc)
        )
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

# The same for an outer glow, fitted on its own against the glow previews,
# since Photoshop need not blur the two alike.
_GLOW_SIGMA_PER_SIZE = 0.45

# The range Photoshop gives a new glow, read where a descriptor has none.
_DEFAULT_GLOW_RANGE = 50.0

# The spread, in pixels, from which an outer glow is cast from the layer's
# coverage rather than its opacity: half a pixel, the least that rounds to a
# whole one. And what the opacity is scaled by, held to one, to read as
# coverage. Both fitted against Photoshop's renders of soft-edged layers;
# the tests pin the values.
_GLOW_SPREAD_PIXELS = 0.5
_GLOW_COVERAGE_GAIN = 20.0

# How far below one a coverage still counts as opaque when the coverage
# reading looks for an antialiased edge: one 8-bit step, so a layer painted
# a step short of solid, or a 32-bit one a rounding short, still has edges.
_OPAQUE_STEP = 1.0 / 255.0

# How far, in pixels along the axes, a partial pixel may lie from an opaque
# one and from a clear one and still read as the shape's antialiased edge:
# two, the width of the band a resampled edge is antialiased over.
_GLOW_EDGE_REACH = 2

# What a glow's range is held above, in the same percent, when its falloff is
# squeezed: a range read as the inverse of the slider would draw a 1% range as
# a hard band, and Photoshop draws a soft halo. Fitted against Photoshop's
# renders of documents outside the fixture corpus, where no glow has a range
# far from 100%; the tests pin the value and the closed form, not the fit.
_GLOW_RANGE_FLOOR = 15.0

# What one pixel of a shadow's or glow's canvas costs at the peak of
# ``draw_drop_shadow()`` or ``glow_falloff()``: the silhouette it is cast
# from, the matte spread grows and the distance transform and masks that grow
# it, the blurred or ramped matte and the shifted one, with the object read or
# group composite the silhouette is taken from. The knockout that follows a
# shadow holds less: the cut silhouette and the stroke bands pasted onto it,
# each drawn on its own stroke's box, which ``stroke_bbox()`` checks on its own.
_SHADOW_BYTES = 64

# What one pixel of a gradient glow's canvas costs while ``glow_paint()`` lays
# the gradient along the falloff: the colour lookup's index and float64 planes
# for each of up to four channels, beside the silhouette and falloff, and the
# arrays held across the call. Those planes put it above a shadow's
# (:data:`_SHADOW_BYTES`) and a gradient fill's (``paint.GRADIENT_FILL_BYTES``).
_GLOW_GRADIENT_BYTES = 192

# Entries in a contour's transfer table. Photoshop states a contour's points
# on a 0-255 grid along both axes.
_CONTOUR_STEPS = 256

# The least coverage Photoshop's 8-bit matte holds. Spread counts a pixel
# as covered from here; a composited silhouette's fainter fringe, which the
# float canvas keeps, counts as clear.
_SPREAD_FLOOR = 1.0 / 255.0


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
    dx, dy = _shadow_offset(distance, angle)
    return _cast_canvas(bbox, dx, dy, size, max_alloc_bytes, _SHADOW_BYTES)


def outer_glow_bbox(
    bbox: tuple[int, int, int, int],
    desc: Descriptor,
    max_alloc_bytes: AllocBudget | None = None,
) -> tuple[int, int, int, int]:
    """The canvas an outer glow is drawn on.

    A shadow's canvas for a shadow cast nowhere: ``bbox`` grown by the
    glow's size on every side, with the same fringe, and raising the same
    way. A gradient glow is checked against what painting its gradient peaks
    at, which is more than the falloff it is laid along. Raises as
    :py:func:`glow_geometry` does for a size it cannot read.
    """
    size, _, _ = glow_geometry(desc)
    per_pixel = _GLOW_GRADIENT_BYTES if Key.Gradient in desc else _SHADOW_BYTES
    return _cast_canvas(bbox, 0.0, 0.0, size, max_alloc_bytes, per_pixel)


def _cast_canvas(
    bbox: tuple[int, int, int, int],
    dx: float,
    dy: float,
    size: float,
    max_alloc_bytes: AllocBudget | None,
    per_pixel: int,
) -> tuple[int, int, int, int]:
    """``bbox`` grown as :py:func:`drop_shadow_bbox` says, at ``per_pixel`` bytes."""
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        return bbox
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
        estimated_bytes=width * height * per_pixel,
        warn=False,
    )
    return grown


def cast_reach(distance: float, angle: float, size: float, spread: float) -> int | None:
    """How far a cast matte reads from each pixel it covers, in pixels.

    The blur's radius and the offset, each rounded out, with the pixel the
    offset interpolates across and the fringe the blur's rounding leaves:
    the silhouette further away than that moves no pixel of the matte, so
    the matte on a box is the matte on any larger box, cut.

    None when the matte is spread; such a matte is drawn on its whole box.
    A spread of a size-0 matte grows nothing.
    """
    if size > 0.0 and spread > 0.0:
        return None
    dx, dy = _shadow_offset(distance, angle)
    return math.ceil(max(size, 0.0)) + math.ceil(max(abs(dx), abs(dy))) + 3


def glow_reach(
    size: float, spread: float, precise: bool, *, outer: bool = True
) -> int | None:
    """How far a glow's falloff reads from each pixel, as :py:func:`cast_reach`.

    A precise glow ramps along the distance to the silhouette and is never
    spread, so it is bounded whatever its spread; when an ``outer`` glow's
    spread reads the coverage (:py:func:`_glow_coverage`), each pixel of the
    silhouette also reads :data:`_GLOW_EDGE_REACH` around it. An inner glow's
    choke does not read it.
    """
    if not precise and size > 0.0 and spread > 0.0:
        return None
    reach = math.ceil(max(size, 0.0)) + 3
    if outer and _reads_coverage(size, spread):
        reach += _GLOW_EDGE_REACH
    return reach


def contour_table(desc: Descriptor | None) -> np.ndarray | None:
    """The transfer table of a contour descriptor, or ``None`` for the identity.

    A contour maps the coverage of a blurred matte to the coverage painted,
    along a curve through the points the descriptor lists. Photoshop draws a
    smooth curve through them, turning sharply at a point marked as a
    corner; this is a monotone cubic through each run of points between two
    corners, which neither overshoots a point nor turns back between two, and
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
    # Photoshop marks a corner point with continuity off.
    smooth = [bool(point.get(Key.Continuity, True)) for point in points]
    order = np.argsort(xs, kind="stable")
    xs, ys = xs[order], ys[order]
    smooth = [smooth[index] for index in order]
    if len(xs) == 2 and np.array_equal(xs, ys) and xs[0] == 0.0 and xs[1] == 1.0:
        return None
    grid = np.linspace(0.0, 1.0, _CONTOUR_STEPS)
    # Straight segments, which hold the end values past the ends on their
    # own; each run of three or more points is then curved over its span.
    table = np.interp(grid, xs, ys)
    curved = [(a, b) for a, b in _contour_runs(xs, smooth) if b - a >= 2]
    if curved:
        _require_scipy("Layer effect contours")
        from scipy.interpolate import PchipInterpolator  # type: ignore[import-untyped]  # noqa: PLC0415

        for a, b in curved:
            span = (grid >= xs[a]) & (grid <= xs[b])
            table[span] = PchipInterpolator(xs[a : b + 1], ys[a : b + 1])(grid[span])
    return np.clip(table, 0.0, 1.0).astype(np.float32)


def _contour_runs(xs: np.ndarray, smooth: list[bool]) -> list[tuple[int, int]]:
    """The first and last index of each run of a contour's sorted points.

    A run ends at a corner, which the next run starts from, and before a
    step, two points at one input that no spline can pass through.
    """
    runs = []
    start = 0
    for index in range(1, len(xs)):
        if xs[index] == xs[index - 1]:
            runs.append((start, index - 1))
            start = index
        elif not smooth[index] and index < len(xs) - 1:
            runs.append((start, index))
            start = index
    runs.append((start, len(xs) - 1))
    return runs


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


def _spread_pixels(size: float, spread: float) -> int:
    """How many whole pixels of ``size`` a ``spread`` percentage grows a matte by.

    Photoshop dilates by whole pixels: a spread that rounds to none leaves
    the matte to the blur, so a feather stays soft under it.
    """
    size = max(float(size), 0.0)
    return math.floor(size * min(max(float(spread), 0.0), 100.0) / 100.0 + 0.5)


def _spread(alpha: np.ndarray, pixels: int) -> np.ndarray:
    """``alpha`` grown by ``pixels`` whole pixels from wherever it is covered.

    Photoshop spreads the matte's coverage, not its coverage levels: every
    pixel covered by at least :data:`_SPREAD_FLOOR` counts as covered, the
    matte is opaque over all of them and out to the grown edge, and the
    edge is antialiased over the pixel it crosses. So a feathered edge is
    cast from its outermost fringe and comes out hard, and a half-covered
    silhouette casts an opaque matte.

    The edge is grown from where the coverage ends, read off the outermost
    covered pixels: a pixel's coverage reaches that share of the way across
    it, so a fully covered pixel's edge is half a pixel past its centre, as
    :py:func:`_signed_distance` puts a hard edge, and a feather's faint
    fringe ends about half a pixel short of its centre.
    """
    from scipy.ndimage import distance_transform_edt  # type: ignore[import-untyped]  # noqa: PLC0415

    # A hair under the floor: the complement of an 8-bit level, taken in
    # float32, lands an ulp short of it.
    inside = alpha >= _SPREAD_FLOOR * (1.0 - 1e-6)
    if inside.all():
        return np.ones(alpha.shape, dtype=np.float32)
    if not inside.any():
        return np.zeros(alpha.shape, dtype=np.float32)
    outside, nearest = distance_transform_edt(~inside, return_indices=True)
    edge = alpha[nearest[0], nearest[1]]
    grown = np.clip(pixels + edge - outside, 0.0, 1.0)
    return np.where(inside, np.float32(1.0), grown).astype(np.float32)


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

    ``spread`` and ``size`` make the matte as :py:func:`_soft_matte` says.
    (The spread key shrinks the silhouette of an inner shadow, where the
    dialog calls it choke.) The offset treats what lies past the canvas as
    empty, as the blur does, since nothing outside it casts a shadow.

    A parameter that is not finite raises ``ValueError``; the file's values
    are checked once in :py:func:`shadow_geometry` before they get here.
    """
    _require_scipy("Drop shadow effects")
    alpha = _cast_matte(
        shape[:, :, 0].astype(np.float32),
        distance=distance,
        angle=angle,
        size=size,
        spread=spread,
        outside=0.0,
        what="Drop shadow",
    )
    if contour is not None:
        alpha = _contoured(alpha, contour, anti_aliased)
    return alpha[:, :, None]


def _cast_matte(
    alpha: np.ndarray,
    *,
    distance: float,
    angle: float,
    size: float,
    spread: float,
    outside: float,
    what: str,
) -> np.ndarray:
    """``alpha`` spread, blurred and offset, as a shadow casts it.

    ``outside`` is the matte past the canvas: 0 for a matte drawn from the
    silhouette, 1 for one drawn from its complement. Raises ``ValueError``
    for a parameter that is not finite, naming ``what``.
    """
    from scipy import ndimage  # type: ignore[import-untyped]  # noqa: PLC0415

    if not all(math.isfinite(value) for value in (distance, angle, size, spread)):
        raise ValueError(
            "%s parameters must be finite: distance=%r angle=%r size=%r spread=%r"
            % (what, distance, angle, size, spread)
        )
    alpha = _soft_matte(alpha, size, spread, _SHADOW_SIGMA_PER_SIZE, outside=outside)
    dx, dy = _shadow_offset(distance, angle)
    if dx or dy:
        alpha = ndimage.shift(alpha, (dy, dx), order=1, mode="constant", cval=outside)
    return np.clip(alpha, 0.0, 1.0)


def draw_inner_shadow(
    viewport: tuple[int, int, int, int],
    shape: np.ndarray,
    *,
    distance: float,
    angle: float,
    size: float,
    choke: float,
    contour: np.ndarray | None = None,
    anti_aliased: bool = False,
) -> np.ndarray:
    """The coverage an inner shadow casts inside ``shape``, on ``viewport``.

    The shadow :py:func:`draw_drop_shadow` casts from the complement of
    ``shape``, cut to ``shape``: the outside falls into the layer from the
    side the light comes from. ``choke`` is a percentage of ``size``, as a
    drop shadow's spread is, and shrinks the silhouette before the blur.
    The canvas past ``viewport`` counts as outside the layer. The result is
    ``(height, width, 1)`` and never exceeds ``shape``.
    """
    _require_scipy("Inner shadow effects")
    alpha = shape[:, :, 0].astype(np.float32)
    matte = _cast_matte(
        1.0 - alpha,
        distance=distance,
        angle=angle,
        size=size,
        spread=choke,
        outside=1.0,
        what="Inner shadow",
    )
    if contour is not None:
        matte = _contoured(matte, contour, anti_aliased)
    return (matte * alpha)[:, :, None]


def _soft_matte(
    alpha: np.ndarray,
    size: float,
    spread: float,
    sigma_per_size: float,
    outside: float = 0.0,
) -> np.ndarray:
    """``alpha`` grown by the spread and blurred by what is left of the size.

    ``spread`` is a percentage of ``size``, which is how Photoshop's dialog
    states it, even though the descriptor labels the value in pixels. Its
    whole pixels (:py:func:`_spread_pixels`) *grow* the silhouette's coverage
    before the blur (:py:func:`_spread`), and the blur takes what is left
    of ``size``, so the matte turns hard-edged at 100%, and a feathered
    edge is hardened by a spread of a pixel.

    The blur is cut off at that remaining radius, rounded to a whole pixel,
    so the matte reaches at most a pixel or two past ``size`` from the
    silhouette, and the canvases the bbox functions ask for hold all of it;
    a gaussian left to its default reach runs on past both. It treats what
    lies past the canvas as ``outside``: empty for a matte drawn from the
    silhouette, since nothing outside it casts anything, full for one drawn
    from its complement. The default for a blur is to reflect the edge
    instead, which doubles back any coverage that reaches it.
    """
    from scipy import ndimage  # type: ignore[import-untyped]  # noqa: PLC0415

    alpha = alpha.astype(np.float32)
    size = max(float(size), 0.0)
    pixels = _spread_pixels(size, spread)
    if pixels > 0:
        alpha = _spread(alpha, pixels)
    radius = max(size - pixels, 0.0)
    sigma = radius * sigma_per_size
    if sigma > 1e-3:
        alpha = ndimage.gaussian_filter(
            alpha, sigma, mode="constant", cval=outside, truncate=radius / sigma
        )
    return alpha


def _precise_matte(alpha: np.ndarray, size: float, spread: float) -> np.ndarray:
    """``alpha`` ramped down with the distance from its boundary.

    Photoshop's precise technique: full out to the spread, then a straight
    ramp to nothing at ``size``, so the glow follows the silhouette's
    corners as a stroke does rather than rounding them off as a blur does.
    A ramp shorter than a pixel is drawn as an antialiased hard edge, the
    way :py:func:`_distance_band` draws one.
    """
    size = max(float(size), 0.0)
    grow = size * min(max(float(spread), 0.0), 100.0) / 100.0
    distance = _signed_distance(alpha, size + 1.0)
    ramp = size - grow
    if ramp < 1.0:
        return np.clip(size + 0.5 - distance, 0.0, 1.0).astype(np.float32)
    return np.clip((size - distance) / ramp, 0.0, 1.0).astype(np.float32)


def _ranged(matte: np.ndarray, range_: float) -> np.ndarray:
    """``matte`` scaled up by the inverse of its range and held to one.

    Photoshop's range picks the part of the falloff the contour is laid over.
    The inverse is taken over the range and :data:`_GLOW_RANGE_FLOOR`
    together, so a small range squeezes no harder than the floor does, and
    100% is the matte itself.
    """
    scale = math.hypot(100.0, _GLOW_RANGE_FLOOR) / math.hypot(range_, _GLOW_RANGE_FLOOR)
    return np.clip(matte * scale, 0.0, 1.0).astype(np.float32)


def _reads_coverage(size: float, spread: float) -> bool:
    """Whether a glow of ``size`` and ``spread`` is cast from the coverage."""
    grow = max(float(size), 0.0) * min(max(float(spread), 0.0), 100.0) / 100.0
    return grow >= _GLOW_SPREAD_PIXELS


def _glow_coverage(alpha: np.ndarray, size: float, spread: float) -> np.ndarray:
    """What a glow of ``size`` and ``spread`` is cast from.

    A glow with no spread is cast from the layer's opacity. One spread by
    half a pixel or more (:data:`_GLOW_SPREAD_PIXELS`) is cast from its
    coverage: the opacity scaled by :data:`_GLOW_COVERAGE_GAIN` and held to
    one, so a soft-edged or faint layer glows as the shape it fades out to
    and only its faintest pixels cast less than a solid one. A partial pixel
    within :data:`_GLOW_EDGE_REACH` of both a clear pixel and an opaque one
    is a hard shape's antialiased edge and keeps its opacity, so text and a
    resampled shape cast from exactly where their edges lie; the neighbours
    are read along the axes, as :py:func:`_signed_distance` reads a hard
    step. Photoshop's renders of such layers split the same way.
    """
    alpha = alpha.astype(np.float32)
    if not _reads_coverage(size, spread):
        return alpha
    covered = alpha > 0.0
    near_clear, near_opaque = ~covered, alpha >= 1.0 - _OPAQUE_STEP
    for _ in range(_GLOW_EDGE_REACH):
        near_clear, near_opaque = _grow(near_clear), _grow(near_opaque)
    edge = covered & near_clear & near_opaque
    coverage = np.clip(alpha * _GLOW_COVERAGE_GAIN, 0.0, 1.0)
    return np.where(edge, alpha, coverage).astype(np.float32)


def glow_falloff(
    shape: np.ndarray,
    *,
    size: float,
    spread: float,
    range_: float = _DEFAULT_GLOW_RANGE,
    precise: bool = False,
    contour: np.ndarray | None = None,
    anti_aliased: bool = False,
) -> np.ndarray:
    """How an outer glow falls off from ``shape``.

    ``shape`` is the layer's silhouette on the canvas
    :py:func:`outer_glow_bbox` sizes; the result is ``(height, width)`` on
    it, one at the silhouette's edge and dying away over ``size`` pixels: a
    shadow's matte cast nowhere, squeezed by the range (:py:func:`_ranged`)
    and then mapped by the contour. A glow spread by half a pixel or more is
    cast from the silhouette's coverage rather than its opacity, so a soft
    edge glows as the shape it fades out to, while a hard shape's
    antialiased edge keeps its opacity. It is not cut under the silhouette.

    ``precise`` ramps the matte by its distance from the silhouette instead
    of blurring it (:py:func:`_precise_matte`). A size of zero draws nothing
    either way.

    A parameter that is not finite raises ``ValueError``, for the reason
    :py:func:`draw_drop_shadow` gives.
    """
    _require_scipy("Outer glow effects")
    if not all(math.isfinite(value) for value in (size, spread, range_)):
        raise ValueError(
            "Outer glow parameters must be finite: size=%r spread=%r range=%r"
            % (size, spread, range_)
        )
    if size <= 0.0:
        return np.zeros(shape.shape[:2], dtype=np.float32)
    alpha = _glow_coverage(shape[:, :, 0], size, spread)
    if not alpha.any():
        return np.zeros(alpha.shape, dtype=np.float32)
    if precise:
        matte = _precise_matte(alpha, size, spread)
    else:
        matte = _soft_matte(alpha, size, spread, _GLOW_SIGMA_PER_SIZE)
    falloff = _ranged(np.clip(matte, 0.0, 1.0), min(max(range_, 1.0), 100.0))
    if contour is not None:
        falloff = _contoured(falloff, contour, anti_aliased)
    return np.clip(falloff, 0.0, 1.0).astype(np.float32)


def _beside(coverage: np.ndarray, shape: np.ndarray, over: float = 0.0) -> np.ndarray:
    """``coverage`` cut away under the silhouette, as ``(height, width, 1)``.

    The glow has what the silhouette leaves of each pixel, ``1 - shape``,
    whatever the layer's own opacity: Photoshop never shows an outer glow
    through its layer. ``over`` is how much of its silhouette a Normal layer
    paints when it is composited on top, its fill opacity and opacity
    together. That compositing keeps ``1 - shape * over`` of what lies
    beneath, so the glow goes on at the ratio of the two and shows at
    ``1 - shape`` once the layer is on, rather than faded by the layer's
    edge twice. At zero it is the plain complement, for a layer that paints
    nothing or blends in another mode.
    """
    silhouette = shape[:, :, 0]
    beside = 1.0 - silhouette
    if over > 0.0:
        under = 1.0 - silhouette * over
        beside = np.divide(beside, under, out=np.zeros_like(beside), where=under > 0.0)
    return np.clip(coverage * beside, 0.0, 1.0).astype(np.float32)[:, :, None]


def glow_paint(
    viewport: tuple[int, int, int, int],
    color_mode: ColorMode,
    desc: Descriptor,
    shape: np.ndarray,
    falloff: np.ndarray,
    over: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """What an outer glow is painted with, on ``viewport``: its colour and coverage.

    ``shape`` is the silhouette and ``falloff`` what :py:func:`glow_falloff`
    drew from it. A solid colour is the descriptor's, covering the falloff.
    A gradient is laid along the falloff instead: its start at the
    silhouette's edge, where the falloff is full, and its end where the
    falloff dies away, so the opaque end of Photoshop's default glow
    gradient sits against the layer. Its transparency stops, read at the
    same place as its colour stops, hold the coverage under them: the glow
    is as opaque as the falloff or the transparency there, whichever is
    less, so a gradient opaque at both ends paints what a solid colour would
    and a translucent stop holds the glow at its opacity for as long as the
    falloff is above it. The gradient's reverse flag turns it around, and its
    stops blend in the method ``gs99`` names, as an overlay's do. Either way
    the coverage is cut away under the silhouette. ``over`` is how much of
    the silhouette a Normal layer composited on top paints, its fill opacity
    and opacity together; the coverage under it is raised so the glow shows
    at ``1 - shape`` once the layer is on. Zero leaves the plain complement.
    A descriptor with neither raises as the solid colour reader does.
    """
    if Key.Gradient not in desc:
        color, _ = paint.draw_solid_color_fill(viewport, color_mode, desc)
        return color, _beside(falloff, shape, over)
    along = 1.0 - falloff
    if bool(desc.get(Key.Reverse, False)):
        along = 1.0 - along
    colors, opacities = paint._make_gradient_color(
        color_mode, desc[Key.Gradient], paint._gradient_method(desc)
    )
    if colors is None:
        raise ValueError("Outer glow gradient has no colour stops.")
    color = np.asarray(colors(along), dtype=np.float32)
    coverage = falloff
    if opacities is not None:
        coverage = np.minimum(coverage, np.asarray(opacities(along), dtype=np.float32))
    return color, _beside(coverage, shape, over)


def _glow_range(matte: np.ndarray, glow_range: float) -> np.ndarray:
    """``matte`` with a glow's range applied, before its contour."""
    share = min(max(float(glow_range), 1.0), 100.0) / 100.0
    return np.clip(matte / share, 0.0, 1.0)


def draw_inner_glow(
    viewport: tuple[int, int, int, int],
    shape: np.ndarray,
    *,
    size: float,
    choke: float,
    center: bool = False,
    precise: bool = False,
    glow_range: float = 50.0,
    contour: np.ndarray | None = None,
    anti_aliased: bool = False,
) -> np.ndarray:
    """The coverage an inner glow puts inside ``shape``, on ``viewport``.

    From the edge, the glow is full at the boundary and fades over ``size``
    inward; from the ``center``, it is the complement, full inside and fading
    out towards the edge. ``choke`` is a percentage of ``size`` held at full
    strength before the fade. The softer technique fades along the blur
    :py:func:`draw_inner_shadow` uses, the ``precise`` one linearly with the
    distance to the edge, read at each pixel's centre. ``glow_range`` is the
    percentage of the fade the contour is spread over. The canvas past
    ``viewport`` counts as outside the layer under either technique. The
    result is ``(height, width, 1)`` and never exceeds ``shape``.
    """
    _require_scipy("Inner glow effects")
    if not all(math.isfinite(value) for value in (size, choke, glow_range)):
        raise ValueError(
            "Inner glow parameters must be finite: size=%r choke=%r range=%r"
            % (size, choke, glow_range)
        )
    alpha = shape[:, :, 0].astype(np.float32)
    size = max(float(size), 0.0)
    if precise:
        grow = size * min(max(float(choke), 0.0), 100.0) / 100.0
        # A ring of clear pixels puts an edge at the canvas, as the blur's
        # ``outside`` does for the softer technique.
        depth = -_signed_distance(np.pad(alpha, 1), size + 1.0)[1:-1, 1:-1]
        fade = max(size - grow, 1e-3)
        matte = np.clip((size - depth) / fade, 0.0, 1.0).astype(np.float32)
    else:
        matte = _cast_matte(
            1.0 - alpha,
            distance=0.0,
            angle=0.0,
            size=size,
            spread=choke,
            outside=1.0,
            what="Inner glow",
        )
    if center:
        matte = 1.0 - matte
    matte = _glow_range(matte, glow_range)
    if contour is not None:
        matte = _contoured(matte, contour, anti_aliased)
    return (matte * alpha)[:, :, None]


# How wide a gaussian stands in for the blur of a satin, and for the blur
# that rounds a smooth bevel, per pixel of the effect's size; how far a
# bevel's soften blurs its shading per pixel of soften; and how steep a
# bevel's faces are at a depth of 100%. Fitted against Photoshop's previews,
# and asserted in the tests.
_SATIN_SIGMA_PER_SIZE = 0.45
_BEVEL_SIGMA_PER_SIZE = 0.45
_SOFTEN_SIGMA_PER_SIZE = 0.45
_BEVEL_SLOPE_PER_DEPTH = 1.0

# How high a bevel's texture stands at a texture depth of 100%, per pixel of
# the effect's size, and how a slope along one axis is smoothed along the
# other (a Sobel operator). Fitted against a Photoshop export of a textured
# bevel, not a preview fixture.
_TEXTURE_HEIGHT_PER_DEPTH = 1.0
_SLOPE_ACROSS = (0.25, 0.5, 0.25)

# How wide the gaussian that rounds a chisel soft bevel's faces is, per pixel
# of size, and how far it reads, in sigmas. Not fitted: no fixture preview
# carries the technique. The reach is what ``bevel_reach`` adds for it.
_SLOPE_LIMIT_SIGMA_PER_SIZE = 0.125
_SLOPE_LIMIT_TRUNCATE = 4.0

# What a pixel of a satin's or a bevel's canvas costs at its peak: the
# silhouette, two shifted and blurred mattes or a height field and its two
# gradients, and the float64 transients of the filters.
_INNER_BYTES = 64

# The deepest bevel Photoshop writes, in percent, and the deepest texture
# either way.
_MAX_BEVEL_DEPTH = 1000.0

# What a satin's distance and size, and a bevel's depth and size, read as
# for a descriptor that leaves them out: the same values as the API getters.
_SATIN_DEFAULT = 120.0
_BEVEL_DEFAULT = 0.0

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
    within: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    """The canvas a satin or a bevel is drawn on: ``bbox`` grown by ``reach``.

    The silhouette is read on this box, so that a blur or an offset near the
    viewport's edge sees the layer past it rather than an edge that is not
    there. The two pixels added on top are the blur's antialiased fringe and
    the offset's interpolation. ``within``, when given, holds the box to no
    further than the same reach past it, since nothing beyond that reaches
    back into it. The box that is left is checked against the per-axis limit
    and ``max_alloc_bytes`` as :py:func:`drop_shadow_bbox` is, raising
    :class:`ValueError`. An empty ``bbox`` is returned untouched, and an empty
    intersection with ``within`` comes back as ``(0, 0, 0, 0)``.
    """
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        return bbox
    margin = math.ceil(max(reach, 0.0)) + 2
    grown = (bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin)
    if within is not None:
        grown = intersect(
            grown,
            (
                within[0] - margin,
                within[1] - margin,
                within[2] + margin,
                within[3] + margin,
            ),
        )
        if grown == (0, 0, 0, 0):
            return grown
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
    """A satin's ``(distance, size)``, read as :py:func:`shadow_geometry` reads.

    Both are held to what Photoshop can write, the distance keeping its sign.
    """
    distance = float(desc.get(Key.Distance, _SATIN_DEFAULT))
    size = float(desc.get(Key.Blur, _SATIN_DEFAULT))
    _finite("Satin", distance=distance, size=size)
    return (
        math.copysign(min(abs(distance), _MAX_STROKE_SIZE), distance),
        min(max(size, 0.0), _MAX_STROKE_SIZE),
    )


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
    more and to what Photoshop can write.
    """
    depth = float(desc.get(Key.StrengthRatio, _BEVEL_DEFAULT))
    size = float(desc.get(Key.Blur, _BEVEL_DEFAULT))
    soften = float(desc.get(Key.Softness, 0.0))
    _finite("Bevel", depth=depth, size=size, soften=soften)
    return (
        min(max(depth, 0.0), _MAX_BEVEL_DEPTH),
        min(max(size, 0.0), _MAX_STROKE_SIZE),
        min(max(soften, 0.0), _MAX_STROKE_SIZE),
    )


def bevel_reach(size: float, soften: float, technique: bytes) -> float:
    """How far a bevel's canvas has to reach past the layer.

    The widest band beside it, the blurs that read across the edge, and for
    a chisel soft bevel the rounding that reads past the band.
    """
    reach = size + soften
    if technique == BevelTechnique.SLOPE_LIMIT_MATTE:
        reach += size * _SLOPE_LIMIT_SIGMA_PER_SIZE * _SLOPE_LIMIT_TRUNCATE
    return reach


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
            height = ndimage.gaussian_filter(
                height, sigma, mode="nearest", truncate=_SLOPE_LIMIT_TRUNCATE
            )
    return height


def texture_height(
    luminance: np.ndarray, *, size: float, depth: float, inverted: bool
) -> np.ndarray:
    """A bevel texture's surface, in pixels of height, from the pattern's luminance.

    The pattern's dark pixels stand up and its light ones sink, the other
    way round when ``inverted``; at a texture ``depth`` of 100% the darkest
    pixel is as high as the bevel's band is wide (``size``), and the depth
    scales that, negative depths turning the texture over. A depth past what
    Photoshop writes is held to it.
    """
    relief = luminance if inverted else 1.0 - luminance
    depth = float(np.clip(depth, -_MAX_BEVEL_DEPTH, _MAX_BEVEL_DEPTH))
    scale = max(float(size), 0.0) * depth / 100.0 * _TEXTURE_HEIGHT_PER_DEPTH
    return (relief * scale).astype(np.float32)


def _slopes(height: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The surface's slope down and across, as a Sobel operator takes it.

    A central difference along each axis, smoothed by ``_SLOPE_ACROSS``
    along the other; the edge pixels repeat past the border.
    """
    height = height.astype(np.float32, copy=False)
    if not height.size:
        return np.zeros_like(height), np.zeros_like(height)
    padded = np.pad(height, 1, mode="edge")
    half = np.float32(0.5)
    dy = (padded[2:, :] - padded[:-2, :]) * half
    dx = (padded[:, 2:] - padded[:, :-2]) * half
    before, middle, after = (np.float32(weight) for weight in _SLOPE_ACROSS)
    gy = dy[:, :-2] * before + dy[:, 1:-1] * middle + dy[:, 2:] * after
    gx = dx[:-2, :] * before + dx[1:-1, :] * middle + dx[2:, :] * after
    return gy, gx


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
    texture: np.ndarray | None = None,
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
    lighting first, the flat layer's with it, so a flat face takes neither
    under any contour. The slope is a Sobel operator's, fitted to one
    Photoshop export: a one pixel step lights the two pixels astride it.

    ``texture`` (from :py:func:`texture_height`, on ``viewport``) is laid on
    the surface before it is lit, so the bevel's faces and the flat layer
    alike carry its bumps.

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
    if texture is not None:
        _finite("Bevel", texture=float(np.abs(texture).max()) if texture.size else 0.0)
        height = height + texture.reshape(height.shape).astype(np.float32)
    slope = max(float(depth), 0.0) / 100.0 * _BEVEL_SLOPE_PER_DEPTH
    if not up:
        slope = -slope
    gy, gx = _slopes(height)
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
    flat = lz
    if contour is not None:
        light = _contoured(light, contour, anti_aliased)
        flat = float(
            _contoured(np.array([lz], dtype=np.float32), contour, anti_aliased)[0]
        )
    highlight = (
        np.clip((light - flat) / (1.0 - flat), 0.0, 1.0)
        if flat < 1.0
        else np.zeros_like(light)
    )
    shadow = (
        np.clip((flat - light) / flat, 0.0, 1.0) if flat > 0.0 else np.zeros_like(light)
    )
    return (
        highlight[:, :, None].astype(np.float32),
        shadow[:, :, None].astype(np.float32),
    )
