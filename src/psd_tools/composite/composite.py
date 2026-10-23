"""Composite implementation for layer rendering and blending."""

import functools
import hashlib
import logging
import math
import warnings
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple, Protocol, cast

import numpy as np
from PIL import Image

from psd_tools.api import numpy_io, pil_io
from psd_tools.api._descriptor import get_blend_mode
from psd_tools.api.effects import (
    BevelEmboss,
    DropShadow,
    InnerGlow,
    InnerShadow,
    OuterGlow,
    Satin,
    Stroke,
)
from psd_tools.api.layers import (
    AdjustmentLayer,
    Artboard,
    FillLayer,
    GroupMixin,
    Layer,
    PixelLayer,
    ShapeLayer,
    SmartObjectLayer,
    TypeLayer,
)
from psd_tools.api.protocols import LayerProtocol, PSDProtocol
from psd_tools.api.psd_image import PSDImage
from psd_tools.api.utils import (
    UNLIMITED,
    check_growth,
    check_pixel_size,
    get_color_channels,
    get_transparency_index,
    has_transparency,
)
from psd_tools.composite import paint, rcache, utils, vector
from psd_tools.composite._compat import HAS_SCIPY
from psd_tools.composite.adjustments import _BT601_LUMINANCE, ADJUSTMENT_FUNC
from psd_tools.composite.blend import fade_source, get_blend_func, normal
from psd_tools.composite.effects import (
    BevelShape,
    bevel_beside,
    bevel_geometry,
    bevel_inside,
    bevel_reach,
    bevel_smooth_inner,
    cast_reach,
    contour_table,
    draw_bevel,
    draw_drop_shadow,
    draw_inner_glow,
    draw_inner_shadow,
    draw_satin,
    draw_stroke_effect_split,
    drop_shadow_bbox,
    glow_falloff,
    glow_geometry,
    glow_paint,
    glow_reach,
    inner_bbox,
    outer_glow_bbox,
    satin_geometry,
    shadow_geometry,
    stroke_band,
    stroke_bbox,
    stroke_reach,
    texture_height,
    uncovered,
)
from psd_tools.composite.text_blend import (
    TEXT_BLEND_GAMMA,
    from_text_space,
    to_text_space,
)
from psd_tools.composite.widen import make_widen
from psd_tools.compression import _decompress
from psd_tools.constants import (
    BevelDirection,
    BevelStyle,
    BevelTechnique,
    BlendMode,
    ChannelID,
    ColorMode,
    GlowSource,
    GlowTechnique,
    Knockout,
    Resource,
    StrokeAlignment,
    StrokePosition,
    Tag,
)
from psd_tools.psd.descriptor import Descriptor
from psd_tools.terminology import Key

logger = logging.getLogger(__name__)


class _StyledEffect(Protocol):
    """What the compositor needs from an overlay or stroke effect.

    ``Effects.find()`` looks effects up by name at runtime and is typed as
    returning the base ``_Effect``, which does not declare ``blend_mode`` --
    that lives on the mixins the concrete classes bring in. Every name looked
    up in this module resolves to a class that has one, so the iterators are
    narrowed to this rather than each use being cast separately.
    """

    @property
    def descriptor(self) -> Descriptor: ...

    @property
    def opacity(self) -> float: ...

    @property
    def blend_mode(self) -> BlendMode: ...


def _styled(effects: Iterator[Any]) -> Iterator[_StyledEffect]:
    return cast(Iterator[_StyledEffect], effects)


# How many times the larger of the viewport and the layer a vector stroke's
# fill, or a layer effect's canvas, may grow to, past the floor check_growth()
# allows.
_CANVAS_GROWTH = 4


# The layer modes whose compositing keeps ``1 - alpha`` of the backdrop, which
# an outer glow's cut under a partial pixel relies on. Dissolve is drawn as
# Normal (``blend.dissolve``).
_COVERS_BENEATH = frozenset({BlendMode.NORMAL, BlendMode.DISSOLVE})

# The channel value each mode blends to itself whatever lies beneath.
_FIXED_POINTS = {BlendMode.MULTIPLY: 0.0, BlendMode.SCREEN: 1.0}


# The levels of an 8-bit document's alpha.
_LEVELS_8 = 255.0


def _held(alpha: np.ndarray, depth: int) -> np.ndarray:
    """``alpha`` rounded to the levels of a ``depth``-bit document.

    Photoshop composites a group into a buffer of the document's depth before
    the group's style reads it, so a child's effect that trails off below
    half a level, such as a soft shadow's tail, is not there to be traced:
    on the float canvas it states a boundary (``_signed_distance()``) and
    moves a stroke out by up to a pixel. Only 8-bit documents are rounded.
    """
    if depth != 8:
        return alpha
    return (np.floor(alpha * _LEVELS_8 + 0.5) / _LEVELS_8).astype(np.float32)


def _covers_beneath(layer: Layer, shown: Callable[[Layer], bool] | None) -> bool:
    """Whether compositing the layer keeps ``1 - alpha`` of what lies beneath.

    A pass-through group puts each child on the backdrop in the child's own
    mode, and the effects beside the child in theirs, so it covers that much
    only when every child it shows, as ``shown`` (the compositor's layer
    filter) picks them, does, and every effect beside one does
    (:py:func:`_beside_covers`). A clipping child paints onto the layer it
    clips to and never reaches the backdrop, and an adjustment layer paints
    nothing.
    """
    if isinstance(layer, AdjustmentLayer):
        return False
    if layer.blend_mode != BlendMode.PASS_THROUGH:
        return layer.blend_mode in _COVERS_BENEATH
    if not layer.is_group():
        return True  # blended as Normal (``blend.get_blend_func``)
    return all(
        _covers_beneath(child, shown) and _beside_covers(child)
        for child in cast(GroupMixin, layer)
        if (shown is None or shown(child)) and not child.clipping
    )


def _beside_covers(layer: Layer) -> bool:
    """Whether what the layer paints beside itself covers.

    That is its drop shadows, outer glows, outer strokes and the bevels that
    reach outside it (:py:func:`_beside_paints`). Each keeps ``1 - alpha``
    of what it lands on in Normal or Dissolve, and as a solid colour its
    mode paints whatever it lands on: every channel 0 in Multiply, 1 in
    Screen, as RGB or grayscale black and white are. A CMYK or Lab black
    does not: some of its channels blend with what it lands on. One at 0%
    paints nothing. An effect that cannot be read does not cover.
    """
    color_mode = layer._psd.color_mode
    for name in ("dropshadow", "outerglow", "stroke", "bevelemboss"):
        for effect in _readable(layer, name):
            try:
                for blend_mode, opacity, desc in _beside_paints(name, effect):
                    if not opacity > 0.0 or blend_mode in _COVERS_BENEATH:
                        continue
                    if (
                        blend_mode not in _FIXED_POINTS
                        or Key.Gradient in desc
                        or Key.Pattern in desc
                        or any(
                            value != _FIXED_POINTS[blend_mode]
                            for value in paint._get_color(color_mode, desc)
                        )
                    ):
                        return False
            except _UNREADABLE as error:
                logger.debug("Cannot read an effect of %s: %s", layer, error)
                return False
    return True


def _beside_paints(
    name: str, effect: _StyledEffect
) -> list[tuple[BlendMode, float, Descriptor]]:
    """The mode, opacity and paint descriptor of each colour an effect puts beside.

    A bevel puts its highlight and its shadow there when its style reaches
    outside the layer, and an inside stroke or an inner bevel nothing: they
    paint over the layer.
    """
    if name == "bevelemboss":
        bevel = cast(BevelEmboss, effect)
        if not bevel_beside(_bevel_style(bevel)):
            return []
        halves = []
        for color, blend_mode, opacity in (
            (bevel.highlight_color, bevel.highlight_mode, bevel.highlight_opacity),
            (bevel.shadow_color, bevel.shadow_mode, bevel.shadow_opacity),
        ):
            desc = Descriptor()
            desc[Key.Color] = color
            halves.append((blend_mode, opacity, desc))
        return halves
    if name == "stroke" and cast(Stroke, effect).position == StrokePosition.INSIDE:
        return []
    if name == "dropshadow":
        blend_mode = get_blend_mode(effect.descriptor, Key.Mode, BlendMode.MULTIPLY)
    else:
        blend_mode = effect.blend_mode
    return [(blend_mode, effect.opacity, effect.descriptor)]


def _stroke_paint(effect: _StyledEffect) -> float:
    """How much of its band a stroke effect paints over what lies beneath.

    Its opacity in Normal or Dissolve; nothing in another mode, which lets
    the backdrop through, or when it cannot be read.
    """
    try:
        if effect.blend_mode not in _COVERS_BENEATH:
            return 0.0
        share = effect.opacity / 100.0
    except _UNREADABLE:
        return 0.0
    return min(max(share, 0.0), 1.0) if math.isfinite(share) else 0.0


def _pixels(bbox: tuple[int, int, int, int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _within_reach(
    bbox: tuple[int, int, int, int],
    within: tuple[int, int, int, int],
    reach: int | None,
) -> tuple[int, int, int, int]:
    """The part of ``bbox`` within ``reach`` of ``within``.

    An effect drawn on that part is, over ``within``, the effect drawn on the
    whole of ``bbox``: each pixel of it reads nothing further than ``reach``
    away, so what lies beyond cannot show there. The effects module states
    the reach of each kind, or None for one that reads its whole box, which
    keeps the box. Empty when nothing of ``bbox`` can show.
    """
    if utils.intersect(bbox, within) == (0, 0, 0, 0):
        return (0, 0, 0, 0)
    if reach is None:
        return bbox
    x0, y0, x1, y1 = within
    return utils.intersect(bbox, (x0 - reach, y0 - reach, x1 + reach, y1 + reach))


# What an effect descriptor psd-tools did not write can raise on the way to
# being read, and then used: a key that is absent, so the read lands on the
# ``None`` ``Descriptor.get()`` returns (AttributeError); a key holding
# something other than what the read expects (TypeError, ValueError); a colour
# class that is none of the five (KeyError); and a number that parses but does
# not survive the arithmetic it is put through -- an infinite stroke size
# (OverflowError out of ``math.ceil``), a gradient scaled to zero
# (ZeroDivisionError). They all mean one thing -- this effect cannot be read --
# and none of them is worth losing the rest of the document over, so every
# guard below catches the whole set rather than the arm it happens to have met.
#
# Not here, deliberately: ``ImportError``, which names the package that would
# have drawn the effect, and ``AssertionError``, which states an invariant of
# this module rather than a property of the file (#826).
_UNREADABLE = (
    AttributeError,
    KeyError,
    OverflowError,
    TypeError,
    ValueError,
    ZeroDivisionError,
)


def _has_enabled_effects(layer: Layer) -> bool:
    """Whether the layer carries a layer style Photoshop would draw."""
    try:
        return layer.has_effects()
    except _UNREADABLE:
        return False


def _readable(layer: Layer, name: str) -> list[_StyledEffect]:
    """The layer's enabled effects of one kind, empty if they cannot be listed.

    In the order they are painted, bottom first: Photoshop shows the instance
    it lists first on top, so the file order is reversed.

    ``Effects`` re-reads the layer's block on every access, so the walk
    ``find()`` makes can fail before it yields anything. The callers guard each
    effect they read; this covers the listing they stand on (#828).
    """
    try:
        return list(_styled(layer.effects.find(name)))[::-1]
    except _UNREADABLE as error:
        # Lazy arguments, not ``%``: a path that degrades must not depend on
        # formatting the layer it is degrading. ``logging`` absorbs a failed
        # format; ``%`` would re-raise it out of the handler.
        logger.debug("Cannot list the %s effects of %s: %s", name, layer, error)
        return []


# How a one-channel array -- a canvas or a source -- becomes a given width.
# Spelled once because it is threaded through most of this module -- see
# ``widen.make_widen()``.
_Widen = Callable[[np.ndarray, int], np.ndarray]

# What each overlay effect draws. Only the pattern needs the compositor's
# channel count, and only the colour overlay always draws -- the pattern and
# the gradient can both decline, returning None -- but a uniform signature is
# what lets the three share one application path: the shape/alpha arithmetic
# around them is identical. Each draws on ``box`` and lays its paint out across
# ``layout`` -- see ``_overlay_boxes()``.
_Box = tuple[int, int, int, int]
_OverlayDraw = Callable[
    [Layer, Any, int, _Box, _Box], tuple[np.ndarray | None, np.ndarray | None]
]


def _overlay_boxes(
    layer: Layer, viewport: _Box, memo: "_ContentBBoxes"
) -> tuple[_Box, _Box]:
    """The box an overlay is drawn on and the box its paint is laid out across.

    A layer's style box for both -- its own box, or an adjustment layer's
    mask -- except for a group: its overlay covers its composited contents,
    effects included, which can run past the group's box. A pass-through
    group lays its overlay out across the document canvas, as Photoshop does.
    """
    if not layer.is_group():
        box = _style_bbox(layer)
        return box, box
    box = utils.intersect(viewport, _content_bbox(layer, memo))
    if layer.blend_mode == BlendMode.PASS_THROUGH:
        return box, layer._psd.viewbox
    return box, layer.bbox


def _reference_point(layer: Layer) -> tuple[float, float] | None:
    """The layer's ``fxrp`` reference point, in document coordinates.

    None without the tag, so that :py:func:`paint.pattern_origin` falls back
    to the layer's box; :py:attr:`~psd_tools.api.layers.Layer.reference_point`
    reads a missing tag as the document's origin.
    """
    point = layer.tagged_blocks.get_data(Tag.REFERENCE_POINT)
    if point is None:
        return None
    return float(point[0]), float(point[1])


def _draw_color_overlay(
    layer: Layer, descriptor: Any, channels: int, box: _Box, layout: _Box
) -> tuple[np.ndarray | None, np.ndarray | None]:
    return paint.draw_solid_color_fill(box, layer._psd.color_mode, descriptor)


def _tiling_origin(
    layer: Layer, descriptor: Descriptor, layout: _Box
) -> tuple[int, int]:
    """Where a pattern of ``layer``'s style laid out across ``layout`` starts.

    A layer's tiling starts at its reference point, a group's at its layout
    box, which is the document's origin for a pass-through group; see
    :py:func:`paint.pattern_origin`.
    """
    point = None if layer.is_group() else _reference_point(layer)
    return paint.pattern_origin(descriptor, layout, point)


def _draw_pattern_overlay(
    layer: Layer, descriptor: Any, channels: int, box: _Box, layout: _Box
) -> tuple[np.ndarray | None, np.ndarray | None]:
    origin = _tiling_origin(layer, descriptor, layout)
    fill, shape = paint.draw_pattern_fill(
        box, layer._psd, descriptor, layout, origin=origin
    )
    if fill is None:
        return None, None
    # A pattern carries its own color mode, so a grayscale one reaches a CMYK
    # document one channel wide and has to be converted rather than replicated.
    # ``_fit_source()`` widens every source at the door, this pattern
    # included (#749, #777).
    #
    # The width check stays. ``_assert_source_fits()`` makes the same one
    # downstream -- paste() preserves the channel count, so it would report the
    # same number -- but this one fires first and names the pattern, so a
    # mismatch points at the pattern rather than at a "source" of unstated
    # provenance. The canvas is what the width is measured against, not the
    # layer color the overlay is drawn over: that color is itself allowed to be
    # single-channel inside a multi-channel document, and measuring against it
    # rejected patterns which in fact matched the canvas exactly (#711). A
    # width that is neither 1 nor the canvas is a real inconsistency: widening
    # reaches 3 from 1, but nothing reaches 3 from 4 (#741).
    assert fill.shape[-1] in (1, channels), (
        "Inconsistent pattern channels: %d, expected 1 or %d"
        % (fill.shape[-1], channels)
    )
    return fill, shape


def _opaque_bounds(layer: Layer, layout: _Box) -> _Box:
    """The box of a pixel, smart object or type layer's pixels over half opaque.

    Photoshop lays such a layer's gradient overlay out on it, so a soft edge
    or a glyph's antialiased rim does not stretch the ramp. Masks are not
    read. Any other layer, one with no pixel that opaque, and one whose pixels
    cannot be read keep ``layout``.
    """
    if not isinstance(layer, (PixelLayer, SmartObjectLayer, TypeLayer)):
        return layout
    try:
        shape = layer.numpy("shape")
    except _UNREADABLE as error:
        logger.debug("Cannot read the shape of %s: %s", layer, error)
        return layout
    if shape is None:
        return layout
    opaque = shape[..., 0] > 0.5
    rows = np.flatnonzero(opaque.any(axis=1))
    if not rows.size:
        return layout
    columns = np.flatnonzero(opaque.any(axis=0))
    left, top = layer.bbox[0], layer.bbox[1]
    return (
        left + int(columns[0]),
        top + int(rows[0]),
        left + int(columns[-1]) + 1,
        top + int(rows[-1]) + 1,
    )


def _draw_gradient_overlay(
    layer: Layer, descriptor: Any, channels: int, box: _Box, layout: _Box
) -> tuple[np.ndarray | None, np.ndarray | None]:
    # With Algn off the ramp spans the document, so the pixels go unread.
    if bool(descriptor.get(Key.Alignment, True)):
        layout = _opaque_bounds(layer, layout)
    return paint.draw_gradient_fill(
        box,
        layer._psd.color_mode,
        descriptor,
        paint.gradient_anchor(layer._psd, descriptor, layout),
    )


# Iteration order is the order the overlays are composited in, unchanged from
# the sequence of calls this replaced.
_OVERLAY_DRAWS: dict[str, _OverlayDraw] = {
    "coloroverlay": _draw_color_overlay,
    "patternoverlay": _draw_pattern_overlay,
    "gradientoverlay": _draw_gradient_overlay,
}


def _path_bbox(layer: Layer) -> tuple[int, int, int, int]:
    """The box of the layer's vector mask in document pixels.

    ``layer.bbox`` follows the layer record, which can be wider than the path.
    A layer without a path to measure keeps ``layer.bbox``.
    """
    mask = layer.vector_mask if layer.has_vector_mask() else None
    if mask is None or not any(len(path) for path in mask.paths):
        return layer.bbox
    left, top, right, bottom = mask.bbox
    width, height = layer._psd.width, layer._psd.height
    try:
        return (
            int(round(left * width)),
            int(round(top * height)),
            int(round(right * width)),
            int(round(bottom * height)),
        )
    except (OverflowError, ValueError):
        return layer.bbox


def _vector_stroke_reach(layer: Layer) -> tuple[int, int, int, int]:
    """``layer.bbox`` grown by the width of a centred or outer vector stroke.

    An inner stroke stays inside the path. A width that cannot be read leaves
    the box as it is, since this runs for every layer the cull visits.
    """
    bbox = layer.bbox
    stroke = layer.stroke if layer.has_vector_mask() else None
    if (
        stroke is None
        or not stroke.enabled
        or stroke.line_alignment is StrokeAlignment.INNER
    ):
        return bbox
    try:
        width = max(math.ceil(float(stroke._data.get("strokeStyleLineWidth", 1.0))), 0)
    except (OverflowError, ValueError, TypeError):
        return bbox
    return (bbox[0] - width, bbox[1] - width, bbox[2] + width, bbox[3] + width)


def _style_bbox(layer: Layer) -> tuple[int, int, int, int]:
    """The box a layer's style is laid out on and its silhouette read from.

    ``layer.bbox``, except for an adjustment layer, which has no pixels of its
    own: its mask is its shape, so the mask's box stands in -- empty when the
    mask hides everything -- and the whole canvas where there is no mask or
    the mask reveals what lies outside its box. A vector mask narrows it as
    it narrows the shape. A mask under full density reveals the layer
    everywhere, as :py:meth:`Compositor._get_mask` folds it in, and narrows
    nothing.
    """
    if not isinstance(layer, AdjustmentLayer):
        return layer.bbox
    psd = layer._psd
    bbox = (0, 0, psd.width, psd.height)
    mask = layer.mask
    parameters = mask.parameters if mask is not None else None
    vector_density = parameters.vector_mask_density if parameters else None
    user_density = parameters.user_mask_density if parameters else None
    if user_density is None:
        user_density = vector_density
    if (
        mask is not None
        and not mask.disabled
        and mask.background_color == 0
        and user_density in (None, 255)
    ):
        bbox = mask.bbox
    vector = layer.vector_mask if layer.has_vector_mask() else None
    if (
        vector is not None
        and not vector.disabled
        and vector_density in (None, 255)
        and any(len(path) for path in vector.paths)
    ):
        bbox = utils.intersect(bbox, _path_bbox(layer))
    return bbox


def _reach_base(layer: Layer) -> tuple[int, int, int, int]:
    """The box a layer's effects reach out from.

    ``layer.bbox``, which is empty for an adjustment layer: one with a style
    reaches out from its style box instead, so the canvas of a group holding
    it is measured to fit the style. One without is not measured at all.
    """
    if isinstance(layer, AdjustmentLayer) and _has_enabled_effects(layer):
        return _style_bbox(layer)
    return layer.bbox


def _stroke_reach(
    layer: Layer, memo: "_ContentBBoxes | None" = None
) -> tuple[int, int, int, int]:
    """``layer.bbox`` grown to every box its strokes are drawn on.

    A stroke, effect or vector, is one of the things in this module that reach
    outside the layer; :py:func:`_beneath_reach` and :py:func:`_bevel_reach`
    measure the others. The three overlays draw on ``layer.bbox`` and are
    pasted from it, and a satin, the inner effects and an inner bevel stay
    inside the layer. :py:func:`_effect_reach` is all of them together, and is
    what the callers below ask.

    ``find("stroke")`` rather than a descriptor walk, so this stays in lockstep
    with the loop in :py:meth:`Compositor._add_stroke_effects` that actually
    draws them: both skip a disabled effect and both skip every effect when the
    layer's master switch is off. Each box is :py:func:`_stroke_box`, the one
    that loop draws on, so a group's stroke is measured around everything its
    contents paint and not around its own box alone.

    Two callers ask it for different reasons: :py:func:`_paint_bbox`, how wide
    a canvas a group has to composite its children on (#808), and
    :py:meth:`Compositor._accepts`, whether a layer paints inside the viewport
    at all (#815). The second reaches every non-group layer the cull visits,
    not just a group's children, so the degradation has to stay total -- and
    total per effect, not for the whole loop: an effect whose box cannot be
    read is skipped rather than taken as a verdict on the layer, so a stroke
    beside it still grows the box. That leaves the box wider than what gets
    drawn, never narrower, since ``_add_stroke_effects`` calls
    ``stroke_bbox()`` on the same descriptor, drops the same effect, and can
    only drop one more over the paint this never reads. Wider is the safe
    direction for both callers: it costs a group canvas nobody draws on, where
    narrower clips a stroke or culls a layer that has one (#826).
    """
    base = _reach_base(layer)
    if base[0] >= base[2] or base[1] >= base[3]:
        return base
    bbox = utils.union_bbox(base, _vector_stroke_reach(layer))
    for effect in _readable(layer, "stroke"):
        try:
            bbox = utils.union_bbox(bbox, _stroke_box(layer, effect.descriptor, memo))
        except _UNREADABLE as error:
            logger.debug("Cannot measure a stroke effect of %s: %s", layer, error)
    return bbox


def _cast_bbox(
    layer: Layer, memo: "_ContentBBoxes | None" = None
) -> tuple[int, int, int, int]:
    """The box a layer's silhouette can cover: what its effects are traced from.

    A layer's own box grown by its vector stroke, which a redrawn shape paints
    past its path, except for a group: its silhouette is its composited
    contents, effects included, and a child's stroke or shadow runs past the
    group's box, which is the union of its children's own. The group's
    content box is every box those contents paint on. An adjustment layer
    casts from its style box, as it has no pixels.
    """
    # A child's effects run past the group's own box (#808).
    if layer.is_group():
        return _content_bbox(layer, memo)
    if isinstance(layer, AdjustmentLayer):
        return _style_bbox(layer)
    return _vector_stroke_reach(layer)


def _stroke_box(
    layer: Layer, desc: Descriptor, memo: "_ContentBBoxes | None" = None
) -> tuple[int, int, int, int]:
    """The box one stroke effect of the layer is drawn on.

    :py:func:`_cast_bbox` grown by the stroke's reach, since the stroke is
    traced from the silhouette and a group's runs past the group's box.
    Spelled once because :py:func:`_stroke_reach`,
    :py:meth:`Compositor._add_stroke_effects` and
    :py:meth:`Compositor._knockout_shape` have to agree on it: a band drawn
    on a box other than the one reserved for it is clipped, and one cut from
    under a shadow on a different box is cut where nothing is painted.
    """
    return stroke_bbox(_cast_bbox(layer, memo), desc, layer._psd._max_alloc_bytes)


def _beneath_reach(
    layer: Layer, memo: "_ContentBBoxes | None" = None
) -> tuple[int, int, int, int]:
    """``layer.bbox`` grown to every box its drop shadows and outer glows are drawn on.

    The half of :py:func:`_effect_reach` that lies beneath the layer,
    measured the way :py:func:`_stroke_reach` measures a stroke and degrading
    the same way: an effect whose box cannot be read is skipped, which leaves
    the box wider than what gets drawn and never narrower, because
    :py:meth:`Compositor._apply_drop_shadows` and
    :py:meth:`Compositor._apply_outer_glows` call the same bbox function on
    the same box and values and drop the same effect.

    The box is grown from :py:func:`_cast_bbox`, not from ``layer.bbox``, so a
    group's shadow is measured from everything the group casts it from. The
    group's own effect lists are read first, so a group with neither never
    measures its contents here.
    """
    bbox = _reach_base(layer)
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        return bbox
    shadows = _readable(layer, "dropshadow")
    glows = _readable(layer, "outerglow")
    if not shadows and not glows:
        return bbox
    cast_from = _cast_bbox(layer, memo)
    budget = layer._psd._max_alloc_bytes
    for effect in shadows:
        shadow = cast(DropShadow, effect)
        try:
            geometry = shadow_geometry(shadow.descriptor, shadow.angle)
            bbox = utils.union_bbox(
                bbox,
                drop_shadow_bbox(
                    cast_from,
                    geometry.distance,
                    geometry.angle,
                    geometry.size,
                    budget,
                ),
            )
        except _UNREADABLE as error:
            logger.debug("Cannot measure a drop shadow of %s: %s", layer, error)
    for effect in glows:
        try:
            glow_box = outer_glow_bbox(cast_from, effect.descriptor, budget)
            bbox = utils.union_bbox(bbox, glow_box)
        except _UNREADABLE as error:
            logger.debug("Cannot measure an outer glow of %s: %s", layer, error)
    return bbox


def _bevel_reach(
    layer: Layer, memo: "_ContentBBoxes | None" = None
) -> tuple[int, int, int, int]:
    """``layer.bbox`` grown to every box its bevels paint beside the layer on.

    An inner bevel stays inside the layer and adds nothing. Degrades as
    :py:func:`_beneath_reach` does, except that no budget is checked here,
    the document's or the module's: the canvas a bevel is drawn on is cut to
    the viewport first (:py:meth:`Compositor._inner_box`) and only that is
    checked, so checking the whole grown box would leave this narrower than
    what is drawn. The size and soften are held to what Photoshop writes, so
    the box is bounded without one.
    """
    bbox = _reach_base(layer)
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        return bbox
    bevels = _readable(layer, "bevelemboss")
    if not bevels:
        return bbox
    for effect in bevels:
        bevel = cast(BevelEmboss, effect)
        try:
            style = _bevel_style(bevel)
            if not bevel_beside(style):
                continue
            _, size, soften = bevel_geometry(bevel.descriptor)
            bbox = utils.union_bbox(
                bbox,
                inner_bbox(
                    _cast_bbox(layer, memo),
                    bevel_reach(size, soften, _bevel_technique(bevel)),
                    UNLIMITED,
                ),
            )
        except _UNREADABLE as error:
            logger.debug("Cannot measure a bevel of %s: %s", layer, error)
    return bbox


def _bevel_style(bevel: BevelEmboss) -> BevelStyle:
    """The bevel's style, inner where the descriptor does not say."""
    return bevel.bevel_style or BevelStyle.INNER_BEVEL


def _bevel_technique(bevel: BevelEmboss) -> BevelTechnique:
    """The bevel's technique, smooth where the descriptor does not say."""
    return bevel.bevel_type or BevelTechnique.SOFT_MATTE


def _effect_reach(
    layer: Layer, memo: "_ContentBBoxes | None" = None
) -> tuple[int, int, int, int]:
    """``layer.bbox`` grown to every box its effects are drawn on.

    Both callers need the whole outward reach, for the reasons
    :py:func:`_stroke_reach` gives: a group canvas that clips a child's
    shadow loses it as surely as one that clips its stroke, and a layer whose
    own box has cleared the viewport can still cast a shadow back into it.
    """
    return utils.union_bbox(
        utils.union_bbox(_stroke_reach(layer, memo), _beneath_reach(layer, memo)),
        _bevel_reach(layer, memo),
    )


_ContentBBoxes = dict[int, tuple[int, int, int, int]]
_GroupAlphas = dict[tuple[int, tuple[int, int, int, int]], np.ndarray]


@dataclass
class _TextBlend:
    """How each type layer's edge blends, settled once per composite pass.

    ``overrides`` pins a layer, by layer ID, to a gamma, or to None for the
    document's encoding. Any other type layer blends in the pass's text gamma
    unless ``reference`` -- Photoshop's own composite of the document, read
    lazily -- shows its edge mixed straight: Photoshop composites a type layer
    straight when it keeps the layer's stored raster as pixels, and nothing
    in the file says which layers it kept. ``chosen`` holds, by ``id()``,
    whether each layer went on straight, so every compositor in the pass
    agrees. A layer whose edge only leans straight where it goes on --
    something composited later still changes it -- goes on in the gamma and
    waits in ``leaning``, by ``id()``, with its edge pixels on the document's
    canvas, for :py:func:`composite` to settle on the finished canvas.
    """

    overrides: Mapping[int, float | None] = field(default_factory=dict)
    reference: Callable[[], np.ndarray | None] | None = None
    chosen: dict[int, bool] = field(default_factory=dict)
    leaning: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    _array: np.ndarray | None = None
    _read: bool = False

    def reference_array(self) -> np.ndarray | None:
        """The reference on the document's canvas, or None."""
        if not self._read:
            self._read = True
            self._array = None if self.reference is None else self.reference()
        return self._array


@dataclass
class _PassCache:
    """What one composite pass measures once and then reuses.

    Every entry is keyed by ``id()`` and none outlives the pass, so unlike a
    cache on the layer they cannot go stale against an edit, and the document
    stays reachable throughout, so no id can be reused under them.

    ``content_bboxes`` holds every ``_content_bbox()`` computed so far. Without
    it the walk repeats at every level: the outermost group measures the whole
    subtree, then compositing descends and each group inside measures its own
    subtree again -- O(nodes x depth), and exactly quadratic down a chain.

    ``group_alphas`` holds every group alpha ``_get_group_alpha()`` has
    composited, keyed by the group and the box it was composited on. Tracing a
    group's stroke composites its whole subtree, and a group inside that
    subtree whose own stroke escapes composites its subtree again, so the work
    can *double* at every level rather than merely repeat. It takes nesting
    where each stroke reaches further than its parent's to get there -- equal
    sizes all the way down leave the inner traces inside the box #808's other
    half already widened, and cost d rather than 2**d -- but at 2**d a chain of
    ten is a thousand composites. Keyed by box and not by layer alone because a
    layer may carry several stroke effects (#798), each traced separately.

    The memo collapses the composites to one per box; the ``paste()`` back to
    each asking viewport still happens per call, so the remaining cost is
    quadratic in depth rather than linear. It also retains one alpha canvas
    per entry until the pass ends: the trade is bounded work for
    bounded-lifetime memory, and on a large document with many traced groups
    the retention is the part that grows. A group's shadows and glows read the
    same memo, on their own boxes.

    ``scipy_warned`` is set once a pass has logged that its effects are left
    out for want of scipy, so a document with many of them says it once.

    ``unfiltered`` is the last smart object whose filter mask
    :py:func:`_unfilter_under_mask` blended, with the result: each effect cast
    from the layer reads it again, and the blend decodes the document's
    ``FEid`` planes. It holds the layer itself, and one layer's pixels.

    ``stroke_bands`` carries a stroke's band from the drop shadow's knockout to
    the stroke drawn after it (:py:func:`effects.stroke_band`).
    ``change_bboxes`` and ``box_safe`` hold :py:func:`_change_bbox` and
    :py:func:`_box_safe` per layer.

    ``rc`` is the cache shared across passes
    (:py:mod:`~psd_tools.composite.rcache`), or None, and ``doc_key`` the
    document's key in it; ``layer_keys`` holds each layer's
    :py:func:`~psd_tools.composite.rcache.layer_key`.
    """

    content_bboxes: _ContentBBoxes = field(default_factory=dict)
    group_alphas: _GroupAlphas = field(default_factory=dict)
    scipy_warned: bool = False
    unfiltered: tuple[Layer, np.ndarray, np.ndarray | None] | None = None
    text: _TextBlend = field(default_factory=_TextBlend)
    stroke_bands: dict = field(default_factory=dict)
    change_bboxes: _ContentBBoxes = field(default_factory=dict)
    box_safe: dict[int, bool] = field(default_factory=dict)
    rc: rcache.RenderCache | None = None
    doc_key: bytes = b""
    layer_keys: dict[int, bytes] = field(default_factory=dict)


def _paint_bbox(layer: Layer, memo: _ContentBBoxes) -> tuple[int, int, int, int]:
    """Every box ``layer`` and everything inside it can put paint on."""
    bbox = _effect_reach(layer, memo)
    if isinstance(layer, GroupMixin):
        bbox = utils.union_bbox(bbox, _content_bbox(layer, memo))
    return bbox


def _content_bbox(
    group: Layer, memo: _ContentBBoxes | None = None
) -> tuple[int, int, int, int]:
    """The box an isolated group has to composite its contents on.

    ``Group.bbox`` is the union of its children's own bounding boxes and
    excludes effect coverage, so a child's outset or centered stroke reaches
    past its own box and therefore past the group's. Compositing the group on
    that box clips the stroke away with no viewport the caller could widen to
    get it back (#808).

    An ``Artboard`` stops the descent. Its ``bbox`` is the artboard rectangle
    rather than a union of its children -- the children routinely run past it
    -- and intersecting with it is how the artboard clip is implemented here.
    Growing it to fit a child would paint into the gap between two artboards,
    which Photoshop's own render leaves empty.

    The children counted are the ones ``Group.extract_bbox()`` counts, so the
    box can only grow and never move. A clipping child is left out because it
    cannot paint outside the layer it clips to:
    :py:meth:`Compositor._apply_clip_layers` keeps its color and discards its
    coverage, and that layer is a non-clipping child already counted.

    Visibility is read the way ``Group.bbox`` reads it and not the way
    ``layer_filter`` does, so a child the filter excludes is measured here. It
    was already inside the old box for the same reason -- ``extract_bbox()``
    does not consult the filter either -- and all this adds on top is that
    child's own stroke margin, so the two stay consistent rather than one
    being narrowed alone.
    """
    bbox = group.bbox
    if isinstance(group, Artboard):
        return bbox
    if memo is None:
        memo = {}
    elif id(group) in memo:
        return memo[id(group)]
    for child in cast(GroupMixin, group):
        if not child.is_visible() or child.clipping:
            continue
        bbox = utils.union_bbox(bbox, _paint_bbox(child, memo))
    memo[id(group)] = bbox
    return bbox


# Whether a layer is composited on the box it changes rather than the whole
# viewport, and the largest share of the viewport such a box may cover.
_LOCAL_BOXES = True
_LOCAL_SHARE = 0.6


def _box_safe(layer: Layer, cache: "_PassCache") -> bool:
    """Whether everything ``layer`` changes stays inside :py:func:`_change_bbox`.

    A knockout reads a backdrop; a group holding one is composited whole.
    """
    memo = cache.box_safe
    known = memo.get(id(layer))
    if known is not None:
        return known
    if _read_knockout(layer):
        safe = False
    elif layer.is_group():
        safe = all(_box_safe(child, cache) for child in cast(GroupMixin, layer))
    else:
        safe = True
    memo[id(layer)] = safe
    return safe


def _drawn_bbox(layer: Layer) -> tuple[int, int, int, int]:
    """:py:func:`vector.drawn_bbox`, or the canvas when the path cannot be measured."""
    try:
        box = vector.drawn_bbox(layer)
    except (OverflowError, TypeError, ValueError):
        return (0, 0, layer._psd.width, layer._psd.height)
    return (0, 0, 0, 0) if box is None else box


def _change_bbox(layer: Layer, cache: "_PassCache") -> tuple[int, int, int, int]:
    """Every box ``layer`` can change the canvas in.

    Its paint box, plus for an adjustment the box its mask reveals (the
    adjustments are pointwise), and for a pass-through group the boxes its
    children change, since an adjustment in it reaches the canvas beneath.
    A path drawn for the layer can cover pixels past the rounded boxes those
    are measured on, so its own box is added.
    """
    memo = cache.change_bboxes
    known = memo.get(id(layer))
    if known is not None:
        return known
    # _paint_bbox() spelled out: the content box is the pass's memo, so no
    # layer is measured twice.
    bbox = _effect_reach(layer, cache.content_bboxes)
    if isinstance(layer, GroupMixin):
        bbox = utils.union_bbox(bbox, _content_bbox(layer, cache.content_bboxes))
    elif layer.has_vector_mask():
        bbox = utils.union_bbox(bbox, _drawn_bbox(layer))
    if isinstance(layer, AdjustmentLayer):
        bbox = utils.union_bbox(bbox, _style_bbox(layer))
    elif layer.is_group() and layer.blend_mode == BlendMode.PASS_THROUGH:
        for child in cast(GroupMixin, layer):
            if isinstance(child, AdjustmentLayer) or child.is_group():
                bbox = utils.union_bbox(bbox, _change_bbox(child, cache))
    memo[id(layer)] = bbox
    return bbox


def composite_pil(
    layer: Layer | PSDImage,
    color: float | Sequence[float] | np.ndarray,
    alpha: float | np.ndarray,
    viewport: tuple[int, int, int, int] | None,
    layer_filter: Callable[[Layer], bool] | None,
    force: bool,
    as_layer: bool = False,
    apply_icc: bool = True,
    text_gamma: float | None = TEXT_BLEND_GAMMA,
    layer_text_gamma: Mapping[int, float | None] | None = None,
    text_preview: bool = True,
) -> Image.Image | None:
    """
    Composite layers and return a PIL Image.

    This function composites the given layer or document into a PIL Image,
    applying blend modes, effects, and color management.

    Args:
        layer: Layer or PSDImage to composite
        color: Initial backdrop color (0.0-1.0). Can be a scalar, a
            per-channel sequence, or an ndarray
        alpha: Initial backdrop alpha (0.0-1.0). Can be scalar or ndarray
        viewport: Bounding box (left, top, right, bottom) to composite. If None, uses layer bounds
        layer_filter: Optional callable to filter which layers to composite. Should return True to include
        force: If True, force re-rendering of all layers (ignore cached pixels)
        as_layer: If True, apply layer blend modes (default: False for document-level compositing)
        apply_icc: If True, apply ICC profile color correction (default: True)
        text_gamma: The gamma a type layer's anti-aliased edges are blended in,
            as Photoshop's "Blend Text Colors Using Gamma" preference; None
            mixes them in the document's encoding like any other layer
        layer_text_gamma: Per type layer, by layer ID, the gamma its edges
            blend in, or None to mix them straight; overrides ``text_gamma``
            and the preview
        text_preview: Read off the document's stored preview which type
            layers Photoshop mixed straight rather than in ``text_gamma``

    Returns:
        PIL Image with composited result, or None if viewport is empty

    Note:
        - Requires optional composite dependencies (aggdraw, scipy)
        - LAB and Duotone color modes have limited blending support
        - Alpha channel handling varies by color mode
        - Multichannel documents come back single-channel. PIL has no
          multichannel mode, so only the first spot channel survives -- a
          warning says so when it happens; use
          :py:func:`psd_tools.composite.composite` to keep every channel.
        - Alpha is packed into the array only for the two reachable modes that
          have an alpha variant, ``"L"`` and ``"RGB"``; there is no ``"1A"``,
          ``"CMYKA"`` or ``"LABA"`` to return it in. For the rest it is offered
          to :py:func:`psd_tools.api.pil_io.post_process` instead, which
          carries it for CMYK by converting to RGB first when an ICC profile is
          applied. Bitmap results carry no alpha either way: ``post_process()``
          applies it only to ``"RGB"`` and ``"L"``, and the ICC conversion that
          reaches those for CMYK cannot help here -- little-cms builds no
          transform for a 1-bit image, so ``_apply_icc()`` logs the failure and
          returns the image still in ``"1"``. Nor do Lab results, for a third
          reason: a Lab document carrying an ICC profile raises inside
          ``_apply_icc()`` before reaching that point (#740).
    """
    UNSUPPORTED_MODES = {
        ColorMode.DUOTONE,
        ColorMode.LAB,
    }
    psd_image = getattr(layer, "_psd", layer)
    assert isinstance(psd_image, PSDImage)
    color_mode = psd_image.color_mode
    if color_mode in UNSUPPORTED_MODES:
        logger.warning("Unsupported blending color space: %s", color_mode)

    backdrop_alpha = alpha
    color, _, alpha = composite(
        layer,
        color=color,
        alpha=alpha,
        viewport=viewport,
        layer_filter=layer_filter,
        force=force,
        as_layer=as_layer,
        text_gamma=text_gamma,
        layer_text_gamma=layer_text_gamma,
        text_preview=text_preview,
    )

    mode = pil_io.get_pil_mode(color_mode)
    if mode == "P":
        mode = "RGB"
    # Narrow the array to what the mode can hold *before* alpha is appended.
    # A multichannel document carries one plane per spot channel and PIL has no
    # mode for that. Narrowing after the concatenation cannot be keyed on
    # `mode`, because alpha has turned "L" into "LA" by then -- and PIL does
    # not reject a four-plane array declared as two, it reads two bytes out of
    # every four and returns planes that correspond to nothing.
    pil_channels = pil_io.get_pil_channels(mode)
    if color.shape[2] > pil_channels:
        logger.warning(
            "%s composited to %d channels; PIL mode %r holds %d. Keeping the "
            "first and dropping the rest -- use psd_tools.composite.composite() "
            "to keep every channel.",
            color_mode,
            color.shape[2],
            mode,
            pil_channels,
        )
        color = color[:, :, :pil_channels]
    # Skip alpha when the color mode requires deferred alpha handling, or
    # when the backdrop is fully opaque (the result is guaranteed opaque).
    delay_alpha_application = color_mode not in (ColorMode.GRAYSCALE, ColorMode.RGB)
    uniform_alpha = _uniform_alpha(backdrop_alpha)
    has_opaque_backdrop = uniform_alpha is not None and uniform_alpha >= 1.0
    skip_alpha = not force and (delay_alpha_application or has_opaque_backdrop)
    # Of the modes reachable here -- "1", "L", "RGB", "CMYK" and "LAB" -- only
    # "L" and "RGB" have an alpha variant. Appending "A" regardless built modes
    # PIL has never heard of, so `force=True` raised outright on a bitmap
    # ("1A"), CMYK ("CMYKA") or Lab ("LABA") document. Leaving it off the array
    # returns the colour that was asked for rather than nothing at all, and for
    # the modes `post_process()` can convert it still arrives, via `putalpha`
    # below.
    if not skip_alpha and mode not in ("L", "RGB"):
        logger.debug("PIL has no alpha variant of %r; not packing it inline.", mode)
        skip_alpha = True
    logger.debug("Skipping alpha: %s", skip_alpha)
    if not skip_alpha:
        color = np.concatenate((color, alpha), 2)
        mode += "A"
    assert isinstance(color, np.ndarray)
    if mode in ("1", "L"):
        color = color[:, :, 0]
    if color.shape[0] == 0 or color.shape[1] == 0:
        return None
    # Clipped, not cast straight: numpy *wraps* an out-of-range float, so 1.2
    # would arrive as byte 50 rather than as white (#757). np.clip passes NaN
    # through. Rounded, not truncated, as Photoshop's export is.
    pixels = 255 * color
    np.clip(pixels, 0, 255, out=pixels)
    np.rint(pixels, out=pixels)
    pixels = pixels.astype(np.uint8)
    if mode == "1":
        # `fromarray(uint8, "1")` does not mean "these bytes, as bilevel". PIL
        # takes the raw mode literally at one bit per pixel, so it consumes one
        # byte per eight columns and expands its bits across the row -- a 4x4
        # document came back as the bits of its first four bytes. Build the
        # plane in "L", where a byte is a pixel, and reduce it afterwards.
        # Rounded, its threshold of byte 128 makes 0.5 and up white.
        image = Image.fromarray(pixels, "L").convert("1", dither=Image.Dither.NONE)
    elif mode == "LAB":
        # The same trap as "1", in a different raw mode. PIL's "LAB" unpacker
        # reads the two chroma planes as *signed* and adds 128, so an array in
        # the encoding the compositor carries -- where byte 128 is ``a = 0``,
        # the file's own convention -- lands 128 off on both axes. That is not
        # a slightly wrong colour but an unrelated one: a flat Lab(60, 25, 25)
        # converted to RGB as (0, 187, 255) where Photoshop puts it at
        # (196, 126, 101). Merging plane by plane writes them verbatim, which
        # is what `pil_io._merge_channels()` does for `topil()`; the two
        # disagreed until #759.
        #
        # `np.asarray()` cannot see the difference -- it reads PIL's internal
        # buffer, which is offset the other way, so a round trip through
        # `fromarray` is the identity. `getpixel()`, `convert()` and `save()`
        # all see it.
        image = Image.merge(
            mode,
            [
                Image.fromarray(np.ascontiguousarray(pixels[:, :, band]), "L")
                for band in range(pil_channels)
            ],
        )
    else:
        image = Image.fromarray(pixels, mode)
    alpha_as_image = None
    # Whenever the alpha did not go into the array, offer it to post_process()
    # instead -- it converts CMYK to RGB before applying it, so `force=True` on
    # a CMYK document keeps the alpha that `force=False` has always returned.
    # Gating this on `not force` dropped it on exactly the paths the branch
    # above had just diverted here.
    if skip_alpha and delay_alpha_application:
        plane = 255 * np.squeeze(alpha, axis=2)
        np.clip(plane, 0, 255, out=plane)
        np.rint(plane, out=plane)
        alpha_as_image = Image.fromarray(plane.astype(np.uint8), "L")
    icc = None
    if apply_icc and Resource.ICC_PROFILE in psd_image.image_resources:
        icc = psd_image.image_resources.get_data(Resource.ICC_PROFILE)
    return pil_io.post_process(image, alpha_as_image, icc)


def composite(
    group: Layer | PSDImage,
    color: float | Sequence[float] | np.ndarray = 1.0,
    alpha: float | np.ndarray = 0.0,
    viewport: tuple[int, int, int, int] | None = None,
    layer_filter: Callable[[Layer], bool] | None = None,
    force: bool = False,
    as_layer: bool = False,
    text_gamma: float | None = TEXT_BLEND_GAMMA,
    layer_text_gamma: Mapping[int, float | None] | None = None,
    text_preview: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Composite layers and return NumPy arrays.

    This function composites the given layer or document into NumPy arrays
    representing color, shape, and alpha channels. It applies layer blending,
    effects, masks, and clipping according to the PSD specification.

    Args:
        group: Layer or PSDImage to composite
        color: Initial backdrop color (0.0-1.0, default: 1.0). Can be a scalar
            applied to all channels, a per-channel sequence, or a full
            (height, width, channels) ndarray.
        alpha: Initial backdrop alpha (0.0-1.0, default: 0.0). Can be scalar or ndarray.
        viewport: Bounding box (left, top, right, bottom) to composite. If None, uses layer bounds
        layer_filter: Optional callable(layer) -> bool to filter which layers to composite
        force: If True, force re-rendering of all layers including vector shapes and fills
        as_layer: If True, treat the group as a layer (apply blend mode to backdrop)
        text_gamma: The gamma a type layer's anti-aliased edges are blended in,
            as Photoshop's "Blend Text Colors Using Gamma" preference; None
            mixes them in the document's encoding like any other layer
        layer_text_gamma: Per type layer, by layer ID, the gamma its edges
            blend in, or None to mix them straight; overrides ``text_gamma``
            and the preview
        text_preview: When compositing a whole document that has a stored
            preview, read off it which type layers Photoshop mixed straight
            rather than in ``text_gamma`` -- it does so for a layer whose
            raster it kept as pixels, which the file does not record

    Returns:
        Tuple of (color, shape, alpha) as float32 ndarrays with shape (height, width, channels):
            - color: RGB/CMYK/Grayscale values in range [0.0, 1.0]
            - shape: Layer shape/coverage mask in range [0.0, 1.0]
            - alpha: Composite alpha channel in range [0.0, 1.0]

    Examples:
        >>> from psd_tools import PSDImage
        >>> psd = PSDImage.open('example.psd')
        >>> color, shape, alpha = composite(psd)
        >>> # Apply custom backdrop
        >>> color, shape, alpha = composite(psd, color=(1.0, 0.0, 0.0), alpha=1.0)
        >>> # Force re-render vector layers
        >>> color, shape, alpha = composite(psd, force=True)
        >>> # Composite only visible layers
        >>> color, shape, alpha = composite(psd, layer_filter=lambda l: l.visible)

    Note:
        - Requires optional composite dependencies (aggdraw, scipy)
          for vector shape rendering, gradient fills, and layer effects.
        - Adjustment layers have limited support.
        - Text rendering is not supported (text layers show as raster if available).
          In an RGB document a type layer's stored raster is blended in
          ``text_gamma``, as Photoshop composites it, unless Photoshop kept
          the raster as pixels (a substituted font, text data without
          ``bounds``) or the document is 32-bit.
        - Multichannel documents are composited over every spot channel the
          file declares. That is more than Photoshop does: Photoshop discards
          the layer records when it opens a multichannel document and displays
          the merged image data instead, so a multichannel document that has
          layers does not render here the way it looks there. Use
          :py:meth:`~psd_tools.api.psd_image.PSDImage.topil` for the merged
          data.
        - Inside ``with RenderCache():`` intermediates are shared across
          passes; see :py:mod:`psd_tools.composite.rcache`.
    """
    args = (
        group,
        color,
        alpha,
        viewport,
        layer_filter,
        force,
        as_layer,
        text_gamma,
        layer_text_gamma,
        text_preview,
    )
    cache = rcache.active()
    if cache is None or not cache.verify:
        return _composite(*args)
    token = rcache.suspended()
    try:
        plain = _composite(*args)
    finally:
        rcache._ACTIVE.reset(token)
    cached = _composite(*args)
    rcache.check(plain, cached, group)
    return cached


def _composite(
    group: Layer | PSDImage,
    color: float | Sequence[float] | np.ndarray,
    alpha: float | np.ndarray,
    viewport: tuple[int, int, int, int] | None,
    layer_filter: Callable[[Layer], bool] | None,
    force: bool,
    as_layer: bool,
    text_gamma: float | None,
    layer_text_gamma: Mapping[int, float | None] | None,
    text_preview: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if viewport is None:
        if isinstance(group, PSDImage):
            viewport = group.viewbox
        else:
            viewport = group.bbox
            if viewport == (0, 0, 0, 0):
                viewport = group._psd.viewbox
    assert viewport is not None

    if isinstance(group, PSDImage) and len(group) == 0:
        # group.numpy() applies check_pixel_size(group.width, group.height) internally
        # for each call (color + shape), so skip the viewport-based check here to
        # avoid an additional warning/raise on top of those already emitted.
        backdrop_color = color
        backdrop_alpha = alpha
        color, shape = group.numpy("color"), group.numpy("shape")
        if viewport != group.viewbox:
            color = paste(viewport, group.bbox, color, 1.0)
            shape = paste(viewport, group.bbox, shape)
        # A wholly transparent backdrop contributes nothing, and blending it
        # in anyway would turn fully uncovered pixels white via divide()'s
        # 0 / 0 fallback. np.any() covers every spelling at once -- scalar,
        # NumPy scalar, or an array of zeros -- and short-circuits on the
        # first nonzero. Normalize only once past that check, so the backdrop
        # is not expanded to a full canvas for the common default.
        if np.any(backdrop_alpha):
            # Sized from the array in hand rather than from the color mode:
            # a multichannel document carries a channel count of its own that
            # EXPECTED_CHANNELS does not predict.
            backdrop = _normalize_backdrop(
                backdrop_color,
                backdrop_alpha,
                color.shape[0],
                color.shape[1],
                color.shape[2],
                widen=make_widen(group),
            )
            color, shape = _blend_backdrop(color, shape, *backdrop)
        return color, shape, shape

    _w = viewport[2] - viewport[0]
    _h = viewport[3] - viewport[1]
    _psd = group if isinstance(group, PSDImage) else group._psd
    # The width the backdrop canvas is allocated at, read from the document
    # rather than from its color mode: EXPECTED_CHANNELS reports 64 for a
    # multichannel document -- the format's maximum, not any file's own count --
    # so reading it from the color mode would allocate the backdrop far wider
    # than the file and hand the first layer a canvas it cannot be blended
    # against.
    _channels = get_color_channels(_psd)
    # The guard is there to reject a file *before* it allocates, so its estimate
    # must never fall below what follows it. `_channels` is the backdrop, and
    # taking the wider of it and the header's own count keeps the modes that
    # expand covered too -- indexed, whose single stored channel becomes three
    # through its palette -- without loosening the ones whose header count is
    # the larger of the pair.
    # It bounds the canvas, not everything downstream: each stored-pixel read is
    # guarded on its own by `Layer.numpy()`, and the canvases a descriptor's size
    # grows -- stroke effect, vector stroke, pattern tiling -- on their own.
    # This keeps the returned-size spelling of the estimate where the two
    # image-data paths moved to a modelled peak (#767), and not for want of
    # trying: what follows this guard grows with the layer count, so there is no
    # expression in `width * height * <constant>` that bounds it. The budget is a
    # per-allocation ceiling on `numpy()` and `topil()`, layers included; here it
    # means the canvas, as it always has.
    _estimate = max(_psd.channels, _channels)
    check_pixel_size(
        _w,
        _h,
        _estimate,
        max_alloc_bytes=_psd._max_alloc_bytes,
    )

    isolated = False
    if not isinstance(group, PSDImage):
        isolated = group.blend_mode != BlendMode.PASS_THROUGH

    # The compositor works exclusively in full-canvas arrays; the scalar and
    # per-channel spellings are a convenience of the public signature, so they
    # are resolved here, once, rather than at each point of use. An isolated
    # group starts transparent, so the caller's alpha is dropped before the
    # canvas is built rather than after.
    # Resolved here and carried through the compositor tree because four of
    # the sites that need it are inside Compositor, which holds no document.
    # (The cost is not the reason -- make_widen() only reads two attributes,
    # and the expensive transform is built lazily and cached globally.)
    _widen_fn = make_widen(_psd)

    backdrop_color, backdrop_alpha = _normalize_backdrop(
        color,
        0.0 if isolated else alpha,
        _h,
        _w,
        _channels,
        widen=_widen_fn,
    )

    layer_filter = layer_filter or Layer.is_visible
    target_group = group if isinstance(group, GroupMixin) and not as_layer else [group]
    layers: list[Layer] = list(target_group)  # type: ignore[arg-type]
    overrides = dict(layer_text_gamma or {})
    reference: Callable[[], np.ndarray | None] | None = None
    if text_preview and isinstance(group, PSDImage) and text_gamma is not None:
        reference = functools.partial(_text_reference, group)

    rc = rcache.active()
    doc_key = None
    if (
        rc is not None
        and layer_filter is Layer.is_visible
        and not (as_layer and isinstance(group, PSDImage))
    ):
        doc_key = rcache.doc_key(cast(PSDImage, _psd))
    # A layer whose type the preview decides records that in the pass, so
    # checkpoints stop below the first one.
    settled_below = len(layers)
    if doc_key is not None and reference is not None:
        settled_below = _first_text_decision(layers, overrides, _psd.color_mode)

    def composite_pass(
        chosen: dict[int, bool], box: tuple[int, int, int, int] = viewport
    ) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], _TextBlend]:
        """One pass over ``box``, a part of the viewport, with ``chosen`` kept."""
        on = _within(viewport, box)
        cache = _PassCache()
        cache.text.overrides = overrides
        cache.text.reference = reference
        cache.text.chosen = dict(chosen)
        compositor = Compositor(
            box,
            backdrop_color[on],
            backdrop_alpha[on],
            layer_filter,
            force,
            document_backdrop=lambda: _document_backdrop(_psd, box),
            widen=_widen_fn,
            color_mode=_psd.color_mode,
            cache=cache,
            text_gamma=text_gamma,
        )
        if doc_key is None:
            for layer in layers:
                compositor.apply(layer)
            return compositor.finish(), cache.text
        cache.rc, cache.doc_key = rc, doc_key
        # What the pass starts from, besides its layers.
        seed = [
            doc_key,
            repr(
                (box, force, as_layer, text_gamma, sorted(overrides.items()))
            ).encode(),
            _array_bytes(color),
            _array_bytes(alpha),
        ]
        if not isinstance(group, PSDImage):
            # Deep knockout reads the document's bottom layer.
            seed.append(rcache.layer_key(group, cache.layer_keys))
            seed.append(rcache.layer_key(cast(Layer, _psd[0]), cache.layer_keys))
            seed.append(b"%d" % group.is_visible())
        _apply_cached(compositor, layers, seed, settled_below)
        return compositor.finish(), cache.text

    result, text = composite_pass({})
    leaning = text.leaning
    reference_array = text.reference_array()
    if not leaning or reference_array is None:
        return result
    # A layer that leaned straight where it went on is settled on the finished
    # canvas, against the same composite with it straight: worked out only
    # over its edge, then over the whole viewport if it does settle.
    rows = np.concatenate([pixels[0] for pixels in leaning.values()])
    columns = np.concatenate([pixels[1] for pixels in leaning.values()])
    box = (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )
    on = _within(viewport, box)
    tried, _ = composite_pass(text.chosen | dict.fromkeys(leaning, True), box)
    settled = {
        key
        for key, pixels in leaning.items()
        if _settles_straight(
            result[0][on], tried[0], result[2][on], box, reference_array, pixels
        )
    }
    if not settled:
        return result
    return composite_pass(text.chosen | dict.fromkeys(settled, True))[0]


def _first_text_decision(
    layers: list[Layer], overrides: Mapping[int, float | None], color_mode: ColorMode
) -> int:
    """The first of ``layers`` holding a type layer the preview may decide.

    See :py:class:`_TextBlend`; a layer pinned in ``overrides`` is decided.
    """
    for index, layer in enumerate(layers):
        tree = [layer]
        if isinstance(layer, GroupMixin):
            tree.extend(layer.descendants())
        if any(
            _blends_in_text_space(item, color_mode) and item.layer_id not in overrides
            for item in tree
        ):
            return index
    return len(layers)


def _within(
    viewport: tuple[int, int, int, int], box: tuple[int, int, int, int]
) -> tuple[slice, slice]:
    """Where ``box``, on the document, falls on a canvas over ``viewport``."""
    return (
        slice(box[1] - viewport[1], box[3] - viewport[1]),
        slice(box[0] - viewport[0], box[2] - viewport[0]),
    )


def _array_bytes(value: float | Sequence[float] | np.ndarray) -> bytes:
    array = np.asarray(value)
    return repr((array.dtype.str, array.shape)).encode() + array.tobytes()


# A checkpoint is the whole compositor state after a top-level layer.
_CHECKPOINT = ("_color", "_alpha", "_shape_g", "_alpha_g")


def _apply_cached(
    compositor: "Compositor",
    layers: list[Layer],
    seed: list[bytes],
    settled_below: int,
) -> None:
    """Apply ``layers``, resuming after the longest prefix already rendered.

    Each key chains the one before it, so it names everything composited so
    far. Only a settled state is stored or restored: a clip layer is
    composited by the layer it clips to, so the state after that layer
    depends on the clip layers above it too. Nor is one at or past
    ``settled_below``, the first layer whose text blend the pass decides.
    """
    cache = compositor._cache
    rc = cache.rc
    assert rc is not None
    digest = hashlib.blake2b(digest_size=16)
    for part in seed:
        digest.update(b"%d:" % len(part) + part)
    chain = digest.digest()
    keys = []
    for layer in layers:
        chain = hashlib.blake2b(
            chain + rcache.layer_key(layer, cache.layer_keys), digest_size=16
        ).digest()
        keys.append(("prefix", chain))
    # The arrays already stored, by the identity of the canvas they copy, so
    # a layer that leaves a canvas as it was stores no second copy.
    stored: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    start = 0
    for index in range(min(settled_below, len(layers)) - 1, -1, -1):
        if _settled(layers, index) and keys[index] in rc:
            state = rc.get(keys[index])
            assert state is not None
            for name, array in zip(_CHECKPOINT, state, strict=True):
                canvas = array.copy()
                setattr(compositor, name, canvas)
                stored[name] = (canvas, array)
            start = index + 1
            break
    else:
        rc.misses["prefix"] += 1
    for index in range(start, len(layers)):
        compositor.apply(layers[index])
        if index >= settled_below or not _settled(layers, index):
            continue
        copies = []
        for name in _CHECKPOINT:
            canvas = getattr(compositor, name)
            known = stored.get(name)
            if known is None or known[0] is not canvas:
                known = stored[name] = (canvas, canvas.copy())
            copies.append(known[1])
        rc.put(keys[index], tuple(copies))


def _settled(layers: list[Layer], index: int) -> bool:
    """Whether no clip layer above ``layers[index]`` is still to come."""
    return index + 1 == len(layers) or not layers[index + 1].clipping


def paste(
    viewport: tuple[int, int, int, int],
    bbox: tuple[int, int, int, int],
    values: np.ndarray,
    background: float | None = None,
) -> np.ndarray:
    """Change to the specified viewport."""
    shape = (viewport[3] - viewport[1], viewport[2] - viewport[0], values.shape[2])
    view = (
        np.full(shape, background, dtype=np.float32)
        if background is not None
        else np.zeros(shape, dtype=np.float32)
    )
    inter = utils.intersect(viewport, bbox)
    if inter == (0, 0, 0, 0):
        return view

    v = (
        inter[0] - viewport[0],
        inter[1] - viewport[1],
        inter[2] - viewport[0],
        inter[3] - viewport[1],
    )
    b = (inter[0] - bbox[0], inter[1] - bbox[1], inter[2] - bbox[0], inter[3] - bbox[1])
    view[v[1] : v[3], v[0] : v[2], :] = values[b[1] : b[3], b[0] : b[2], :]
    return view


def _is_background_layer(layer: "LayerProtocol") -> bool:
    """Whether the layer is Photoshop's Background layer.

    A Background layer is opaque by construction and carries no transparency
    channel, which is what distinguishes it from an ordinary bottom layer that
    merely happens to be fully opaque.
    """
    record = getattr(layer, "_record", None)
    if record is None:
        return False
    return not any(
        channel.id == ChannelID.TRANSPARENCY_MASK for channel in record.channel_info
    )


def _read_knockout(layer: Layer) -> Knockout:
    """Read a layer's knockout setting, tolerating undefined values.

    The tagged block is a raw byte, so a corrupt file -- or one written by a
    future Photoshop or a third-party tool -- can carry a value outside the
    enum. Fall back to NONE with a warning rather than raising: compositing a
    malformed document should degrade, not crash.
    """
    value = layer.tagged_blocks.get_data(Tag.KNOCKOUT_SETTING, 0)
    try:
        return Knockout(value)
    except ValueError:
        logger.warning(
            "Unknown knockout setting %r in %s; compositing without knockout",
            value,
            layer,
        )
        return Knockout.NONE


def _excluded_channels(layer: Layer, channels: int) -> list[int]:
    """The colour channels a layer's Blending Options > Channels leaves out.

    Indices into a canvas ``channels`` wide; one beyond it names no colour
    channel and is ignored.
    """
    restricted = layer.tagged_blocks.get_data(Tag.CHANNEL_BLENDING_RESTRICTIONS_SETTING)
    if not restricted:
        return []
    return sorted({index for index in restricted if index < channels})


# Each document's preview as _text_reference() read it, with the stored bytes
# it was read from: a save that regenerates the preview replaces them.
_KeptReference = tuple[bytes, np.ndarray | None]
_TEXT_REFERENCES: weakref.WeakKeyDictionary[PSDImage, _KeptReference] = (
    weakref.WeakKeyDictionary()
)


def _text_reference(psd: PSDImage) -> np.ndarray | None:
    """The document's stored preview, 8-bit, or None without one to read.

    Read once and kept with the document until its stored preview changes.
    """
    stored = psd._record.image_data.data
    kept = _TEXT_REFERENCES.get(psd)
    if kept is not None and kept[0] is stored:
        return kept[1]
    array = _read_text_reference(psd)
    _TEXT_REFERENCES[psd] = (stored, array)
    return array


def _read_text_reference(psd: PSDImage) -> np.ndarray | None:
    """:py:meth:`PSDImage.topil` without the profile, as an array, or None.

    A side read: a preview that does not decode, or that the document's
    allocation budget does not allow, is not read, and says nothing. The
    decode raises rather than warns, as a warning filter would be
    process-global.
    """
    header = psd._record.header
    if (
        psd.color_mode != ColorMode.RGB
        or psd.depth == 32
        or header.channels < 3
        or not psd.has_preview()
    ):
        return None
    image_data = psd._record.image_data
    try:
        check_pixel_size(
            psd.width,
            psd.height,
            psd.channels,
            max_alloc_bytes=psd._max_alloc_bytes,
            estimated_bytes=pil_io._image_data_peak_bytes(psd, None, False),
            warn=False,
        )
        data = _decompress(
            image_data.data,
            image_data.compression,
            header.width,
            header.height * header.channels,
            header.depth,
            header.version,
            strict=True,
        )
        plane = len(data) // header.channels
        planes = [
            pil_io._create_image(
                (psd.width, psd.height),
                data[index * plane : (index + 1) * plane],
                psd.depth,
            )
            for index in range(header.channels)
        ]
        image = Image.merge("RGB", planes[:3])
        if has_transparency(psd):
            image.putalpha(planes[get_transparency_index(psd)])
        array = np.asarray(pil_io._remove_white_background(image, psd.color_mode))
    except ValueError:
        return None
    if array.ndim != 3 or array.shape[:2] != (psd.height, psd.width):
        return None
    return array


def _document_backdrop(
    psd: "PSDProtocol | None", viewport: tuple[int, int, int, int]
) -> tuple[np.ndarray, np.ndarray] | None:
    """The backdrop that deep knockout knocks out to.

    Photoshop treats the Background layer as the canvas rather than as an
    ordinary layer: deep knockout removes every ordinary layer beneath the
    knockout group and stops there. Verified against Photoshop 2026 -- with the
    Background converted to an ordinary layer, the same document knocks all the
    way through to transparency instead.

    Returns None when the document has no Background layer, in which case deep
    knockout falls back to the compositor's own initial backdrop.
    """
    if psd is None or len(psd) == 0:
        return None
    bottom = psd[0]
    if not _is_background_layer(bottom):
        return None
    color = bottom.numpy("color")
    if color is None:
        return None
    color = paste(viewport, bottom.bbox, color, 1.0)
    alpha = np.ones((color.shape[0], color.shape[1], 1), dtype=np.float32)
    return color, alpha


def _uniform_alpha(alpha: float | np.ndarray) -> float | None:
    """The backdrop alpha as a plain float when it is a single scalar.

    None for an array-valued backdrop, which is not scanned: an all-ones array
    reported as opaque would drop the alpha channel from images that carry one.
    Tested by dimensionality so ``1``, ``1.0``, ``np.float32(1.0)`` and
    ``np.array(1.0)`` behave alike.
    """
    return float(alpha) if np.ndim(alpha) == 0 else None


def _widen(color: np.ndarray, channels: int) -> np.ndarray:
    """Replicate a single-channel array across ``channels``.

    A backdrop is widened once where it is handed over, since canvases keep a
    fixed width; a source is widened there too because what a grey means is the
    document's answer, not the blend arithmetic's (#749).

    Replication is wrong for CMYK and Lab, so this is only the fallback when no
    document is available; everything else uses :py:func:`widen.make_widen` (#722).
    """
    if color.shape[2] == 1 and channels > 1:
        return np.repeat(color, channels, axis=2)
    return color


def _to_canvas(
    value: float | Sequence[float] | np.ndarray,
    shape: tuple[int, int, int],
    name: str,
    exact_channels: bool = False,
    widen: _Widen = _widen,
) -> np.ndarray:
    """Expand a backdrop component to a full ``(height, width, channels)`` array.

    An array that is already per-pixel is taken as given, except that a
    single-channel one is widened to the document's channel count: the
    compositor's canvases keep a fixed width for their whole lifetime.
    ``exact_channels`` rejects any mismatch outright instead, for alpha, which
    is single-channel by definition and would otherwise reach
    ``composite_pil()`` and be concatenated into a color array of the wrong
    width.

    A scalar is one color component, exactly as that single-channel array is,
    so it is widened the same way rather than being broadcast across channels
    that do not share an axis (#753). A per-channel sequence is broadcast as
    given: its width is the caller's statement of the colour, not something to
    reinterpret.
    """
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 3:
        if exact_channels:
            if array.shape != shape:
                raise ValueError(
                    f"Backdrop {name} has shape {array.shape!r}, expected {shape!r}"
                )
        elif array.shape[:2] != shape[:2]:
            raise ValueError(
                f"Backdrop {name} covers {array.shape[:2]!r}, expected {shape[:2]!r} to match the viewport"
            )
        elif array.shape[2] not in (1, shape[2]):
            # A single channel replicates; any other width has no reading, and
            # left alone it would surface as a broadcast error from inside the
            # blend arithmetic instead.
            raise ValueError(
                "Backdrop %s has %d channels, expected 1 or %d for this "
                "color mode" % (name, array.shape[2], shape[2])
            )
        else:
            array = widen(array, shape[2])
        return array
    if not exact_channels and array.size == 1 and shape[2] > 1:
        # One color component, which is what the single-channel array above is
        # too -- so it takes the same conversion rather than being broadcast
        # across channels that do not share an axis. Broadcasting would make the
        # two spellings of one backdrop disagree: on a Lab document
        # `color=1.0` -- the public default -- would come out (255, 255, 255),
        # maximum chroma at maximum lightness, where the one-channel array
        # spelling of it gives the (255, 128, 128) that is white (#753).
        single = np.full((shape[0], shape[1], 1), array, dtype=np.float32)
        return widen(single, shape[2])
    try:
        return np.full(shape, array, dtype=np.float32)
    except ValueError:
        raise ValueError(
            f"Backdrop {name} {value!r} cannot be expanded to {shape!r}; pass a scalar, a "
            "per-channel sequence, or a full (height, width, channels) array"
        ) from None


def _normalize_backdrop(
    color: float | Sequence[float] | np.ndarray,
    alpha: float | np.ndarray,
    height: int,
    width: int,
    channels: int | None,
    widen: _Widen = _widen,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve a backdrop given in any accepted spelling to a pair of arrays.

    The public entry points accept the backdrop as a scalar, a per-channel
    sequence or a ready-made array because that is convenient to call; the
    compositor only ever works with arrays. Converting once here keeps the
    three spellings from being re-interpreted -- inconsistently -- at each use
    site.

    This does materialize a full canvas even for a constant backdrop, exactly
    as ``Compositor`` did before: the scalar spelling is an API convenience,
    not an allocation-avoidance path. (``_get_mask``/``_get_const`` keep their
    scalars for that reason instead.)

    ``channels`` is the width of the canvas to build -- the document's own
    color channel count, per ``get_color_channels()`` -- or None to infer it
    from ``color`` where no document is available to ask, the same defensive
    fallback ``check_pixel_size()`` is given in ``composite()``.
    """
    if channels is None:
        channels = 1 if np.ndim(color) == 0 else int(np.shape(color)[-1])
    return (
        _to_canvas(color, (height, width, channels), "color", widen=widen),
        _to_canvas(alpha, (height, width, 1), "alpha", exact_channels=True),
    )


def _blend_backdrop(
    color: np.ndarray,
    shape: np.ndarray,
    backdrop_color: np.ndarray,
    backdrop_alpha: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend foreground color/shape over a backdrop using standard "over" compositing."""
    result_alpha = utils.union(shape, backdrop_alpha)
    result_color = utils.clip(
        utils.divide(
            color * shape + backdrop_color * backdrop_alpha * (1.0 - shape),
            result_alpha,
        )
    )
    return result_color, result_alpha


# A pixel the path covers whole, which the rasterizer reports a float32 step
# short of 1.0.
_PATH_FULL = 1e-5


def _is_shape_layer(layer: Layer) -> bool:
    """A fill clipped to a path, whose stored coverage is rasterized from it."""
    return (
        not isinstance(layer, GroupMixin)
        and layer.has_pixels()
        and layer.vector_mask is not None
        and not layer.vector_mask.disabled
        and utils.has_fill(layer)
    )


@dataclass(frozen=True)
class _Source:
    """A layer resolved into the operands the blend equations take.

    All three canvases cover the full viewport, but they carry different
    factors: ``shape`` has the layer's mask folded in, ``alpha`` has the mask
    and the layer opacity, and ``color`` has neither -- it is the layer's own
    color, with any clip layers already composited onto it.

    ``fill_opacity`` is folded into none of them, because the two composite
    paths apply it differently -- see ``Compositor._composite_source()``. A
    source that has been through :py:class:`_EffectCanvas` is the exception:
    the layer's effects are not scaled by it and its own paint is, so there is
    no one factor left to apply and the canvas folds it in, leaving 1.0 here
    and the coverage a knockout punches with in ``knockout_shape``.
    """

    color: np.ndarray
    shape: np.ndarray
    alpha: np.ndarray
    # The mask coverage scaled by the layer opacity: how much of the source
    # replaces the backdrop along the pass-through path.
    mask: float | np.ndarray
    # The mask coverage on its own. A stroke effect on a layer that has no
    # pixels of its own traces this rather than ``shape``.
    shape_mask: float | np.ndarray
    # Tag.BLEND_FILL_OPACITY, as a 0-1 fraction.
    fill_opacity: float
    # The layer opacity, as a 0-1 fraction. Already in ``alpha`` and in
    # ``mask``; carried on its own because an effect composited into the layer
    # is scaled by it too, and ``alpha / shape`` is not it for a group.
    opacity: float
    knockout: Knockout
    # Whether the group isolated its adjustments; None for a layer that is not
    # a group. Only an explicit False makes a pass-through group composite as
    # one -- see _composite_source().
    adjustment_isolated: bool | None
    # What a knockout punches its hole with. ``shape`` before fill opacity is
    # applied, which is the same array until an effect canvas has folded that
    # in: fill opacity fades the layer's own paint without narrowing the hole
    # it punches, which is what makes "fill 0 under a knockout" a hole at all.
    knockout_shape: np.ndarray
    # What the layer's effects cover, under its mask and before its opacity:
    # ``shape`` for an object, the composited alpha of a group, so a child
    # faded to nothing carries none of the group's style. Only an effect
    # canvas reads it, so a group with no style to draw carries ``shape``.
    region: np.ndarray
    # The gamma a type layer's edge coverage blends in, or None to blend in
    # the document's encoding like every other source.
    text_gamma: float | None = None
    # The type layer's ``id()`` while its blend is still to be read off the
    # reference: see :py:class:`_TextBlend`.
    text_key: int | None = None


def _blends_in_text_space(layer: Layer, color_mode: ColorMode | None) -> bool:
    """Whether Photoshop mixes this layer's glyph edges in the text gamma.

    Only a type layer in an RGB document, and only one Photoshop renders
    itself: a layer whose font was substituted (``substitutesUsed``) or whose
    text data predates the ``bounds`` key keeps its stored raster and is
    composited straight, like a pixel layer. A 32-bit document's channels
    are linear already and mix straight too.
    """
    if not isinstance(layer, TypeLayer):
        return False
    if color_mode not in (None, ColorMode.RGB) or layer._psd.depth == 32:
        return False
    # Unreadable text data leaves nothing but the stored pixels.
    if layer._type_data is None:
        return False
    text_data = layer._type_data.text_data
    return b"bounds" in text_data and not text_data.get(b"substitutesUsed", False)


# A type layer's edge pixel counts where the straight and gamma mixes differ by
# more than this, and only a layer with at least the count of them is read.
# Straight wins on that share of them with a median miss within the fit (the
# reference is 8-bit); a layer that only leans straight, on the lean's share,
# is settled on the finished canvas instead.
_TEXT_DECISIVE = 3.0 / 255.0
_TEXT_MIN_PIXELS = 16
_TEXT_STRAIGHT_SHARE = 0.9
_TEXT_FIT = 2.0 / 255.0
_TEXT_LEAN_SHARE = 0.5


def _favours_straight(
    gamma: np.ndarray, straight: np.ndarray, reference: np.ndarray
) -> tuple[float, float]:
    """How the reference splits between two mixes of the same edge pixels.

    Returns the share of pixels the straight mix is nearer on, and its median
    miss. All three are ``(pixels, channels)``.
    """
    miss_gamma = np.abs(gamma - reference).max(axis=1)
    miss_straight = np.abs(straight - reference).max(axis=1)
    return float((miss_straight < miss_gamma).mean()), float(np.median(miss_straight))


def _edge_evidence(
    viewport: tuple[int, int, int, int],
    box: tuple[int, int, int, int],
    reference: np.ndarray,
    mixed: np.ndarray,
    canvas: np.ndarray,
    color: np.ndarray,
    shape: np.ndarray,
    alpha: np.ndarray,
    alpha_previous: np.ndarray,
    alpha_new: np.ndarray,
    color_b: np.ndarray,
    alpha_b: np.ndarray,
    blend_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> tuple[float, float, np.ndarray, np.ndarray] | None:
    """How ``reference`` splits between a type layer's two mixes, where it goes on.

    ``mixed`` is the canvas with the layer's edge mixed in the text gamma; the
    straight mix is worked out beside it on ``box``, (top, left, bottom,
    right) on the canvas, which stands at ``viewport`` on the reference.
    Returns :py:func:`_favours_straight`'s share and fit, and the rows and
    columns on the document of the pixels read; None for too few of them.
    """
    top, left, bottom, right = box
    height, width = reference.shape[:2]
    top = max(top, -viewport[1])
    left = max(left, -viewport[0])
    bottom = min(bottom, height - viewport[1])
    right = min(right, width - viewport[0])
    if top >= bottom or left >= right:
        return None
    on = (slice(top, bottom), slice(left, right))
    seen = (
        reference[
            top + viewport[1] : bottom + viewport[1],
            left + viewport[0] : right + viewport[0],
        ]
        / 255.0
    )
    channels = canvas.shape[2]
    edge = (alpha[on][..., 0] > 0.0) & (alpha[on][..., 0] < 1.0)
    edge &= alpha_new[on][..., 0] > 0.99
    if seen.shape[2] > channels:
        edge &= seen[..., channels] > 0.99
    if not edge.any():
        return None
    backdrop = color_b[on]
    shape_on, alpha_on, alpha_b_on = shape[on], alpha[on], alpha_b[on]
    lerped = (shape_on - alpha_on) * alpha_b_on * backdrop + alpha_on * (
        (1.0 - alpha_b_on) * color[on] + alpha_b_on * blend_fn(backdrop, color[on])
    )
    lerped = utils.clip(
        utils.divide(
            (1.0 - shape_on) * alpha_previous[on] * canvas[on] + lerped, alpha_new[on]
        )
    )
    gamma = mixed[on]
    read = edge & (np.abs(gamma - lerped).max(axis=2) > _TEXT_DECISIVE)
    if int(read.sum()) < _TEXT_MIN_PIXELS:
        return None
    share, fit = _favours_straight(
        gamma[read], lerped[read], seen[..., :channels][read]
    )
    rows, columns = np.nonzero(read)
    return share, fit, rows + top + viewport[1], columns + left + viewport[0]


def _settles_straight(
    first: np.ndarray,
    second: np.ndarray,
    alpha: np.ndarray,
    viewport: tuple[int, int, int, int],
    reference: np.ndarray,
    pixels: tuple[np.ndarray, np.ndarray],
) -> bool:
    """Whether the finished canvas shows a leaning layer's edge straight.

    ``first`` is the canvas with the layer in the gamma, ``second`` with it
    straight, both on ``viewport``; ``pixels`` are its edge pixels on the
    document.
    """
    rows, columns = pixels[0] - viewport[1], pixels[1] - viewport[0]
    inside = (
        (rows >= 0)
        & (rows < first.shape[0])
        & (columns >= 0)
        & (columns < first.shape[1])
    )
    rows, columns = rows[inside], columns[inside]
    channels = first.shape[2]
    seen = reference[rows + viewport[1], columns + viewport[0]] / 255.0
    gamma, straight = first[rows, columns], second[rows, columns]
    read = (alpha[rows, columns, 0] > 0.99) & (
        np.abs(gamma - straight).max(axis=1) > _TEXT_DECISIVE
    )
    if seen.shape[1] > channels:
        read &= seen[:, channels] > 0.99
    if int(read.sum()) < _TEXT_MIN_PIXELS:
        return False
    share, fit = _favours_straight(gamma[read], straight[read], seen[read, :channels])
    return share >= _TEXT_STRAIGHT_SHARE and fit <= _TEXT_FIT


def _composites_as_passthrough(source: _Source, blend_mode: BlendMode) -> bool:
    """Whether this source goes on by interpolation rather than by compositing.

    ``is False``, not a truth test: when a group isolates its adjustments,
    pass-through falls back to over composing, and None means not a group.
    Spelled once because the composite and the outer-effect ordering must agree.
    """
    return blend_mode == BlendMode.PASS_THROUGH and source.adjustment_isolated is False


# The effects "Blend Interior Effects as Group" decides about, as ``_readable``
# names them. Adobe names inner glow, satin and the colour and gradient
# overlays "such as"; a pattern overlay and an inner shadow paint inside the
# layer the same way, and no corpus layer tests them (#846).
_INTERIOR_EFFECTS = (
    "coloroverlay",
    "gradientoverlay",
    "patternoverlay",
    "satin",
    "innerglow",
    "innershadow",
)


def _blends_interior_apart(layer: Layer, source: _Source) -> bool:
    """Whether the layer's interior effects go on over it rather than into it.

    "Blend Interior Effects as Group" off, Photoshop's default, puts them on
    the layer once it is composited with its mode
    (:py:meth:`Compositor._apply_apart`). Under Normal at full fill the two
    ways agree, so only another mode or a lower fill takes this path.
    """
    if source.knockout or source.opacity <= 0.0:
        return False
    if _composites_as_passthrough(source, layer.blend_mode):
        return False
    if (
        layer.blend_mode in _COVERS_BENEATH | {BlendMode.PASS_THROUGH}
        and source.fill_opacity >= 1.0
    ):
        return False
    if layer.tagged_blocks.get_data(Tag.BLEND_INTERIOR_ELEMENTS, 0):
        # TODO: blended as a group, fill opacity fades the interior effects
        # too, which _EffectCanvas does not do (#846).
        return False
    return _has_enabled_effects(layer) and any(
        _readable(layer, name) for name in _INTERIOR_EFFECTS
    )


class _OuterEffect(NamedTuple):
    """What an effect paints *beside* the layer, held until the layer is ready.

    An outer effect goes on before the layer and keeps its own blend mode
    against the backdrop, so unlike an inner one it is not folded into the
    layer's source -- see :py:meth:`Compositor._apply_outer_effects`. It is
    collected rather than applied on the spot because the scale it goes on at
    is read off the layer's finished alpha, which the inner effects are still
    changing while the stroke is being drawn.
    """

    color: np.ndarray
    coverage: np.ndarray
    # The effect's own opacity with the layer's already in it. The layer
    # opacity fades a layer's style along with the layer, which the inner
    # effects get from being composited inside the source it scales; an outer
    # effect is not in that source, so it carries the factor itself. Per
    # pixel for merged stroke bands painted at different opacities.
    opacity: float | np.ndarray
    blend_mode: BlendMode
    # The alpha the layer's outer effects listed above this one paint on the
    # pixel, which this one is disjoint from -- see
    # :py:meth:`Compositor._apply_outer_effects`.
    above: float | np.ndarray = 0.0


class _StrokeBand(NamedTuple):
    """A stroke effect's paint and coverage, measured before any is painted.

    Or several bands as one: adjacent bands of one blend mode are disjoint
    once each is cut from under the ones above it, so they add, and one
    source carries their summed coverage, the opacity each pixel is painted
    at and the colour weighted by it -- see :py:class:`_BandSum`.
    """

    color: np.ndarray
    coverage: np.ndarray
    # The part of ``coverage`` inside the boundary the band was traced from.
    inside: np.ndarray
    opacity: float | np.ndarray
    blend_mode: BlendMode


class _StrokeHalves(NamedTuple):
    """A run of stroke bands split at the layer's region, before it is painted.

    ``share`` is the half inside the region as a share of it, ``beside`` the
    half outside; ``above_share`` and ``above_beside`` are what the runs
    listed above take off each.
    """

    band: _StrokeBand
    share: np.ndarray
    beside: np.ndarray
    above_share: float | np.ndarray
    above_beside: float | np.ndarray


class _BandSum:
    """Adjacent stroke bands of one blend mode, added as they are measured.

    The bands are disjoint, so one source carrying their sum blends each
    against the backdrop alone, where painted one over the other the upper
    would blend against the lower's paint on a pixel the two share. The sums
    are kept in place of the bands, so the memory held is per run of one
    blend mode and not per stroke, and a lone band is handed back as it is.
    """

    def __init__(self, band: _StrokeBand) -> None:
        self.blend_mode = band.blend_mode
        self._lone: _StrokeBand | None = band
        self._coverage = band.coverage
        self._inside = band.inside
        self._weight: np.ndarray | None = None
        self._color: np.ndarray | None = None

    def add(self, band: _StrokeBand) -> None:
        assert band.blend_mode == self.blend_mode
        if self._lone is not None:
            self._weight = self._lone.coverage * self._lone.opacity
            self._color = self._weight * self._lone.color
            self._lone = None
        assert self._weight is not None and self._color is not None
        weight = band.coverage * band.opacity
        self._coverage = self._coverage + band.coverage
        self._inside = self._inside + band.inside
        self._weight = self._weight + weight
        self._color = self._color + weight * band.color

    def band(self) -> _StrokeBand:
        if self._lone is not None:
            return self._lone
        assert self._weight is not None and self._color is not None
        # Bands that overrun a pixel between them are clamped to it, and the
        # weight with them so the opacity painted stays at most 1.
        coverage = np.minimum(self._coverage, 1.0)
        weight = np.minimum(self._weight, coverage)
        return _StrokeBand(
            utils.divide(self._color, self._weight, fill=0.0),
            coverage,
            np.minimum(self._inside, 1.0),
            utils.divide(weight, coverage, fill=0.0),
            self.blend_mode,
        )


class _EffectSink(Protocol):
    """Where the effects drawn inside a layer's region are put.

    :py:class:`_EffectCanvas` composites them into the layer, which is then
    composited as one source; :py:class:`_AdjustmentCanvas` puts them straight
    on the compositor, which has no layer paint to composite them into. The
    drawing code hands either one a share of ``region`` and the effect's own
    colour, blend mode and opacity, and does not know which it has.
    """

    @property
    def region(self) -> np.ndarray: ...

    @property
    def opacity(self) -> float: ...

    def within(self, coverage: np.ndarray) -> np.ndarray: ...

    def over(
        self,
        coverage: np.ndarray,
        color: np.ndarray,
        blend_mode: BlendMode,
        opacity: float | np.ndarray,
        fades_source: bool = False,
    ) -> None: ...

    def replace(
        self, coverage: np.ndarray, color: np.ndarray, opacity: float | np.ndarray
    ) -> None: ...

    @property
    def knocks_out(self) -> bool:
        """Whether :py:meth:`replace` takes all it covers off what is under it.

        Otherwise it takes its opacity's share, as :py:meth:`over` does.
        """
        ...


class _EffectCanvas:
    """The layer's own canvas, where the effects inside it composite into it.

    Photoshop composites a layer's effects with the layer and puts the result
    on the backdrop, so an inset stroke renders at ``alpha*S + (1 - alpha)*B``.
    Putting the layer on the backdrop and each effect on top of that instead
    leaves a backdrop term in every pixel where the layer's own coverage is
    partial -- ``t*S + (1 - t)*(alpha*L + (1 - alpha)*B)`` -- knocking the
    layer out from under itself (#846).

    What the effect paints is the same either way; where it goes is not. An
    *inner* effect -- an overlay, or the part of a stroke within the layer's
    boundary -- covers a share of the layer's own region, which is
    ``source.region``: it paints over the layer, knocks out what it covers, and
    never widens the region. The coverage it is given is that share rather
    than a coverage of the pixel, so that the layer's opacity and whatever
    alpha it has of its own stay out of the arithmetic until
    :py:meth:`source` puts them back, and an effect's own blend mode blends it
    against the layer it lands on.

    ``fill_opacity`` fades the layer's paint but not its effects, so it is
    folded in here rather than left for
    :py:meth:`Compositor._composite_source` to apply to both, and the coverage
    a knockout punches with travels in ``knockout_shape`` instead.

    The *outer* half of a stroke does not come through here at all: it is
    disjoint from the layer rather than inside it, and it keeps its own blend
    mode against the backdrop, which one merged source composited with the
    layer's blend mode could not express.
    """

    def __init__(self, compositor: "Compositor", source: _Source) -> None:
        self._compositor = compositor
        self._source = source
        self._painted = False

    def _start(self) -> None:
        """Build the canvases, once, when an effect actually draws on them.

        Deferred because most layers carry no effect that draws, and an effect
        that declines -- a pattern whose data will not decode -- must leave the
        source it was handed untouched rather than a rebuilt copy of it, which
        the round trip through :py:meth:`source` below would not be.
        """
        if self._painted:
            return
        source = self._source
        self._color = self._compositor._fit_source(source.color)
        # Inside the region and relative to it, with the layer opacity divided
        # back out: that opacity fades the layer and its effects alike, so it
        # applies once, to the whole stack, in source() below. Leaving it in
        # here and scaling each effect by it as well is what made an overlay
        # at full strength lift a half-opaque layer to 0.75.
        #
        # What is left is the layer's own alpha against its region, which is 1
        # inside it, faded by fill opacity.
        opacity = source.opacity
        self._alpha = utils.divide(source.alpha, self.region, fill=0.0) * (
            source.fill_opacity / opacity if opacity > 0.0 else 0.0
        )
        self._premultiplied = self._alpha * self._color
        # A scalar until something widens it, which is the same
        # allocation-avoidance path ``_get_mask()`` takes and for the same
        # reason: this canvas is held for the whole of the effect stack rather
        # than freed between effects the way a source is.
        self._shape: float | np.ndarray = source.fill_opacity
        self._painted = True

    @property
    def region(self) -> np.ndarray:
        """What the layer's effects cover, which ``over()`` is a share of."""
        return self._source.region

    @property
    def opacity(self) -> float:
        """The layer opacity, which fades its effects along with the layer."""
        return self._source.opacity

    def within(self, coverage: np.ndarray) -> np.ndarray:
        """A coverage of the pixel, as the share of the region it is.

        A stroke band is measured on the pixel, an inner effect on the region.
        Where the layer's boundary cuts an inset band, the result is the whole
        region, so the inset stroke knocks the layer out rather than blending.
        """
        return utils.clip(utils.divide(coverage, self.region, fill=0.0))

    def over(
        self,
        coverage: np.ndarray,
        color: np.ndarray,
        blend_mode: BlendMode,
        opacity: float | np.ndarray,
        fades_source: bool = False,
    ) -> None:
        """Paint an effect over the layer, inside its region.

        ``coverage`` is a share of the region, and ``opacity`` the effect's
        own; ``fades_source`` is whether that opacity fades the colour in the
        modes :py:func:`~psd_tools.composite.blend.fade_source` names. The
        layer's opacity is not applied here -- see :py:meth:`_start`.
        """
        if not coverage.any():
            return
        self._start()
        color = self._compositor._fit_source(color)
        blend_fn = get_blend_func(blend_mode, self._compositor._color_mode)
        if fades_source:
            # Only an overlay or a satin fades its colour, at one opacity.
            assert not isinstance(opacity, np.ndarray)
            blend_fn = fade_source(blend_fn, blend_mode, opacity)
        alpha = coverage * opacity
        # ``normal`` ignores what is under it, and is also what an effect whose
        # blend mode this module does not know answers, so the un-premultiply
        # is skipped rather than computed for a function that will drop it.
        under = (
            color
            if blend_fn is normal
            else utils.divide(self._premultiplied, self._alpha, fill=color)
        )
        self._premultiplied = (1.0 - alpha) * self._premultiplied + alpha * (
            (1.0 - self._alpha) * color + self._alpha * blend_fn(under, color)
        )
        self._alpha = alpha + (1.0 - alpha) * self._alpha
        # Coverage and opacity are separate everywhere in this module: the
        # effect's opacity fades what it paints without narrowing what it
        # covers, so the shape takes the coverage it was drawn on.
        self._shape = utils.union(self._shape, coverage)

    knocks_out = True

    def replace(
        self, coverage: np.ndarray, color: np.ndarray, opacity: float | np.ndarray
    ) -> None:
        """Paint an effect in place of the layer, inside its region.

        What ``coverage`` covers of the layer and the effects under this one
        is knocked out, so the effect's opacity shows the backdrop rather
        than the layer.
        """
        if not coverage.any():
            return
        self._start()
        color = self._compositor._fit_source(color)
        kept = 1.0 - coverage
        alpha = coverage * opacity
        self._premultiplied = kept * self._premultiplied + alpha * color
        self._alpha = kept * self._alpha + alpha
        self._shape = utils.union(self._shape, coverage)

    def source(self) -> _Source:
        """The layer and the effects inside it, as one source to composite."""
        if not self._painted:
            return self._source
        opacity = self._source.opacity
        alpha = self.region * self._alpha * opacity
        premultiplied = self.region * self._premultiplied * opacity
        return replace(
            self._source,
            color=utils.clip(utils.divide(premultiplied, alpha, fill=self._color)),
            shape=utils.clip(self.region * self._shape),
            alpha=utils.clip(alpha),
            # Unchanged: fill opacity, which the shape above now carries,
            # fades the layer's own paint without narrowing the hole it
            # punches -- that is what makes "fill 0 under a knockout" a hole
            # at all -- and an effect that draws inside the layer does not
            # move the hole either.
            fill_opacity=1.0,
        )


class _AdjustmentCanvas:
    """Where an adjustment layer's effects paint: straight onto the compositor.

    An adjustment layer has no paint of its own for an :py:class:`_EffectCanvas`
    to composite an effect into; its mask is its shape, and the adjustment has
    already rewritten the canvas under that shape by the time its effects are
    drawn. So each effect goes onto the compositor's canvas by itself, with its
    own blend mode against what is there, which is how Photoshop composites a
    layer's interior effects when they are not blended as a group -- and with
    nothing of the layer's to blend them into, the one way they can go on at
    all. The layer opacity is not applied here: it fades the adjustment and
    every effect as one, once they are all on
    (:py:meth:`Compositor._apply_styled_adjustment`).
    """

    def __init__(self, compositor: "Compositor", region: np.ndarray) -> None:
        self._compositor = compositor
        self._region = region

    @property
    def region(self) -> np.ndarray:
        """The mask coverage, which is what ``over()`` is a share of."""
        return self._region

    @property
    def opacity(self) -> float:
        """Whole: the layer opacity is applied to the finished style instead."""
        return 1.0

    def within(self, coverage: np.ndarray) -> np.ndarray:
        """A coverage of the pixel, as the share of the region it is."""
        return utils.clip(utils.divide(coverage, self._region, fill=0.0))

    def over(
        self,
        coverage: np.ndarray,
        color: np.ndarray,
        blend_mode: BlendMode,
        opacity: float | np.ndarray,
        fades_source: bool = False,
    ) -> None:
        """Composite an effect onto the canvas, inside the region."""
        if not coverage.any():
            return
        shape = self._region * coverage
        faded = 1.0
        if fades_source:
            assert not isinstance(opacity, np.ndarray)
            faded = opacity
        self._compositor._apply_source(
            color, shape, shape * opacity, blend_mode, faded=faded
        )

    knocks_out = False

    def replace(
        self, coverage: np.ndarray, color: np.ndarray, opacity: float | np.ndarray
    ) -> None:
        """As :py:meth:`over` Normal.

        The adjustment is already applied to the backdrop under the band and
        stays there; only the stroke's own opacity fades it.
        """
        self.over(coverage, color, BlendMode.NORMAL, opacity)


class _ApartCanvas(_AdjustmentCanvas):
    """Where the effects of a layer not blended as a group paint.

    Onto the compositor, over the layer's paint put on at full coverage, each
    at the share of the region it covers: the layer's coverage and opacity
    fade the lot once, afterwards (:py:meth:`Compositor._apply_apart`). What
    paints beside the layer is not faded there, so it is drawn at the layer
    opacity, as on an :py:class:`_EffectCanvas`.
    """

    def __init__(
        self, compositor: "Compositor", region: np.ndarray, opacity: float
    ) -> None:
        super().__init__(compositor, region)
        self._opacity = opacity

    @property
    def opacity(self) -> float:
        """The layer opacity, which fades what paints beside the layer."""
        return self._opacity

    def over(
        self,
        coverage: np.ndarray,
        color: np.ndarray,
        blend_mode: BlendMode,
        opacity: float | np.ndarray,
        fades_source: bool = False,
    ) -> None:
        """Composite an effect onto the canvas at its share of the region."""
        if not coverage.any():
            return
        faded = 1.0
        if fades_source:
            assert not isinstance(opacity, np.ndarray)
            faded = opacity
        self._compositor._apply_source(
            color, coverage, coverage * opacity, blend_mode, faded=faded
        )

    knocks_out = True

    def replace(
        self, coverage: np.ndarray, color: np.ndarray, opacity: float | np.ndarray
    ) -> None:
        """Paint an effect in place of the layer and the effects under it.

        A stroke is no interior effect, so it knocks out as on an
        :py:class:`_EffectCanvas`: what ``coverage`` covers goes back to the
        canvas the compositor was seeded with, and the effect goes on that.
        """
        if not coverage.any():
            return
        canvas = self._compositor
        color = canvas._fit_source(color)
        kept = 1.0 - coverage
        alpha = coverage * opacity
        premultiplied = (
            kept * canvas._alpha * canvas._color
            + (coverage - alpha) * canvas._alpha_0 * canvas._color_0
            + alpha * color
        )
        canvas._shape_g = cast(np.ndarray, utils.union(canvas._shape_g, coverage))
        canvas._alpha_g = kept * canvas._alpha_g + alpha
        canvas._alpha = cast(np.ndarray, utils.union(canvas._alpha_0, canvas._alpha_g))
        canvas._color = utils.clip(utils.divide(premultiplied, canvas._alpha))


@dataclass
class _Isolation:
    """How opaque an enclosing isolated group's content is, and what it stands on.

    Handed to a pass-through group nested inside an isolated one, on the
    nested group's viewport: its adjustments read the enclosing group's
    content rather than the canvas it was composited over, which takes the
    backdrop the enclosing group was seeded with -- see
    :py:meth:`Compositor._apply_adjustment`. A nested group that is not
    pass-through is seeded transparent, so its adjustments read its own
    content and it is handed none.
    """

    alpha: np.ndarray
    backdrop_color: np.ndarray
    backdrop_alpha: np.ndarray


class _HeldChannels(NamedTuple):
    """The channels a layer is excluded from, as they were before it went on."""

    channels: list[int]
    # The canvas's colour in those channels.
    color: np.ndarray
    # The alpha of the content they belong to.
    alpha: np.ndarray
    # ``color`` as a share of that content, or None to keep it as it is.
    premultiplied: np.ndarray | None


def _unmix(
    color: np.ndarray,
    backdrop_color: np.ndarray,
    backdrop_alpha: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """The content of ``alpha`` that, over the backdrop, composited to ``color``.

    Exact where the content went on Normal; where it blended, the backdrop's
    share is taken out as if it had. Where ``alpha`` is zero there is no
    content to divide out, and the canvas is handed back as it is.
    """
    return utils.clip(
        color
        + (color - backdrop_color)
        * (
            # A ratio of two alphas rather than a color, so the fill means
            # "fully opaque" here and not "white".
            utils.divide(backdrop_alpha, alpha, fill=1.0) - backdrop_alpha
        )
    )


def _remix(
    content: np.ndarray,
    backdrop_color: np.ndarray,
    backdrop_alpha: np.ndarray,
    alpha: np.ndarray,
    canvas_alpha: np.ndarray,
) -> np.ndarray:
    """``content`` at ``alpha`` composited Normal over the backdrop: _unmix() undone.

    ``canvas_alpha`` is the union of the two alphas, which the caller already
    holds.
    """
    return utils.clip(
        utils.divide(
            alpha * content + (1.0 - alpha) * backdrop_alpha * backdrop_color,
            canvas_alpha,
            fill=content,
        )
    )


# A bevel texture's colour planes, by its pattern's mode. An indexed pattern's
# one plane holds palette indices, not grey levels.
_TEXTURE_PLANES = {ColorMode.RGB: 3, ColorMode.GRAYSCALE: 1}

# Float32 planes of ``layer.bbox`` that _blend_unfiltered() allocates at its
# peak, beside the arrays it is given.
_UNFILTER_PLANES = 8


def _unfilter_under_mask(
    layer: SmartObjectLayer, color: np.ndarray, shape: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray | None]:
    """A filtered smart object's pixels with its filter mask applied.

    Blends the stored pixels with
    :py:meth:`~psd_tools.api.smart_object.SmartObject.filter_source` through
    the mask, in premultiplied space on ``layer.bbox``. A missing, disabled or
    all-white mask, a document that is not RGB, or any error reading the
    block leaves the stored pixels as they are.
    """
    try:
        so = layer.smart_object
        if (
            layer._psd.color_mode != ColorMode.RGB
            or color.shape[-1] != 3
            or not (so.has_enabled_filters and so.filter_mask_enabled)
        ):
            return color, shape
        held = sum(numpy_io._backing_bytes(a) for a in (color, shape) if a is not None)
        mask = so._filter_mask(held)
        if mask is None:
            return color, shape
        box = layer.bbox
        fill = 255.0 if so.filter_mask_extends_white else 0.0
        weight = paste(box, mask[0], mask[1][..., None], background=fill)
        del mask
        if weight.min() == 255:
            return color, shape
        source = so._filter_source(held + numpy_io._backing_bytes(weight))
        if source is None:
            return color, shape
        weight /= 255
        return _blend_unfiltered(layer, color, shape, weight, source)
    except Exception as exc:  # the stored pixels are the safe fallback
        logger.warning("Smart filter mask of %r not applied: %s", layer.name, exc)
        return color, shape


def _blend_unfiltered(
    layer: SmartObjectLayer,
    color: np.ndarray,
    shape: np.ndarray | None,
    weight: np.ndarray,
    source: tuple[tuple[int, int, int, int], np.ndarray, np.ndarray | None],
) -> tuple[np.ndarray, np.ndarray]:
    """``lerp(source, stored, weight)`` with straight colour, on ``layer.bbox``.

    *weight* is the filter mask on the box, 1 = filtered, and is overwritten.
    The filtered pixels stand where the source is not stored.
    """
    box = layer.bbox
    sbox, src_color, src_alpha = source
    height, width = color.shape[:2]
    inputs = (color, shape, weight, src_color, src_alpha)
    check_pixel_size(
        width,
        height,
        3,
        max_alloc_bytes=layer._psd._max_alloc_bytes,
        estimated_bytes=sum(numpy_io._backing_bytes(a) for a in inputs if a is not None)
        + _UNFILTER_PLANES * height * width * 4,
        warn=False,
    )
    stored = np.broadcast_to(np.float32(1), (*src_color.shape[:2], 1))
    unfiltered = paste(box, sbox, stored)
    unfiltered *= 1 - weight
    np.subtract(1, unfiltered, out=weight)
    if src_alpha is not None:
        src_shape = paste(box, sbox, src_alpha[..., None])
        src_shape /= 255
        unfiltered *= src_shape
        del src_shape
    if shape is not None:
        weight *= shape
    alpha = weight + unfiltered
    out = paste(box, sbox, src_color)
    unfiltered /= 255
    out *= unfiltered
    out += weight * color
    np.divide(out, alpha, out=out, where=alpha > 0)
    np.copyto(out, color, where=alpha <= 0)
    logger.debug("Smart filter mask applied to %r", layer.name)
    return out, alpha


class Compositor:
    """Composite context.

    ``color`` and ``alpha`` are the initial backdrop, and must already be
    ``(height, width, channels)`` arrays covering ``viewport``; the public
    entry points normalize the spellings they accept via
    ``_normalize_backdrop()``. An isolated group composites against a
    transparent backdrop, which the caller expresses by passing a zero
    ``alpha`` -- so that canvas is never built only to be discarded.

    ``color``'s channel count becomes ``self.channels`` and is fixed for this
    compositor's lifetime, so a caller handing over a canvas narrower than the
    document must widen it first. A narrow *source* is fine and must not change
    the width: ``_fit_source()`` widens it at the door, so the blend arithmetic
    only ever sees arrays of this compositor's own width.

    ``widen`` is how that is done, here and in every sub-compositor this one
    builds. Pass the document's own, from
    :py:func:`~psd_tools.composite.widen.make_widen`: widening is a colour
    conversion outside RGB, and the default is the mode-blind replication that
    is only correct when there is no document to ask (#722). A sub-compositor
    that does not pass it on returns its whole subtree to that fallback.

    ``color_mode`` is the document's, and travels the same way for the same
    reason: the six non-separable blend modes need to know what their operands'
    channels are and not merely how many, since a multichannel document's spot
    plates are neither RGB nor CMYK (#746). A sub-compositor that does not pass
    it on returns its whole subtree to deciding by width alone.

    Example::

        widen = make_widen(psd)
        color, alpha = _normalize_backdrop(
            1.0, 0.0, height, width, channels, widen=widen
        )
        compositor = Compositor(
            group.bbox, color, alpha, widen=widen, color_mode=psd.color_mode
        )
        for layer in group:
            compositor.apply(layer)
        color, shape, alpha = compositor.finish()
    """

    def __init__(
        self,
        viewport: tuple[int, int, int, int],
        color: np.ndarray,
        alpha: np.ndarray,
        layer_filter: Callable[[Layer], bool] | None = None,
        force: bool = False,
        adjustment_isolated: bool | _Isolation = False,
        document_backdrop: Callable[[], tuple[np.ndarray, np.ndarray] | None]
        | None = None,
        widen: _Widen = _widen,
        color_mode: ColorMode | None = None,
        cache: _PassCache | None = None,
        text_gamma: float | None = TEXT_BLEND_GAMMA,
        text_isolated: bool = False,
    ):
        self._viewport = viewport
        self._layer_filter = layer_filter
        self._force = force
        # The space a type layer's edge blends in, or None for the document's
        # encoding. ``text_isolated`` is whether that edge meets only what
        # this compositor has painted itself: Photoshop renders a group it
        # cannot pass through -- one that carries a layer style -- as one
        # layer, so the text inside never meets the backdrop the group was
        # seeded with here for its children's blend modes.
        self._text_gamma = text_gamma
        self._text_isolated = text_isolated
        self._mask_repeats: dict[int, bool] = {}
        # How a one-channel array becomes this compositor's width. Carried
        # rather than derived because the four sites that need it below are
        # inside this class, which holds no document handle (#722, #749).
        self._widen = widen
        # The document's colour mode, for the blend functions that need to know
        # what their operands' channels are and not just how many there are.
        # Carried for the same reason as ``widen``: ``_apply_source()`` holds no
        # document handle either. None means "no document to ask", and leaves
        # the width to decide alone (#746).
        self._color_mode = color_mode
        # Shared with every sub-compositor this one builds, so a group's
        # contents are measured once per composite pass rather than once per
        # enclosing group. A compositor built without one starts a pass of its
        # own, which is what the public entry points and the tests do.
        self._cache: _PassCache = _PassCache() if cache is None else cache
        # Whether adjustments stop at what the enclosing isolated group has
        # painted. An _Isolation is how opaque that group's content is so far
        # and the backdrop it stands on, handed down to a nested group that
        # does not isolate on its own; True starts a fresh isolated group,
        # whose adjustments stop at its own content.
        self._adjustment_isolated = adjustment_isolated is not False
        self._enclosing: _Isolation | None = (
            adjustment_isolated if isinstance(adjustment_isolated, _Isolation) else None
        )
        # What Knockout.DEEP knocks out to. Inherited by pass-through
        # sub-compositors and reset at every isolation boundary, so deep
        # knockout escapes pass-through groups but stops at an isolated one.
        # None means "this compositor's own initial backdrop".
        #
        # Resolved lazily: it decodes the Background layer, and the vast
        # majority of documents never contain a deep knockout at all.
        self._document_backdrop_fn = document_backdrop
        self._document_backdrop_resolved = False
        self._document_backdrop: tuple[np.ndarray, np.ndarray] | None = None

        # Preconditions rather than documentation: the whole point of fixing
        # the channel count is that nothing downstream has to re-check it.
        assert color.ndim == 3, "backdrop color must be a (h, w, c) canvas"
        assert color.shape[:2] == (self.height, self.width), (
            f"backdrop color {color.shape!r} does not cover viewport {self._viewport!r}"
        )
        assert alpha.shape == (self.height, self.width, 1), (
            f"backdrop alpha {alpha.shape!r} must be single-channel over viewport {self._viewport!r}"
        )

        self._alpha_0 = alpha
        self._color_0 = color
        # The channel count is fixed for this compositor's lifetime; every
        # canvas it hands to or takes from the blend equations is this wide.
        self._channels = color.shape[2]

        self._shape_g = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self._alpha_g = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self._color = self._color_0
        self._alpha = self._alpha_0

    def apply(self, layer: Layer, clip_compositing: bool = False) -> None:
        """Composite a layer into the stack."""
        logger.debug("Compositing %s", layer)

        if not self._accepts(layer, clip_compositing):
            return
        excluded = _excluded_channels(layer, self._channels)
        if not excluded:
            box = self._local_box(layer) if _LOCAL_BOXES else None
            if box is not None:
                self._apply_in_box(layer, box)
                return
            self._apply_layer(layer)
            return
        held = self._hold_channels(excluded, clip_compositing)
        self._apply_layer(layer)
        self._restore_channels(held)

    def _local_box(self, layer: Layer) -> tuple[int, int, int, int] | None:
        """The box ``layer`` paints within, when compositing it there is safe."""
        if not _box_safe(layer, self._cache):
            return None
        box = utils.intersect(self._viewport, _change_bbox(layer, self._cache))
        if box == (0, 0, 0, 0):
            return None
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area > _LOCAL_SHARE * self.width * self.height:
            return None
        return box

    def _apply_in_box(self, layer: Layer, box: tuple[int, int, int, int]) -> None:
        """:py:meth:`_apply_layer` on ``box``; the canvases outside it stay as they are."""
        x0, y0 = self._viewport[0], self._viewport[1]
        sl = (slice(box[1] - y0, box[3] - y0), slice(box[0] - x0, box[2] - x0))
        sub = object.__new__(Compositor)
        sub.__dict__.update(self.__dict__)
        sub._viewport = box
        for name in (
            "_color_0",
            "_alpha_0",
            "_color",
            "_alpha",
            "_alpha_g",
            "_shape_g",
        ):
            setattr(sub, name, getattr(self, name)[sl])
        if self._enclosing is not None:
            sub._enclosing = _Isolation(
                alpha=self._enclosing.alpha[sl],
                backdrop_color=self._enclosing.backdrop_color[sl],
                backdrop_alpha=self._enclosing.backdrop_alpha[sl],
            )
        parent_resolve = self._resolve_document_backdrop

        def document_backdrop() -> tuple[np.ndarray, np.ndarray] | None:
            resolved = parent_resolve()
            return None if resolved is None else (resolved[0][sl], resolved[1][sl])

        sub._document_backdrop_fn = document_backdrop
        sub._document_backdrop_resolved = False
        sub._document_backdrop = None
        before = {
            name: getattr(sub, name)
            for name in ("_color", "_alpha", "_alpha_g", "_shape_g")
        }
        sub._apply_layer(layer)
        for name, old in before.items():
            new = getattr(sub, name)
            if new is old:
                continue
            canvas = getattr(self, name).copy()
            canvas[sl] = new
            setattr(self, name, canvas)

    def _apply_layer(self, layer: Layer) -> None:
        """Composite an accepted layer, its effects and its clip layers."""
        if isinstance(layer, AdjustmentLayer):
            # An adjustment contributes no source of its own; it rewrites the
            # canvas that is already there. Its style is drawn from its mask,
            # the shape it has, and goes on around that rewrite the way a
            # layer's goes on around its paint -- see
            # _apply_styled_adjustment(). A mask that hides everything leaves
            # no shape to draw from.
            style = _style_bbox(layer)
            if (
                _has_enabled_effects(layer)
                and style[0] < style[2]
                and style[1] < style[3]
            ):
                self._apply_styled_adjustment(layer)
            else:
                self._apply_adjustment(layer)
            return

        # Beneath everything else the layer paints, and before the layer is
        # resolved rather than only before it is composited: a pass-through
        # group's contents blend with the canvas they are resolved over, and
        # its shadow and glow are part of that canvas. A pass-through group
        # that knocks out loses their colour the way it loses that of every
        # layer beneath it, because it is resolved over its knockout backdrop
        # and re-applied against this canvas (#707).
        self._apply_beneath_effects(layer)
        source = self._resolve_source(layer)
        if _blends_interior_apart(layer, source):
            self._apply_apart(layer, source)
            return
        source, outer = self._compose_effects(layer, source)
        if outer and _composites_as_passthrough(source, layer.blend_mode):
            # A pass-through group is re-applied by interpolating this canvas
            # against the group's own result, which has no outer effect in it,
            # so anything painted beside the group first is interpolated away
            # again. This branch therefore puts the source on before the outer
            # effects; nothing in the fixture corpus reaches it (#846).
            self._composite_source(source, layer.blend_mode)
            self._apply_outer_effects(outer, None)
            return
        # ``alpha * fill_opacity`` is what _composite_source() is about to put
        # on, and it is what the band has to leave room beside. The two agree
        # only once an effect canvas has folded fill opacity in, which an
        # outset-only stroke never reaches: it puts nothing inside the layer,
        # so the canvas is never started and hands its source back untouched,
        # fill opacity and all.
        self._apply_outer_effects(outer, source.alpha * source.fill_opacity)
        self._composite_source(source, layer.blend_mode)

    def _accepts(self, layer: Layer, clip_compositing: bool) -> bool:
        """Whether this layer contributes to the composite at all.

        The cull measures ``_effect_reach()`` and not ``layer.bbox``, because
        a stroke reaches outside the layer: a layer whose own box has cleared
        the viewport can still paint the part of its stroke that falls back
        inside, and rejecting it here would lose the stroke along with the
        layer (#815). A drop shadow reaches out the same way. ``_effect_reach()``
        rather than ``_paint_bbox()`` because the group half of that
        measurement is unreachable from here: a group is exempt from the cull,
        so the box is only ever measured for a layer that has no contents to
        recurse into. That is also why the exemption is tested first -- the
        reach is not worth measuring for a layer that is exempt from the test
        it feeds.

        ``is_group()`` and not ``isinstance(layer, GroupMixin)``, for the
        reason ``Layer._invalidate_moved_bbox()`` and its neighbours already
        name: ``GroupMixin`` is a ``runtime_checkable`` protocol whose
        ``isinstance`` runs ``hasattr(x, "bbox")`` on Python <= 3.11, which
        recomputes a group's box from its children just to answer a question
        about the layer's type. The exemption is tested first, so the question
        is asked of every layer; the two spellings agree on every layer in the
        fixture corpus and only one of them is cheap.
        """
        if self._layer_filter is not None and not self._layer_filter(layer):
            logger.debug("Ignore %s", layer)
            return False
        if not (
            isinstance(layer, AdjustmentLayer) or layer.is_group()
        ) and utils.intersect(self._viewport, _effect_reach(layer)) == (0, 0, 0, 0):
            logger.debug("Out of viewport %s", layer)
            return False
        if not clip_compositing and layer.clipping:
            # Composited by the layer it clips to, through _apply_clip_layers().
            return False
        return True

    def _hold_channels(
        self, channels: list[int], clip_compositing: bool
    ) -> _HeldChannels:
        """What the excluded channels hold before the layer goes on.

        A layer keeps an excluded channel's premultiplied value in the content
        it joins: the enclosing isolated group's own when adjustments read
        that (see :py:meth:`_apply_adjustment`), the canvas otherwise. Over an
        opaque backdrop the channel stays as it was; where the layer covers
        transparency it adds coverage and leaves that channel at zero. A clip
        layer keeps its base's colour, since the base's alpha is the result's.
        """
        color = self._color[..., channels]
        if clip_compositing:
            return _HeldChannels(channels, color, self._alpha, None)
        alpha = self._content_alpha()
        content = color
        if self._adjustment_isolated:
            backdrop_color, backdrop_alpha = self._isolation_backdrop()
            content = _unmix(
                color, backdrop_color[..., channels], backdrop_alpha, alpha
            )
        return _HeldChannels(channels, color, alpha, alpha * content)

    def _restore_channels(self, held: _HeldChannels) -> None:
        """Put the excluded channels back as :py:meth:`_hold_channels` held them."""
        kept = held.color
        if held.premultiplied is not None:
            alpha = self._content_alpha()
            premultiplied = held.premultiplied
            if self._adjustment_isolated:
                backdrop_color, backdrop_alpha = self._isolation_backdrop()
                premultiplied = (
                    premultiplied
                    + (1.0 - alpha)
                    * backdrop_alpha
                    * backdrop_color[..., held.channels]
                )
            # Where the content's alpha did not move, neither did the channel.
            kept = np.where(
                alpha == held.alpha,
                held.color,
                utils.clip(utils.divide(premultiplied, self._alpha, fill=held.color)),
            )
        color = np.array(self._color, copy=True)
        color[..., held.channels] = kept
        self._color = color

    def _content_alpha(self) -> np.ndarray:
        """How opaque the content a layer joins is: see :py:meth:`_hold_channels`."""
        if self._adjustment_isolated:
            return self._isolation_alpha()
        return self._alpha

    def _resolve_source(self, layer: Layer) -> _Source:
        """Resolve a layer into the operands the blend equations take.

        Reads the layer -- group, object, clip layers, mask, opacity -- and
        reads this compositor's canvases where a group needs a backdrop to
        stand on, but writes none of them. ``_composite_source()`` is the write.
        """
        knockout = _read_knockout(layer)
        adjustment_isolated: bool | None = None
        stroke: tuple[np.ndarray, np.ndarray] | None = None
        is_group = isinstance(layer, GroupMixin)
        if is_group:
            color, shape, alpha, adjustment_isolated = self._get_group(layer, knockout)
            if layer.has_clip_layers():
                color = self._apply_clip_layers(layer, color)
        else:
            color, shape, alpha, stroke = self._get_object(layer, clip_layers=True)

        shape_mask = self._get_mask(layer)
        fill_opacity, opacity = self._get_const(layer)
        mask = shape_mask * opacity
        # In place: both canvases were built for this layer by the calls above
        # and are not shared with anything else.
        shape *= shape_mask
        region = None
        if is_group and _has_enabled_effects(layer):
            region = alpha * shape_mask
        alpha *= mask
        if stroke is not None:
            stroke_mask = self._get_mask(layer, clip_to_path=False)
            # The fill's share and the stroke's are disjoint, so they add.
            shape = np.minimum(shape + stroke[0] * stroke_mask, 1.0)
            alpha = np.minimum(alpha + stroke[1] * stroke_mask * opacity, 1.0)
        if region is None:
            # An object's effects cover its shape, the redrawn stroke included.
            # A group with no style to draw never has its region read.
            region = shape

        text_gamma, text_key = self._text_blend(layer)
        return _Source(
            color=color,
            shape=shape,
            alpha=alpha,
            mask=mask,
            shape_mask=shape_mask,
            fill_opacity=fill_opacity,
            opacity=opacity,
            knockout=knockout,
            adjustment_isolated=adjustment_isolated,
            knockout_shape=shape,
            region=region,
            text_gamma=text_gamma,
            text_key=text_key,
        )

    def _text_blend(self, layer: Layer) -> tuple[float | None, int | None]:
        """The gamma ``layer``'s edge blends in, and its key while undecided.

        The key is set only while the pass's reference is still to decide
        between the text gamma and straight; see :py:class:`_TextBlend`.
        """
        if not _blends_in_text_space(layer, self._color_mode):
            return None, None
        blend = self._cache.text
        layer_id = layer.layer_id
        if layer_id in blend.overrides:
            return blend.overrides[layer_id], None
        if self._text_gamma is None:
            return None, None
        straight = blend.chosen.get(id(layer))
        if straight is not None:
            return (None if straight else self._text_gamma), None
        if blend.reference is None:
            return self._text_gamma, None
        return self._text_gamma, id(layer)

    def _mixes_straight(
        self,
        key: int,
        mixed: np.ndarray,
        color: np.ndarray,
        shape: np.ndarray,
        alpha: np.ndarray,
        alpha_previous: np.ndarray,
        color_b: np.ndarray,
        alpha_b: np.ndarray,
        blend_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> bool:
        """Whether the reference shows this type layer's edge mixed straight.

        ``mixed`` is the canvas with the edge mixed in the text gamma. Straight
        needs a clear majority of the edge and a close fit where the layer
        goes on; a layer that only leans straight there is left in the gamma
        for :py:func:`composite` to settle. The answer is kept for the pass.
        """
        blend = self._cache.text
        reference = blend.reference_array()
        rows = np.flatnonzero(shape.any(axis=(1, 2)))
        columns = np.flatnonzero(shape.any(axis=(0, 2)))
        evidence = None
        if reference is not None and len(rows) and len(columns):
            evidence = _edge_evidence(
                self._viewport,
                (rows[0], columns[0], rows[-1] + 1, columns[-1] + 1),
                reference,
                mixed,
                self._color,
                color,
                shape,
                alpha,
                alpha_previous,
                self._alpha,
                color_b,
                alpha_b,
                blend_fn,
            )
        straight = False
        if evidence is not None:
            share, fit, edge_rows, edge_columns = evidence
            straight = share >= _TEXT_STRAIGHT_SHARE and fit <= _TEXT_FIT
            if not straight and share >= _TEXT_LEAN_SHARE:
                blend.leaning[key] = (edge_rows, edge_columns)
        blend.chosen[key] = straight
        return straight

    def _composite_source(self, source: _Source, blend_mode: BlendMode) -> None:
        """Composite a resolved source into this compositor's canvases."""
        if _composites_as_passthrough(source, blend_mode):
            self._apply_passthrough_source(
                source.color,
                source.shape * source.fill_opacity,
                source.alpha * source.fill_opacity,
                source.mask * source.fill_opacity,
            )
            return

        # Fill opacity scales how much the source contributes, but under
        # knockout the hole is punched by the source's full coverage -- so
        # ``shape`` stays unscaled there and ``(1 - shape)`` in _apply_source()
        # removes the whole backdrop.
        self._apply_source(
            source.color,
            source.knockout_shape
            if source.knockout
            else source.shape * source.fill_opacity,
            source.alpha * source.fill_opacity,
            blend_mode,
            source.knockout,
            text_gamma=source.text_gamma,
            text_key=source.text_key,
        )

    def _compose_effects(
        self, layer: Layer, source: _Source
    ) -> tuple[_Source, list[_OuterEffect]]:
        """The layer's effects: the inner ones in its source, the outer beside.

        A drop shadow and an outer glow are neither, and are not composed
        here: they lie beneath the layer and go on before the layer is
        resolved -- see :py:meth:`_apply_beneath_effects`.

        A satin, then the inner glows and inner shadows go on over the
        overlays and under the stroke, as Photoshop stacks them. A bevel goes
        on over the stroke, so it shades the stroke too; the part of a bevel
        beside the layer is an outer effect, after the stroke's.
        """
        canvas = _EffectCanvas(self, source)
        outer = self._paint_effects(layer, canvas, source.shape, source.shape_mask)
        return canvas.source(), outer

    def _paint_effects(
        self,
        layer: Layer,
        canvas: _EffectSink,
        shape: np.ndarray,
        shape_mask: float | np.ndarray,
    ) -> list[_OuterEffect]:
        """Draw the layer's effects into ``canvas``, returning the outer ones.

        ``shape`` is the layer's coverage on this viewport and ``shape_mask``
        its mask alone; a stroke traces one or the other.
        """
        for effect_name in _OVERLAY_DRAWS:
            self._add_overlay(layer, effect_name, canvas)
        self._add_satins(layer, canvas)
        self._add_inner_effects(layer, canvas)

        traces_mask = self._traces_mask(layer)
        traced: float | np.ndarray = shape
        if traces_mask:
            traced = shape_mask
        elif _is_shape_layer(layer) and _readable(layer, "stroke"):
            traced = self._path_interior(layer, self._viewport, shape_mask, shape)
        outer: list[_OuterEffect] = []
        self._add_stroke_effects(
            layer,
            traced,
            canvas,
            outer,
            traces_mask,
        )
        self._add_bevels(layer, canvas, outer)
        return outer

    def _traces_mask(self, layer: Layer) -> bool:
        """Whether the layer's strokes follow its mask rather than its coverage.

        A layer drawn from a fill, or force-redrawn from its vector mask, has
        no source shape worth tracing; an adjustment layer has no shape but
        its mask.
        """
        return (
            isinstance(layer, AdjustmentLayer)
            or (self._force and layer.has_vector_mask())
            or (not layer.has_pixels() and utils.has_fill(layer))
        )

    def _apply_outer_effects(
        self, effects: Sequence[_OuterEffect], covered: np.ndarray | None
    ) -> None:
        """Composite what paints beside the layer, before the layer goes on.

        An outer band is disjoint from the layer rather than over or under it:
        a band drawn from the layer's own coverage is complementary to it, so
        on a pixel the boundary cuts an outset band comes out at ``1 - alpha``
        and the two together fill the pixel. Their coverages therefore *add*,
        which is not what compositing one over the other gives.

        Painting the band first and the layer over it is how that addition is
        reached without merging the two into one source -- which would hand
        the band the layer's blend mode instead of its own. The layer then
        attenuates the band by ``1 - covered``, so the band goes on at
        ``t / (1 - covered)``, and the ``union`` the layer composites with
        resolves to ``covered + t`` exactly. ``covered`` is the layer's
        finished alpha, effects inside it included, so it is read after they
        are composited and not before. The effects listed above a band are
        disjoint from it in the same way and go on over it, so what they
        paint on the pixel, ``effect.above``, comes off the divisor too.

        Where the layer is already opaque there is no room beside it and no
        band either, which is the ``fill=0`` below: the quotient is 0/0 there
        rather than large.

        ``covered`` is None where the layer is composited first after all, and
        the band goes on at what it was drawn at -- see :py:meth:`apply`.
        """
        for effect in effects:
            coverage = effect.coverage
            room: float | np.ndarray = 1.0 - effect.above
            if covered is not None:
                room = room - covered
            if isinstance(room, np.ndarray):
                coverage = utils.clip(utils.divide(coverage, room, fill=0.0))
            self._apply_source(
                effect.color,
                coverage,
                coverage * effect.opacity,
                effect.blend_mode,
            )

    def _apply_passthrough_source(
        self,
        color: np.ndarray,
        shape: np.ndarray,
        alpha: np.ndarray,
        mask: float | np.ndarray,
    ) -> None:
        color = self._fit_source(color)
        # ``color`` is the group already composited over this backdrop, because a
        # pass-through group is rendered by a non-isolated sub-compositor seeded
        # with the backdrop. Re-applying the group therefore means interpolating
        # between the backdrop and that result by the group opacity/mask, which
        # must happen in premultiplied space so a transparent (or partially
        # transparent) backdrop does not bleed its color into the result.
        #
        # TODO(#707): knockout is ignored here while _get_group() honors it, so
        # a knockout pass-through group interpolates against a different backdrop
        # than it was composited over, and the result varies with nesting depth.
        color_b = self._color
        alpha_b = self._alpha

        self._shape_g = cast(np.ndarray, utils.union(self._shape_g, shape))
        self._alpha_g = cast(np.ndarray, utils.union(self._alpha_g, alpha))
        # union(alpha_b, alpha) -- ``alpha`` is the group alpha already scaled by
        # ``mask``, which is what the premultiplied interpolation resolves to.
        self._alpha = cast(np.ndarray, utils.union(self._alpha_0, self._alpha_g))

        color_t = (1.0 - mask) * alpha_b * color_b + (
            mask * alpha_b + alpha * (1.0 - alpha_b)
        ) * color
        # Where the result is fully transparent the color is arbitrary, so fall
        # back to the group's own color rather than to white.
        self._color = utils.clip(utils.divide(color_t, self._alpha, fill=color))

    def _fit_source(self, color: np.ndarray) -> np.ndarray:
        """Bring a source to this compositor's width, checking the invariant.

        Both doors a source colour comes through call this, so what a grey means
        in this document is answered in one place: a single-channel source (a
        pattern, or a fill on a multichannel document) is widened by ``widen``,
        not broadcast by the blend arithmetic, which is wrong for CMYK (#749).
        """
        self._assert_source_fits(color)
        return self._widen(color, self._channels)

    def _assert_source_fits(self, color: np.ndarray) -> None:
        """Check the fixed-width invariant where a source meets the canvases.

        A single-channel source is widened by ``_fit_source()`` above, but a
        *wider* one would silently widen ``_color`` and leave ``channels``
        stale. This fires instead, if a caller ever builds a compositor
        narrower than the sources it will be given.
        """
        assert self._color.shape[2] == self._channels, (
            "canvas widened to %d channels, expected %d"
            % (self._color.shape[2], self._channels)
        )
        assert color.shape[2] in (1, self._channels), (
            "source has %d channels, expected 1 or %d"
            % (color.shape[2], self._channels)
        )

    def _resolve_document_backdrop(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not self._document_backdrop_resolved:
            self._document_backdrop_resolved = True
            if self._document_backdrop_fn is not None:
                resolved = self._document_backdrop_fn()
                if resolved is not None:
                    # The Background layer can be narrower than the document
                    # it sits in; widen once here so knockout hands back a
                    # backdrop of this compositor's width like any other.
                    color_b, alpha_b = resolved
                    resolved = (self._widen(color_b, self.channels), alpha_b)
                self._document_backdrop = resolved
        return self._document_backdrop

    def _knockout_backdrop(self, knockout: Knockout) -> tuple[np.ndarray, np.ndarray]:
        """The (color, alpha) pair that ``knockout`` knocks out to.

        SHALLOW knocks out to this compositor's own initial backdrop, i.e. the
        enclosing group's. DEEP knocks out to the document backdrop, which is
        inherited across pass-through groups and reset at isolation boundaries;
        where there is none it degrades to the same backdrop as SHALLOW.
        """
        if knockout == Knockout.DEEP:
            resolved = self._resolve_document_backdrop()
            if resolved is not None:
                return resolved
        return self._color_0, self._alpha_0

    def _apply_source(
        self,
        color: np.ndarray,
        shape: np.ndarray,
        alpha: np.ndarray,
        blend_mode: BlendMode,
        knockout: Knockout = Knockout.NONE,
        text_gamma: float | None = None,
        faded: float = 1.0,
        text_key: int | None = None,
    ) -> None:
        """Put a source on the canvases.

        ``text_gamma`` is the space a type layer's coverage mixes its colour
        with the backdrop in -- see :py:mod:`psd_tools.composite.text_blend`.
        The blend mode still sees the document's values; only the mix by
        coverage moves into that space.

        ``faded`` is the opacity in ``alpha`` that fades the colour instead,
        for the modes :py:func:`~psd_tools.composite.blend.fade_source` names.
        """
        color = self._fit_source(color)
        knockout_color, knockout_alpha = self._knockout_backdrop(knockout)

        alpha_g_previous = self._alpha_g
        self._shape_g = cast(np.ndarray, utils.union(self._shape_g, shape))
        if knockout:
            self._alpha_g = (
                (1.0 - shape) * self._alpha_g + (shape - alpha) * knockout_alpha + alpha
            )
        else:
            self._alpha_g = cast(np.ndarray, utils.union(self._alpha_g, alpha))
        alpha_previous = self._alpha
        self._alpha = cast(np.ndarray, utils.union(self._alpha_0, self._alpha_g))

        alpha_b = knockout_alpha if knockout else alpha_previous
        color_b = knockout_color if knockout else self._color

        blend_fn = fade_source(
            get_blend_func(blend_mode, self._color_mode), blend_mode, faded
        )
        if text_gamma is not None:
            mixed = self._mix_in_text_space(
                color,
                shape,
                alpha,
                alpha_previous,
                alpha_g_previous,
                color_b,
                alpha_b,
                blend_fn,
                text_gamma,
                isolated=self._text_isolated and not knockout,
            )
            if text_key is None or not self._mixes_straight(
                text_key,
                mixed,
                color,
                shape,
                alpha,
                alpha_previous,
                color_b,
                alpha_b,
                blend_fn,
            ):
                self._color = mixed
                return
        color_t = (shape - alpha) * alpha_b * color_b + alpha * (
            (1.0 - alpha_b) * color + alpha_b * blend_fn(color_b, color)
        )
        self._color = utils.clip(
            utils.divide(
                (1.0 - shape) * alpha_previous * self._color + color_t, self._alpha
            )
        )

    def _mix_in_text_space(
        self,
        color: np.ndarray,
        shape: np.ndarray,
        alpha: np.ndarray,
        alpha_previous: np.ndarray,
        alpha_g_previous: np.ndarray,
        color_b: np.ndarray,
        alpha_b: np.ndarray,
        blend_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        text_gamma: float,
        isolated: bool,
    ) -> np.ndarray:
        """The colour canvas after a type layer goes on.

        The same mix as :py:meth:`_apply_source`, with every colour taken into
        the text blend space first and the result brought back. Computed in
        double precision and only on the rows and columns the source covers:
        the transform is a round trip for every pixel the source leaves
        alone, which single precision does not return exactly.

        ``isolated`` mixes against what this compositor has painted on its own
        -- the canvas with the seeded backdrop taken back out, as
        :py:meth:`result_isolated` does -- and puts the result back over that
        backdrop, so the canvas keeps standing over it for the layers to come.
        The blend mode still sees the canvas: a pass-through group's children
        blend with the backdrop whether or not the group is styled.
        """
        rows = np.flatnonzero(shape.any(axis=(1, 2)))
        columns = np.flatnonzero(shape.any(axis=(0, 2)))
        if not len(rows) or not len(columns):
            return self._color
        box = (slice(rows[0], rows[-1] + 1), slice(columns[0], columns[-1] + 1))
        color = color[box].astype(np.float64)
        previous: np.ndarray = self._color[box].astype(np.float64)
        shape = shape[box]
        alpha = alpha[box]
        alpha_new = self._alpha[box]
        alpha_b = alpha_b[box].astype(np.float64)
        backdrop = (
            previous if color_b is self._color else color_b[box].astype(np.float64)
        )
        source = to_text_space(
            (1.0 - alpha_b) * color + alpha_b * blend_fn(backdrop, color), text_gamma
        )
        if isolated:
            color_0 = self._color_0[box].astype(np.float64)
            alpha_0 = self._alpha_0[box].astype(np.float64)
            alpha_b = alpha_g_previous[box].astype(np.float64)
            # The content so far, un-mixed from the backdrop it stands over;
            # where there is none its colour carries no weight below.
            previous = utils.divide(
                previous * alpha_previous[box] - (1.0 - alpha_b) * alpha_0 * color_0,
                alpha_b,
                fill=0.0,
            )
            backdrop = previous
            alpha_new = self._alpha_g[box]
        previous = to_text_space(previous, text_gamma)
        backdrop = (
            previous if backdrop is previous else to_text_space(backdrop, text_gamma)
        )
        if isolated:
            kept = (1.0 - alpha) * alpha_b * previous
        else:
            kept = (1.0 - shape) * alpha_previous[box] * previous + (
                shape - alpha
            ) * alpha_b * backdrop
        mixed = from_text_space(
            utils.divide(kept + alpha * source, alpha_new), text_gamma
        )
        if isolated:
            mixed = utils.divide(
                alpha_new * mixed + (1.0 - alpha_new) * alpha_0 * color_0,
                self._alpha[box],
                fill=mixed,
            )
        result = self._color.copy()
        result[box] = utils.clip(mixed)
        return result

    def _apply_adjustment(
        self, layer: AdjustmentLayer, opacity: float | None = None
    ) -> None:
        """Rewrite the canvas under the layer's mask.

        ``opacity`` stands in for the layer's own where the caller fades the
        layer later, with its style (:py:meth:`_apply_styled_adjustment`).
        """
        adjustment_fn = ADJUSTMENT_FUNC.get(layer.kind)
        colormode = layer._psd.color_mode

        if adjustment_fn is None:
            logger.debug("Unsupported adjustment layer: %s", layer.kind)
            return
        if colormode not in (ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB):
            logger.debug(
                "Unsupported color mode for adjustment %s: %s", layer.kind, colormode
            )
            return

        # No block to apply, so nothing to blend either; a neutral one still blends.
        if layer._data is None:
            logger.debug("Adjustment layer without data: %s", layer.kind)
            return

        # An adjustment inside an isolated group reads the group's own
        # content, not the canvas that content was composited over: a
        # half-transparent layer under it is adjusted at its own colour and
        # stays half-transparent over the backdrop. The content is divided
        # back out of the canvas, adjusted, and composited over the backdrop
        # again, which confines the adjustment to where the group has
        # painted; adjusting the canvas and dividing afterwards cannot undo
        # an adjustment that is not linear.
        if self._adjustment_isolated:
            backdrop_color, backdrop_alpha = self._isolation_backdrop()
            content_alpha = self._isolation_alpha()
            content = _unmix(self._color, backdrop_color, backdrop_alpha, content_alpha)
        else:
            content_alpha = self._alpha
            content = self._color
        transformed_color = adjustment_fn(content, colormode, layer)
        # An adjustment that is not rendered leaves the content as it is; its
        # clip layers still go over that, as they go over a neutral one. With
        # none there is nothing to blend: the content would blend with itself.
        if transformed_color is None:
            logger.debug("Adjustment layer not rendered: %s", layer.kind)
            if not layer.has_clip_layers():
                return
            transformed_color = content

        if layer.has_clip_layers():
            transformed_color = self._apply_clip_layers(layer, transformed_color)

        shape_mask = self._get_mask(layer)
        shape_const, opacity_const = self._get_const(layer)
        if opacity is None:
            opacity = opacity_const

        strength = shape_mask * shape_const * opacity

        # Passed for uniformity with _apply_source() rather than for effect.
        # The guard above returns for every mode whose blending the colour mode
        # changes, multichannel included -- though it reads the *layer's*
        # document where this binds the compositor's, which are the same
        # document on every path the library builds. Where they could differ, a
        # compositor built by hand, the canvas is this one's and so is the mode
        # that describes it.
        blend_fn = get_blend_func(layer.blend_mode, self._color_mode)
        blended = blend_fn(content, transformed_color)
        content = utils.clip(content + strength * (blended - content))

        if self._adjustment_isolated:
            self._color = _remix(
                content, backdrop_color, backdrop_alpha, content_alpha, self._alpha
            )
        else:
            self._color = content

    def _apply_styled_adjustment(self, layer: AdjustmentLayer) -> None:
        """Composite an adjustment layer that carries a style.

        Its shadows and glows go beneath, then the adjustment, then the rest
        of its effects over that: the mask is the layer's shape, so it is the
        region the effects are a share of and the silhouette the inner
        effects and strokes are drawn from. Each goes onto the canvas on its
        own (:py:class:`_AdjustmentCanvas`); what paints beside the region
        follows, at what it was drawn at, since there is no layer source
        about to go over it.

        The layer opacity fades the adjustment and its effects as one, as it
        fades a layer and its effects: they go on whole, and the canvas is
        then drawn back towards what it was. The shadows and glows are faded
        on their own, before that mark is taken.
        """
        self._apply_beneath_effects(layer)
        _, opacity = self._get_const(layer)
        if opacity <= 0.0:
            return
        color, alpha = self._color, self._alpha
        shape_g, alpha_g = self._shape_g, self._alpha_g
        self._apply_adjustment(layer, opacity=1.0)
        region = self._adjustment_shape(layer, self._viewport)
        canvas = _AdjustmentCanvas(self, region)
        outer = self._paint_effects(layer, canvas, region, region)
        self._apply_outer_effects(outer, None)
        if opacity < 1.0:
            self._fade(opacity, color, alpha, shape_g, alpha_g)

    def _apply_apart(self, layer: Layer, source: _Source) -> None:
        """Composite a layer whose interior effects are not blended as a group.

        Photoshop puts the layer's paint on with its mode at full coverage,
        faded by fill opacity alone, then each effect inside the layer onto
        that with the effect's own mode, and draws the result back towards
        the canvas by the layer's coverage and opacity once. That is built on
        a compositor seeded with this canvas; what it adds goes on here
        Normal, at the share of the pixel it holds, so a type layer's edge
        mixes in the text gamma and what paints beside the layer is
        attenuated as :py:meth:`_apply_outer_effects` expects.
        """
        region = source.region
        paint = utils.clip(
            utils.divide(source.alpha, region * source.opacity, fill=0.0)
            * source.fill_opacity
        )
        scratch = Compositor(
            self._viewport,
            self._color,
            self._alpha,
            self._layer_filter,
            self._force,
            widen=self._widen,
            color_mode=self._color_mode,
            cache=self._cache,
            text_gamma=None,
        )
        scratch._apply_source(source.color, paint, paint, layer.blend_mode)
        outer = scratch._paint_effects(
            layer,
            _ApartCanvas(scratch, region, source.opacity),
            source.shape,
            source.shape_mask,
        )
        # The scratch canvas is ``own`` of what went on over ``1 - own`` of
        # this one; taking that back out leaves what went on.
        own = scratch._alpha_g
        color = utils.clip(
            utils.divide(
                scratch._alpha * scratch._color
                - (1.0 - own) * self._alpha * self._color,
                own,
                fill=scratch._color,
            )
        )
        alpha = source.alpha * own
        self._apply_outer_effects(outer, alpha)
        self._apply_source(
            color,
            utils.clip(region * scratch._shape_g),
            alpha,
            BlendMode.NORMAL,
            text_gamma=source.text_gamma,
            text_key=source.text_key,
        )

    def _fade(
        self,
        share: float,
        color: np.ndarray,
        alpha: np.ndarray,
        shape_g: np.ndarray,
        alpha_g: np.ndarray,
    ) -> None:
        """Keep ``share`` of what went onto the canvas since the state given.

        What a layer's opacity does to the layer and its effects together.
        Interpolated premultiplied, which is exact for what went on Normal
        and is what fading each piece by ``share`` as it went on would have
        summed to for the alphas; the group's shape and alpha interpolate
        the same way, as ``union`` is affine in what joins it.
        """
        premultiplied = alpha * color + share * (
            self._alpha * self._color - alpha * color
        )
        self._alpha = alpha + share * (self._alpha - alpha)
        self._color = utils.clip(
            utils.divide(premultiplied, self._alpha, fill=self._color)
        )
        self._shape_g = shape_g + share * (self._shape_g - shape_g)
        self._alpha_g = alpha_g + share * (self._alpha_g - alpha_g)

    def _adjustment_shape(
        self, layer: AdjustmentLayer, viewport: tuple[int, int, int, int]
    ) -> np.ndarray:
        """An adjustment layer's silhouette on ``viewport``: its mask, in the document.

        The layer has no pixels, so its mask is its shape; where the mask
        leaves it uncovered -- no mask, or one that reveals its outside --
        the layer runs to the document's edge, which is then the edge its
        strokes and inner effects are drawn from.
        """
        height, width = viewport[3] - viewport[1], viewport[2] - viewport[0]
        shape = np.zeros((height, width, 1), dtype=np.float32)
        x0, y0, x1, y1 = utils.intersect(viewport, layer._psd.viewbox)
        if x0 < x1 and y0 < y1:
            shape[
                y0 - viewport[1] : y1 - viewport[1], x0 - viewport[0] : x1 - viewport[0]
            ] = 1.0
        return utils.clip(shape * self._get_mask(layer, viewport))

    def _isolation_alpha(self) -> np.ndarray:
        """How opaque what the isolated group has painted so far is."""
        if self._enclosing is None:
            return self._alpha_g
        return cast(np.ndarray, utils.union(self._enclosing.alpha, self._alpha_g))

    def _isolation_backdrop(self) -> tuple[np.ndarray, np.ndarray]:
        """The backdrop the isolated group's content stands on."""
        if self._enclosing is None:
            return self._color_0, self._alpha_0
        return self._enclosing.backdrop_color, self._enclosing.backdrop_alpha

    def finish(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the isolated result, the shape and the alpha."""
        return self.result_isolated(), self.shape, self.alpha

    @property
    def viewport(self) -> tuple[int, int, int, int]:
        """Viewport as `(left, top, right, bottom)`."""
        return self._viewport

    @property
    def width(self) -> int:
        """Width of the viewport."""
        return self._viewport[2] - self._viewport[0]

    @property
    def height(self) -> int:
        """Height of the viewport."""
        return self._viewport[3] - self._viewport[1]

    @property
    def channels(self) -> int:
        """The channel count every canvas in this compositor carries."""
        return self._channels

    def result_isolated(self) -> np.ndarray:
        """The composited color with the initial backdrop's contribution removed.

        For a result handed back as a *source* -- a group its parent composites,
        or the return value of ``composite()`` -- which the caller composites
        over that same backdrop again.
        """
        return _unmix(self._color, self._color_0, self._alpha_0, self._alpha_g)

    def result_over_backdrop(self) -> np.ndarray:
        """The composited color *as it stands over* the initial backdrop.

        For a caller that keeps compositing onto the backdrop this compositor
        was seeded with: the pass-through, clip-layer and vector-stroke paths.
        """
        return self._color

    @property
    def shape(self) -> np.ndarray:
        """Accumulated shape of the composite."""
        return self._shape_g

    @property
    def alpha(self) -> np.ndarray:
        """Accumulated alpha of the composite."""
        return self._alpha_g

    def _get_group(
        self, layer: Layer, knockout: Knockout
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        is_passthrough = layer.blend_mode == BlendMode.PASS_THROUGH
        viewport = (
            self._viewport
            if is_passthrough
            else utils.intersect(
                self._viewport, _content_bbox(layer, self._cache.content_bboxes)
            )
        )
        if knockout:
            color_b, alpha_b = self._knockout_backdrop(knockout)
        else:
            color_b = self._color
            alpha_b = self._alpha

        # Photoshop keeps an adjustment inside a group that has clip layers, a
        # fill under 100 or a layer style within the group's shape instead of
        # passing it through; the children's own blend modes still reach the
        # backdrop. A non-pass-through group is composited on its own anyway.
        shape_const, _ = self._get_const(layer)
        isolate_adjustments = (
            shape_const < 1.0
            or layer.has_clip_layers()
            or (is_passthrough and _has_enabled_effects(layer))
        )

        # A pass-through group is not an isolation boundary, so deep knockout
        # inside it still reaches the document backdrop; an isolated group is,
        # so deep knockout there stops at the group's own backdrop.
        document_backdrop = None
        if is_passthrough and self._document_backdrop_fn is not None:
            parent_viewport = self._viewport
            resolve_parent = self._resolve_document_backdrop

            def document_backdrop() -> tuple[np.ndarray, np.ndarray] | None:
                resolved = resolve_parent()
                if resolved is None:
                    return None
                backdrop_color, backdrop_alpha = resolved
                return (
                    paste(viewport, parent_viewport, backdrop_color, 1.0),
                    paste(viewport, parent_viewport, backdrop_alpha),
                )

        # Only a pass-through group inherits the enclosing alpha; an isolated
        # one starts transparent, so that paste is skipped rather than made and
        # thrown away.
        group_alpha = (
            paste(viewport, self._viewport, alpha_b)
            if is_passthrough
            else np.zeros(
                (viewport[3] - viewport[1], viewport[2] - viewport[0], 1),
                dtype=np.float32,
            )
        )
        # A non-pass-through group is seeded transparent, so its colour is its
        # own content and the enclosing group's shape does not weigh on it:
        # only a pass-through child inherits the enclosing isolation.
        adjustment_isolated: bool | _Isolation = isolate_adjustments
        if not isolate_adjustments and is_passthrough and self._adjustment_isolated:
            backdrop_color, backdrop_alpha = self._isolation_backdrop()
            adjustment_isolated = _Isolation(
                alpha=paste(viewport, self._viewport, self._isolation_alpha()),
                backdrop_color=paste(viewport, self._viewport, backdrop_color, 1.0),
                backdrop_alpha=paste(viewport, self._viewport, backdrop_alpha),
            )
        group_compositor = Compositor(
            viewport,
            color=paste(viewport, self._viewport, color_b, 1.0),
            alpha=group_alpha,
            layer_filter=self._layer_filter,
            force=self._force,
            adjustment_isolated=adjustment_isolated,
            document_backdrop=document_backdrop,
            widen=self._widen,
            color_mode=self._color_mode,
            cache=self._cache,
            text_gamma=self._text_gamma,
            # A group composited as an ordinary source is rendered as one
            # layer, and so is every group inside it.
            text_isolated=self._text_isolated or isolate_adjustments,
        )

        for sublayer in cast(GroupMixin, layer):
            group_compositor.apply(sublayer)

        # When adjustments are isolated the group is composited as an ordinary
        # source, so the backdrop it was seeded with has to come back out
        # first; otherwise it stays in, because _apply_passthrough_source()
        # interpolates the result against that same backdrop. For a
        # non-pass-through group the two agree anyway -- it was seeded
        # transparent, and the correction is a no-op there.
        if isolate_adjustments:
            color = group_compositor.result_isolated()
        else:
            color = group_compositor.result_over_backdrop()
        shape = group_compositor.shape
        alpha = group_compositor.alpha

        color = paste(self._viewport, viewport, color, 1.0)
        shape = paste(self._viewport, viewport, shape)
        alpha = paste(self._viewport, viewport, alpha)

        return color, shape, alpha, isolate_adjustments

    def _read_object(self, layer: Layer) -> tuple[np.ndarray | None, np.ndarray | None]:
        """The layer's own color and coverage, on ``layer.bbox`` and unpasted.

        Split out of :py:meth:`_get_object` so a stroke effect can re-read the
        coverage on the box it draws on rather than on this compositor's
        viewport (#804), without restating which of the two sources -- stored
        pixels or a redrawn fill -- this layer's arrays come from.
        """
        color = layer.numpy("color")
        # The colour array stays live through the shape read, so it is charged
        # to that read's allocation guard.
        if color is not None:
            numpy_io.check_shape_read(layer, numpy_io._backing_bytes(color))
        shape = layer.numpy("shape")
        if (self._force or not layer.has_pixels()) and utils.has_fill(layer):
            color, shape = paint.create_fill(layer, layer.bbox)
            if shape is None:
                shape = np.ones((layer.height, layer.width, 1), dtype=np.float32)
        elif color is not None and isinstance(layer, SmartObjectLayer):
            memo = self._cache.unfiltered
            if memo is not None and memo[0] is layer:
                return memo[1], memo[2]
            stored = color
            color, shape = _unfilter_under_mask(layer, color, shape)
            if color is not stored:
                self._cache.unfiltered = (layer, color, shape)
        return color, shape

    @staticmethod
    def _place_object_shape(
        color: np.ndarray | None,
        shape: np.ndarray | None,
        bbox: tuple[int, int, int, int],
        viewport: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Place :py:meth:`_read_object`'s coverage on ``viewport``.

        A layer with no shape channel is opaque over its own box and absent
        outside it; one with neither color nor shape is an empty pixel layer
        and covers nothing.

        The opaque case covers ``bbox`` and not ``viewport``. The two agree
        only while ``bbox`` contains ``viewport``, which is the shape a layer
        with no transparency channel usually has -- a Background spanning the
        canvas -- and that is why filling the viewport went unnoticed. It is
        not guaranteed: #815 lets the cull accept a layer whose box misses the
        viewport entirely, and filling the viewport for one of those paints it
        opaque end to end.

        Written onto a viewport-sized canvas rather than filled on ``bbox``
        and pasted, so the allocation is bounded by the viewport. ``bbox`` is
        the layer's and can be far larger -- the case this widening newly
        admits is precisely a big layer against a small viewport -- and
        :py:func:`paste` would have copied only this intersection out of it
        anyway.
        """
        if shape is not None:
            return paste(viewport, bbox, shape)
        height, width = viewport[3] - viewport[1], viewport[2] - viewport[0]
        covered = np.zeros((height, width, 1), dtype=np.float32)
        if color is None:
            return covered
        inter = utils.intersect(viewport, bbox)
        if inter != (0, 0, 0, 0):
            covered[
                inter[1] - viewport[1] : inter[3] - viewport[1],
                inter[0] - viewport[0] : inter[2] - viewport[0],
                :,
            ] = 1.0
        return covered

    def _get_object_coverage(
        self, layer: Layer, viewport: tuple[int, int, int, int], alpha: bool = False
    ) -> np.ndarray:
        """What the layer paints on ``viewport``, under its mask.

        The coverage :py:meth:`_resolve_source` gives a layer's source, read
        on a box of the caller's: the fill, gone where the fill is switched
        off and the layer is redrawn, plus what a redrawn vector stroke covers
        of its own. With ``alpha`` it is how opaque that paint is instead, the
        stroke at its own opacity. A stroke effect is traced from the
        coverage and a shadow, glow, satin or bevel cast from the alpha, so
        both follow the layer as it is painted and not the raw fill under it.
        """
        _, shape, own_alpha, stroke = self._get_object(layer, viewport)
        own = own_alpha if alpha else shape
        own = own * self._get_mask(layer, viewport)
        if stroke is not None:
            stroke_mask = self._get_mask(layer, viewport, clip_to_path=False)
            own = np.minimum(own + stroke[1 if alpha else 0] * stroke_mask, 1.0)
        return own

    def _has_redrawn_stroke(self, layer: Layer) -> bool:
        """Whether this layer's vector stroke is drawn from the path.

        A layer read from stored pixels has its stroke in them already.
        """
        return bool(
            (self._force or not layer.has_pixels())
            and layer.has_vector_mask()
            and layer.stroke is not None
            and layer.stroke.enabled
        )

    def _get_object(
        self,
        layer: Layer,
        viewport: tuple[int, int, int, int] | None = None,
        clip_layers: bool = False,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray] | None
    ]:
        """Get object attributes, on ``viewport`` or this compositor's.

        The last item is the coverage and alpha a redrawn stroke has of its own,
        where the fill leaves any. Unlike the fill it is not clipped to the
        vector mask, which a centred or outer stroke reaches past.

        ``clip_layers`` paints the layer's clip layers into the color, on this
        compositor's viewport. They go on the fill and under a redrawn stroke,
        which Photoshop draws over them; with the fill off they go on the
        stroke, the layer's only paint.
        """
        if clip_layers and layer.has_clip_layers():
            assert viewport is None, "clip layers paint on this compositor's viewport"
            clip: Layer | None = layer
        else:
            clip = None
        if viewport is None:
            viewport = self._viewport
        height, width = viewport[3] - viewport[1], viewport[2] - viewport[0]
        color, own_shape = self._read_object(layer)
        shape = self._place_object_shape(color, own_shape, layer.bbox, viewport)

        if color is None:
            color = np.ones((height, width, 1), dtype=np.float32)
        else:
            color = paste(viewport, layer.bbox, color, 1.0)

        alpha = shape * 1.0  # Constant factor is always 1.
        fill_off = utils.is_fill_disabled(layer)

        if self._has_redrawn_stroke(layer):
            assert layer.stroke is not None
            path = None if fill_off else vector.draw_vector_mask(layer, viewport)
            color_s, shape_s, alpha_s = self._get_stroke(layer, path, viewport)
            # An inner stroke stays inside the path, so with the fill off the
            # path's own coverage is the bound.
            inner = layer.stroke.line_alignment is StrokeAlignment.INNER
            if fill_off:
                # No fill to paint onto: the layer is the stroke.
                if clip is not None:
                    color_s = self._apply_clip_layers(clip, color_s)
                if inner:
                    return color_s, shape * shape_s, alpha * alpha_s, None
                return color_s, shape * 0.0, alpha * 0.0, (shape_s, alpha_s)
            # What the fill covers of the path. An inner stroke keeps the
            # fill's own alpha, so the stroke is painted onto it whole (#883).
            assert path is not None
            covered = alpha if inner else alpha * path
            if clip is not None:
                # Under the stroke: Photoshop draws it over the clip layers.
                color = self._apply_clip_layers(clip, color)
            compositor = Compositor(
                viewport,
                self._widen(color, self.channels),
                covered,
                widen=self._widen,
                color_mode=self._color_mode,
            )
            # Photoshop paints a vector stroke Normal whatever its
            # ``strokeStyleBlendMode`` says, against the fill and the backdrop alike.
            compositor._apply_source(color_s, shape_s, alpha_s, BlendMode.NORMAL)
            # Seeded with the fill's color and alpha, so the result is wanted
            # as it stands on that seed: the stroke is painted onto the fill it
            # outlines, and a pixel it covers in part is that much of the
            # stroke over that fill (#883).
            color = compositor.result_over_backdrop()
            # What the fill leaves uncovered -- outside the path, or under a
            # transparent part of it -- is the stroke's own.
            uncovered = 1.0 - covered
            if inner:
                # The pen's antialiasing can reach past the path; the stroke
                # has nothing out there.
                # Alpha scales with coverage so the stroke keeps its opacity.
                bounded = np.minimum(shape_s, path)
                alpha_s = alpha_s * np.divide(
                    bounded, shape_s, out=np.zeros_like(bounded), where=shape_s > 0
                )
                shape_s = bounded
            return color, shape, alpha, (shape_s * uncovered, alpha_s * uncovered)

        if fill_off and (self._force or not layer.has_pixels()):
            shape, alpha = shape * 0.0, alpha * 0.0
        if clip is not None:
            color = self._apply_clip_layers(clip, color)
        return color, shape, alpha, None

    def _apply_clip_layers(self, layer: Layer, color: np.ndarray) -> np.ndarray:
        """``color`` with ``layer``'s clip layers painted on it.

        The clip layers blend with the base's color as though the base were
        opaque, and the base keeps its own alpha: a partly covered base pixel
        is that much of the blended color, never the clip layer's own color
        mixed in by the base's transparency. That is Blend Clipped Layers as
        Group, Photoshop's default.
        """
        # TODO: Consider Tag.BLEND_CLIPPING_ELEMENTS, which this reads as on.
        compositor = Compositor(
            self._viewport,
            self._widen(color, self.channels),
            np.ones((self.height, self.width, 1), dtype=np.float32),
            layer_filter=self._layer_filter,
            force=self._force,
            widen=self._widen,
            color_mode=self._color_mode,
            cache=self._cache,
            text_gamma=self._text_gamma,
        )
        for clip_layer in layer.clip_layers:
            compositor.apply(clip_layer, clip_compositing=True)
        # Seeded with ``color`` itself, so the result is wanted as it stands on
        # that seed -- the clipped layers paint onto the base layer's color.
        return compositor.result_over_backdrop()

    def _mask_repeats_shape(self, layer: Layer) -> bool:
        """Whether the mask is the stored shape channel again, pixel for pixel.

        Applying it would square the layer's edge coverage (#885). Only shape
        and fill layers qualify: elsewhere the mask is a user's, even one made
        from the layer's transparency. Memoized per layer, as stroke tracing
        asks again.
        """
        known = self._mask_repeats.get(id(layer))
        if known is None:
            known = self._mask_repeats[id(layer)] = self._compare_mask_to_shape(layer)
        return known

    def _compare_mask_to_shape(self, layer: Layer) -> bool:
        mask = layer.mask
        if (
            mask is None
            or not isinstance(layer, (ShapeLayer, FillLayer))
            or self._force
            or not layer.has_pixels()
            or mask.real_flags is not None
            or mask.bbox != layer.bbox
        ):
            return False
        if mask.parameters and any(
            d not in (None, 255)
            for d in (
                mask.parameters.user_mask_density,
                mask.parameters.vector_mask_density,
            )
        ):
            return False
        stored = layer.numpy("mask", real_mask=True)
        if stored is None:
            return False
        # The mask stays live through the shape read, so it is charged to it.
        numpy_io.check_shape_read(layer, numpy_io._backing_bytes(stored))
        shape = layer.numpy("shape")
        return (
            shape is not None
            and stored is not None
            and shape.shape == stored.shape
            and bool(np.array_equal(shape, stored))
        )

    def _get_mask(
        self,
        layer: Layer,
        viewport: tuple[int, int, int, int] | None = None,
        clip_to_path: bool = True,
    ) -> float | np.ndarray:
        """The layer's mask coverage, with any mask density already folded in.

        ``viewport`` defaults to this compositor's; a stroke effect passes the
        box it draws on, which may reach outside it (#804). ``clip_to_path=False``
        leaves the vector mask out: a stroke is not clipped to its own path.

        The scalar 1.0 default is an allocation-avoidance path, not an API
        convenience like the backdrop spellings: most layers have no mask, and
        materializing an all-ones canvas for each of them is pure waste.
        """
        if viewport is None:
            viewport = self._viewport
        shape: float | np.ndarray = 1.0
        if (
            layer.mask is not None
            and not layer.mask.disabled
            and not self._mask_repeats_shape(layer)
        ):
            # TODO: When force, ignore real mask.
            mask = layer.numpy("mask", real_mask=not self._force)
            if mask is not None:
                shape = paste(
                    viewport,
                    layer.mask.bbox,
                    mask,
                    layer.mask.background_color / 255.0,
                )
            if layer.mask.parameters:
                density = layer.mask.parameters.user_mask_density
                if density is None:
                    density = layer.mask.parameters.vector_mask_density
                if density is None:
                    density = 255

                share = float(density) / 255.0
                shape = share * shape + (1 - share)

        if (
            clip_to_path
            and layer.vector_mask is not None
            and not layer.vector_mask.disabled
            and (
                self._force
                or not layer.has_pixels()
                or (
                    not utils.has_fill(layer)
                    and layer.mask is not None
                    and not layer.mask.has_real()
                )
            )
        ):
            shape *= vector.draw_vector_mask(layer, viewport)

            if layer.mask is not None and layer.mask.parameters:
                density_v = layer.mask.parameters.vector_mask_density
            else:
                density_v = None

            if density_v is not None:
                d = float(density_v) / 255.0
                shape = d * shape + (1.0 - d)

        return shape

    def _get_const(self, layer: Layer) -> tuple[float, float]:
        """Get constant attributes."""
        shape = layer.tagged_blocks.get_data(Tag.BLEND_FILL_OPACITY, 255) / 255.0
        opacity = layer.opacity / 255.0
        return float(shape), opacity

    def _get_stroke(
        self,
        layer: Layer,
        path: np.ndarray | None = None,
        viewport: tuple[int, int, int, int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get stroke source, on ``viewport`` or this compositor's.

        The fill is drawn on the layer's box grown by the stroke width and
        pasted onto the viewport, but laid out across the path's own box:
        Photoshop maps a pattern's phase and a gradient's ramp across the
        path whatever the width. The coverage is rasterized straight onto
        that viewport: a path is placed in document coordinates,
        so asking for the viewport is what puts the stroke where the fill
        already is, whether or not the viewport happens to share the
        document's dimensions (#807).
        """
        if viewport is None:
            viewport = self._viewport
        if layer.stroke is None:
            raise ValueError("Layer does not have stroke data.")
        desc = layer.stroke._data
        try:
            width = math.ceil(desc.get("strokeStyleLineWidth", 1.0))
        except OverflowError as error:
            raise ValueError("Stroke width is not finite.") from error
        fill_bbox = cast(
            tuple[int, int, int, int],
            tuple(
                x + d
                for x, d in zip(layer.bbox, (-width, -width, width, width), strict=True)
            ),
        )
        # The width is the file's, and it grows the box the fill is drawn on.
        if fill_bbox[0] < fill_bbox[2] and fill_bbox[1] < fill_bbox[3]:
            check_growth(
                (fill_bbox[2] - fill_bbox[0]) * (fill_bbox[3] - fill_bbox[1]),
                max(_pixels(viewport), _pixels(layer.bbox)),
                _CANVAS_GROWTH,
            )
            check_pixel_size(
                fill_bbox[2] - fill_bbox[0],
                fill_bbox[3] - fill_bbox[1],
                self.channels,
                layer._psd._max_alloc_bytes,
                estimated_bytes=(fill_bbox[2] - fill_bbox[0])
                * (fill_bbox[3] - fill_bbox[1])
                * (
                    paint.GRADIENT_FILL_BYTES
                    if desc.get("strokeStyleContent").classID == b"gradientLayer"
                    else self.channels * 4
                ),
                warn=False,
            )
        color, _ = paint.create_fill_desc(
            layer, desc.get("strokeStyleContent"), fill_bbox, _path_bbox(layer)
        )
        if color is None:
            raise ValueError(
                "Unsupported stroke fill descriptor in layer strokeStyleContent"
            )
        color = paste(viewport, fill_bbox, color, 1.0)
        shape = vector.draw_stroke(layer, viewport, path)
        opacity = desc.get("strokeStyleOpacity", 100.0) / 100.0
        alpha = shape * opacity
        return color, shape, alpha

    def _add_overlay(self, layer: Layer, effect_name: str, canvas: _EffectSink) -> None:
        """Composite every overlay effect of one kind into the layer.

        The three overlay kinds differ only in what they draw, which is what
        ``_OVERLAY_DRAWS`` holds; the coverage arithmetic below is shared.

        An overlay covers the layer's whole region, so what it hands the
        canvas is its own mask and opacity alone -- the layer's coverage is
        what the canvas already measures a share of. It is drawn and laid out
        on the boxes ``_overlay_boxes()`` picks.

        An effect is skipped twice over: when its draw declines, which the
        pattern and the gradient do for a fill they cannot make sense of, and
        when reading its descriptor raises, which is the same degradation one
        step earlier (#826).
        """
        draw = _OVERLAY_DRAWS[effect_name]
        effects = _readable(layer, effect_name)
        if not effects:
            return
        box, layout = _overlay_boxes(layer, self._viewport, self._cache.content_bboxes)
        if box == (0, 0, 0, 0):
            return
        for effect in effects:
            # The draw is guarded and the coverage arithmetic below is not,
            # which keeps the shared half of this loop -- identical for all
            # three kinds -- out of the clause, so a fault there still
            # surfaces. The draw is not purely a descriptor read: a pattern
            # fill decodes the pattern's own pixels, and a file whose pattern
            # data is corrupt loses the effect here rather than the document.
            try:
                fill, shape_e = draw(
                    layer, effect.descriptor, self.channels, box, layout
                )
                opacity = effect.opacity / 100.0
                blend_mode = effect.blend_mode
            except _UNREADABLE as error:
                logger.debug(
                    "Skipping an unreadable %s effect in %s: %s",
                    effect_name,
                    layer,
                    error,
                )
                continue
            if fill is None:
                logger.debug("Skipping undrawable %s effect in %s", effect_name, layer)
                continue
            color = paste(self._viewport, box, fill, 1.0)
            if shape_e is None:
                shape_e = np.ones((self.height, self.width, 1), dtype=np.float32)
            else:
                shape_e = paste(self._viewport, box, shape_e)
            canvas.over(shape_e, color, blend_mode, opacity, fades_source=True)

    def _trace_shape(
        self,
        layer: Layer,
        viewport: tuple[int, int, int, int],
        shape: np.ndarray,
        traces_mask: bool,
    ) -> np.ndarray:
        """The layer's own coverage on ``viewport``, read past the canvas edge.

        ``shape`` is that same coverage on this compositor's viewport, and is
        the answer wherever ``viewport`` stays inside it. Where it does not,
        paste() zero-filled the rest, and a stroke traced from that copy
        follows the viewport edge as if it were the layer's own -- so the
        coverage is read again, on the box actually asked for (#804).

        A group's stroke follows its contents' alpha composited on their own,
        which ``shape`` is not, so it is always read again -- unless it traces
        the group's mask, which is the one branch of :py:meth:`_read_traced`
        that never consults a layer's contents at all (#808).
        """
        x0, y0, x1, y1 = viewport
        vx0, vy0, vx1, vy1 = self._viewport
        inside = vx0 <= x0 and vy0 <= y0 and x1 <= vx1 and y1 <= vy1
        if inside and (traces_mask or not isinstance(layer, GroupMixin)):
            return paste(viewport, self._viewport, shape)
        return self._read_traced(layer, viewport, traces_mask)

    def _read_traced(
        self, layer: Layer, viewport: tuple[int, int, int, int], traces_mask: bool
    ) -> np.ndarray:
        """What a stroke effect of the layer is traced from, read on ``viewport``.

        The coverage :py:meth:`_compose_effects` hands :py:meth:`_add_stroke_effects`
        on this compositor's viewport, read afresh on a box of the caller's:
        the mask when the stroke follows it, otherwise what the layer paints
        under its mask, with a shape layer's path interior held whole and a
        group's alpha held at the document's depth (:py:func:`_held`). The
        drop shadow's knockout and the outer glow's cut read their stroke
        bands from this too, so a band is cut from under either exactly where
        it is drawn.
        """
        x0, y0, x1, y1 = viewport
        # TODO: a group whose stroke traces its mask -- ``force`` with a vector
        # mask on the group -- gets the mask outline alone, which for a group
        # bears no relation to where its contents are. Pre-existing, unreachable
        # in the corpus (the two groups it applies to carry no stroke effect),
        # and the same inside the viewport as outside; ``_get_group_alpha()``
        # is what a fix would multiply in.
        if isinstance(layer, AdjustmentLayer):
            return self._adjustment_shape(layer, viewport)
        traced = mask = self._get_mask(layer, viewport)
        if not traces_mask:
            # Reading a group as an object finds no pixel data and no fill, so
            # the only route to its alpha is to composite its contents again
            # on the box asked for.
            if isinstance(layer, GroupMixin):
                traced = _held(
                    self._get_group_alpha(layer, viewport) * mask, layer._psd.depth
                )
            else:
                traced = self._get_object_coverage(layer, viewport)
            if _is_shape_layer(layer):
                traced = self._path_interior(layer, viewport, mask, traced)
        if not isinstance(traced, np.ndarray):
            # An unmasked layer whose stroke traces its mask: _get_mask()
            # yields a bare 1.0, and draw_stroke_effect() needs a canvas.
            traced = np.full((y1 - y0, x1 - x0, 1), traced, dtype=np.float32)
        return traced

    @staticmethod
    def _path_interior(
        layer: Layer,
        viewport: tuple[int, int, int, int],
        mask: float | np.ndarray,
        stored: np.ndarray,
    ) -> np.ndarray:
        """``stored`` coverage, with the path's interior held at ``mask``.

        Photoshop strokes the path, so a fill that fades inside it does not
        thin the stroke (#886). The path's own edge is left to the stored
        coverage: Photoshop's rasterizer does not agree with the exact area
        that :py:func:`vector.draw_vector_mask` computes there.
        """
        inside = vector.draw_vector_mask(layer, viewport) >= 1.0 - _PATH_FULL
        return np.where(inside, mask, stored).astype(np.float32, copy=False)

    def _rkey(self, kind: str, layer: Layer, *args: object) -> tuple | None:
        """A key in the cross-pass cache, or None when there is none."""
        cache = self._cache
        if cache.rc is None:
            return None
        return (
            kind,
            rcache.layer_key(layer, cache.layer_keys),
            cache.doc_key,
            self._force,
            self._channels,
            self._color_mode,
            args,
        )

    def _get_group_alpha(
        self, layer: Layer, viewport: tuple[int, int, int, int]
    ) -> np.ndarray:
        """The group's own alpha on ``viewport``, composited a second time.

        What a group's stroke is traced from and its shadows and glows are
        cast from: how opaque its contents are, not what they cover, so a
        child faded to nothing has no stroke round it. The object path
        re-reads a layer's stored pixels or redraws its fill. A group has
        neither -- its alpha *is* a composite -- so the only way to have it on
        a box of the caller's is to composite its children again on that box
        (#808).

        The box is narrowed the way :py:meth:`_get_group` narrows it, and for
        the same reason: ``_content_bbox()`` returns an ``Artboard``'s frame
        verbatim, and intersecting with it is how the artboard clip is
        implemented, so a trace that skipped it would follow the children's
        coverage out past the artboard rectangle. The pass-through branch is
        not symmetry for its own sake either -- :py:meth:`_get_group` gives a
        pass-through group this compositor's viewport untouched, so a
        pass-through artboard gets no frame clip today, and the trace has to
        reproduce that rather than improve on it.

        Narrowing is also what bounds the cost. For an isolated group ``inner``
        follows the contents rather than the stroke's margin, which is read
        unvalidated from the descriptor, so a forged size grows the ``paste()``
        back -- which ``draw_stroke_effect()`` already needed -- and not the
        composite. A pass-through group has no such bound and does pay for the
        whole box.

        The backdrop is transparent and carries neither knockout nor the
        document backdrop: the contents are composited on their own, as an
        isolated group composites them, so a knockout inside the group can
        only cut the group's own alpha away, pass-through or not.
        Effects, on the other hand, must run -- an overlay or an inset stroke
        reaches that canvas in the source its layer is composited from and an
        outset one beside it, which ``group-clips-child-stroke.psd`` measures.
        """
        inner = viewport
        passthrough = layer.blend_mode == BlendMode.PASS_THROUGH
        if not passthrough:
            inner = utils.intersect(
                viewport, _content_bbox(layer, self._cache.content_bboxes)
            )
        height, width = viewport[3] - viewport[1], viewport[2] - viewport[0]
        if inner == (0, 0, 0, 0):
            return np.zeros((height, width, 1), dtype=np.float32)

        # Shared across passes, the alpha is composited on the whole content
        # box, so another viewport crops it from there. Not a pass-through
        # artboard's: it paints outside its frame.
        rkey = None
        if self._cache.rc is not None and not (
            passthrough and isinstance(layer, Artboard)
        ):
            inner = utils.intersect(
                _content_bbox(layer, self._cache.content_bboxes),
                utils.union_bbox(layer._psd.viewbox, viewport),
            )
            rkey = self._rkey("group_alpha", layer, inner)
        key = (id(layer), inner)
        alpha = self._cache.group_alphas.get(key)
        if alpha is None and rkey is not None:
            assert self._cache.rc is not None
            hit = self._cache.rc.get(rkey)
            if hit is not None:
                alpha = self._cache.group_alphas[key] = hit[0]
        if alpha is None:
            inner_h, inner_w = inner[3] - inner[1], inner[2] - inner[0]
            group_compositor = Compositor(
                inner,
                np.ones((inner_h, inner_w, self._channels), dtype=np.float32),
                np.zeros((inner_h, inner_w, 1), dtype=np.float32),
                layer_filter=self._layer_filter,
                force=self._force,
                widen=self._widen,
                color_mode=self._color_mode,
                cache=self._cache,
                # Only the alpha is read, which text gamma leaves.
                text_gamma=None,
            )
            for sublayer in cast(GroupMixin, layer):
                group_compositor.apply(sublayer)
            alpha = self._cache.group_alphas[key] = group_compositor.alpha
            if rkey is not None:
                assert self._cache.rc is not None
                self._cache.rc.put(rkey, (alpha,))
        return paste(viewport, inner, alpha)

    def _cast_shape(
        self, layer: Layer, viewport: tuple[int, int, int, int]
    ) -> np.ndarray:
        """What casts the layer's drop shadow or outer glow, on ``viewport``.

        How opaque the layer is, before its opacity and fill opacity, and
        under its mask: an object's own alpha as it is painted
        (:py:meth:`_get_object_coverage`), and a group's composited alpha
        held at the document's depth, as its stroke traces it (:py:func:`_held`),
        so a child faded to nothing casts nothing. Not the outline of a vector
        mask, which a stroke follows under ``force``: on a pixel layer that
        would cast a shadow from the mask's empty margin.

        Read afresh on ``viewport`` rather than taken from the resolved
        source, which :py:meth:`_trace_shape` pastes from where it can. Both
        go on before the layer is resolved, so there is no source yet; and a
        group resolved over this canvas can take alpha from it through a
        knockout, which would make the shadow depend on the viewport. The
        group's contents are composited over nothing, once per box and pass,
        by :py:meth:`_get_group_alpha`.

        A shape redrawn from its path casts what :py:meth:`_get_object` paints
        of it: its stroke as well as its fill, and the stroke alone with the
        fill switched off. An adjustment layer has no pixels: its mask is the
        whole silhouette.
        """
        if isinstance(layer, AdjustmentLayer):
            return self._adjustment_shape(layer, viewport)
        if isinstance(layer, GroupMixin):
            alpha = self._get_group_alpha(layer, viewport)
            masked = utils.clip(alpha * self._get_mask(layer, viewport))
            return _held(masked, layer._psd.depth)
        return utils.clip(self._get_object_coverage(layer, viewport, alpha=True))

    def _knockout_shape(
        self,
        layer: Layer,
        viewport: tuple[int, int, int, int],
        silhouette: np.ndarray,
        covers: bool = False,
    ) -> np.ndarray:
        """What "layer knocks out drop shadow" cuts the shadow from under.

        An outer glow of a stroked layer is cut by it too, knockout or not
        (:py:meth:`_apply_outer_glows`).

        The silhouette the shadow is cast from, and the bands of the layer's
        stroke effects, whole whatever their opacity: Photoshop cuts under
        the stroke's shape, so a faint stroke shows the backdrop through
        itself and not the shadow. The part of a band outside the boundary
        *adds* to the silhouette rather than unioning with it, for the reason
        :py:meth:`_apply_outer_effects` gives: it is complementary to the
        coverage it is traced from, so on a pixel the edge cuts the layer
        covers one part and the stroke the rest. The part inside the boundary
        paints over the layer, not beside it, and on such a pixel comes out
        equal to the layer's own coverage; added it would count that coverage
        twice, so it is left out.

        That is the whole cut for a layer whose paint does not cover what
        lies beneath it. One that ``covers`` (:py:func:`_covers_beneath`)
        hides part of the shadow itself once composited on top: its paint at
        its fill opacity, and a Normal stroke's at the stroke's opacity, both
        faded by the layer opacity. The cut takes only the rest, so the
        shadow shows at what the shape leaves once the layer is on
        (:py:func:`~psd_tools.composite.effects.uncovered`): a layer at full
        fill and opacity, its strokes opaque, cuts nothing.

        Each band is drawn as :py:meth:`_add_stroke_effects` draws it, on its
        own :py:func:`_stroke_box` and traced from :py:meth:`_read_traced`,
        then pasted onto the shadow's canvas: an object's silhouette is its
        alpha and its stroke follows its coverage, so a band traced from the
        silhouette would cut the shadow where no stroke is painted. The
        stroke's box is the one ``stroke_bbox()`` checked against the
        budget, so the band costs what a stroke does and no more, and a
        stroke too large to draw cuts nothing. Its paint is not drawn here,
        so a stroke dropped only because its paint will not draw still cuts.
        """
        strokes = _readable(layer, "stroke")
        traces_mask = self._traces_mask(layer) if strokes else False
        # The bands are nested intervals on one side of the boundary, as
        # :py:meth:`_add_stroke_effects` cuts them, so together they claim the
        # widest of them of a pixel, and that is what adds to the silhouette.
        outside: np.ndarray | None = None
        painted: np.ndarray | None = None
        for effect in strokes:
            try:
                bbox = _within_reach(
                    _stroke_box(layer, effect.descriptor, self._cache.content_bboxes),
                    viewport,
                    stroke_reach(effect.descriptor),
                )
            except _UNREADABLE as error:
                logger.debug("Cannot measure a stroke effect of %s: %s", layer, error)
                continue
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                continue
            traced = self._read_traced(layer, bbox, traces_mask)
            try:
                band, inside = stroke_band(
                    bbox, traced, effect.descriptor, self._cache.stroke_bands, keep=True
                )
            except _UNREADABLE as error:
                logger.debug(
                    "Cannot cut a drop shadow under a stroke of %s: %s", layer, error
                )
                continue
            band = paste(viewport, bbox, band - inside)
            outside = band if outside is None else np.maximum(outside, band)
            if covers:
                share = band * _stroke_paint(effect)
                painted = share if painted is None else np.maximum(painted, share)
        shape = silhouette if outside is None else np.minimum(silhouette + outside, 1.0)
        if not covers:
            return shape
        fill_opacity, opacity = self._get_const(layer)
        hidden = silhouette * fill_opacity
        if painted is not None:
            hidden = hidden + painted
        return 1.0 - np.clip(uncovered(shape, hidden * opacity), 0.0, 1.0)

    def _has_scipy(self, what: str) -> bool:
        """Whether scipy is there to draw the layer's ``what``, warning when it is not.

        A shadow, glow, satin or bevel is common enough that a document
        carrying one composites without it, as it did before they were drawn
        at all, rather than failing for a missing package: each kind of effect
        warns, and the pass logs the loss once. That is not the per-effect
        guard: a missing package is a property of the environment, not of the
        file, which is why ``ImportError`` stays out of ``_UNREADABLE``.
        """
        if HAS_SCIPY:
            return True
        warnings.warn(
            "%s are not drawn without scipy. Install with: "
            "pip install 'psd-tools[composite]'" % what,
            stacklevel=2,
        )
        if not self._cache.scipy_warned:
            self._cache.scipy_warned = True
            logger.warning(
                "Layer effects that need scipy are not drawn; install scipy to "
                "draw them: pip install 'psd-tools[composite]'"
            )
        return False

    def _apply_beneath_effects(self, layer: Layer) -> None:
        """Composite what lies beneath the layer: its drop shadows, then its glows.

        Called before the layer is resolved: see :py:meth:`apply`. The outer
        glows go over the shadows, as Photoshop stacks them.
        """
        self._apply_drop_shadows(layer)
        self._apply_outer_glows(layer)

    def _apply_drop_shadows(self, layer: Layer) -> None:
        """Composite every drop shadow of the layer, beneath the layer.

        Called before the layer is resolved: see :py:meth:`apply`. The shadow
        is drawn on its own box, grown from :py:func:`_cast_bbox` so a group's
        has room for its children's effects, and cast from
        :py:meth:`_cast_shape` read on that box, so a layer that runs past the
        viewport still casts from the part of it out of view. The box is cut
        to what can reach the viewport (:py:func:`_within_reach`), which is
        what bounds the cost of a small viewport on a large layer.

        The layer opacity fades the shadow with the layer, as it fades an
        outer stroke. Fill opacity does not: it fades the layer's own paint
        and not its effects, which is what lets a layer at fill 0 cast a
        shadow and nothing else. With "layer knocks out drop shadow" on, a
        layer faded by fill opacity shows the backdrop through itself and not
        its shadow: the shadow is cut under the silhouette it is cast from by
        what the layer's paint leaves uncovered, and under the layer's stroke
        bands (:py:meth:`_knockout_shape`).

        Each effect is guarded as the strokes are, with the silhouette read
        outside both guards so that a fault compositing a group's contents is
        not taken for an unreadable effect. Without scipy the shadows are left
        out (:py:meth:`_has_scipy`).
        """
        shadows = _readable(layer, "dropshadow")
        if not shadows or not self._has_scipy("Drop shadows"):
            return
        _, opacity = self._get_const(layer)
        if opacity <= 0.0:
            return
        covers: bool | None = None
        cast_from = _cast_bbox(layer, self._cache.content_bboxes)
        for effect in shadows:
            shadow = cast(DropShadow, effect)
            try:
                geometry = shadow_geometry(shadow.descriptor, shadow.angle)
                effect_opacity = shadow.opacity / 100.0
                if not math.isfinite(effect_opacity):
                    raise ValueError("opacity %r is not finite" % effect_opacity)
                blend_mode = get_blend_mode(
                    shadow.descriptor, Key.Mode, BlendMode.MULTIPLY
                )
                knocks_out = shadow.layer_knocks_out
                contour = contour_table(shadow.contour)
                anti_aliased = shadow.anti_aliased
                noise = shadow.noise
                bbox = drop_shadow_bbox(
                    cast_from,
                    geometry.distance,
                    geometry.angle,
                    geometry.size,
                    layer._psd._max_alloc_bytes,
                )
                self._check_effect_growth(bbox, cast_from)
                bbox = _within_reach(
                    bbox,
                    self._viewport,
                    cast_reach(
                        geometry.distance,
                        geometry.angle,
                        geometry.size,
                        geometry.spread,
                    ),
                )
            except _UNREADABLE as error:
                logger.debug("Cannot measure a drop shadow of %s: %s", layer, error)
                continue
            if bbox == (0, 0, 0, 0):
                continue
            if noise > 0.0:
                logger.debug("The noise of a drop shadow of %s is not applied", layer)
            silhouette = self._cast_shape(layer, bbox)
            try:
                coverage = draw_drop_shadow(
                    bbox,
                    silhouette,
                    distance=geometry.distance,
                    angle=geometry.angle,
                    size=geometry.size,
                    spread=geometry.spread,
                    contour=contour,
                    anti_aliased=anti_aliased,
                )
                color, _ = paint.draw_solid_color_fill(
                    self._viewport, layer._psd.color_mode, shadow.descriptor
                )
            except _UNREADABLE as error:
                logger.debug("Cannot draw a drop shadow of %s: %s", layer, error)
                continue
            if knocks_out:
                if covers is None:
                    covers = _covers_beneath(layer, self._layer_filter)
                coverage = coverage * (
                    1.0 - self._knockout_shape(layer, bbox, silhouette, covers)
                )
            coverage = paste(self._viewport, bbox, coverage)
            if coverage.any():
                self._apply_source(
                    color, coverage, coverage * effect_opacity * opacity, blend_mode
                )

    def _apply_outer_glows(self, layer: Layer) -> None:
        """Composite every outer glow of the layer, beneath the layer.

        As :py:meth:`_apply_drop_shadows`, from the same silhouette on a box
        grown the same way, faded by the layer opacity and not by its fill
        opacity, each effect guarded the same way. There is no knockout
        switch: ``glow_paint()`` always cuts the glow away under the
        silhouette, leaving what the layer's paint will not cover once it is
        composited over the glow. Only a layer that covers that much of what
        lies beneath (:py:func:`_covers_beneath`) has its glow raised to
        match; any other leaves its glow the plain complement of its
        silhouette.

        A layer with a stroke has its glow cut under the stroke's band too,
        as a knocked-out shadow is (:py:meth:`_knockout_shape`): a faint or
        Lighten stroke shows the backdrop through itself, not the glow. That
        cut replaces ``glow_paint()``'s, which it equals without a stroke, and
        is read outside the guard as the shadow's is.

        The paint is drawn on the glow's own box, since a gradient glow's
        colour follows its falloff, and pasted beside it; where the glow
        covers nothing the colour pasted there is never read.
        """
        glows = _readable(layer, "outerglow")
        if not glows or not self._has_scipy("Outer glows"):
            return
        fill_opacity, opacity = self._get_const(layer)
        if opacity <= 0.0:
            return
        covers = _covers_beneath(layer, self._layer_filter)
        over = fill_opacity * opacity if covers else 0.0
        banded = bool(_readable(layer, "stroke"))
        cast_from = _cast_bbox(layer, self._cache.content_bboxes)
        for effect in glows:
            glow = cast(OuterGlow, effect)
            try:
                size, spread, range_ = glow_geometry(glow.descriptor)
                effect_opacity = glow.opacity / 100.0
                if not math.isfinite(effect_opacity):
                    raise ValueError("opacity %r is not finite" % effect_opacity)
                blend_mode = glow.blend_mode
                precise = glow.glow_type == GlowTechnique.PRECISE_MATTE
                contour = contour_table(glow.contour)
                anti_aliased = glow.anti_aliased
                noise = glow.noise
                jitter = glow.quality_jitter
                bbox = _within_reach(
                    outer_glow_bbox(
                        cast_from, glow.descriptor, layer._psd._max_alloc_bytes
                    ),
                    self._viewport,
                    glow_reach(size, spread, precise),
                )
            except _UNREADABLE as error:
                logger.debug("Cannot measure an outer glow of %s: %s", layer, error)
                continue
            if bbox == (0, 0, 0, 0):
                continue
            if noise > 0.0:
                logger.debug("The noise of an outer glow of %s is not applied", layer)
            if jitter > 0.0:
                logger.debug("The jitter of an outer glow of %s is not applied", layer)
            silhouette = self._cast_shape(layer, bbox)
            try:
                falloff = glow_falloff(
                    silhouette,
                    size=size,
                    spread=spread,
                    range_=range_,
                    precise=precise,
                    contour=contour,
                    anti_aliased=anti_aliased,
                )
                color, coverage = glow_paint(
                    bbox,
                    layer._psd.color_mode,
                    glow.descriptor,
                    np.zeros_like(silhouette) if banded else silhouette,
                    falloff,
                    over=0.0 if banded else over,
                )
            except _UNREADABLE as error:
                logger.debug("Cannot draw an outer glow of %s: %s", layer, error)
                continue
            if banded:
                coverage = coverage * (
                    1.0 - self._knockout_shape(layer, bbox, silhouette, covers)
                )
            coverage = paste(self._viewport, bbox, coverage)
            if coverage.any():
                self._apply_source(
                    paste(self._viewport, bbox, color),
                    coverage,
                    coverage * effect_opacity * opacity,
                    blend_mode,
                )

    def _add_inner_effects(self, layer: Layer, canvas: _EffectSink) -> None:
        """Composite every inner glow, then every inner shadow, into the layer.

        Each is drawn on the cast box grown by its size, from the silhouette
        :py:meth:`_cast_shape` reads there, so an edge past the viewport is
        not taken for the layer's own.
        """
        glows = _readable(layer, "innerglow")
        shadows = _readable(layer, "innershadow")
        if not (glows or shadows) or not self._has_scipy("Inner shadows and glows"):
            return
        cast_from = _cast_bbox(layer, self._cache.content_bboxes)
        silhouettes: dict[tuple[int, int, int, int], np.ndarray] = {}
        for effect in [*glows, *shadows]:
            try:
                bbox, draw = self._inner_effect(layer, effect, cast_from)
                opacity = effect.opacity / 100.0
                if not math.isfinite(opacity):
                    raise ValueError("opacity %r is not finite" % opacity)
                default = (
                    BlendMode.MULTIPLY
                    if isinstance(effect, InnerShadow)
                    else BlendMode.SCREEN
                )
                blend_mode = get_blend_mode(effect.descriptor, Key.Mode, default)
            except _UNREADABLE as error:
                logger.debug("Cannot measure an inner effect of %s: %s", layer, error)
                continue
            if bbox == (0, 0, 0, 0):
                continue
            silhouette = silhouettes.get(bbox)
            if silhouette is None:
                silhouette = silhouettes[bbox] = self._cast_shape(layer, bbox)
            try:
                coverage = draw(bbox, silhouette)
                color, _ = paint.draw_solid_color_fill(
                    self._viewport, layer._psd.color_mode, effect.descriptor
                )
            except _UNREADABLE as error:
                logger.debug("Cannot draw an inner effect of %s: %s", layer, error)
                continue
            coverage = paste(self._viewport, bbox, coverage)
            canvas.over(canvas.within(coverage), color, blend_mode, opacity)

    def _inner_box(self, layer: Layer, reach: float) -> tuple[int, int, int, int]:
        """The box an inner effect reaching ``reach`` past an edge is drawn on.

        The silhouette is read on it: the layer's box grown by the reach. An
        object's goes no further than the reach past the viewport, since
        nothing beyond that reaches back into view. A group's is not cut to
        the viewport, so it is the same box from every compositor that asks
        and a nested group's contents are composited for it once per pass.
        The other effects cut a group's box as an object's
        (:py:func:`_within_reach`), which draws on less but composites a
        nested group again for each compositor above it. A group's box that
        is over the budget is cut as an object's is, so the effect is still
        drawn where what reaches the viewport fits.
        Raises as :py:func:`inner_bbox` does, and when the box is out of
        proportion to the layer and the viewport.
        """
        cast_from = _cast_bbox(layer, self._cache.content_bboxes)
        budget = layer._psd._max_alloc_bytes
        bbox: tuple[int, int, int, int] | None = None
        if isinstance(layer, GroupMixin):
            try:
                bbox = inner_bbox(cast_from, reach, budget)
            except ValueError:
                pass
        if bbox is None:
            bbox = inner_bbox(cast_from, reach, budget, within=self._viewport)
        self._check_effect_growth(bbox, cast_from)
        return bbox

    def _check_effect_growth(
        self, bbox: tuple[int, int, int, int], cast_from: tuple[int, int, int, int]
    ) -> None:
        """Raise ``ValueError`` when an effect's canvas is out of proportion.

        Its distance and size are the file's, so the canvas they grow is held
        to the larger of the viewport and what it is cast from, as a vector
        stroke's is.
        """
        check_growth(
            _pixels(bbox),
            max(_pixels(self._viewport), _pixels(cast_from)),
            _CANVAS_GROWTH,
        )

    def _add_satins(self, layer: Layer, canvas: _EffectSink) -> None:
        """Composite every satin of the layer into it, over its overlays.

        Drawn from :py:meth:`_cast_shape` read on a box grown by the satin's
        reach, and guarded as the drop shadows are.
        """
        satins = _readable(layer, "satin")
        if not satins or not self._has_scipy("Satins"):
            return
        for effect in satins:
            satin = cast(Satin, effect)
            try:
                distance, size = satin_geometry(satin.descriptor)
                angle = satin.angle
                opacity = satin.opacity / 100.0
                if not math.isfinite(opacity):
                    raise ValueError("opacity %r is not finite" % opacity)
                blend_mode = satin.blend_mode
                inverted = satin.inverted
                contour = contour_table(satin.contour)
                anti_aliased = satin.anti_aliased
                bbox = self._inner_box(layer, abs(distance) + size)
            except _UNREADABLE as error:
                logger.debug("Cannot measure a satin of %s: %s", layer, error)
                continue
            if utils.intersect(self._viewport, bbox) == (0, 0, 0, 0):
                continue
            silhouette = self._cast_shape(layer, bbox)
            try:
                coverage = draw_satin(
                    bbox,
                    silhouette,
                    distance=distance,
                    angle=angle,
                    size=size,
                    inverted=inverted,
                    contour=contour,
                    anti_aliased=anti_aliased,
                )
                color, _ = paint.draw_solid_color_fill(
                    self._viewport, layer._psd.color_mode, satin.descriptor
                )
            except _UNREADABLE as error:
                logger.debug("Cannot draw a satin of %s: %s", layer, error)
                continue
            coverage = paste(self._viewport, bbox, coverage)
            canvas.over(
                canvas.within(coverage), color, blend_mode, opacity, fades_source=True
            )

    def _inner_effect(
        self,
        layer: Layer,
        effect: _StyledEffect,
        cast_from: tuple[int, int, int, int],
    ) -> tuple[
        tuple[int, int, int, int],
        Callable[[tuple[int, int, int, int], np.ndarray], np.ndarray],
    ]:
        """An inner effect's box, cut to what reaches the viewport, and its draw."""
        inner = cast(InnerShadow | InnerGlow, effect)
        desc = inner.descriptor
        what = "Inner shadow" if isinstance(inner, InnerShadow) else "Inner glow"
        angle = inner.angle if isinstance(inner, InnerShadow) else 0.0
        geometry = shadow_geometry(desc, angle, what)
        distance, size, choke = geometry.distance, geometry.size, geometry.spread
        contour = contour_table(inner.contour)
        anti_aliased = inner.anti_aliased
        if inner.noise > 0.0:
            logger.debug("The noise of an inner effect of %s is not applied", layer)
        bbox = drop_shadow_bbox(cast_from, 0.0, 0.0, size, layer._psd._max_alloc_bytes)
        if isinstance(inner, InnerShadow):
            bbox = _within_reach(
                bbox, self._viewport, cast_reach(distance, angle, size, choke)
            )

            def draw_shadow(
                box: tuple[int, int, int, int], shape: np.ndarray
            ) -> np.ndarray:
                return draw_inner_shadow(
                    box,
                    shape,
                    distance=distance,
                    angle=angle,
                    size=size,
                    choke=choke,
                    contour=contour,
                    anti_aliased=anti_aliased,
                )

            return bbox, draw_shadow
        glow = inner
        if glow.gradient is not None and desc.get(Key.Color) is None:
            raise ValueError("a gradient glow is not drawn")
        center = glow.glow_source == GlowSource.CENTER
        precise = glow.glow_type == GlowTechnique.PRECISE_MATTE
        glow_range = float(desc.get(Key.InputRange, 50.0))
        bbox = _within_reach(
            bbox, self._viewport, glow_reach(size, choke, precise, outer=False)
        )

        def draw_glow(box: tuple[int, int, int, int], shape: np.ndarray) -> np.ndarray:
            return draw_inner_glow(
                box,
                shape,
                size=size,
                choke=choke,
                center=center,
                precise=precise,
                glow_range=glow_range,
                contour=contour,
                anti_aliased=anti_aliased,
            )

        return bbox, draw_glow

    def _add_bevels(
        self, layer: Layer, canvas: _EffectSink, outer: list[_OuterEffect]
    ) -> None:
        """Composite every bevel of the layer: inside it, and beside it.

        The highlight and the shadow are two effects with a colour, blend mode
        and opacity each. What falls inside the silhouette goes into the
        layer, over everything else inside it; what an outer bevel, an emboss or
        a pillow emboss puts beside it is an outer effect, as an outset
        stroke is. A stroke emboss, which bevels the stroke rather than the
        layer, is not drawn, nor is a shape contour but a smooth inner
        bevel's; a texture is (:py:meth:`_bevel_texture`).
        """
        bevels = _readable(layer, "bevelemboss")
        if not bevels or not self._has_scipy("Bevels"):
            return
        for effect in bevels:
            bevel = cast(BevelEmboss, effect)
            try:
                style = _bevel_style(bevel)
                if style == BevelStyle.STROKE_EMBOSS:
                    logger.debug("A stroke emboss of %s is not drawn", layer)
                    continue
                depth, size, soften = bevel_geometry(bevel.descriptor)
                technique = _bevel_technique(bevel)
                up = bevel.direction != BevelDirection.STAMP_OUT
                angle, altitude = bevel.angle, bevel.altitude
                contour = contour_table(bevel.contour)
                anti_aliased = bevel.anti_aliased
                halves = []
                for color_desc, blend_mode, opacity in (
                    (
                        bevel.highlight_color,
                        bevel.highlight_mode,
                        bevel.highlight_opacity,
                    ),
                    (bevel.shadow_color, bevel.shadow_mode, bevel.shadow_opacity),
                ):
                    opacity /= 100.0
                    if not math.isfinite(opacity):
                        raise ValueError("opacity %r is not finite" % opacity)
                    paint_desc = Descriptor()
                    paint_desc[Key.Color] = color_desc
                    color, _ = paint.draw_solid_color_fill(
                        self._viewport, layer._psd.color_mode, paint_desc
                    )
                    halves.append((color, blend_mode, opacity))
                bbox = self._inner_box(layer, bevel_reach(size, soften, technique))
            except _UNREADABLE as error:
                logger.debug("Cannot measure a bevel of %s: %s", layer, error)
                continue
            if utils.intersect(self._viewport, bbox) == (0, 0, 0, 0):
                continue
            bevel_shape = None
            if bevel.use_shape:
                if bevel_smooth_inner(style, technique):
                    try:
                        bevel_shape = BevelShape(
                            contour_table(bevel.descriptor.get(Key.MappingShape)),
                            float(bevel.descriptor.get(Key.InputRange, 50.0)),
                            bool(bevel.descriptor.get(Key.AntiAlias, False)),
                        )
                    except _UNREADABLE as error:
                        logger.debug(
                            "Cannot read a bevel's shape of %s: %s", layer, error
                        )
                else:
                    logger.debug(
                        "The shape contour of a bevel of %s is not applied", layer
                    )
            texture = None
            if bevel.use_texture:
                texture = self._bevel_texture(layer, bevel, bbox, size)
            silhouette = self._cast_shape(layer, bbox)
            try:
                amounts = draw_bevel(
                    bbox,
                    silhouette,
                    style=style,
                    technique=technique,
                    up=up,
                    depth=depth,
                    size=size,
                    soften=soften,
                    angle=angle,
                    altitude=altitude,
                    contour=contour,
                    anti_aliased=anti_aliased,
                    texture=texture,
                    bevel_shape=bevel_shape,
                )
            except _UNREADABLE as error:
                logger.debug("Cannot draw a bevel of %s: %s", layer, error)
                continue
            for amount, (color, blend_mode, opacity) in zip(
                amounts, halves, strict=True
            ):
                if bevel_inside(style):
                    inside = paste(self._viewport, bbox, amount * silhouette)
                    canvas.over(canvas.within(inside), color, blend_mode, opacity)
                if bevel_beside(style):
                    beside = paste(self._viewport, bbox, amount * (1.0 - silhouette))
                    beside = np.minimum(beside, 1.0 - canvas.region)
                    if beside.any():
                        outer.append(
                            _OuterEffect(
                                color, beside, opacity * canvas.opacity, blend_mode
                            )
                        )

    def _bevel_texture(
        self,
        layer: Layer,
        bevel: BevelEmboss,
        bbox: tuple[int, int, int, int],
        size: float,
    ) -> np.ndarray | None:
        """A bevel's texture on ``bbox``: its pattern's luminance as a height field.

        The pattern tiles from where a pattern overlay of the layer would
        (:py:func:`_tiling_origin`), at the texture's scale;
        :py:func:`texture_height` turns its luminance into height; the
        pattern's alpha is not read. None, and the bevel goes on untextured,
        when the pattern is not in the document or cannot be read, or the
        document or the pattern is neither RGB nor grayscale.
        """
        if layer._psd.color_mode not in (ColorMode.RGB, ColorMode.GRAYSCALE):
            logger.debug(
                "A bevel texture is drawn in RGB and grayscale only: %s", layer
            )
            return None
        try:
            pattern = paint._find_pattern(layer._psd, bevel.descriptor)
            if pattern is None:  # _find_pattern has logged it
                return None
            planes = _TEXTURE_PLANES.get(pattern.image_mode)
            if planes is None:
                logger.debug(
                    "The texture pattern of a bevel of %s is neither RGB nor grey",
                    layer,
                )
                return None
            _, layout = _overlay_boxes(
                layer, self._viewport, self._cache.content_bboxes
            )
            origin = _tiling_origin(layer, bevel.descriptor, layout)
            fill, _ = paint.draw_pattern_fill(
                bbox, layer._psd, bevel.descriptor, origin=origin
            )
        except _UNREADABLE as error:
            logger.debug("Cannot read the texture of a bevel of %s: %s", layer, error)
            return None
        if fill is None or fill.shape[2] != planes:
            logger.debug("Cannot read the texture of a bevel of %s", layer)
            return None
        luminance = fill[:, :, 0] if planes == 1 else fill @ _BT601_LUMINANCE
        return texture_height(
            np.clip(luminance, 0.0, 1.0),
            size=size,
            depth=bevel.texture_depth,
            inverted=bevel.texture_inverted,
        )

    def _add_stroke_effects(
        self,
        layer: Layer,
        shape: float | np.ndarray,
        canvas: _EffectSink,
        outer: list[_OuterEffect],
        traces_mask: bool,
    ) -> None:
        # ``shape`` is the layer's coverage on this compositor's viewport, or
        # -- when the stroke traces the mask -- _get_mask()'s output, which is
        # a bare 1.0 for a layer with no mask. _trace_shape() hands that to
        # paste(), which needs something with a channel axis. broadcast_to
        # gives a stride-0 view rather than a full canvas, which is all
        # paste() requires since it only reads from it.
        if not isinstance(shape, np.ndarray):
            shape = np.broadcast_to(np.float32(shape), (self.height, self.width, 1))
        # Photoshop traces every stroke from the layer, so ``shape`` stays the
        # layer's coverage for the whole loop and each effect's own mask gets a
        # separate name. Assigning the mask back over ``shape`` made the second
        # stroke outline the first stroke rather than the layer (#798).
        # Each effect is guarded on its own, so a layer carrying two strokes
        # still draws the one it can read (#798). The two guards are kept
        # apart so that _trace_shape() sits outside both: for a group it
        # composites the whole subtree, and a fault raised down there is not
        # this effect being unreadable. What is inside the second guard is the
        # drawing as well as the reading, so a stroke whose pattern data will
        # not decode is lost here too, rather than taking the document.
        # Each band is cut from under the ones listed above it, whatever
        # their opacity: Photoshop shows the backdrop through a faint stroke,
        # not the next stroke down. So the bands are measured top first, in
        # file order, and painted bottom first. Every band is measured from
        # the same boundary, so on either side of it the bands are nested
        # intervals: what the ones above claim of a pixel is the widest of
        # them, on each side separately, and a band keeps what exceeds that.
        runs: list[_BandSum] = []
        # The topmost band, whole, and the (inside, outside) the bands above
        # claim, built from it once there is a band to cut.
        top: _StrokeBand | None = None
        claimed: tuple[np.ndarray, np.ndarray] | None = None
        for index, effect in enumerate(reversed(_readable(layer, "stroke"))):
            try:
                # Effect must happen at the layer viewport, grown so an outset
                # or centered stroke has room for the part of itself that
                # falls outside the layer (#792), then cut to what can show on
                # this compositor's viewport.
                bbox = _within_reach(
                    _stroke_box(layer, effect.descriptor, self._cache.content_bboxes),
                    self._viewport,
                    stroke_reach(effect.descriptor),
                )
            except _UNREADABLE as error:
                logger.debug("Cannot measure a stroke effect of %s: %s", layer, error)
                continue
            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                continue
            rkey = self._rkey("stroke", layer, bbox, traces_mask, index)
            hit = None
            if rkey is not None:
                assert self._cache.rc is not None
                hit = self._cache.rc.get(rkey)
            if hit is None:
                shape_in_bbox = self._trace_shape(layer, bbox, shape, traces_mask)
            try:
                if hit is not None:
                    color, mask_in_bbox, inside_in_bbox = hit
                else:
                    color, mask_in_bbox, inside_in_bbox = draw_stroke_effect_split(
                        bbox,
                        shape_in_bbox,
                        effect.descriptor,
                        layer._psd,
                        self._cache.stroke_bands,
                    )
                    if rkey is not None:
                        assert self._cache.rc is not None
                        self._cache.rc.put(rkey, (color, mask_in_bbox, inside_in_bbox))
                opacity = effect.opacity / 100.0
                blend_mode = effect.blend_mode
            except _UNREADABLE as error:
                logger.debug("Cannot draw a stroke effect of %s: %s", layer, error)
                continue
            color = paste(self._viewport, bbox, color)
            mask = paste(self._viewport, bbox, mask_in_bbox)
            inside = paste(self._viewport, bbox, inside_in_bbox)
            if top is None:
                top = _StrokeBand(color, mask, inside, opacity, blend_mode)
                runs.append(_BandSum(top))
                continue
            if claimed is None:
                claimed = (top.inside, top.coverage - top.inside)
            claimed_inside, claimed_outside = claimed
            outside = mask - inside
            claimed = (
                np.maximum(claimed_inside, inside),
                np.maximum(claimed_outside, outside),
            )
            inside = np.maximum(inside - claimed_inside, 0.0)
            mask = inside + np.maximum(outside - claimed_outside, 0.0)
            band = _StrokeBand(color, mask, inside, opacity, blend_mode)
            if runs[-1].blend_mode == blend_mode:
                runs[-1].add(band)
            else:
                runs.append(_BandSum(band))

        # The runs are disjoint, so on a pixel two share they add; painted
        # one over the other, the upper would fade the lower by what it
        # paints there. So each half of a run goes on at its share of what
        # the runs above leave of the pixel, as :py:meth:`_apply_outer_effects`
        # puts a band beside the layer, and the sum comes out exact.
        halves: list[_StrokeHalves] = []
        above_share: float | np.ndarray = 0.0
        above_beside: float | np.ndarray = 0.0
        for run in runs:
            band = run.band()
            # The two halves of the band go to different places, and an inset
            # or outset stroke is all of one of them -- only a centered stroke
            # is split. The rest of the band, not ``mask - inside``, so that
            # the two are exactly what was drawn however they divide.
            overlap = np.minimum(band.inside, canvas.region)
            share = canvas.within(overlap)
            # ``inside`` is inside the boundary the stroke was traced from,
            # which is the layer's own coverage only while the two are the
            # same shape. A stroke that traces the *mask* -- a fill layer's,
            # or any layer's under ``force`` -- can land where the layer has
            # no paint to knock out, and that part of it paints beside the
            # layer like the outer half does rather than being dropped.
            #
            # Clamped to what the layer's region leaves of the pixel: the two
            # are disjoint by construction, and a band that overran would
            # otherwise carry the pixel past opaque.
            beside = np.minimum(band.coverage - overlap, 1.0 - canvas.region)
            halves.append(_StrokeHalves(band, share, beside, above_share, above_beside))
            # A Normal band replaces what is under its inner half, so it
            # takes all of its share off the bands below, not its opacity's.
            if band.blend_mode is BlendMode.NORMAL and canvas.knocks_out:
                above_share = above_share + share
            else:
                above_share = above_share + share * band.opacity
            # What the run's outer half is painted at, layer opacity included.
            above_beside = above_beside + beside * band.opacity * canvas.opacity
        del runs

        for band, share, beside, above_share, above_beside in reversed(halves):
            if isinstance(above_share, np.ndarray):
                share = utils.clip(utils.divide(share, 1.0 - above_share, fill=0.0))
            # A Normal stroke replaces the layer under its inner half.
            if band.blend_mode is BlendMode.NORMAL:
                canvas.replace(share, band.color, band.opacity)
            else:
                canvas.over(share, band.color, band.blend_mode, band.opacity)
            if beside.any():
                outer.append(
                    _OuterEffect(
                        band.color,
                        beside,
                        band.opacity * canvas.opacity,
                        band.blend_mode,
                        above_beside,
                    )
                )
