"""Splice: redraw only what a text edit changed.

:func:`capture` keeps a type layer's layout and stored raster before an
edit; :func:`splice` renders the edited layer and keeps the stored pixels
wherever no changed glyph reaches, so Photoshop's raster survives on every
line whose glyphs, styles and positions are unchanged.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from psd_tools.composite.text.constants import FAUX_ITALIC_SHEAR, SPLICE_PAD
from psd_tools.composite.text.geometry import _cell, _ink, _map_box
from psd_tools.composite.text.layout import Layout, PlacedGlyph, Rule, layout
from psd_tools.composite.text.model import TextModel, Transform
from psd_tools.composite.text.raster import Bbox, Options
from psd_tools.composite.text.report import RenderResult, Splice
from psd_tools.composite.text.verify import stored_raster
from psd_tools.constants import ColorMode

if TYPE_CHECKING:
    from psd_tools.api.layers import TypeLayer
    from psd_tools.composite.text.fonts import FontMap


@dataclass(frozen=True, eq=False)
class Before:
    """A type layer before an edit: its model, layout and stored raster."""

    model: TextModel
    layout: Layout
    rgba: np.ndarray | None  #: straight RGBA ``float32`` as stored
    bbox: Bbox | None
    rgb: bool  #: an RGB document, whose channels round-trip exactly


def capture(layer: TypeLayer, fonts: FontMap | None = None) -> Before:
    """``layer`` as it is now, for :func:`splice` after an edit."""
    if fonts is None:
        from psd_tools.composite.text import default_font_map  # noqa: PLC0415

        fonts = default_font_map()
    model = TextModel.from_layer(layer)
    rgba, bbox = stored_raster(layer)
    rgb = layer._psd.color_mode == ColorMode.RGB
    return Before(model, layout(model, fonts), rgba, bbox, rgb)


def _glyph_key(g: PlacedGlyph) -> tuple:
    """Everything that draws a glyph's pixels."""
    r = round
    return (
        g.face.key,
        g.gid,
        r(g.x, 6),
        r(g.y, 6),
        r(g.em_w, 6),
        r(g.em_h, 6),
        g.fill,
        g.faux_bold,
        g.faux_italic,
    )


def _rule_key(rule: Rule) -> tuple:
    return (
        round(rule.x0, 6),
        round(rule.x1, 6),
        round(rule.y, 6),
        round(rule.thickness, 6),
        rule.fill,
    )


def _lines_of(lay: Layout) -> list[int]:
    """Each glyph's line index."""
    starts = [line.start for line in lay.lines]
    return [max(bisect_right(starts, g.cluster) - 1, 0) for g in lay.glyphs]


def _kept(old: Layout, new: Layout) -> tuple[list[bool], list[bool]]:
    """Per glyph of ``old`` and of ``new``: whether the other layout draws it alike."""
    old_keys = [_glyph_key(g) for g in old.glyphs]
    new_keys = [_glyph_key(g) for g in new.glyphs]
    old_set, new_set = set(old_keys), set(new_keys)
    return [k in new_set for k in old_keys], [k in old_set for k in new_keys]


def _glyph_box(g: PlacedGlyph, transform: Transform, pad: int) -> Bbox | None:
    """The pixels a glyph can ink, its outline box grown by ``pad``.

    A colour font's glyph is drawn from its own document, so its cell counts too.
    """
    xx, xy, yx, yy = transform[:4]
    lean = FAUX_ITALIC_SHEAR * math.hypot(yx, yy) / (math.hypot(xx, xy) or 1.0)
    box = _ink(g, lean)
    if g.face.metrics.has_svg:
        cell = _cell(g)
        box = cell if box is None else _union_f(box, cell)
    if box is None:
        return None
    x0, y0, x1, y1 = _map_box(transform, box)
    return (
        math.floor(x0) - pad,
        math.floor(y0) - pad,
        math.ceil(x1) + pad,
        math.ceil(y1) + pad,
    )


def _rule_box(rule: Rule, transform: Transform, pad: int) -> Bbox:
    h = rule.thickness / 2
    x0, y0, x1, y1 = _map_box(transform, (rule.x0, rule.y - h, rule.x1, rule.y + h))
    return (
        math.floor(x0) - pad,
        math.floor(y0) - pad,
        math.ceil(x1) + pad,
        math.ceil(y1) + pad,
    )


def _union_f(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _union(boxes: Sequence[Bbox]) -> Bbox:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _clip(box: Bbox, canvas: Bbox) -> Bbox:
    return (
        max(box[0], canvas[0]),
        max(box[1], canvas[1]),
        min(box[2], canvas[2]),
        min(box[3], canvas[3]),
    )


def _mask(boxes: Sequence[Bbox], canvas: Bbox) -> np.ndarray:
    cx0, cy0, cx1, cy1 = canvas
    out = np.zeros((cy1 - cy0, cx1 - cx0), bool)
    for x0, y0, x1, y1 in boxes:
        out[
            max(y0 - cy0, 0) : max(y1 - cy0, 0), max(x0 - cx0, 0) : max(x1 - cx0, 0)
        ] = True
    return out


def _place(rgba: np.ndarray | None, bbox: Bbox | None, canvas: Bbox) -> np.ndarray:
    """Straight RGBA at ``bbox`` on ``canvas``, transparent elsewhere."""
    cx0, cy0, cx1, cy1 = canvas
    out = np.zeros((cy1 - cy0, cx1 - cx0, 4), np.float32)
    if rgba is not None and bbox is not None and rgba.size:
        x0, y0, x1, y1 = bbox
        out[y0 - cy0 : y1 - cy0, x0 - cx0 : x1 - cx0] = rgba
    return out


def _incomparable(before: Before, model: TextModel) -> str | None:
    """Why the two layouts' glyphs cannot be matched, if they cannot."""
    if model.transform != before.model.transform:
        return "the transform changed"
    if model.antialias != before.model.antialias:
        return "the anti-aliasing changed"
    return None


def splice(
    layer: TypeLayer,
    before: Before,
    fonts: FontMap | None = None,
    options: Options | None = None,
) -> RenderResult:
    """Render ``layer`` keeping ``before``'s stored pixels where nothing changed.

    A glyph is unchanged when the old layout drew the same glyph of the same
    face, size, fill and faux styles at the same place. The engine's pixels
    fill the outline boxes, grown by ``SPLICE_PAD``, of the changed glyphs old
    and new, and the stored ones stay everywhere else. The whole layer is rendered instead (``fallback`` says
    why) when the transform or anti-aliasing changed, the document is not
    RGB, the layer has no stored raster, or stored ink lies outside those
    boxes and the unchanged glyphs' (the layout does not place Photoshop's
    glyphs). Nothing changed: the stored raster comes back as it is, ``zone``
    ``None``.
    """
    from psd_tools.composite.text import _render  # noqa: PLC0415

    if fonts is None:
        from psd_tools.composite.text import default_font_map  # noqa: PLC0415

        fonts = default_font_map()
    result, lay, model = _render(layer, fonts, options)
    lines = tuple(range(len(lay.lines)))

    def done(
        kept: tuple[int, ...],
        zone: Bbox | None,
        fallback: str | None = None,
        rgba: np.ndarray = result.rgba,
        bbox: Bbox = result.bbox,
    ) -> RenderResult:
        redrawn = tuple(j for j in lines if j not in kept)
        info = Splice(kept, redrawn, zone, fallback)
        return RenderResult(rgba, bbox, replace(result.report, splice=info))

    why = _incomparable(before, model)
    if why is not None:
        return done((), result.bbox, why)
    t = model.transform
    keep_old, keep_new = _kept(before.layout, lay)
    old_rules = {_rule_key(r) for r in before.layout.rules}
    new_rules = {_rule_key(r) for r in lay.rules}
    changed = [
        _glyph_box(g, t, SPLICE_PAD)
        for glyphs, keep in ((before.layout.glyphs, keep_old), (lay.glyphs, keep_new))
        for g, k in zip(glyphs, keep, strict=True)
        if not k
    ]
    changed += [
        _rule_box(r, t, SPLICE_PAD)
        for r in before.layout.rules
        if _rule_key(r) not in new_rules
    ]
    changed += [
        _rule_box(r, t, SPLICE_PAD) for r in lay.rules if _rule_key(r) not in old_rules
    ]
    of = _lines_of(lay)
    moved = {of[n] for n, k in enumerate(keep_new) if not k}
    kept = tuple(j for j in lines if j not in moved)
    boxes = [b for b in changed if b is not None]
    if not boxes:
        if before.rgba is None or before.bbox is None:
            x, y = result.bbox[0], result.bbox[1]
            return done(kept, None, None, result.rgba[:0, :0], (x, y, x, y))
        return done(kept, None, None, before.rgba, before.bbox)
    if not before.rgb:
        return done((), result.bbox, "not an RGB document")
    if before.rgba is None or before.bbox is None:
        return done((), result.bbox, "no stored raster")

    canvas = _union([before.bbox, result.bbox])
    zone = _mask(boxes, canvas)
    held = [
        _glyph_box(g, t, SPLICE_PAD)
        for g, k in zip(lay.glyphs, keep_new, strict=True)
        if k
    ]
    held += [
        _rule_box(r, t, SPLICE_PAD) for r in lay.rules if _rule_key(r) in old_rules
    ]
    out = _place(before.rgba, before.bbox, canvas)
    stray = (
        (out[..., 3] > 0) & ~zone & ~_mask([b for b in held if b is not None], canvas)
    )
    if stray.any():
        return done(
            (), result.bbox, f"stored ink outside the layout: {int(stray.sum())} px"
        )
    out[zone] = _place(result.rgba, result.bbox, canvas)[zone]
    ys, xs = np.nonzero(out[..., 3] > 0)
    if not len(ys):
        x, y = canvas[0], canvas[1]
        return done(kept, _clip(_union(boxes), canvas), None, out[:0, :0], (x, y, x, y))
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    cx, cy = canvas[0], canvas[1]
    bbox = (cx + x0, cy + y0, cx + x1, cy + y1)
    zone_box = _clip(_union(boxes), canvas)
    return done(kept, zone_box, None, np.ascontiguousarray(out[y0:y1, x0:x1]), bbox)
