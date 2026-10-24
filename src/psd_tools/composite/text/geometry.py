"""Where a type layer's lines and glyphs sit, in document pixels.

The layout the renderer draws from, mapped through the layer's transform.
Nothing is rasterized.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from typing import TYPE_CHECKING, Any

import freetype

from psd_tools.composite.text import layout as _layout
from psd_tools.composite.text.constants import FAUX_BOLD_EM, FAUX_ITALIC_SHEAR
from psd_tools.composite.text.fonts import FontMap
from psd_tools.composite.text.model import TextModel, Transform

if TYPE_CHECKING:
    from psd_tools.api.layers import TypeLayer

Box = tuple[float, float, float, float]


def _map_box(transform: Transform, box: Box) -> list[float]:
    """The axis-aligned bounds of a text-space rectangle in document px."""
    xx, xy, yx, yy, tx, ty = transform
    x0, y0, x1, y1 = box
    corners = ((x0, y0), (x1, y0), (x0, y1), (x1, y1))
    xs = [xx * x + yx * y + tx for x, y in corners]
    ys = [xy * x + yy * y + ty for x, y in corners]
    return [round(v, 2) for v in (min(xs), min(ys), max(xs), max(ys))]


def _union(boxes: list[Box]) -> Box:
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _cell(glyph: _layout.PlacedGlyph) -> Box:
    """Pen advance by the face's ascent and descent, in text-space pt."""
    face = glyph.face
    upem = face.metrics.upem
    ascent = face.metrics.ascent_units / upem * glyph.em_h
    descent = max(-face.ft.descender, 0) / upem * glyph.em_h
    return (glyph.x, glyph.y - ascent, glyph.x + glyph.advance, glyph.y + descent)


def _ink(glyph: _layout.PlacedGlyph, lean: float) -> Box | None:
    """The outline's bounds in text-space pt; ``None`` for a blank glyph.

    ``lean`` is the faux italic shear in text space, x per unit of height.
    """
    face = glyph.face
    points = face.instructed(glyph.gid)
    with face.lock:
        try:
            face.ft.load_glyph(glyph.gid, freetype.FT_LOAD_NO_SCALE)
        except freetype.FT_Exception:
            return None
        m = face.ft.glyph.metrics
        left, top, width, height = m.horiBearingX, m.horiBearingY, m.width, m.height
    if points is not None:
        (left, bottom), (right, top) = points.min(axis=0), points.max(axis=0)
        width, height = right - left, top - bottom
    if width == 0 or height == 0:
        return None
    sx = glyph.em_w / face.metrics.upem
    sy = glyph.em_h / face.metrics.upem
    x0, x1 = glyph.x + left * sx, glyph.x + (left + width) * sx
    up, down = top * sy, (top - height) * sy  # heights above the baseline
    if glyph.faux_bold:
        # The renderer dilates the outline by half the stroke on every side.
        bx, by = FAUX_BOLD_EM / 2 * glyph.em_w, FAUX_BOLD_EM / 2 * glyph.em_h
        x0, x1, up, down = x0 - bx, x1 + bx, up + by, down - by
    if glyph.faux_italic:
        x0, x1 = x0 + lean * down, x1 + lean * up
    return (x0, glyph.y - up, x1, glyph.y - down)


def text_geometry(layer: TypeLayer, fonts: FontMap) -> dict[str, Any]:
    """The lines and glyphs of ``layer`` in document pixels, as JSON-ready data.

    ``text`` is the string every index points into. Each line has its
    ``text``, ``start`` and ``end`` (``text[start:end]``, the break character
    excluded), its ``baseline`` from pen start to pen end as
    ``[x0, y0, x1, y1]``, the ``box`` around its glyphs' cells and ink
    (``None`` when it draws nothing) and whether it was ``drawn``. Each glyph
    has ``i`` and ``end``, the slice of ``text`` it draws (several characters
    for a ligature), ``ch`` (``text[i]``), its ``line`` and style ``run``
    indices, the ``font`` it was drawn with, its ``cell`` (pen advance by the
    face's ascent and descent) and its ``ink`` box (``None`` when blank).
    Boxes are ``[x0, y0, x1, y1]``.

    Under the ``skip`` font policy the runs of a font listed in
    ``missing_fonts`` have no glyphs; under ``raise`` a missing font raises
    :class:`~psd_tools.composite.text.MissingFontError`.
    """
    model = TextModel.from_layer(layer)
    lay = _layout.layout(model, fonts)
    t = model.transform
    xx, xy, yx, yy, tx, ty = t
    lean = FAUX_ITALIC_SHEAR * math.hypot(yx, yy) / (math.hypot(xx, xy) or 1.0)
    starts = [line.start for line in lay.lines]

    def point(x: float, y: float) -> list[float]:
        return [round(xx * x + yx * y + tx, 2), round(xy * x + yy * y + ty, 2)]

    rows = []
    clusters: dict[int, set[int]] = {}
    for g in lay.glyphs:
        line = max(bisect_right(starts, g.cluster) - 1, 0)
        rows.append((line, g, _cell(g), _ink(g, lean)))
        clusters.setdefault(line, set()).add(g.cluster)
    ordered = {line: sorted(found) for line, found in clusters.items()}

    glyphs = []
    boxes: dict[int, list[Box]] = {}
    for line, g, cell, ink in rows:
        on_line = ordered[line]
        k = bisect_right(on_line, g.cluster)
        boxes.setdefault(line, []).append(cell)
        if ink is not None:
            boxes[line].append(ink)
        name = g.face.ft.postscript_name
        glyphs.append(
            {
                "i": g.cluster,
                "end": on_line[k] if k < len(on_line) else lay.lines[line].end,
                "ch": model.text[g.cluster],
                "line": line,
                "run": g.run,
                "font": name.decode() if name else "",
                "cell": _map_box(t, cell),
                "ink": None if ink is None else _map_box(t, ink),
            }
        )

    lines = []
    for j, info in enumerate(lay.lines):
        pen_end = info.x0 + info.width + info.n_spaces * info.stretch
        line_boxes = boxes.get(j)
        lines.append(
            {
                "text": info.text,
                "start": info.start,
                "end": info.end,
                "baseline": point(info.x0, info.y) + point(pen_end, info.y),
                "box": _map_box(t, _union(line_boxes)) if line_boxes else None,
                "drawn": info.drawn,
            }
        )
    return {
        "layer": layer.name,
        "text": model.text,
        "bbox": list(layer.bbox),
        "lines": lines,
        "glyphs": glyphs,
        "missing_fonts": [n for n, r in lay.fonts.items() if r.how == "missing"],
        "warnings": list(lay.warnings),
    }
