"""Where a type layer's characters and lines sit, in document pixels (research prototype).

The layout the engine draws from, mapped through the layer's transform: one
box per line (its characters' range, baseline, ascent to descent) and one per
glyph (its ``str`` index, pen cell and ink box). Nothing is rasterized.
"""

from __future__ import annotations

from typing import Any

import freetype

from psd_tools.composite.text import layout as _layout
from psd_tools.composite.text.fonts import FontMap
from psd_tools.composite.text.model import TextModel, Transform

Box = tuple[float, float, float, float]


def _map_box(transform: Transform, x0: float, y0: float, x1: float, y1: float) -> Box:
    """The axis-aligned box of a text-space rectangle in document pixels."""
    xx, xy, yx, yy, tx, ty = transform
    xs, ys = [], []
    for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        xs.append(xx * x + yx * y + tx)
        ys.append(xy * x + yy * y + ty)
    return (min(xs), min(ys), max(xs), max(ys))


def _ink(glyph: _layout.PlacedGlyph) -> Box | None:
    """The glyph's outline box in text-space pt, or None for a blank glyph."""
    face = glyph.face
    with face.lock:
        face.ft.load_glyph(glyph.gid, freetype.FT_LOAD_NO_SCALE)
        m = face.ft.glyph.metrics
        left, top, width, height = m.horiBearingX, m.horiBearingY, m.width, m.height
    if width == 0 or height == 0:
        return None
    sx = glyph.em_w / face.metrics.upem
    sy = glyph.em_h / face.metrics.upem
    x0 = glyph.x + left * sx
    return (x0, glyph.y - top * sy, x0 + width * sx, glyph.y - (top - height) * sy)


def _round(box: Box) -> list[float]:
    return [round(v, 2) for v in box]


def text_geometry(layer: Any, fonts: FontMap) -> dict:
    """Lines and glyphs of ``layer`` as JSON-ready data in document pixels.

    ``glyphs[i]["i"]`` indexes the layer's text, so a token's characters find
    their boxes; ``lines[j]["start"]:["end"]`` is the line's slice of it.
    """
    model = TextModel.from_layer(layer)
    lay = _layout.layout(model, fonts)
    t = model.transform
    glyphs = []
    by_line: dict[int, list[Box]] = {}
    starts = [line.start for line in lay.lines]
    for g in lay.glyphs:
        ink = _ink(g)
        upem = g.face.metrics.upem
        ascent = g.face.metrics.ascent_units / upem * g.em_h
        cell = (g.x, g.y - ascent, g.x + g.advance, g.y + 0.25 * g.em_h)
        line = max(0, sum(1 for s in starts if s <= g.cluster) - 1)
        by_line.setdefault(line, []).append(cell)
        glyphs.append(
            {
                "i": g.cluster,
                "ch": model.text[g.cluster] if 0 <= g.cluster < len(model.text) else "",
                "line": line,
                "run": g.run,
                "font": g.face.ft.postscript_name.decode()
                if g.face.ft.postscript_name
                else "",
                "cell": _round(_map_box(t, *cell)),
                "ink": None if ink is None else _round(_map_box(t, *ink)),
            }
        )
    lines = []
    for j, info in enumerate(lay.lines):
        cells = by_line.get(j)
        box = None
        if cells:
            box = _round(
                _map_box(
                    t,
                    min(c[0] for c in cells),
                    min(c[1] for c in cells),
                    max(c[2] for c in cells),
                    max(c[3] for c in cells),
                )
            )
        lines.append(
            {
                "text": info.text,
                "start": info.start,
                "end": info.end,
                "baseline": round(t[1] * info.x0 + t[3] * info.y + t[5], 2),
                "box": box,
                "drawn": info.drawn,
            }
        )
    return {
        "layer": layer.name,
        "text": model.text,
        "bbox": list(layer.bbox),
        "lines": lines,
        "glyphs": glyphs,
        "warnings": list(lay.warnings),
    }
