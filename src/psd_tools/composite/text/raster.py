"""Rasterization: a :class:`Layout` under a transform to straight RGBA.

``transform`` maps text-space pt to document px (a caller folds a non-72 ppi
resolution into it). Every glyph is loaded at its final pixel size and
FreeType's own transform carries only the column-normalised rotation, flip
or shear: loading at the text-space em and scaling afterwards quantises the
outline at the small size, and FreeType adopts an embedded bitmap strike's
integer ppem when the em rounds onto one.
"""

from __future__ import annotations

import ctypes
import math
from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeVar

import freetype
import numpy as np

from psd_tools.composite.scanline import fill_coverage
from psd_tools.composite.text.constants import (
    FAUX_BOLD_EM,
    FAUX_ITALIC_SHEAR,
    GENERAL_TRANSFORM_EPS,
    GLYPH_CACHE_SIZE,
    MAX_CANVAS_PIXELS,
    NONE_THRESHOLD,
    STRONG_EMBOLDEN_26_6,
)
from psd_tools.composite.text.fonts import Face, TextRenderError
from psd_tools.composite.text.layout import Layout, PlacedGlyph, Rule
from psd_tools.composite.text.model import RGB, Transform

#: ``(x0, y0, x1, y1)`` in document pixels.
Bbox = tuple[int, int, int, int]
T = TypeVar("T")

#: Unhinted outline, bitmap strikes ignored: the outline is box-filtered as is.
_UNHINTED = (
    freetype.FT_LOAD_NO_HINTING
    | freetype.FT_LOAD_NO_BITMAP
    | freetype.FT_LOAD_NO_AUTOHINT
)
#: AntA Sharp: FreeType's light hinter.
_LIGHT = freetype.FT_LOAD_TARGET_LIGHT | freetype.FT_LOAD_NO_BITMAP
_IDENTITY = (0x10000, 0, 0, 0x10000)


@dataclass(frozen=True)
class Options:
    """Rendering options; the defaults are the measured recipe."""

    #: Applies to the default :class:`FontMap`; an explicit map keeps its own policy.
    missing_font: Literal["raise", "skip", "substitute"] = "raise"
    #: Non-diagonal transforms: FreeType matrix, or axis-aligned then resampled.
    general_transform: Literal["matrix", "resample"] = "matrix"
    #: Draw the outlines of a colour (``SVG ``) font as a silhouette; when
    #: false its glyphs are skipped.
    monochrome_emoji: bool = True
    max_canvas_pixels: int = MAX_CANVAS_PIXELS


#: Where a glyph origin lands: as computed, on whole pixels, or on whole
#: pixels only under a scale or flip (Crisp snaps to the grid Photoshop's
#: stem hinter works on, which a rotation or shear does not have).
_Origin = Literal["fractional", "integer", "integer_diagonal"]


@dataclass(frozen=True)
class _Mode:
    """How one AntA value loads, dilates and places a glyph."""

    flags: int
    embolden: int  #: 26.6 total widening in x and in y
    origin: _Origin
    threshold: float | None  #: binarise the final coverage above this

    def snaps(self, general: bool) -> bool:
        return self.origin == "integer" or (
            self.origin == "integer_diagonal" and not general
        )


_STRONG = _Mode(_UNHINTED, STRONG_EMBOLDEN_26_6, "fractional", None)
_MODES: dict[int, _Mode] = {
    0: _Mode(_UNHINTED, 0, "fractional", NONE_THRESHOLD),
    1: _Mode(_UNHINTED, 0, "integer_diagonal", None),
    2: _STRONG,
    3: _STRONG,
    4: _Mode(_LIGHT, 0, "integer", None),
}


class GlyphCache:
    """LRU of rendered glyph bitmaps, keyed by everything that shapes one."""

    def __init__(self, maxsize: int = GLYPH_CACHE_SIZE):
        self.maxsize = maxsize
        self._items: OrderedDict[Hashable, object] = OrderedDict()

    def __len__(self) -> int:
        return len(self._items)

    def get(self, key: Hashable, make: Callable[[], T]) -> T:
        """The cached value for ``key``, rendering it with ``make`` once."""
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]  # type: ignore[return-value]
        value = make()
        self._items[key] = value
        while len(self._items) > self.maxsize:
            self._items.popitem(last=False)
        return value


def is_general(transform: Transform) -> bool:
    """A rotation or shear rather than a scale or flip."""
    return (
        abs(transform[1]) > GENERAL_TRANSFORM_EPS
        or abs(transform[2]) > GENERAL_TRANSFORM_EPS
    )


def strike_scale(px_h: float, strikes: frozenset[int]) -> int:
    """Char-size multiplier that keeps ``px_h`` off an embedded bitmap strike.

    FreeType selects a strike whose ppem the requested height rounds to even
    under ``FT_LOAD_NO_BITMAP``; loading at a power of two of the size and
    scaling the matrix back is the same outline at a size no strike matches.
    """
    k = 1
    while round(px_h * k) in strikes:
        k *= 2
    return k


def raster_warnings(layout: Layout, transform: Transform) -> tuple[str, ...]:
    """Warnings :func:`rasterize` would raise for ``layout`` under ``transform``."""
    out: list[str] = []
    if is_general(transform):
        out.append("generic_transform")
    cy = math.hypot(transform[2], transform[3])
    if any(
        strike_scale(g.em_h * cy, g.face.metrics.strikes) > 1 for g in layout.glyphs
    ):
        out.append("strike_guard")
    if any(g.face.metrics.has_svg for g in layout.glyphs):
        out.append("monochrome_emoji")
    return tuple(out)


def _bitmap(slot: freetype.GlyphSlot) -> np.ndarray:
    """The slot's rendered coverage as ``uint8`` (rows, width)."""
    bm = slot.bitmap
    rows, width, pitch = bm.rows, bm.width, bm.pitch
    if rows == 0 or width == 0:
        return np.zeros((0, 0), np.uint8)
    n = rows * abs(pitch)
    raw = ctypes.cast(bm._FT_Bitmap.buffer, ctypes.POINTER(ctypes.c_ubyte * n)).contents
    a = np.frombuffer(raw, dtype=np.uint8).reshape(rows, abs(pitch))
    if pitch < 0:
        a = a[::-1]
    if bm.pixel_mode == freetype.FT_PIXEL_MODE_MONO:
        return np.unpackbits(a, axis=1)[:, :width] * np.uint8(255)
    return a[:, :width].copy()


def _render_glyph(
    face: Face,
    gid: int,
    size66: tuple[int, int],
    matrix: tuple[int, int, int, int],
    delta: tuple[int, int],
    flags: int,
    embolden: tuple[int, int],
) -> tuple[int, int, np.ndarray]:
    """``(bitmap_left, bitmap_top, coverage uint8)`` of one glyph."""
    ft = face.ft
    try:
        ft.set_char_size(size66[0], size66[1], 72, 72)
        ft.set_transform(freetype.FT_Matrix(*matrix), freetype.FT_Vector(*delta))
        ft.load_glyph(gid, flags)
        slot = ft.glyph
        if embolden != (0, 0) and slot.format == freetype.FT_GLYPH_FORMAT_OUTLINE:
            # FreeType grows the outline towards +x and +y by the whole
            # strength; translating back by half of it centres the dilation.
            outline = ctypes.byref(slot.outline._FT_Outline)
            freetype.FT_Outline_EmboldenXY(outline, embolden[0], embolden[1])
            freetype.FT_Outline_Translate(
                outline, -(embolden[0] // 2), -(embolden[1] // 2)
            )
        slot.render(freetype.FT_RENDER_MODE_NORMAL)
    except freetype.FT_Exception as ex:
        raise TextRenderError(f"glyph {gid} of {face.key[0]}: {ex}") from ex
    return slot.bitmap_left, slot.bitmap_top, _bitmap(slot)


def _fixed(v: float) -> int:
    return int(round(v * 0x10000))


def _empty(transform: Transform) -> tuple[np.ndarray, Bbox]:
    x, y = round(transform[4]), round(transform[5])
    return np.zeros((0, 0, 4), dtype=np.float32), (x, y, x, y)


#: A coverage array pasted with its top-left pixel at ``(x, y)`` in one fill.
_Item = tuple[int, int, np.ndarray, RGB]


def _glyph_items(
    glyphs: Sequence[PlacedGlyph],
    transform: Transform,
    mode: _Mode,
    options: Options,
    cache: GlyphCache,
) -> list[_Item]:
    xx, xy, yx, yy, tx, ty = transform
    cx, cy = math.hypot(xx, xy), math.hypot(yx, yy)
    # Text space is y-down and FreeType y-up, so the two off-diagonals flip sign.
    base = (xx / cx, -yx / cy, -xy / cx, yy / cy)
    snap = mode.snaps(is_general(transform))
    items: list[_Item] = []
    for g in glyphs:
        if g.face.metrics.has_svg and not options.monochrome_emoji:
            continue
        px_w, px_h = g.em_w * cx, g.em_h * cy
        if px_w * px_h > options.max_canvas_pixels:
            raise TextRenderError(
                f"glyph of {px_w:.0f}x{px_h:.0f} px exceeds the canvas limit"
            )
        k = strike_scale(px_h, g.face.metrics.strikes)
        size66 = (round(64 * px_w * k), round(64 * px_h * k))
        if not size66[0] or not size66[1]:
            continue
        m = base
        if g.faux_italic:
            s = FAUX_ITALIC_SHEAR
            m = (m[0], m[0] * s + m[1], m[2], m[2] * s + m[3])
        matrix = (
            _fixed(m[0] / k),
            _fixed(m[1] / k),
            _fixed(m[2] / k),
            _fixed(m[3] / k),
        )
        x = xx * g.x + yx * g.y + tx
        y = xy * g.x + yy * g.y + ty
        if snap:
            x, y = float(round(x)), float(round(y))
        ix, iy = math.floor(x), math.floor(y)
        dx, dy = round((x - ix) * 64), round((y - iy) * 64)
        if dx == 64:
            ix, dx = ix + 1, 0
        if dy == 64:
            iy, dy = iy + 1, 0
        ex = ey = mode.embolden
        if g.faux_bold:
            ex += round(64 * px_w * FAUX_BOLD_EM)
            ey += round(64 * px_h * FAUX_BOLD_EM)
        key = (g.face.key, g.gid, size66, matrix, dx, -dy, mode.flags, ex, ey)
        left, top, cov = cache.get(
            key,
            lambda: _render_glyph(
                g.face, g.gid, size66, matrix, (dx, -dy), mode.flags, (ex, ey)
            ),
        )
        if cov.size:
            items.append((ix + left, iy - top, cov, g.fill))
    return items


def _rule_items(rules: Sequence[Rule], transform: Transform) -> list[_Item]:
    xx, xy, yx, yy, tx, ty = transform
    items: list[_Item] = []
    for r in rules:
        h = r.thickness / 2
        quad = np.array(
            [(r.x0, r.y - h), (r.x1, r.y - h), (r.x1, r.y + h), (r.x0, r.y + h)]
        )
        pts = np.column_stack(
            [
                xx * quad[:, 0] + yx * quad[:, 1] + tx,
                xy * quad[:, 0] + yy * quad[:, 1] + ty,
            ]
        )
        x0, y0 = int(math.floor(pts[:, 0].min())), int(math.floor(pts[:, 1].min()))
        x1, y1 = int(math.ceil(pts[:, 0].max())), int(math.ceil(pts[:, 1].max()))
        cov = fill_coverage([pts - (x0, y0)], x1 - x0, y1 - y0)
        if cov.size and cov.any():
            items.append((x0, y0, (cov * 255 + 0.5).astype(np.uint8), r.fill))
    return items


def _compose(
    items: list[_Item], transform: Transform, mode: _Mode, limit: int
) -> tuple[np.ndarray, Bbox]:
    if not items:
        return _empty(transform)
    x0 = min(x for x, _, _, _ in items)
    y0 = min(y for _, y, _, _ in items)
    x1 = max(x + c.shape[1] for x, _, c, _ in items)
    y1 = max(y + c.shape[0] for _, y, c, _ in items)
    w, h = x1 - x0, y1 - y0
    if w * h > limit:
        raise TextRenderError(f"canvas {w}x{h} exceeds {limit} pixels")
    alpha = np.zeros((h, w), np.float32)
    rgb = np.zeros((h, w, 3), np.float32)
    fills = {f for _, _, _, f in items}
    best = np.zeros((h, w), np.float32) if len(fills) > 1 else None
    for x, y, cov8, fill in items:
        cov = cov8.astype(np.float32) / 255
        ys, xs = (
            slice(y - y0, y - y0 + cov.shape[0]),
            slice(x - x0, x - x0 + cov.shape[1]),
        )
        alpha[ys, xs] += cov
        if best is None:
            continue
        # Where fills overlap the larger coverage keeps the pixel.
        win = cov > best[ys, xs]
        rgb[ys, xs][win] = fill
        np.maximum(best[ys, xs], cov, out=best[ys, xs])
    if best is None:
        rgb[alpha > 0] = fills.pop()
    if mode.threshold is not None:
        alpha = (alpha > mode.threshold).astype(np.float32)
    else:
        np.clip(alpha, 0, 1, out=alpha)
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
    return rgba, (x0, y0, x1, y1)


def _rasterize_matrix(
    layout: Layout,
    transform: Transform,
    mode: _Mode,
    options: Options,
    cache: GlyphCache,
) -> tuple[np.ndarray, Bbox]:
    items = _glyph_items(layout.glyphs, transform, mode, options, cache)
    items += _rule_items(layout.rules, transform)
    return _compose(items, transform, mode, options.max_canvas_pixels)


def _resample(
    layout: Layout,
    transform: Transform,
    mode: _Mode,
    options: Options,
    cache: GlyphCache,
) -> tuple[np.ndarray, Bbox]:
    """Render axis-aligned at the column lengths, then warp to ``transform``."""
    from scipy import ndimage  # type: ignore[import-untyped]  # noqa: PLC0415

    xx, xy, yx, yy, tx, ty = transform
    cx, cy = math.hypot(xx, xy), math.hypot(yx, yy)
    axis: Transform = (cx, 0.0, 0.0, cy, 0.0, 0.0)
    rgba, (ax0, ay0, ax1, ay1) = _rasterize_matrix(layout, axis, mode, options, cache)
    if not rgba.size:
        return _empty(transform)
    # Axis-aligned (u, v) -> document (X, Y).
    m = np.array([[xx / cx, yx / cy], [xy / cx, yy / cy]])
    corners = np.array([[ax0, ay0], [ax1, ay0], [ax0, ay1], [ax1, ay1]], float) @ m.T
    corners += (tx, ty)
    bx0, by0 = (
        int(math.floor(corners[:, 0].min())),
        int(math.floor(corners[:, 1].min())),
    )
    bx1, by1 = int(math.ceil(corners[:, 0].max())), int(math.ceil(corners[:, 1].max()))
    w, h = bx1 - bx0, by1 - by0
    if w * h > options.max_canvas_pixels:
        raise TextRenderError(
            f"canvas {w}x{h} exceeds {options.max_canvas_pixels} pixels"
        )
    # Output index (row, col) -> input index, sampling at pixel centres.
    inv = np.linalg.inv(m)[::-1, ::-1]
    offset = inv @ (by0 + 0.5 - ty, bx0 + 0.5 - tx) - (ay0 + 0.5, ax0 + 0.5)
    pre = rgba.copy()
    pre[..., :3] *= pre[..., 3:]
    out = np.zeros((h, w, 4), np.float32)
    for c in range(4):
        out[..., c] = ndimage.affine_transform(
            pre[..., c],
            inv,
            offset,
            output_shape=(h, w),
            order=1,
            mode="constant",
            cval=0.0,
        )
    alpha = out[..., 3]
    if mode.threshold is not None:
        alpha = (alpha > mode.threshold).astype(np.float32)
    ink = alpha > 0
    rgb = np.zeros((h, w, 3), np.float32)
    rgb[ink] = np.clip(
        out[..., :3][ink] / np.maximum(out[..., 3][ink], 1e-6)[:, None], 0, 1
    )
    return np.concatenate([rgb, alpha[..., None]], axis=-1), (bx0, by0, bx1, by1)


def rasterize(
    layout: Layout,
    transform: Transform,
    antialias: int,
    options: Options,
    cache: GlyphCache | None = None,
) -> tuple[np.ndarray, Bbox]:
    """Render ``layout`` to ``float32`` HxWx4 straight RGBA and its bbox.

    ``antialias`` is the layer's ``AntA`` value. The bbox is the integer
    union of the pasted glyph bitmaps and rule quads; an empty render is
    ``(0, 0, 4)`` at the rounded translation.
    """
    mode = _MODES.get(antialias)
    if mode is None:
        raise ValueError(f"unknown AntA value {antialias!r}")
    if (
        math.hypot(transform[0], transform[1]) == 0
        or math.hypot(transform[2], transform[3]) == 0
    ):
        return _empty(transform)
    cache = cache if cache is not None else GlyphCache()
    if options.general_transform == "resample" and is_general(transform):
        return _resample(layout, transform, mode, options, cache)
    return _rasterize_matrix(layout, transform, mode, options, cache)
