"""OpenType ``SVG `` colour glyphs: the SVG subset and its rasterization.

The fixture colour font is the rectangle TrueType fixture with an ``SVG ``
table: ``A`` is a badge (a rounded rectangle with a vertical gradient, a
stroke and a white label), ``V`` an embedded PNG; ``I`` has no document and
stays an outline glyph.
"""

from __future__ import annotations

import base64
import io
import os

import numpy as np
import pytest

pytest.importorskip("freetype")
pytest.importorskip("uharfbuzz")

from psd_tools.composite.scanline import fill_coverage
from psd_tools.composite.text import svg
from psd_tools.composite.text.fonts import Face, load_face
from psd_tools.composite.text.layout import Layout, PlacedGlyph
from psd_tools.composite.text.model import RGB, Transform
from psd_tools.composite.text.raster import (
    Options,
    raster_warnings,
    rasterize,
)

from .conftest import build_font

STRONG, SHARP = 2, 4
IDENTITY: Transform = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
RED: RGB = (1.0, 0.0, 0.0)
TOP, BOTTOM, STROKE = "#4f86f7", "#2563eb", "#1e3a8a"
BADGE = (
    '<svg version="1.1" xmlns="http://www.w3.org/2000/svg"><g id="glyph{gid}">'
    '<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">'
    f'<stop offset="0" stop-color="{TOP}"/><stop offset="1" stop-color="{BOTTOM}"/>'
    "</linearGradient></defs>"
    '<rect x="0" y="-800" width="1600" height="800" rx="400" ry="400" '
    f'fill="url(#g)" stroke="{STROKE}" stroke-width="40"/>'
    '<path d="M600 -600 H1000 V-200 H600 Z" fill="#ffffff"/></g></svg>'
)


def _png(rgba: np.ndarray) -> str:
    from PIL import Image  # noqa: PLC0415

    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


#: 4 x 4 opaque blue inside a transparent black ring: 6 x 6.
_ICON = np.zeros((6, 6, 4), np.uint8)
_ICON[1:5, 1:5] = (0, 0, 255, 255)
IMAGE = (
    '<svg version="1.1" xmlns="http://www.w3.org/2000/svg" '
    'xmlns:xlink="http://www.w3.org/1999/xlink"><g id="glyph{gid}">'
    '<image x="0" y="-600" width="600" height="600" '
    f'xlink:href="data:image/png;base64,{_png(_ICON)}"/></g></svg>'
)


def _color_font(path: str, docs: dict[str, str], compress: bool = False) -> str:
    from fontTools.ttLib import TTFont, newTable  # noqa: PLC0415
    from fontTools.ttLib.tables.S_V_G_ import SVGDocument  # noqa: PLC0415

    build_font(path, cff=False)
    font = TTFont(path)
    order = font.getGlyphOrder()
    table = newTable("SVG ")
    table.docList = []
    for name, doc in sorted(docs.items(), key=lambda kv: order.index(kv[0])):
        gid = order.index(name)
        table.docList.append(SVGDocument(doc.format(gid=gid), gid, gid, compress))
    font["SVG "] = table
    font.save(path)
    return path


@pytest.fixture(scope="module")
def color_face(tmp_path_factory: pytest.TempPathFactory) -> Face:
    d = tmp_path_factory.mktemp("color-font")
    path = _color_font(str(d / "Color.ttf"), {"A": BADGE, "V": IMAGE})
    return load_face(path)


def _glyph(face: Face, char: str, x: float, y: float, em: float) -> PlacedGlyph:
    gid = face.ft.get_char_index(ord(char))
    return PlacedGlyph(face, gid, x, y, em, em, RED, False, False, 0)


def _render(
    glyphs: list[PlacedGlyph], antialias: int = STRONG, options: Options | None = None
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    layout = Layout(tuple(glyphs), (), (), {}, ())
    return rasterize(layout, IDENTITY, antialias, options or Options())


def _hex(color: str) -> np.ndarray:
    return np.array([int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)])


# -- the font table --------------------------------------------------------


def test_face_reads_the_svg_table(color_face: Face) -> None:
    table = color_face.svg
    assert table is not None
    gid = color_face.ft.get_char_index(ord("A"))
    assert gid in table
    assert color_face.ft.get_char_index(ord("I")) not in table
    doc = table.document(gid)
    assert doc is not None and doc.startswith(b"<svg")


def test_a_face_without_the_table_has_none(ttf_font_path: str) -> None:
    assert load_face(ttf_font_path).svg is None


def test_compressed_documents(tmp_path: pytest.TempPathFactory) -> None:
    path = _color_font(os.path.join(str(tmp_path), "Gz.ttf"), {"A": BADGE}, True)
    face = load_face(path)
    gid = face.ft.get_char_index(ord("A"))
    assert face.svg is not None
    assert b"\x1f\x8b" in face.svg._data  # stored gzipped
    doc = face.svg.document(gid)
    assert doc is not None and doc.startswith(b"<svg")
    assert face.svg.drawing(gid) is not None


# -- parsing ---------------------------------------------------------------


def test_colors() -> None:
    ctx = (0.1, 0.2, 0.3, 1.0)
    assert svg.parse_color("#fff", ctx) == (1.0, 1.0, 1.0, 1.0)
    assert svg.parse_color("#ff000080", ctx) == pytest.approx((1, 0, 0, 128 / 255))
    assert svg.parse_color("rgb(255, 0, 51)", ctx) == pytest.approx((1, 0, 0.2, 1))
    assert svg.parse_color("navy", ctx) == pytest.approx((0, 0, 128 / 255, 1))
    assert svg.parse_color("none", ctx) is None
    assert svg.parse_color("currentColor", ctx) == ctx
    with pytest.raises(svg.Unsupported):
        svg.parse_color("hsl(0, 100%, 50%)", ctx)


def test_transforms() -> None:
    m = svg.parse_transform("translate(10 20) scale(2)")
    assert m == pytest.approx((2, 0, 0, 2, 10, 20))
    r = svg.parse_transform("rotate(90 5 5)")
    x, y = r[0] * 10 + r[2] * 5 + r[4], r[1] * 10 + r[3] * 5 + r[5]
    assert (x, y) == pytest.approx((5, 10))


@pytest.mark.parametrize(
    "d",
    [
        "M0 0 H10 V10 H0 Z",
        "m0 0 h10 v10 h-10 z",
        "M0 0 L10 0 L10 10 L0 10 Z",
        "M0 0 Q5 0 10 0 L10 10 L0 10 Z",
        "M0 0 C3 0 7 0 10 0 L10 10 C7 10 3 10 0 10 Z",
    ],
)
def test_path_data_square(d: str) -> None:
    contours = svg.parse_path(d)
    assert len(contours) == 1
    pts = contours[0].reshape(-1, 2)
    assert pts.min(axis=0) == pytest.approx((0, 0))
    assert pts.max(axis=0) == pytest.approx((10, 10))
    cov = fill_coverage([c.reshape(-1, 2)[::4] for c in contours], 12, 12)
    assert cov.sum() == pytest.approx(100)


def test_smooth_curves_reflect_the_last_control_point() -> None:
    (quad,) = svg.parse_path("M0 0 Q5 5 10 0 T20 0")
    # T's control is (15, -5); as a cubic its first handle is 2/3 of the way.
    assert quad[1, 1] == pytest.approx((10 + 2 / 3 * 5, -2 / 3 * 5))
    (cubic,) = svg.parse_path("M0 0 C0 5 5 5 10 0 S20 -5 20 0")
    assert cubic[1, 1] == pytest.approx((15, -5))


def test_path_arc_is_a_half_disc() -> None:
    contours = svg.parse_path("M0 10 A10 10 0 0 1 20 10 Z")
    shape = svg.Shape(
        tuple(contours), svg.IDENTITY, (1, 1, 1, 1), 1.0, True, (0, 0, 20, 10)
    )
    cov = fill_coverage(svg._device_contours(shape, svg.IDENTITY), 20, 20)
    assert cov.sum() == pytest.approx(np.pi * 100 / 2, rel=1e-3)


@pytest.mark.parametrize(
    "body",
    [
        '<use href="#a"/>',
        "<text>x</text>",
        '<rect width="10" height="10" clip-path="url(#c)"/>',
        '<path d="M0 0 H10" stroke="#000"/>',
        '<image width="5" height="5" href="file.png"/>',
        '<g opacity="0.5"><rect width="2" height="2"/><rect width="1" height="1"/></g>',
        '<svg><rect width="1" height="1"/></svg>',
    ],
)
def test_unsupported_documents_draw_nothing(body: str) -> None:
    doc = f'<svg xmlns="http://www.w3.org/2000/svg"><g id="glyph1">{body}</g></svg>'
    assert _table({1: doc}).drawing(1) is None


def test_opacity_of_a_single_shape_group() -> None:
    drawing = _drawing('<g opacity="0.5"><rect width="2" height="2" fill="#f00"/></g>')
    _, _, rgba = svg.rasterize(drawing, svg.IDENTITY)
    assert rgba[..., 3] == pytest.approx(np.full((2, 2), 0.5))


def test_oversized_images_are_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(svg, "_MAX_IMAGE_PIXELS", 35)
    assert _table({1: IMAGE.format(gid=1)}).drawing(1) is None
    monkeypatch.setattr(svg, "_MAX_IMAGE_PIXELS", 36)
    assert _table({1: IMAGE.format(gid=1)}).drawing(1) is not None


def test_images_share_one_pixel_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    two = IMAGE.replace(
        "</g>", IMAGE.split('<g id="glyph{gid}">')[1].split("</g>")[0] + "</g>"
    )
    monkeypatch.setattr(svg, "_MAX_IMAGE_PIXELS", 71)
    assert _table({1: two.format(gid=1)}).drawing(1) is None
    monkeypatch.setattr(svg, "_MAX_IMAGE_PIXELS", 72)
    drawing = _table({1: two.format(gid=1)}).drawing(1)
    assert drawing is not None and len(drawing.items) == 2


def _image_doc(data: bytes) -> str:
    return (
        '<svg xmlns="http://www.w3.org/2000/svg"><g id="glyph1">'
        '<image width="6" height="6" href="data:image/png;base64,'
        f'{base64.b64encode(data).decode()}"/></g></svg>'
    )


def test_undecodable_images_are_not_drawn(monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image  # noqa: PLC0415

    png = base64.b64decode(_png(_ICON))
    assert _table({1: _image_doc(png)}).drawing(1) is not None
    assert _table({1: _image_doc(b"not an image")}).drawing(1) is None
    assert _table({1: _image_doc(png[:-30])}).drawing(1) is None
    # A decompression bomb is refused by Pillow before the size check.
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    assert _table({1: _image_doc(png)}).drawing(1) is None


def test_a_corrupt_compressed_document_is_not_drawn() -> None:
    table = _table({1: "x" * 40})
    start = table._index[0][2]
    table._data = table._data[:start] + b"\x1f\x8b\x08\x00" + b"\xff" * 36
    assert table.document(1) is None
    assert table.drawing(1) is None


def test_deep_nesting_is_not_drawn() -> None:
    body = "<g>" * 3000 + '<rect width="1" height="1"/>' + "</g>" * 3000
    doc = f'<svg xmlns="http://www.w3.org/2000/svg">{body}</svg>'
    assert _table({1: doc}).drawing(1) is None
    doc = '<svg xmlns="http://www.w3.org/2000/svg"><g id="glyph1"/></svg>'
    deep = doc.replace(
        '<g id="glyph1"/>', "<g>" * 3000 + '<g id="glyph1"/>' + "</g>" * 3000
    )
    assert _table({1: deep}).drawing(1) is None


@pytest.mark.parametrize(
    "body",
    [
        '<rect width="1e999" height="1"/>',
        '<rect x="1e9" width="1" height="1"/>',
        '<path d="M0 0 H1e400 V1 Z"/>',
        '<g transform="scale(1e999)"><rect width="1" height="1"/></g>',
    ],
)
def test_non_finite_or_far_geometry_is_not_drawn(body: str) -> None:
    doc = f'<svg xmlns="http://www.w3.org/2000/svg"><g id="glyph1">{body}</g></svg>'
    assert _table({1: doc}).drawing(1) is None


def test_style_sheets_are_not_drawn() -> None:
    sheet = "<style>rect { fill: #f00 }</style>"
    rect = '<rect class="a" width="1" height="1"/>'
    doc = '<svg xmlns="http://www.w3.org/2000/svg"><g id="glyph1">{}</g></svg>'
    assert _table({1: doc.format(sheet + rect)}).drawing(1) is None
    assert _table({1: doc.format("<style/>" + rect)}).drawing(1) is not None


def test_opacity_on_a_filled_stroke_is_not_drawn() -> None:
    stroked = '<rect width="4" height="4" fill="#f00" stroke="#00f" stroke-width="1"'
    doc = '<svg xmlns="http://www.w3.org/2000/svg"><g id="glyph1">{}</g></svg>'
    assert _table({1: doc.format(stroked + ' opacity="0.5"/>')}).drawing(1) is None
    group = '<g opacity="0.5">' + stroked + "/></g>"
    assert _table({1: doc.format(group)}).drawing(1) is None
    drawing = _table({1: doc.format(stroked + "/>")}).drawing(1)
    assert drawing is not None and len(drawing.items) == 2
    lone = '<rect width="4" height="4" fill="none" stroke="#00f" opacity="0.5"/>'
    assert _table({1: doc.format(lone)}).drawing(1) is not None


def _glyph_doc(outer: str, inner: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg"{outer}><g{inner}>'
        f'<g id="glyph1">{body}</g></g></svg>'
    )


def test_ancestors_pass_on_transform_and_opacity() -> None:
    rect = _square("#ff0000", 0, 0, 2)
    doc = _glyph_doc(' opacity="0.5"', ' transform="translate(3 1)"', rect)
    drawing = _table({1: doc}).drawing(1)
    assert drawing is not None
    left, top, rgba = svg.rasterize(drawing, svg.IDENTITY)
    assert (left, top, rgba.shape[:2]) == (3, 1, (2, 2))
    assert rgba[..., 3] == pytest.approx(np.full((2, 2), 0.5))


@pytest.mark.parametrize(
    "outer, inner",
    [(' opacity="0.5"', ""), ("", ' opacity="0.5"'), ("", ' clip-path="url(#c)"')],
)
def test_ancestors_of_a_group_glyph(outer: str, inner: str) -> None:
    two = _square("#ff0000", 0, 0, 2) + _square("#00ff00", 1, 1, 2)
    assert _table({1: _glyph_doc(outer, inner, two)}).drawing(1) is None


def test_a_hidden_ancestor_draws_nothing() -> None:
    doc = _glyph_doc("", ' display="none"', _square("#ff0000", 0, 0, 2))
    assert _table({1: doc}).drawing(1) == svg.Drawing(())


def test_arc_flags_may_run_together() -> None:
    spaced = svg.parse_path("M0 10 a10 10 0 0 1 20 0 Z")
    compact = svg.parse_path("M0 10 a10 10 0 0120 0 Z")
    assert len(spaced) == len(compact) == 1
    assert np.array_equal(spaced[0], compact[0])
    with pytest.raises(svg.Unsupported):
        svg.parse_path("M0 10 a10 10 0 2 1 20 0 Z")


def test_object_bounding_box_is_the_tight_bounds() -> None:
    drawing = _drawing('<path d="M0 0 C0 -10 10 -10 10 0 Z"/>')
    (shape,) = drawing.items
    assert isinstance(shape, svg.Shape)
    # The curve peaks at t = 1/2, three quarters of the way to its handles.
    assert shape.bbox == pytest.approx((0, -7.5, 10, 0))


def test_a_singular_gradient_paints_nothing() -> None:
    drawing = _drawing(
        '<defs><linearGradient id="g" gradientTransform="scale(0)">'
        '<stop offset="0" stop-color="#f00"/><stop offset="1" stop-color="#00f"/>'
        '</linearGradient></defs><rect width="2" height="2" fill="url(#g)"/>'
    )
    _, _, rgba = svg.rasterize(drawing, svg.IDENTITY)
    assert rgba.shape[:2] == (2, 2) and not rgba.any()


@pytest.mark.parametrize("grid", [0, 4])
def test_peak_bytes_bounds_the_allocations(grid: int) -> None:
    import tracemalloc  # noqa: PLC0415

    drawing = _table({1: BADGE.format(gid=1)}).drawing(1)
    assert drawing is not None
    m: svg.Affine = (0.25, 0.0, 0.0, 0.25, 0.3, 200.6)
    box = svg.device_bbox(drawing, m)
    assert box is not None
    w, h = box[2] - box[0], box[3] - box[1]
    tracemalloc.start()
    try:
        svg.rasterize(drawing, m, grid)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak <= svg.peak_bytes(drawing, w, h, grid)


def test_peak_bytes_counts_images_and_the_alpha_grid() -> None:
    drawing = _table({1: IMAGE.format(gid=1)}).drawing(1)
    assert drawing is not None
    plain = svg.Drawing(())
    assert svg.peak_bytes(drawing, 10, 10) > svg.peak_bytes(plain, 10, 10)
    assert svg.peak_bytes(plain, 10, 10, 4) > svg.peak_bytes(plain, 10, 10)


def test_entities_are_refused() -> None:
    doc = (
        '<?xml version="1.0"?><!DOCTYPE svg [<!ENTITY a "aaaa">]>'
        '<svg xmlns="http://www.w3.org/2000/svg"><rect width="1" height="1"/></svg>'
    )
    assert _table({1: doc}).drawing(1) is None


def _table(docs: dict[int, str]) -> svg.SvgTable:
    """A raw ``SVG `` table: one document per glyph id."""
    body = b""
    entries = b""
    for gid, doc in sorted(docs.items()):
        data = doc.encode()
        start = 2 + 12 * len(docs) + len(body)
        entries += gid.to_bytes(2, "big") * 2
        entries += start.to_bytes(4, "big") + len(data).to_bytes(4, "big")
        body += data
    index = len(docs).to_bytes(2, "big") + entries + body
    return svg.SvgTable(b"\0\0" + (10).to_bytes(4, "big") + b"\0" * 4 + index)


def test_table_index() -> None:
    table = _table({3: "<svg/>", 7: "<svg></svg>"})
    assert 3 in table and 7 in table and 5 not in table
    assert table.document(7) == b"<svg></svg>"


# -- rasterization ---------------------------------------------------------


def _square(color: str, x0: float, y0: float, size: float) -> str:
    return f'<rect x="{x0}" y="{y0}" width="{size}" height="{size}" fill="{color}"/>'


def _drawing(*bodies: str) -> svg.Drawing:
    doc = f'<svg xmlns="http://www.w3.org/2000/svg">{"".join(bodies)}</svg>'
    drawing = _table({1: doc}).drawing(1)
    assert drawing is not None
    return drawing


def test_exact_coverage_at_a_fractional_origin() -> None:
    m: svg.Affine = (1.0, 0.0, 0.0, 1.0, 0.25, 0.0)
    left, top, rgba = svg.rasterize(_drawing(_square("#ff0000", 0, 0, 4)), m)
    assert (left, top) == (0, 0)
    assert rgba[1, :, 3] == pytest.approx([0.75, 1, 1, 1, 0.25])
    straight = rgba[1, 0, :3] / rgba[1, 0, 3]
    assert straight == pytest.approx([1, 0, 0], abs=1e-6)


def test_overlaps_are_composited_in_linear_light() -> None:
    """Photoshop mixes a colour glyph's own shapes in linear light."""
    blue = "#2563eb"
    m: svg.Affine = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    # White covers the left half of the middle column of a blue square.
    _, _, rgba = svg.rasterize(
        _drawing(
            _square(blue, 0, 0, 3),
            '<rect x="0" y="0" width="1.5" height="3" fill="#fff"/>',
        ),
        m,
    )
    expect = svg.to_srgb(0.5 * svg.to_linear(_hex(blue)) + 0.5)
    assert rgba[1, 1, :3] == pytest.approx(expect, abs=1e-5)
    assert rgba[1, 1, :3] != pytest.approx(0.5 * _hex(blue) + 0.5, abs=0.01)


def test_alpha_grid_samples_points() -> None:
    """An edge 0.3 px into a pixel leaves 3 of 4 sample columns inside."""
    m: svg.Affine = (1.0, 0.0, 0.0, 1.0, 0.3, 0.0)
    drawing = _drawing(_square("#00ff00", 0, 0, 4))
    _, _, exact = svg.rasterize(drawing, m)
    _, _, sampled = svg.rasterize(drawing, m, alpha_grid=4)
    assert exact[1, 0, 3] == pytest.approx(0.7)
    assert sampled[1, 0, 3] == pytest.approx(0.75)
    assert sampled[1, 0, :3] / sampled[1, 0, 3] == pytest.approx([0, 1, 0], abs=1e-6)


def test_gradient_in_object_bounding_box_units() -> None:
    drawing = _drawing(
        '<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{TOP}"/><stop offset="1" stop-color="{BOTTOM}"/>'
        '</linearGradient></defs><rect x="0" y="0" width="2" height="10" fill="url(#g)"/>'
    )
    _, _, rgba = svg.rasterize(drawing, svg.IDENTITY)
    # Interpolated in sRGB at each pixel centre.
    for row in (0, 4, 9):
        t = (row + 0.5) / 10
        expect = (1 - t) * _hex(TOP) + t * _hex(BOTTOM)
        assert rgba[row, 0, :3] == pytest.approx(expect, abs=1e-5)


def test_image_snaps_to_whole_pixels() -> None:
    """An image fills ``ceil(size)`` whole pixels from ``floor(left + bias)``."""
    m: svg.Affine = (0.01, 0.0, 0.0, 0.01, 10.3, 20.0)
    drawing = _table({1: IMAGE.format(gid=1)}).drawing(1)
    assert drawing is not None
    left, top, rgba = svg.rasterize(drawing, m)
    # 600 units at 0.01 px: 6 x 6 px from (10.3, 14.0), one source pixel per
    # device pixel, drawn into the whole pixels from (10, 14) and sampled
    # half a source pixel up and left.
    assert (left, top) == (10, 14)
    assert rgba.shape[:2] == (6, 6)
    assert rgba[2, :, 3] == pytest.approx([0.5, 1, 1, 1, 0.5, 0])
    assert rgba[:, 2, 3] == pytest.approx([0.5, 1, 1, 1, 0.5, 0])
    # Colour and alpha are resampled apart: the transparent black ring
    # darkens the edge pixel's colour to half its blue.
    edge = rgba[2, 0]
    assert edge[2] / edge[3] == pytest.approx(0.5)


# -- in the text engine ----------------------------------------------------


def test_badge_renders_in_color(color_face: Face) -> None:
    em = 100.0
    rgba, bbox = _render([_glyph(color_face, "A", 0.0, 100.0, em)])
    # 1600 x 800 units at 0.1 px/unit, stroke 2 px centred on the edge.
    assert bbox == (-2, 18, 162, 102)
    x0, y0 = bbox[0], bbox[1]

    def at(x: float, y: float) -> np.ndarray:
        return rgba[int(y) - y0, int(x) - x0]

    # Gradient fill inside, top towards TOP and bottom towards BOTTOM.
    assert at(40, 25)[:3] == pytest.approx(_hex(TOP), abs=0.05)
    assert at(40, 95)[:3] == pytest.approx(_hex(BOTTOM), abs=0.05)
    # The label is white, the stroke dark, all opaque.
    assert at(80, 60)[:3] == pytest.approx([1, 1, 1], abs=1e-6)
    assert at(80, 20.5)[:3] == pytest.approx(_hex(STROKE), abs=1e-3)
    assert at(80, 60)[3] == 1.0
    # Not the run's red.
    assert not np.allclose(rgba[..., :3][rgba[..., 3] > 0], RED)


def test_no_warning_when_drawn_in_color(color_face: Face) -> None:
    layout = Layout((_glyph(color_face, "A", 0.0, 100.0, 50.0),), (), (), {}, ())
    assert raster_warnings(layout, IDENTITY) == ()
    off = Options(color_glyphs=False)
    assert raster_warnings(layout, IDENTITY, off) == ("monochrome_emoji",)


def test_outline_glyphs_of_a_color_font_stay_outlines(color_face: Face) -> None:
    glyphs = [_glyph(color_face, "I", 0.25, 100.0, 40.5)]
    layout = Layout(tuple(glyphs), (), (), {}, ())
    assert raster_warnings(layout, IDENTITY) == ()
    rgba, _ = _render(glyphs, options=Options(monochrome_emoji=False))
    assert rgba[..., 3].any()
    assert (rgba[..., :3][rgba[..., 3] > 0] == RED).all()


def test_color_glyphs_off_is_the_silhouette(color_face: Face) -> None:
    glyphs = [_glyph(color_face, "A", 0.25, 100.0, 50.0)]
    rgba, _ = _render(glyphs, options=Options(color_glyphs=False))
    assert (rgba[..., :3][rgba[..., 3] > 0] == RED).all()
    skipped, _ = _render(
        glyphs, options=Options(color_glyphs=False, monochrome_emoji=False)
    )
    assert skipped.shape == (0, 0, 4)


def test_alpha_grid_follows_the_antialias_mode(color_face: Face) -> None:
    glyphs = [_glyph(color_face, "A", 0.3, 100.0, 50.0)]
    strong, _ = _render(glyphs, STRONG)
    sharp, _ = _render(glyphs, SHARP)
    partial = strong[..., 3][(strong[..., 3] > 0) & (strong[..., 3] < 1)]
    # Strong: alpha in sixteenths; Sharp keeps the exact coverage.
    assert np.allclose(partial * 16, np.round(partial * 16), atol=1e-4)
    sharp_partial = sharp[..., 3][(sharp[..., 3] > 0) & (sharp[..., 3] < 1)]
    assert not np.allclose(sharp_partial * 16, np.round(sharp_partial * 16), atol=1e-4)


def test_the_glyph_budget_counts_the_alpha_grid(color_face: Face) -> None:
    glyph = _glyph(color_face, "A", 0.0, 100.0, 100.0)
    drawing = color_face.svg.drawing(glyph.gid, (*RED, 1.0)) if color_face.svg else None
    assert drawing is not None
    w, h = 202, 102  # the badge's device box: 2 x 1 em and its stroke
    budget = svg.peak_bytes(drawing, w, h, 0) * 2
    assert budget < svg.peak_bytes(drawing, w, h, 4)
    layout = Layout((glyph,), (), (), {}, ())
    rasterize(layout, IDENTITY, SHARP, Options(), max_alloc_bytes=budget)
    with pytest.raises(ValueError, match="over the configured budget"):
        rasterize(layout, IDENTITY, STRONG, Options(), max_alloc_bytes=budget)


def test_image_glyph_and_text_compose(color_face: Face) -> None:
    glyphs = [
        _glyph(color_face, "I", 0.0, 100.0, 40.5),
        _glyph(color_face, "V", 20.0, 100.0, 100.0),
    ]
    rgba, bbox = _render(glyphs)
    blue = rgba[..., 3] > 0.99
    blue &= rgba[..., 2] > 0.99
    assert blue.any()
    red = (rgba[..., 0] > 0.99) & (rgba[..., 3] > 0.99)
    assert red.any()


def test_viewbox_maps_onto_the_em_square() -> None:
    """A root viewBox is scaled onto ``upem`` units from the glyph origin."""
    doc = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="-5 -5 10 10">'
        '<rect x="-5" y="-5" width="10" height="5" fill="#0f0"/></svg>'
    )
    drawing = _table({1: doc}).drawing(1)
    assert drawing is not None
    m: svg.Affine = (0.01, 0.0, 0.0, 0.01, 0.0, 0.0)
    left, top, rgba = svg.rasterize(drawing, m)
    # 1000 units at 0.01 px: the top half of a 10 px square at the origin.
    assert (left, top, rgba.shape[:2]) == (0, 0, (5, 10))
    assert rgba[..., 3] == pytest.approx(np.ones((5, 10)))


def test_viewbox_keeps_its_aspect_ratio() -> None:
    doc = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 10">'
        '<rect width="20" height="10" fill="#0f0"/></svg>'
    )
    drawing = _table({1: doc}).drawing(1)
    assert drawing is not None
    left, top, rgba = svg.rasterize(drawing, (0.01, 0.0, 0.0, 0.01, 0.0, 0.0))
    # Fitted to the width and centred: 10 x 5 px from y 2.5.
    assert (left, top, rgba.shape[:2]) == (0, 2, (6, 10))
    assert rgba[:, 0, 3] == pytest.approx([0.5, 1, 1, 1, 1, 0.5])


def test_rotated_color_glyph(color_face: Face) -> None:
    rotate: Transform = (0.0, 1.0, -1.0, 0.0, 100.0, 0.0)
    layout = Layout((_glyph(color_face, "A", 0.0, 0.0, 50.0),), (), (), {}, ())
    rgba, bbox = rasterize(layout, rotate, STRONG, Options())
    # 80 x 40 px badge turned a quarter: 40 wide, 80 tall.
    assert bbox[2] - bbox[0] == pytest.approx(42, abs=1)
    assert bbox[3] - bbox[1] == pytest.approx(82, abs=1)
    assert rgba[..., 3].max() == 1.0
    assert not np.allclose(rgba[..., :3][rgba[..., 3] > 0], RED)
