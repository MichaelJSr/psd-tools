"""Rasterization on the rectangle fixture fonts: coverages are hand-computed."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np
import pytest

freetype = pytest.importorskip("freetype")
pytest.importorskip("uharfbuzz")

from psd_tools.composite.scanline import fill_coverage  # noqa: E402
from psd_tools.composite.text.constants import (  # noqa: E402
    FAUX_BOLD_EM,
    FAUX_ITALIC_SHEAR,
    GENERAL_TRANSFORM_EPS,
    STRONG_EMBOLDEN_26_6,
)
from psd_tools.composite.text.fonts import Face, TextRenderError, load_face  # noqa: E402
from psd_tools.composite.text.layout import Layout, PlacedGlyph, Rule  # noqa: E402
from psd_tools.composite.text.model import RGB, Transform  # noqa: E402
from psd_tools.composite.text.raster import (  # noqa: E402
    GlyphCache,
    Options,
    is_general,
    raster_warnings,
    rasterize,
    strike_scale,
)

from .conftest import GLYPHS, UPEM  # noqa: E402

NONE, CRISP, STRONG, SMOOTH, SHARP = 0, 1, 2, 3, 4
IDENTITY: Transform = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
RED: RGB = (1.0, 0.0, 0.0)
BLUE: RGB = (0.0, 0.0, 1.0)
GREEN: RGB = (0.0, 1.0, 0.0)
#: At this em the 100-unit stem of ``I`` is 4.05 px wide (26.6: 4.046875)
#: and 700 units tall are 28.35 px (26.6: 28.34375).
EM = 40.5
STEM_W = 259 / 64
STEM_H = 1814 / 64
#: Strong moves each edge outward by half the 26.6 strength, floored.
DILATE = (STRONG_EMBOLDEN_26_6 // 2) / 64
#: 8-bit coverage plus the 26.6 placement of an edge.
TOL = 1 / 255 + 1 / 64


@pytest.fixture(scope="module")
def face(ttf_font_path: str) -> Face:
    return load_face(ttf_font_path)


@pytest.fixture(scope="module")
def cff_face(cff_font_path: str) -> Face:
    return load_face(cff_font_path)


def glyph(
    face: Face,
    char: str,
    x: float,
    y: float,
    em: float = EM,
    fill: RGB = RED,
    faux_bold: bool = False,
    faux_italic: bool = False,
) -> PlacedGlyph:
    gid = face.ft.get_char_index(ord(char))
    return PlacedGlyph(face, gid, x, y, em, em, fill, faux_bold, faux_italic, 0)


def layout(glyphs: Sequence[PlacedGlyph] = (), rules: Sequence[Rule] = ()) -> Layout:
    return Layout(tuple(glyphs), tuple(rules), (), {}, ())


def render(
    glyphs: Sequence[PlacedGlyph] = (),
    antialias: int = STRONG,
    transform: Transform = IDENTITY,
    rules: Sequence[Rule] = (),
    options: Options | None = None,
    cache: GlyphCache | None = None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    return rasterize(
        layout(glyphs, rules), transform, antialias, options or Options(), cache
    )


def same(
    a: tuple[np.ndarray, tuple[int, int, int, int]],
    b: tuple[np.ndarray, tuple[int, int, int, int]],
) -> bool:
    return a[1] == b[1] and a[0].shape == b[0].shape and bool(np.allclose(a[0], b[0]))


def stem_oracle(
    transform: Transform, pen: tuple[float, float], em: float, bbox: Sequence[int]
) -> np.ndarray:
    """Exact coverage of the transformed ``I`` rectangle over ``bbox``."""
    xx, xy, yx, yy, tx, ty = transform
    gx0, gy0, gx1, gy1 = GLYPHS["I"][1]  # type: ignore[misc]
    poly = []
    for gx, gy in ((gx0, gy0), (gx1, gy0), (gx1, gy1), (gx0, gy1)):
        x, y = pen[0] + gx / UPEM * em, pen[1] - gy / UPEM * em
        poly.append((xx * x + yx * y + tx - bbox[0], xy * x + yy * y + ty - bbox[1]))
    return fill_coverage([np.array(poly)], bbox[2] - bbox[0], bbox[3] - bbox[1])


def test_phase_columns_strong(face: Face) -> None:
    rgba, bbox = render([glyph(face, "I", 0.25, 100.0)])
    assert bbox[0] == 0 and bbox[2] == 5
    cols = rgba[5, :, 3]
    # Sub-pixel placement: 0.75, 1, 1, 1, 0.30 before the symmetric dilation.
    assert cols == pytest.approx([0.75 + DILATE, 1, 1, 1, 0.296875 + DILATE], abs=TOL)
    assert cols[0] - 0.75 == pytest.approx(cols[4] - 0.296875, abs=TOL)
    assert cols.sum() == pytest.approx(STEM_W + 2 * DILATE, abs=3 * TOL)
    assert (rgba[..., :3][rgba[..., 3] > 0] == RED).all()


def test_delta_sign(face: Face) -> None:
    """A pen lower on the page moves ink down: ``delta.y`` is ``-frac_y``."""
    for frac in (0.25, 0.75):
        rgba, bbox = render([glyph(face, "I", 0.0, 100 + frac)])
        assert bbox[3] == 101
        assert rgba[100 - bbox[1], 1, 3] == pytest.approx(frac + DILATE, abs=TOL)


def test_smooth_is_strong(face: Face) -> None:
    g = [glyph(face, "I", 0.3, 50.7)]
    assert same(render(g, SMOOTH), render(g, STRONG))


def test_none_threshold(face: Face) -> None:
    rgba, _ = render([glyph(face, "I", 0.4, 100.0)], NONE)
    assert rgba[5, :, 3].tolist() == [1, 1, 1, 1, 0]  # 0.6 keeps, 0.447 drops
    rgba, _ = render([glyph(face, "I", 0.5, 100.0)], NONE)
    assert rgba[5, :, 3].tolist() == [0, 1, 1, 1, 0]  # 0.5 drops
    assert set(np.unique(rgba[..., 3])) <= {0.0, 1.0}


def test_crisp_rounds_origin(face: Face) -> None:
    rgba, bbox = render([glyph(face, "I", 0.25, 100.25)], CRISP)
    assert (bbox[1], bbox[3]) == (71, 100)
    assert rgba[5, :, 3] == pytest.approx([1, 1, 1, 1, STEM_W - 4], abs=TOL)
    assert rgba[-1, 1, 3] == 1.0


@pytest.mark.parametrize(
    "em, x, cols",
    [
        (32.0, 0.3, [1, 1, 1, 0.2]),  # whole pixels from 32 px up
        (32 * (1 - 1e-12), 0.3, [1, 1, 1, 0.2]),  # 32 px within 26.6 precision
        (31.0, 0.3, [0.5, 1, 1, 0.6]),  # half pixels below 32 px: 0.5
        (24.0, 0.3, [0.5, 1, 0.9]),  # 2.4 px stem at 0.5
        (23.0, 0.3, [0.75, 1, 0.55]),  # quarter pixels below 24 px: 0.25
        (20.0, 0.6, [0.5, 1, 0.5]),  # 0.6 rounds to 0.5
    ],
)
def test_crisp_subpixel_x_grows_finer_with_small_text(
    face: Face, em: float, x: float, cols: list[float]
) -> None:
    rgba, bbox = render([glyph(face, "I", x, 100.3, em=em)], CRISP)
    assert bbox[0] == 0
    assert rgba[5, :, 3] == pytest.approx(cols, abs=TOL)
    # y snaps to whole pixels at every size.
    assert bbox[3] == 100
    assert rgba[-1, 1, 3] == 1.0


def test_crisp_keeps_fraction_under_shear(face: Face) -> None:
    shear: Transform = (1.0, 0.0, 2 * GENERAL_TRANSFORM_EPS, 1.0, 0.25, 100.0)
    rgba, bbox = render([glyph(face, "I", 0.0, 0.0)], CRISP, shear)
    assert 0.7 < rgba[-1, -bbox[0], 3] < 0.8


def test_sharp_integer_origin(face: Face) -> None:
    rgba, bbox = render([glyph(face, "I", 0.25, 100.0)], SHARP)
    assert bbox[0] == 0
    cols = rgba[rgba.shape[0] // 2, :, 3]
    assert cols[:4].tolist() == [1, 1, 1, 1]
    assert cols[4] < 0.1


@pytest.mark.parametrize(
    "transform",
    [
        (3.0, 0.0, 0.0, 3.0, 20.0, 50.0),  # scale
        (-3.0, 0.0, 0.0, -3.0, 20.0, 50.0),  # flipped (180 degrees)
        (-3.0, 0.0, 0.0, 3.0, 20.0, 50.0),  # mirrored
        (0.0, 3.0, -3.0, 0.0, 20.0, 50.0),  # rot90
        (3.0, 0.0, 0.6, 3.0, 20.0, 50.0),  # skew
        (2.0, 1.0, -1.5, 3.0, 20.0, 50.0),  # general
    ],
)
def test_matrix_matches_exact_coverage(face: Face, transform: Transform) -> None:
    em, pen = 10.0, (0.0, 0.0)
    rgba, bbox = render([glyph(face, "I", *pen, em=em)], CRISP, transform)
    oracle = stem_oracle(transform, pen, em, bbox)
    assert oracle.sum() == pytest.approx(rgba[..., 3].sum(), rel=0.02)
    assert np.abs(rgba[..., 3] - oracle).max() < 0.05
    # The bbox is the ink's integer extent.
    assert (
        oracle[0].any(),
        oracle[-1].any(),
        oracle[:, 0].any(),
        oracle[:, -1].any(),
    ) == (
        True,
        True,
        True,
        True,
    )


def test_mirror_dilates(face: Face) -> None:
    plain = render(
        [glyph(face, "I", 0.0, 0.0, em=10)], STRONG, (3.0, 0, 0, 3.0, 20, 50)
    )
    mirror = render(
        [glyph(face, "I", 0.0, 0.0, em=10)], STRONG, (-3.0, 0, 0, 3.0, 20, 50)
    )
    assert mirror[0][..., 3].sum() == pytest.approx(plain[0][..., 3].sum(), rel=0.01)
    assert (
        mirror[0][..., 3].sum()
        > stem_oracle((3.0, 0, 0, 3.0, 20, 50), (0, 0), 10, plain[1]).sum()
    )


def test_strike_scale() -> None:
    assert strike_scale(40.2, frozenset({40})) == 2
    assert strike_scale(40.0, frozenset({40, 80})) == 4
    assert strike_scale(41.0, frozenset({40})) == 1
    assert strike_scale(12.4, frozenset()) == 1


def test_strike_guard_render(face: Face) -> None:
    strikes = frozenset({round(EM)})
    guarded = Face(
        face.hb,
        face.ft,
        dataclasses.replace(face.metrics, strikes=strikes),
        (face.key[0], 9),
    )
    g = [glyph(face, "I", 0.25, 100.25)]
    plain, bbox = render(g)
    out, bbox2 = render([glyph(guarded, "I", 0.25, 100.25)])
    assert bbox2 == bbox
    assert np.abs(out - plain).max() < 2 * TOL
    assert "strike_guard" in raster_warnings(
        layout([glyph(guarded, "I", 0, 0)]), IDENTITY
    )
    assert "strike_guard" not in raster_warnings(layout(g), IDENTITY)


def test_faux_bold(face: Face) -> None:
    rgba, bbox = render([glyph(face, "I", 0.0, 100.0, faux_bold=True)], CRISP)
    extra = round(64 * EM * FAUX_BOLD_EM) / 64
    row = rgba[5, :, 3]
    assert bbox[0] == -1
    assert row.sum() == pytest.approx(STEM_W + extra, abs=3 * TOL)
    assert row[0] == pytest.approx(extra / 2, abs=TOL)  # grows symmetrically
    assert row[-1] == pytest.approx(STEM_W - 4 + extra / 2, abs=TOL)


def test_faux_italic(face: Face) -> None:
    rgba, bbox = render([glyph(face, "I", 0.0, 100.0, faux_italic=True)], CRISP)
    a = rgba[..., 3]
    cols = np.arange(a.shape[1])
    top = (a[1] * cols).sum() / a[1].sum()
    bottom = (a[-1] * cols).sum() / a[-1].sum()
    rows_apart = a.shape[0] - 2
    assert top - bottom == pytest.approx(FAUX_ITALIC_SHEAR * rows_apart, abs=0.2)
    assert bbox[2] - bbox[0] == pytest.approx(
        STEM_W + FAUX_ITALIC_SHEAR * STEM_H, abs=1.5
    )


def test_two_colour_overlap(face: Face) -> None:
    rgba, bbox = render(
        [
            glyph(face, "I", 0.0, 100.0, fill=RED),
            glyph(face, "I", 0.5, 100.0, fill=BLUE),
        ]
    )
    assert rgba[..., 3].max() == 1.0  # summed coverage is clipped
    c0, c4 = -bbox[0], 4 - bbox[0]
    assert tuple(rgba[5, c0, :3]) == RED  # red covers column 0 more
    assert tuple(rgba[5, c4, :3]) == BLUE  # blue covers column 4 more
    assert rgba[5, c4, 3] == pytest.approx(
        2 * (STEM_W - 4) + 0.5 + 2 * DILATE, abs=2 * TOL
    )


def test_rules(face: Face) -> None:
    rgba, bbox = render(rules=[Rule(0.0, 10.0, 5.0, 2.0, GREEN)])
    assert bbox == (0, 4, 10, 6)
    assert (rgba[..., 3] == 1).all()
    assert (rgba[..., :3] == GREEN).all()
    rgba, bbox = render(rules=[Rule(0.0, 10.0, 5.25, 1.0, GREEN)])
    assert bbox == (0, 4, 10, 6)
    assert rgba[:, 3, 3].tolist() == pytest.approx([0.25, 0.75], abs=1 / 255)
    rgba, bbox = render(
        rules=[Rule(0.0, 10.0, 5.0, 2.0, GREEN)], transform=(3.0, 0, 0, 3.0, 1, 1)
    )
    assert bbox == (1, 13, 31, 19)
    rgba, _ = render(rules=[Rule(0.0, 10.0, 5.25, 1.0, GREEN)], antialias=NONE)
    assert rgba[:, 3, 3].tolist() == [0, 1]


def test_rule_and_glyph_share_canvas(face: Face) -> None:
    rgba, bbox = render(
        [glyph(face, "I", 0.0, 100.0)], rules=[Rule(0.0, 20.0, 103.0, 2.0, RED)]
    )
    assert bbox == (-1, 71, 20, 104)  # the dilation reaches column -1
    assert rgba[-1, 10, 3] == 1.0 and rgba[0, 1, 3] > 0


def test_bbox_is_ink_extent(face: Face) -> None:
    rgba, bbox = render([glyph(face, "I", 10.25, 100.0), glyph(face, "A", 40.0, 100.0)])
    a = rgba[..., 3]
    assert rgba.shape == (bbox[3] - bbox[1], bbox[2] - bbox[0], 4)
    assert rgba.dtype == np.float32
    assert a[0].any() and a[-1].any() and a[:, 0].any() and a[:, -1].any()
    assert bbox[:2] == (10, 71)


def test_empty(face: Face) -> None:
    t: Transform = (3.0, 0.0, 0.0, 3.0, 10.4, 20.6)
    rgba, bbox = render([], transform=t)
    assert rgba.shape == (0, 0, 4) and rgba.dtype == np.float32
    assert bbox == (10, 21, 10, 21)
    rgba, bbox = render([glyph(face, " ", 0.0, 0.0)], transform=t)
    assert rgba.shape == (0, 0, 4) and bbox == (10, 21, 10, 21)
    rgba, bbox = render(
        [glyph(face, "I", 0.0, 0.0)], transform=(0.0, 0.0, 0.0, 1.0, 1.0, 2.0)
    )
    assert rgba.shape == (0, 0, 4) and bbox == (1, 2, 1, 2)


def test_canvas_guard(face: Face) -> None:
    far = [glyph(face, "I", 0.0, 0.0), glyph(face, "I", 1000.0, 1000.0)]
    with pytest.raises(TextRenderError):
        render(far, options=Options(max_canvas_pixels=10_000))
    render(far, options=Options(max_canvas_pixels=2_000_000))
    with pytest.raises(TextRenderError):
        render(
            [glyph(face, "I", 0.0, 0.0, em=1e5)],
            options=Options(max_canvas_pixels=10_000),
        )


def test_glyph_cache(face: Face) -> None:
    cache = GlyphCache()
    g = [
        glyph(face, "I", 0.25, 0.0),
        glyph(face, "I", 10.25, 0.0),
        glyph(face, "I", 20.5, 0.0),
    ]
    first = render(g, cache=cache)
    assert len(cache) == 2  # two distinct sub-pixel phases
    second = render(g, cache=cache)
    assert len(cache) == 2
    assert second[1] == first[1] and (second[0] == first[0]).all()
    small = GlyphCache(maxsize=1)
    render(g, cache=small)
    assert len(small) == 1
    assert small.get("k", lambda: 7) == 7 and small.get("k", lambda: 8) == 7


def test_resample_matches_matrix(luckiest_guy_path: str) -> None:
    pytest.importorskip("scipy")
    real = load_face(luckiest_guy_path)
    skew: Transform = (3.0, 0.4, 0.6, 3.0, 20.0, 50.0)
    glyphs = [
        glyph(real, c, x, 0.0, em=12) for c, x in (("A", 0.0), ("V", 9.0), ("d", 18.0))
    ]
    assert is_general(skew)
    matrix, mb = render(glyphs, STRONG, skew)
    warped, wb = render(
        glyphs, STRONG, skew, options=Options(general_transform="resample")
    )
    ub = (min(mb[0], wb[0]), min(mb[1], wb[1]), max(mb[2], wb[2]), max(mb[3], wb[3]))
    place = np.zeros((2, ub[3] - ub[1], ub[2] - ub[0]), np.float32)
    for k, (im, b) in enumerate(((matrix, mb), (warped, wb))):
        place[k, b[1] - ub[1] : b[3] - ub[1], b[0] - ub[0] : b[2] - ub[0]] = im[..., 3]
    either = (place > 0.1).any(axis=0)
    assert either.any()
    assert np.abs(place[0] - place[1])[either].mean() < 0.05
    diagonal: Transform = (3.0, 0.0, 0.0, 3.0, 20.0, 50.0)
    assert same(
        render(glyphs, STRONG, diagonal, options=Options(general_transform="resample")),
        render(glyphs, STRONG, diagonal),
    )


def test_emoji_silhouette_option(face: Face) -> None:
    svg = Face(
        face.hb,
        face.ft,
        dataclasses.replace(face.metrics, has_svg=True),
        (face.key[0], 8),
    )
    g = [glyph(svg, "I", 0.25, 100.0)]
    assert same(render(g), render([glyph(face, "I", 0.25, 100.0)]))
    rgba, _ = render(g, options=Options(monochrome_emoji=False))
    assert rgba.shape == (0, 0, 4)
    assert raster_warnings(layout(g), IDENTITY) == ("monochrome_emoji",)


def test_warnings_and_general(face: Face) -> None:
    g = layout([glyph(face, "I", 0.0, 0.0)])
    assert raster_warnings(g, IDENTITY) == ()
    assert raster_warnings(g, (1.0, 0.0, 2 * GENERAL_TRANSFORM_EPS, 1.0, 0.0, 0.0)) == (
        "generic_transform",
    )
    assert not is_general((1.0, GENERAL_TRANSFORM_EPS / 2, 0.0, -1.0, 0.0, 0.0))
    assert is_general((0.0, 3.0, -3.0, 0.0, 0.0, 0.0))


def test_unknown_antialias(face: Face) -> None:
    with pytest.raises(ValueError):
        render([glyph(face, "I", 0.0, 0.0)], antialias=7)


def test_cff_matches_truetype(face: Face, cff_face: Face) -> None:
    tt, tb = render([glyph(face, "I", 0.25, 100.75)])
    cff, cb = render([glyph(cff_face, "I", 0.25, 100.75)])
    assert cb == tb
    assert np.abs(cff - tt).max() < 2 * TOL


def test_real_font(luckiest_guy_path: str) -> None:
    real = load_face(luckiest_guy_path)
    rgba, bbox = render([glyph(real, "A", 0.0, 100.0, em=40)])
    a = rgba[..., 3]
    assert a.max() == 1.0 and 0 < a.min() < 1 or a.min() == 0
    assert bbox[1] < 100 - 0.5 * 40 and bbox[3] >= 100
    assert math.isclose(float(a.sum()), float(a.sum()))  # finite
    assert np.isfinite(rgba).all()
