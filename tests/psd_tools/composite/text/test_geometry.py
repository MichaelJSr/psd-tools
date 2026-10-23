import json
import os
from typing import Any

import numpy as np
import pytest

pytest.importorskip("freetype")
pytest.importorskip("uharfbuzz")

from psd_tools import PSDImage
from psd_tools.api.layers import TypeLayer
from psd_tools.composite.text import (
    FontMap,
    MissingFontError,
    render,
    text_geometry,
)
from psd_tools.composite.text import fonts as fonts_module
from psd_tools.composite.text.constants import FAUX_BOLD_EM, FAUX_ITALIC_SHEAR
from psd_tools.composite.text.layout import layout
from psd_tools.composite.text.model import TextModel

from ...utils import full_name
from .conftest import GLYPHS, UPEM

#: ``text.psd`` laid out with ArialMT drawn as Luckiest Guy, one line or glyph
#: per row; rewrite it from ``text_geometry`` after a deliberate layout change.
GOLDEN = os.path.join(os.path.dirname(__file__), "text-psd-geometry.json")
BREAKS = "\r\x03 "


@pytest.fixture(autouse=True)
def _no_platform_fonts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fonts_module, "platform_font_dirs", lambda: [])


def type_layer(psd: str, name: str | None = None, index: int = 0) -> TypeLayer:
    found = [
        x
        for x in PSDImage.open(full_name(psd)).descendants()
        if isinstance(x, TypeLayer) and name in (None, x.name)
    ]
    return found[index]


def as_fixture_font(path: str, *names: str) -> FontMap:
    return FontMap(aliases={name: path for name in names})


def assert_lines_cover(geometry: dict[str, Any]) -> None:
    """Lines tile the text in order; only break characters fall between them."""
    text = geometry["text"]
    done = 0
    for line in geometry["lines"]:
        assert line["start"] >= done
        assert text[done : line["start"]].strip(BREAKS) == ""
        assert line["text"] == text[line["start"] : line["end"]]
        assert line["text"][-1:] not in ("\r", "\x03")
        done = line["end"]
    assert text[done:].strip(BREAKS) == ""
    for g in geometry["glyphs"]:
        line = geometry["lines"][g["line"]]
        assert line["start"] <= g["i"] < g["end"] <= line["end"]
        assert g["ch"] == text[g["i"]]


def assert_close(actual: Any, expected: Any, path: str = "") -> None:
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            assert_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for k, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert_close(a, e, f"{path}[{k}]")
    elif isinstance(expected, float):
        # An outline or advance one FreeType/HarfBuzz release apart.
        assert actual == pytest.approx(expected, abs=0.02), path
    else:
        assert actual == expected, path


def test_ligature_resolves_to_its_characters(ttf_font_path: str) -> None:
    layer = type_layer("text.psd")
    layer.set_text("fid AI")
    geometry = text_geometry(layer, as_fixture_font(ttf_font_path, "ArialMT"))
    text = geometry["text"]
    spans = [text[g["i"] : g["end"]] for g in geometry["glyphs"]]
    assert spans == ["fi", "d", " ", "A", "I"]

    # The f_i ligature's ink is its outline rectangle at the layer's size.
    size = TextModel.from_layer(layer).runs[0].style.size
    x, y = layer.transform[4], layer.transform[5]
    x0, y0, x1, y1 = GLYPHS["f_i"][1]  # type: ignore[misc]
    expected = [x + x0 / UPEM * size, y - y1 / UPEM * size]
    expected += [x + x1 / UPEM * size, y - y0 / UPEM * size]
    assert geometry["glyphs"][0]["ink"] == pytest.approx(expected, abs=0.01)
    assert geometry["glyphs"][2]["ink"] is None  # the space
    assert_lines_cover(geometry)


def test_lines_cover_the_text(luckiest_guy_path: str) -> None:
    geometry = text_geometry(
        type_layer("text.psd"), as_fixture_font(luckiest_guy_path, "ArialMT")
    )
    assert [(line["start"], line["end"]) for line in geometry["lines"]] == [
        (0, 6),
        (7, 13),
        (14, 29),
    ]
    assert_lines_cover(geometry)


def test_wrapped_lines_exclude_breaks(ttf_font_path: str) -> None:
    layer = type_layer("text-story.psd", "Ability")
    layer.set_text("IA " * 20 + "\x03AV\rVA")
    geometry = text_geometry(layer, as_fixture_font(ttf_font_path, "ArialMT"))
    lines = geometry["lines"]
    assert len(lines) >= 4
    # A soft wrap drops the space it breaks at; a forced break drops ``\x03``.
    assert all(not line["text"].endswith(" ") for line in lines)
    assert [line["text"] for line in lines[-2:]] == ["AV", "VA"]
    assert_lines_cover(geometry)
    # Each line's box holds its glyphs, below the line before.
    for j, line in enumerate(lines):
        cells = [g["cell"] for g in geometry["glyphs"] if g["line"] == j]
        assert line["box"][0] <= min(c[0] for c in cells)
        assert line["box"][2] >= max(c[2] for c in cells)
    tops = [line["box"][1] for line in lines]
    assert tops == sorted(tops)


@pytest.mark.parametrize(
    ("index", "uniform"), [(4, True), (2, False)], ids=["scaled", "stretched"]
)
def test_transformed_layer(luckiest_guy_path: str, index: int, uniform: bool) -> None:
    # masks.psd scales its centred box text 2x, some layers less in x than in y.
    layer = type_layer("masks.psd", index=index)
    xx, _, _, yy, tx, ty = layer.transform
    assert yy == 2.0 and (xx == yy) == uniform
    fonts = as_fixture_font(luckiest_guy_path, "ProximaNova-Regular")
    geometry = text_geometry(layer, fonts)
    lay = layout(TextModel.from_layer(layer), fonts)
    assert len(geometry["glyphs"]) == len(lay.glyphs)
    for g, placed in zip(geometry["glyphs"], lay.glyphs, strict=True):
        x0, _, x1, _ = g["cell"]
        assert x0 == pytest.approx(xx * placed.x + tx, abs=0.01)
        assert x1 - x0 == pytest.approx(xx * placed.advance, abs=0.02)
    line = lay.lines[0]
    assert geometry["lines"][0]["baseline"][1] == pytest.approx(
        yy * line.y + ty, abs=0.01
    )
    assert_ink_covers_render(layer, fonts, geometry)


def assert_ink_covers_render(
    layer: TypeLayer, fonts: FontMap, geometry: dict[str, Any]
) -> None:
    """Every pixel the renderer inks is inside a glyph's ink box, give or take one."""
    result = render(layer, fonts=fonts)
    bx, by = result.bbox[0], result.bbox[1]
    mask = np.zeros(result.rgba.shape[:2], bool)
    for g in geometry["glyphs"]:
        if g["ink"] is None:
            continue
        x0, y0, x1, y1 = g["ink"]
        rows = slice(max(int(np.floor(y0)) - 1 - by, 0), int(np.ceil(y1)) + 1 - by)
        cols = slice(max(int(np.floor(x0)) - 1 - bx, 0), int(np.ceil(x1)) + 1 - bx)
        mask[rows, cols] = True
    ink = result.rgba[..., 3] > 0.5
    assert ink.any()
    assert not (ink & ~mask).any()


@pytest.mark.parametrize(
    "style",
    [{}, {"FauxBold": True}, {"FauxItalic": True}],
    ids=["plain", "bold", "italic"],
)
def test_ink_covers_render(luckiest_guy_path: str, style: dict[str, Any]) -> None:
    layer = type_layer("text.psd")
    if style:
        layer.set_span_style(0, len(layer.text), **style)
    fonts = as_fixture_font(luckiest_guy_path, "ArialMT")
    assert_ink_covers_render(layer, fonts, text_geometry(layer, fonts))


def test_faux_styles_move_ink(luckiest_guy_path: str) -> None:
    fonts = as_fixture_font(luckiest_guy_path, "ArialMT")
    layer = type_layer("text.psd")
    x0, y0, x1, y1 = text_geometry(layer, fonts)["glyphs"][0]["ink"]
    layer.set_span_style(0, 1, FauxBold=True)
    bold = text_geometry(layer, fonts)["glyphs"][0]["ink"]
    half = FAUX_BOLD_EM / 2 * TextModel.from_layer(layer).runs[0].style.size
    assert np.subtract(bold, [x0, y0, x1, y1]) == pytest.approx(
        [-half, -half, half, half], abs=0.02
    )
    layer.set_span_style(0, 1, FauxBold=False, FauxItalic=True)
    italic = text_geometry(layer, fonts)["glyphs"][0]["ink"]
    # The shear leans each edge by its height above the baseline.
    baseline = layer.transform[5]
    lean = [
        FAUX_ITALIC_SHEAR * (baseline - y1),
        0,
        FAUX_ITALIC_SHEAR * (baseline - y0),
        0,
    ]
    assert np.subtract(italic, [x0, y0, x1, y1]) == pytest.approx(lean, abs=0.02)


def test_missing_fonts() -> None:
    layer = type_layer("masks.psd", "text replaced")
    geometry = text_geometry(layer, FontMap(policy="skip"))
    assert geometry["missing_fonts"] == ["ProximaNova-Regular"]
    assert geometry["glyphs"] == []
    assert [line["box"] for line in geometry["lines"]] == [None]
    with pytest.raises(MissingFontError):
        text_geometry(layer, FontMap())


def test_golden(luckiest_guy_path: str) -> None:
    geometry = text_geometry(
        type_layer("text.psd"), as_fixture_font(luckiest_guy_path, "ArialMT")
    )
    assert json.loads(json.dumps(geometry)) == geometry
    with open(GOLDEN) as f:
        expected = json.load(f)
    # The font is the fixture's, not Photoshop's: the file pins the layout only.
    assert_close(geometry, expected)
