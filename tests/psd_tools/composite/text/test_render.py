import subprocess
import sys
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
    Options,
    default_font_map,
    font_names,
    render,
    render_type_layer,
    verify,
)
from psd_tools.composite.text.model import TextModel
from psd_tools.constants import Resource

from ...utils import full_name
from .conftest import PS_TT


@pytest.fixture
def layer() -> TypeLayer:
    psd = PSDImage.open(full_name("text.psd"))
    found = [x for x in psd.descendants() if isinstance(x, TypeLayer)]
    return found[0]


@pytest.fixture
def arial_as_luckiest_guy(luckiest_guy_path: str) -> FontMap:
    return FontMap(aliases={"ArialMT": luckiest_guy_path})


def test_font_names(layer: TypeLayer) -> None:
    assert font_names(layer) == ["ArialMT"]


def test_missing_font_is_none(layer: TypeLayer, fixture_fonts_dir: str) -> None:
    assert render_type_layer(layer, FontMap(dirs=[fixture_fonts_dir])) is None
    with pytest.raises(MissingFontError):
        render(layer, fonts=FontMap(dirs=[fixture_fonts_dir]))


def test_substitute_policy_renders(layer: TypeLayer, fixture_fonts_dir: str) -> None:
    fm = FontMap(dirs=[fixture_fonts_dir], policy="substitute", fallback=PS_TT)
    assert render_type_layer(layer, fm, Options()) is not None
    report = render(layer, fonts=fm).report
    assert report.fonts["ArialMT"][0] == "substitute"
    assert "font_substitute" in report.warnings


def test_default_font_map_follows_policy() -> None:
    assert default_font_map("substitute").policy == "substitute"
    assert default_font_map("substitute") is default_font_map("substitute")


def test_text_psd_renders(layer: TypeLayer, arial_as_luckiest_guy: FontMap) -> None:
    out = render_type_layer(layer, arial_as_luckiest_guy)
    assert out is not None
    rgba, bbox = out
    assert rgba.dtype == np.float32
    assert rgba.shape == (bbox[3] - bbox[1], bbox[2] - bbox[0], 4)
    assert rgba[..., 3].max() == 1.0
    assert rgba[..., 3].min() == 0.0
    # Black text: the fill is black wherever there is ink.
    assert not rgba[..., :3][rgba[..., 3] > 0].any()
    # The pen starts at the transform's translation.
    x, y = layer.transform[4], layer.transform[5]
    assert bbox[0] <= x + 1 and bbox[1] < y < bbox[3]


def test_text_psd_report(layer: TypeLayer, arial_as_luckiest_guy: FontMap) -> None:
    result = render(layer, fonts=arial_as_luckiest_guy)
    report = result.report
    assert report.fonts["ArialMT"][0] == "exact"
    assert report.warnings == ()
    assert [line.text for line in report.lines] == [
        "Line 1",
        "Line 2",
        "Line 3 and text",
    ]
    model = TextModel.from_layer(layer)
    size = model.runs[0].style.size
    pitch = model.auto_leading * size
    assert [line.y for line in report.lines] == pytest.approx([0.0, pitch, 2 * pitch])
    assert all(line.decision == "paragraph" for line in report.lines)


def test_type_layer_psd_renders(luckiest_guy_path: str) -> None:
    psd = PSDImage.open(full_name("layers/type-layer.psd"))
    layer = next(x for x in psd.descendants() if isinstance(x, TypeLayer))
    out = render_type_layer(layer, FontMap(aliases={"ArialMT": luckiest_guy_path}))
    assert out is not None
    rgba, _bbox = out
    assert rgba.shape[0] > 0 and rgba.shape[1] > 0
    assert rgba[..., 3].max() == 1.0


def test_resolution_does_not_scale_the_render(luckiest_guy_path: str) -> None:
    # A 150 ppi document: FontSize and the transform are already in pixels.
    psd = PSDImage.open(full_name("adjustment-fillers.psd"))
    layer = next(x for x in psd.descendants() if isinstance(x, TypeLayer))
    info = psd.image_resources.get_data(Resource.RESOLUTION_INFO)
    assert info.horizontal == 150 << 16
    fonts = FontMap(aliases={"ArialMT": luckiest_guy_path})
    at_150 = render(layer, fonts=fonts)
    x0, y0, x1, y1 = layer.bbox
    assert at_150.bbox[3] - at_150.bbox[1] == y1 - y0
    assert abs(at_150.bbox[1] - y0) <= 1 and abs(at_150.bbox[0] - x0) < (x1 - x0) / 4
    info.horizontal = 72 << 16
    at_72 = render(layer, fonts=fonts)
    assert at_72.bbox == at_150.bbox
    np.testing.assert_array_equal(at_72.rgba, at_150.rgba)


def test_platform_antialias_falls_back(fixture_fonts_dir: str) -> None:
    # ``antiAliasPlatformLCD`` has no transfer of its own; it renders as the
    # default mode with a warning instead of failing.
    psd = PSDImage.open(full_name("masks.psd"))
    layer = next(x for x in psd.descendants() if isinstance(x, TypeLayer))
    assert layer._data.text_data[b"AntA"].enum == b"antiAliasPlatformLCD"
    fonts = FontMap(dirs=[fixture_fonts_dir], policy="substitute", fallback=PS_TT)
    result = render(layer, fonts=fonts)
    assert "antialias_fallback" in result.report.warnings
    assert verify(layer, fonts=FontMap(dirs=[fixture_fonts_dir])).status == "missing"


def test_verify_text_psd(layer: TypeLayer, arial_as_luckiest_guy: FontMap) -> None:
    fidelity = verify(layer, fonts=arial_as_luckiest_guy)
    assert fidelity.status == "ok"
    # A different font than Photoshop used: some agreement, not identity.
    assert 0.0 < fidelity.iou < 1.0
    assert fidelity.bad_pct > 0
    assert fidelity.cdx is not None and fidelity.dx0 is not None


def test_verify_missing_font(layer: TypeLayer, fixture_fonts_dir: str) -> None:
    fidelity = verify(layer, fonts=FontMap(dirs=[fixture_fonts_dir]))
    assert fidelity.status == "missing"
    assert fidelity.iou == 0.0 and fidelity.bad_pct == 100.0


def test_composite_import_does_not_load_text() -> None:
    code = (
        "import sys, psd_tools.composite; "
        "assert 'psd_tools.composite.text' not in sys.modules; "
        "assert 'freetype' not in sys.modules; "
        "assert 'uharfbuzz' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_unedited_document_saves_byte_identical(
    tmp_path: Any, fixture_fonts_dir: str
) -> None:
    path = full_name("text.psd")
    psd = PSDImage.open(path)
    layer = next(x for x in psd.descendants() if isinstance(x, TypeLayer))
    render_type_layer(layer, FontMap(dirs=[fixture_fonts_dir]))
    psd.save(tmp_path / "copy.psd")
    with open(path, "rb") as f, open(tmp_path / "copy.psd", "rb") as g:
        assert f.read() == g.read()
