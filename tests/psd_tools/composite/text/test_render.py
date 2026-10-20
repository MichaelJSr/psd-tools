import builtins
import dataclasses
import os
import subprocess
import sys
from typing import Any, cast

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
    document_transform,
    font_names,
    render,
    render_type_layer,
    verify,
)
from psd_tools.composite.text import fonts as fonts_module
from psd_tools.composite.text.model import TextModel

from ...utils import full_name
from .conftest import PS_TT

#: Photoshop-rendered cards copied out of the TTSCardMaker corpus; the tests
#: on them skip when the directory is absent.
FIXTURES = os.environ.get("PSD_TOOLS_TEXT_FIXTURES", "/tmp/wt-text-fixtures")
#: (psd, index path, layer name, bad% ceiling, IoU floor)
CORPUS = [
    # point Strong, one subscript run, centred
    ("Creatures/Mario/3 Stars/Vivian.psd", (5, 2), "Name", 5.0, 0.95),
    # box Strong, ten runs in several colours, three paragraphs
    ("Creatures/Undertale/4 Stars/Toriel.psd", (4, 0), "Ability", 6.0, 0.93),
    # box Crisp, justified
    ("Auxiliary/Rulebook/10_Attacking.psd", (2, 3), "Phase 1", 30.0, 0.88),
    # point Crisp
    ("Auxiliary/Rulebook/17_Death.psd", (7, 1), "17", 10.0, 0.90),
    # flipped transform, two paragraphs of point text
    ("Creatures/Isaac/3 Stars/Tainted_Lazarus.psd", (2, 1, 1), "Name", 8.0, 0.95),
    # box Strong with BaselineShift runs and AutoKerning off
    ("Creatures/Pandas/2 Stars/Minecraft_Panda.psd", (4, 0), "Ability", 6.0, 0.90),
]


@pytest.fixture(autouse=True)
def _no_platform_fonts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fonts_module, "platform_font_dirs", lambda: [])


@pytest.fixture
def layer() -> TypeLayer:
    psd = PSDImage.open(full_name("text.psd"))
    found = [x for x in psd.descendants() if isinstance(x, TypeLayer)]
    return found[0]


@pytest.fixture
def arial_as_luckiest_guy(luckiest_guy_path: str) -> FontMap:
    return FontMap(aliases={"ArialMT": luckiest_guy_path})


@pytest.fixture(scope="module")
def corpus_fonts() -> FontMap:
    fonts = os.path.join(FIXTURES, "fonts")
    if not os.path.isdir(fonts):
        pytest.skip(f"corpus fixtures not found under {FIXTURES}")
    return FontMap(dirs=[fonts])


def corpus_layer(psd: str, index: tuple[int, ...]) -> TypeLayer:
    path = os.path.join(FIXTURES, psd)
    if not os.path.isfile(path):
        pytest.skip(f"{path} not found")
    node: Any = PSDImage.open(path)
    for i in index:
        node = node[i]
    assert isinstance(node, TypeLayer)
    return node


def test_font_names(layer: TypeLayer) -> None:
    assert font_names(layer) == ["ArialMT"]


def test_missing_font_is_none(layer: TypeLayer, fixture_fonts_dir: str) -> None:
    assert render_type_layer(layer, FontMap(dirs=[fixture_fonts_dir])) is None
    assert render_type_layer(layer, FontMap(policy="skip")) is None
    with pytest.raises(MissingFontError):
        render(layer, fonts=FontMap(dirs=[fixture_fonts_dir]))


def test_skip_policy_reports_missing(layer: TypeLayer) -> None:
    result = render(layer, fonts=FontMap(policy="skip"))
    assert result.report.fonts == {"ArialMT": ("missing", None)}
    assert result.rgba.shape == (0, 0, 4)


def test_substitute_policy_renders(layer: TypeLayer, fixture_fonts_dir: str) -> None:
    fm = FontMap(dirs=[fixture_fonts_dir], policy="substitute", fallback=PS_TT)
    assert render_type_layer(layer, fm, Options()) is not None
    report = render(layer, fonts=fm).report
    assert report.fonts["ArialMT"][0] == "substitute"
    assert "font_substitute" in report.warnings


def test_default_font_map_follows_policy() -> None:
    assert default_font_map("skip").policy == "skip"
    assert default_font_map("skip") is default_font_map("skip")


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
    assert report.ms > 0


def test_breaks_override(layer: TypeLayer, arial_as_luckiest_guy: FontMap) -> None:
    # Break the third paragraph before "text".
    start = layer.text.index("text")
    result = render(layer, fonts=arial_as_luckiest_guy, breaks=[start])
    texts = [line.text for line in result.report.lines]
    assert texts == ["Line 1", "Line 2", "Line 3 and ", "text"]
    assert result.report.lines[2].decision == "break"


def test_type_layer_psd_renders(luckiest_guy_path: str) -> None:
    psd = PSDImage.open(full_name("layers/type-layer.psd"))
    layer = next(x for x in psd.descendants() if isinstance(x, TypeLayer))
    out = render_type_layer(layer, FontMap(aliases={"ArialMT": luckiest_guy_path}))
    assert out is not None
    rgba, bbox = out
    assert rgba.shape[0] > 0 and rgba.shape[1] > 0
    assert rgba[..., 3].max() == 1.0


def test_document_transform_scales_the_linear_part(layer: TypeLayer) -> None:
    model = TextModel.from_layer(layer)
    assert document_transform(model) == model.transform
    doubled = document_transform(dataclasses.replace(model, resolution=144.0))
    xx, xy, yx, yy, tx, ty = model.transform
    assert doubled == (2 * xx, 2 * xy, 2 * yx, 2 * yy, tx, ty)


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
    assert verify(layer, fonts=FontMap(policy="skip")).status == "missing"


@pytest.mark.parametrize(("psd", "index", "name", "bad_max", "iou_min"), CORPUS)
def test_corpus_layer_matches_photoshop(
    corpus_fonts: FontMap,
    psd: str,
    index: tuple[int, ...],
    name: str,
    bad_max: float,
    iou_min: float,
) -> None:
    layer = corpus_layer(psd, index)
    assert layer.name == name
    result = render(layer, fonts=corpus_fonts)
    assert all(how == "exact" for how, _ in result.report.fonts.values())
    fidelity = verify(layer, fonts=corpus_fonts)
    assert fidelity.status == "ok"
    assert fidelity.bad_pct <= bad_max
    assert fidelity.iou >= iou_min
    assert abs(fidelity.cdx or 0.0) < 1.0 and abs(fidelity.cdy or 0.0) < 1.0
    out = render_type_layer(layer, corpus_fonts)
    assert out is not None
    assert out[1] == result.bbox


def test_corpus_box_report(corpus_fonts: FontMap) -> None:
    psd, index, *_ = CORPUS[1]
    layer = corpus_layer(psd, index)
    report = render(layer, fonts=corpus_fonts).report
    assert len(report.lines) >= 3
    assert "".join(line.text for line in report.lines).replace(" ", "") == (
        layer.text.replace("\r", "").replace(" ", "")
    )
    assert report.lines[-1].decision == "paragraph"


def test_composite_import_does_not_load_text() -> None:
    code = (
        "import sys, psd_tools.composite; "
        "assert 'psd_tools.composite.text' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def _quantised(rgba: np.ndarray) -> np.ndarray:
    return np.round(rgba * 255) / 255


def test_rasterize_round_trip(
    layer: TypeLayer, arial_as_luckiest_guy: FontMap, tmp_path: Any
) -> None:
    psd = cast(PSDImage, layer._psd)
    layer.set_text("Hello\rworld")
    assert layer._text_dirty
    report = layer.rasterize(fonts=arial_as_luckiest_guy)
    assert not layer._text_dirty
    assert [line.text for line in report.lines] == ["Hello", "world"]
    out = render_type_layer(layer, arial_as_luckiest_guy)
    assert out is not None
    rgba, bbox = out
    assert tuple(layer.bbox) == bbox
    shape = layer.numpy("shape")
    color = layer.numpy("color")
    assert shape is not None and color is not None
    np.testing.assert_allclose(shape[..., 0], _quantised(rgba[..., 3]), atol=1e-6)
    ink = rgba[..., 3] > 0
    np.testing.assert_allclose(color[ink], _quantised(rgba[..., :3][ink]), atol=1e-6)
    psd.save(tmp_path / "edited.psd")
    again = PSDImage.open(tmp_path / "edited.psd")
    reopened = next(x for x in again.descendants() if isinstance(x, TypeLayer))
    assert reopened.text == "Hello\rworld"
    assert tuple(reopened.bbox) == bbox
    shape2 = reopened.numpy("shape")
    assert shape2 is not None
    np.testing.assert_allclose(shape2[..., 0], _quantised(rgba[..., 3]), atol=1e-6)
    image = reopened.composite()
    assert image is not None and image.size == (bbox[2] - bbox[0], bbox[3] - bbox[1])
    alpha = np.asarray(image.convert("RGBA"), dtype=np.float32)[..., 3] / 255
    np.testing.assert_allclose(alpha, _quantised(rgba[..., 3]), atol=1e-6)
    whole = np.asarray(again.composite().convert("RGBA"), dtype=np.float32)
    x0, y0, x1, y1 = bbox
    assert (whole[y0:y1, x0:x1, :3].min(axis=-1) < 128).any()


def test_rasterize_rewrites_text_bounds(luckiest_guy_path: str) -> None:
    psd = PSDImage.open(full_name("layers/type-layer.psd"))
    layer = next(x for x in psd.descendants() if isinstance(x, TypeLayer))
    box = layer._data.text_data[b"boundingBox"]
    before = {k: v.value for k, v in box.items()}
    layer.rasterize(fonts=FontMap(aliases={"ArialMT": luckiest_guy_path}))
    after = layer._data.text_data[b"boundingBox"]
    assert any(after[k].value != before[k] for k in before)
    assert after[b"Rght"].value > after[b"Left"].value
    # Box text keeps its frame.
    assert layer._data.text_data[b"bounds"][b"Rght"].value == pytest.approx(32.0)


def test_rasterize_without_the_extra(
    layer: TypeLayer, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "psd_tools.composite.text":
            raise ImportError("no text extra")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"psd-tools\[text\]"):
        layer.rasterize()


def test_unedited_document_saves_byte_identical(tmp_path: Any) -> None:
    path = full_name("text.psd")
    psd = PSDImage.open(path)
    layer = next(x for x in psd.descendants() if isinstance(x, TypeLayer))
    render_type_layer(layer, FontMap(policy="skip"))
    psd.save(tmp_path / "copy.psd")
    with open(path, "rb") as f, open(tmp_path / "copy.psd", "rb") as g:
        assert f.read() == g.read()


@pytest.mark.parametrize("which", [0, 1, 4])
def test_corpus_edit_round_trip(
    corpus_fonts: FontMap, which: int, tmp_path: Any
) -> None:
    psd_name, index, name, *_ = CORPUS[which]
    layer = corpus_layer(psd_name, index)
    psd = cast(PSDImage, layer._psd)
    layer.set_text(layer.text + " edited")
    report = layer.rasterize(fonts=corpus_fonts)
    assert report.lines[-1].text.endswith("edited")
    out = render_type_layer(layer, corpus_fonts)
    assert out is not None
    rgba, bbox = out
    assert tuple(layer.bbox) == bbox
    psd.save(tmp_path / "card.psd")
    again = PSDImage.open(tmp_path / "card.psd")
    node: Any = again
    for i in index:
        node = node[i]
    assert isinstance(node, TypeLayer)
    assert node.text == layer.text
    shape = node.numpy("shape")
    assert shape is not None
    np.testing.assert_allclose(shape[..., 0], _quantised(rgba[..., 3]), atol=1e-6)
    image = node.composite()
    assert image is not None
    alpha = np.asarray(image.convert("RGBA"), dtype=np.float32)[..., 3] / 255
    np.testing.assert_allclose(alpha, _quantised(rgba[..., 3]), atol=1e-6)
