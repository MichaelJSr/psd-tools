import io
import math
from typing import Any, cast

import pytest

pytest.importorskip("freetype")
pytest.importorskip("uharfbuzz")

from psd_tools import PSDImage
from psd_tools.api.layers import TypeLayer
from psd_tools.composite.text import FontMap, MissingFontError, fit_text
from psd_tools.composite.text import fonts as fonts_module
from psd_tools.composite.text.layout import layout
from psd_tools.composite.text.model import TextModel

from ...utils import full_name


@pytest.fixture(autouse=True)
def _no_platform_fonts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fonts_module, "platform_font_dirs", lambda: [])


@pytest.fixture
def fonts(luckiest_guy_path: str) -> FontMap:
    return FontMap(aliases={"ArialMT": luckiest_guy_path})


def _layer(name: str) -> TypeLayer:
    # text-story.psd: "Name" is point text "B.F.B" (ArialMT), "Ability" box text.
    psd = PSDImage.open(full_name("text-story.psd"))
    layer = next(x for x in psd.descendants() if x.name == name)
    assert isinstance(layer, TypeLayer)
    return layer


def _width(layer: TypeLayer, fonts: FontMap) -> float:
    model = TextModel.from_layer(layer)
    lines = layout(model, fonts).lines
    return max(line.width for line in lines) * math.hypot(*model.transform[:2])


def _sizes(layer: TypeLayer) -> list[float]:
    runs = layer.engine_dict["StyleRun"]["RunArray"]
    return [float(r["StyleSheet"]["StyleSheetData"]["FontSize"].value) for r in runs]


def _story_sizes(layer: TypeLayer) -> list[float]:
    blocks = cast(PSDImage, layer._psd)._record.layer_and_mask_information.tagged_blocks
    assert blocks is not None
    story = blocks.get_data(b"Txt2")["1"]["1"][int(layer._data.text_data[b"TextIndex"])]
    return [float(r["0"]["0"]["6"]["1"].value) for r in story["0"]["6"]["0"]]


def _raw(layer: TypeLayer) -> bytes:
    buf = io.BytesIO()
    layer._record.write(buf)
    return buf.getvalue()


def test_text_that_fits_is_left_alone(fonts: FontMap) -> None:
    layer = _layer("Name")
    before = _raw(layer)
    assert fit_text(layer, _width(layer, fonts) + 1, 0.5, fonts=fonts) == 1.0
    assert _raw(layer) == before and not layer._psd.is_updated()


def test_fit_text_shrinks_to_the_width(fonts: FontMap) -> None:
    layer = _layer("Name")
    natural, (size,) = _width(layer, fonts), _sizes(layer)
    scale = layer.fit_text(natural * 0.6, 0.5, fonts=fonts)
    assert scale == pytest.approx(0.6, abs=0.01) and scale <= 0.6
    assert _sizes(layer) == [math.floor(size * scale * 100) / 100]
    assert _story_sizes(layer) == _sizes(layer)  # Photoshop reads the story
    assert natural * 0.59 < _width(layer, fonts) <= natural * 0.6
    assert layer.story_text == layer.text + "\r" and layer.text == "B.F.B"


def test_fit_text_refuses_below_min_scale(fonts: FontMap) -> None:
    layer = _layer("Name")
    before = _raw(layer)
    with pytest.raises(ValueError, match="min_scale 0.5"):
        fit_text(layer, _width(layer, fonts) * 0.4, 0.5, fonts=fonts)
    assert _raw(layer) == before


def test_fit_text_refuses_box_text(fonts: FontMap) -> None:
    with pytest.raises(ValueError, match="box text"):
        fit_text(_layer("Ability"), 10, 0.5, fonts=fonts)


def test_fit_text_needs_the_fonts(luckiest_guy_path: Any) -> None:
    layer = _layer("Name")
    for policy in ("raise", "skip"):
        with pytest.raises(MissingFontError, match="ArialMT"):
            fit_text(layer, 1, 0.5, fonts=FontMap(policy=policy))
