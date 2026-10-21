"""Tests of :py:meth:`TypeLayer.rasterize`.

The text engine itself is tested under ``tests/psd_tools/composite/text``.
"""

import builtins
import os
from typing import Any, cast

import numpy as np
import pytest

pytest.importorskip("freetype")
pytest.importorskip("uharfbuzz")

from psd_tools import PSDImage
from psd_tools.api.layers import TypeLayer
from psd_tools.composite.text import FontMap, render_type_layer
from psd_tools.composite.text import fonts as fonts_module
from psd_tools.constants import ChannelID

from ..utils import full_name

pytestmark = pytest.mark.text

LUCKIEST_GUY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "fonts",
    "LuckiestGuy-Regular.ttf",
)


@pytest.fixture(autouse=True)
def _no_platform_fonts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PSD_TOOLS_FONT_DIRS", raising=False)
    monkeypatch.setattr(fonts_module, "platform_font_dirs", lambda: [])


@pytest.fixture
def luckiest_guy_path() -> str:
    return LUCKIEST_GUY


@pytest.fixture
def layer() -> TypeLayer:
    psd = PSDImage.open(full_name("text.psd"))
    return next(x for x in psd.descendants() if isinstance(x, TypeLayer))


@pytest.fixture
def arial_as_luckiest_guy(luckiest_guy_path: str) -> FontMap:
    return FontMap(aliases={"ArialMT": luckiest_guy_path})


def _quantised(rgba: np.ndarray) -> np.ndarray:
    return np.round(rgba * 255) / 255


def test_rasterize_round_trip(
    layer: TypeLayer, arial_as_luckiest_guy: FontMap, tmp_path: Any
) -> None:
    psd = cast(PSDImage, layer._psd)
    layer.set_text("Hello\rworld")
    report = layer.rasterize(fonts=arial_as_luckiest_guy)
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


def test_rasterize_keeps_the_mask(luckiest_guy_path: str, tmp_path: Any) -> None:
    psd = PSDImage.open(full_name("transparency/fill-opacity.psd"))
    layer = next(x for x in psd.descendants() if isinstance(x, TypeLayer))
    assert layer.mask is not None
    before = layer.mask.topil()
    assert before is not None
    bbox = layer.mask.bbox
    layer.rasterize(fonts=FontMap(aliases={"ArialMT": luckiest_guy_path}))
    ids = [info.id for info in layer._record.channel_info]
    assert ids.count(ChannelID.USER_LAYER_MASK) == 1
    assert len(ids) == len(layer._channels)
    assert layer.mask is not None and layer.mask.bbox == bbox
    after = layer.mask.topil()
    assert after is not None and after.tobytes() == before.tobytes()
    psd.save(tmp_path / "masked.psd")
    again = PSDImage.open(tmp_path / "masked.psd")
    reopened = next(x for x in again.descendants() if isinstance(x, TypeLayer))
    assert reopened.mask is not None and reopened.mask.bbox == bbox
    saved = reopened.mask.topil()
    assert saved is not None and saved.tobytes() == before.tobytes()
    assert reopened.composite() is not None


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


def test_rasterize_keeps_float_precision_in_a_32_bit_document(tmp_path: Any) -> None:
    psd = PSDImage.open(full_name("pen-text.psd"))
    layer = next(x for x in psd.descendants() if isinstance(x, TypeLayer))
    fonts = FontMap(aliases={name: LUCKIEST_GUY for name in _font_names(layer)})
    layer.rasterize(fonts=fonts)
    out = render_type_layer(layer, fonts)
    assert out is not None
    rgba, bbox = out
    assert tuple(layer.bbox) == bbox
    shape = layer.numpy("shape")
    assert shape is not None
    # The stored alpha is the render's own float, not an 8-bit widening.
    np.testing.assert_array_equal(shape[..., 0], rgba[..., 3])
    color = layer.numpy("color")
    assert color is not None
    ink = rgba[..., 3] > 0
    np.testing.assert_array_equal(color[ink], rgba[..., :3][ink])
    psd.save(tmp_path / "deep.psd")
    again = PSDImage.open(tmp_path / "deep.psd")
    reopened = next(x for x in again.descendants() if isinstance(x, TypeLayer))
    shape2 = reopened.numpy("shape")
    assert shape2 is not None
    np.testing.assert_array_equal(shape2[..., 0], rgba[..., 3])


def _font_names(layer: TypeLayer) -> list[str]:
    from psd_tools.composite.text import font_names  # noqa: PLC0415

    return font_names(layer)
