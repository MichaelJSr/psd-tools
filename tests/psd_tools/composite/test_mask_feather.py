"""A user mask's Properties feather: a gaussian blur of the whole mask."""

import math
import sys
import threading
import time
from collections import OrderedDict

import numpy as np
import pytest
from PIL import Image

from psd_tools.api.mask import Mask
from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import composite
from psd_tools.composite.composite import _style_bbox
from psd_tools.psd.layer_and_mask import MaskParameters

from ..utils import full_name

pytest.importorskip("scipy")
from scipy import ndimage  # type: ignore[import-untyped]

_module = sys.modules["psd_tools.composite.composite"]

_SIGMA = 3.0


def _edge(t: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """A solid run ``[lo, hi)`` blurred: the gaussian's mass over it at pixel centres."""

    def phi(x: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 + np.vectorize(math.erf)(x / (_SIGMA * math.sqrt(2.0))))

    return phi(t + 0.5 - lo) - phi(t + 0.5 - hi)


def _square() -> tuple[PSDImage, Mask]:
    """An opaque layer under a solid 20 px mask at (22, 22)."""
    psd = PSDImage.new("RGB", (64, 64))
    layer = psd.create_pixel_layer(Image.new("RGB", (64, 64), (255, 0, 0)))
    return psd, layer.create_mask(Image.new("L", (20, 20), 255), top=22, left=22)


def test_a_feathered_mask_is_its_box_blurred_whole() -> None:
    """The layer's alpha is the mask's box blurred, held at 8-bit levels.

    The blur reaches past the mask's box, and a box render reads the whole
    render's values. Density goes on after the feather.
    """
    psd, mask = _square()
    mask._data.parameters = MaskParameters(user_mask_feather=_SIGMA)

    _, _, alpha = composite(psd, force=True)
    t = np.arange(64, dtype=np.float64)
    expected = np.outer(_edge(t, 22, 42), _edge(t, 22, 42))
    assert np.allclose(alpha[..., 0], expected, atol=2e-3 + 0.5 / 255)
    assert alpha[30, 19, 0] > 0.1

    box = (10, 15, 30, 40)
    _, _, part = composite(psd, viewport=box, force=True)
    assert np.array_equal(part, alpha[15:40, 10:30])

    mask._data.parameters = MaskParameters(
        user_mask_density=128, user_mask_feather=_SIGMA
    )
    _, _, dense = composite(psd, force=True)
    share = 128 / 255
    assert np.allclose(dense, share * alpha + (1.0 - share), atol=1 / 255)


@pytest.mark.parametrize("feather", [1e308, 900.0])
def test_a_forged_or_too_wide_feather_leaves_the_mask_as_stored(feather: float) -> None:
    psd, mask = _square()
    _, _, hard = composite(psd, force=True)
    mask._data.parameters = MaskParameters(user_mask_feather=feather)
    _, _, alpha = composite(psd, force=True)
    assert np.array_equal(alpha, hard)


def test_a_feathered_adjustment_mask_grows_its_style_box() -> None:
    psd = PSDImage.open(full_name("adjustments/exposure_rgb.psd"))
    layer = psd[3]
    assert layer.mask is not None and _style_bbox(layer) == (150, 0, 200, 200)
    layer.mask._data.parameters = MaskParameters(user_mask_feather=2.0)
    assert _style_bbox(layer) == (142, -8, 208, 208)


@pytest.fixture
def blurs(monkeypatch: pytest.MonkeyPatch) -> list:
    """An empty memo; the gaussian calls, counted."""
    calls: list = []
    real = ndimage.gaussian_filter

    def counted(*args: object, **kwargs: object) -> np.ndarray:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(ndimage, "gaussian_filter", counted)
    monkeypatch.setattr(_module, "_FEATHERED", OrderedDict())
    return calls


def test_a_feathered_mask_is_blurred_once_per_content(blurs: list) -> None:
    psd, mask = _square()
    mask._data.parameters = MaskParameters(user_mask_feather=_SIGMA)
    _, _, alpha = composite(psd, force=True)
    other, other_mask = _square()
    other_mask._data.parameters = MaskParameters(user_mask_feather=_SIGMA)
    _, _, part = composite(other, viewport=(10, 15, 30, 40), force=True)
    assert np.array_equal(part, alpha[15:40, 10:30]) and len(blurs) == 1

    _module._FEATHERED.clear()
    _, _, again = composite(psd, force=True)
    assert np.array_equal(again, alpha) and len(blurs) == 2

    other_mask._data.parameters = MaskParameters(user_mask_feather=2 * _SIGMA)
    composite(other, force=True)
    assert len(blurs) == 3


def test_the_memo_evicts_past_its_cap(
    blurs: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    def make() -> np.ndarray:
        return np.zeros((32, 32, 1), np.float32)  # 4 KiB

    monkeypatch.setattr(_module, "_FEATHERED_MAX", 6 << 10)
    first = _module._feathered(("a",), make)
    assert not first.flags.writeable and _module._feathered(("a",), make) is first
    _module._feathered(("b",), make)
    assert list(_module._FEATHERED) == [("b",)]


def test_threads_wait_for_one_blur() -> None:
    started, go, calls = threading.Event(), threading.Event(), []

    def slow() -> np.ndarray:
        calls.append(1)
        started.set()
        go.wait(5)
        return np.ones((2, 2, 1), np.float32)

    key = ("threads", time.time())
    results: list = []
    first = threading.Thread(
        target=lambda: results.append(_module._feathered(key, slow))
    )
    first.start()
    started.wait(5)
    second = threading.Thread(
        target=lambda: results.append(_module._feathered(key, slow))
    )
    second.start()
    time.sleep(0.1)
    go.set()
    first.join(5)
    second.join(5)
    assert len(calls) == 1 and results[0] is results[1]
