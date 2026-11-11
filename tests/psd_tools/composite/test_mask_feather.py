"""A user mask's Properties feather: a gaussian blur of the whole mask."""

import math

import numpy as np
import pytest
from PIL import Image

from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import composite
from psd_tools.composite.composite import _style_bbox
from psd_tools.psd.layer_and_mask import MaskParameters

from ..utils import full_name

pytest.importorskip("scipy")

_SIGMA = 3.0


def _edge(t: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """A solid run ``[lo, hi)`` blurred: the gaussian's mass over it at pixel centres."""

    def phi(x: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 + np.vectorize(math.erf)(x / (_SIGMA * math.sqrt(2.0))))

    return phi(t + 0.5 - lo) - phi(t + 0.5 - hi)


def test_a_feathered_mask_is_its_box_blurred_whole() -> None:
    """An opaque layer under a solid 20 px mask: its alpha is the box blurred.

    The blur reaches past the mask's box, and a box render reads the whole
    render's values. Density goes on after the feather.
    """
    psd = PSDImage.new("RGB", (64, 64))
    layer = psd.create_pixel_layer(Image.new("RGB", (64, 64), (255, 0, 0)))
    mask = layer.create_mask(Image.new("L", (20, 20), 255), top=22, left=22)
    mask._data.parameters = MaskParameters(user_mask_feather=_SIGMA)

    _, _, alpha = composite(psd, force=True)
    t = np.arange(64, dtype=np.float64)
    expected = np.outer(_edge(t, 22, 42), _edge(t, 22, 42))
    assert np.allclose(alpha[..., 0], expected, atol=2e-3)
    assert alpha[30, 19, 0] > 0.1

    box = (10, 15, 30, 40)
    _, _, part = composite(psd, viewport=box, force=True)
    assert np.array_equal(part, alpha[15:40, 10:30])

    mask._data.parameters = MaskParameters(
        user_mask_density=128, user_mask_feather=_SIGMA
    )
    _, _, dense = composite(psd, force=True)
    share = 128 / 255
    assert np.allclose(dense, share * alpha + (1.0 - share), atol=1e-6)


def test_a_feathered_adjustment_mask_grows_its_style_box() -> None:
    psd = PSDImage.open(full_name("adjustments/exposure_rgb.psd"))
    layer = psd[3]
    assert layer.mask is not None and _style_bbox(layer) == (150, 0, 200, 200)
    layer.mask._data.parameters = MaskParameters(user_mask_feather=2.0)
    assert _style_bbox(layer) == (142, -8, 208, 208)
