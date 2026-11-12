"""A knockout pass-through group replaces what lies under it (#707).

``knockout-passthrough-masks.psb`` is a stat bar's embedded document as
Photoshop 2023 saved it: 'Dark', an opaque grey row, under 'Light', a
pass-through group with shallow knockout and Transparency Shapes Layer off
holding a half-transparent yellow row, its mask hiding x >= 200. With the
mask moved 100 px left (its comp '2') Photoshop drew yellow at alpha 128 for
x < 100 and the grey row from there on.
"""

import numpy as np

from psd_tools.api.layers import Layer
from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import composite
from psd_tools.constants import Tag

from ..utils import full_name

_YELLOW = [250, 200, 50, 128]
_GREY = [40, 40, 40, 255]


def _bar() -> tuple[PSDImage, Layer]:
    psd = PSDImage.open(full_name("transparency/knockout-passthrough-masks.psb"))
    return psd, next(layer for layer in psd if layer.name == "Light")


def _row(psd: PSDImage) -> np.ndarray:
    color, _, alpha = composite(psd)
    return np.round(np.concatenate([color, alpha], axis=2)[30] * 255).astype(int)


def test_the_mask_shapes_the_knockout() -> None:
    psd, _ = _bar()
    row = _row(psd)
    np.testing.assert_allclose(row[:200], [_YELLOW] * 200, atol=1)
    np.testing.assert_allclose(row[200:], [_GREY] * 200, atol=1)


def test_a_moved_mask_draws_photoshops_comp() -> None:
    psd, light = _bar()
    mask = light.mask
    assert mask is not None
    mask.offset = (mask.left - 100, mask.top)
    row = _row(psd)
    np.testing.assert_allclose(row[:100], [_YELLOW] * 100, atol=1)
    np.testing.assert_allclose(row[100:], [_GREY] * 300, atol=1)


def test_with_shapes_layer_on_the_content_knocks_out() -> None:
    """Half-transparent pixels are half the shape: half the grey stays."""
    psd, light = _bar()
    light.tagged_blocks.set_data(Tag.TRANSPARENCY_SHAPES_LAYER, 1)
    row = _row(psd)
    share = 128 / 255
    mixed = [
        round(250 * share + 40 * (1 - share)),
        round(200 * share + 40 * (1 - share)),
        round(50 * share + 40 * (1 - share)),
        255,
    ]
    np.testing.assert_allclose(row[:200], [mixed] * 200, atol=1)
    np.testing.assert_allclose(row[200:], [_GREY] * 200, atol=1)
