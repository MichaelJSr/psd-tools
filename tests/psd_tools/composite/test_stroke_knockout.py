"""The inner half of a stroke effect replaces the layer under it.

A Normal stroke below full opacity shows what is beneath the layer through
its inner band, not the layer's own paint.
"""

import numpy as np
import pytest

from psd_tools.api.layers import Layer
from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import composite
from psd_tools.psd.descriptor import Double, Enumerated

from ..utils import full_name

pytest.importorskip("scipy")

_RED = [1.0, 0.0, 0.0]
_BLUE = [0.0, 0.0, 1.0]
_OPACITY = 0.4


def _square(
    psd: PSDImage, blend_mode: bytes = b"Nrml", opacity: float = _OPACITY
) -> Layer:
    """``Size 3``: a blue 16x16 square under a red 3 px stroke."""
    layer = next(x for x in psd if x.name == "Size 3")
    desc = layer.effects[0].descriptor
    desc[b"Opct"] = Double(100.0 * opacity)
    desc[b"Md  "] = Enumerated(typeID=b"BlnM", enum=blend_mode)
    return layer


def test_a_faded_inset_stroke_shows_the_backdrop_not_the_layer() -> None:
    psd = PSDImage.open(full_name("effects/inset-stroke-sizes.psd"))
    layer = _square(psd)
    assert layer.bbox == (80, 16, 96, 32)
    color, _, alpha = composite(psd)
    row = 24
    np.testing.assert_allclose(color[row, 80:83], [_RED] * 3, atol=1e-6)
    np.testing.assert_allclose(alpha[row, 80:83, 0], [_OPACITY] * 3, atol=1e-6)
    np.testing.assert_allclose(color[row, 83], _BLUE, atol=1e-6)
    assert alpha[row, 83, 0] == pytest.approx(1.0)


def test_a_faded_centred_stroke_knocks_out_only_its_inner_half() -> None:
    """``center-stroke-sizes.psd``: a 2 px centred stroke, 1 px each side."""
    psd = PSDImage.open(full_name("effects/center-stroke-sizes.psd"))
    layer = next(x for x in psd if x.name == "Size 2")
    assert layer.bbox == (48, 16, 64, 32)
    layer.effects[0].descriptor[b"Opct"] = Double(100.0 * _OPACITY)
    color, _, alpha = composite(psd)
    row = 24
    np.testing.assert_allclose(alpha[row, 46, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(color[row, 47:49], [_RED] * 2, atol=1e-6)
    np.testing.assert_allclose(alpha[row, 47:49, 0], [_OPACITY] * 2, atol=1e-6)
    np.testing.assert_allclose(color[row, 49], _BLUE, atol=1e-6)
    assert alpha[row, 49, 0] == pytest.approx(1.0)


def test_a_faded_multiply_inset_stroke_still_blends_with_the_layer() -> None:
    """Pins the fork's rendering: Photoshop's is not calibrated for this."""
    psd = PSDImage.open(full_name("effects/inset-stroke-sizes.psd"))
    _square(psd, b"Mltp")
    color, _, alpha = composite(psd)
    # Red multiplied onto blue is black, at the stroke's opacity.
    np.testing.assert_allclose(color[24, 81], [0.0, 0.0, 1.0 - _OPACITY], atol=1e-6)
    assert alpha[24, 81, 0] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "viewport", [None, (0, 0, 8, 48)], ids=["document", "edge-viewport"]
)
def test_an_inset_stroke_follows_a_layer_past_the_canvas(
    viewport: tuple[int, int, int, int] | None,
) -> None:
    """The square dragged half off the left edge keeps its stroke off there.

    The canvas edge is not an edge of the layer: the column at x = 0 is the
    square's interior, and only its top, bottom and right bands are stroked.
    The stroke is left opaque so that this holds apart from the knockout.
    """
    psd = PSDImage.open(full_name("effects/inset-stroke-sizes.psd"))
    layer = _square(psd, opacity=1.0)
    layer.offset = (-8, 16)
    assert layer.bbox == (-8, 16, 8, 32)
    color, _, alpha = composite(psd, viewport=viewport)
    row = 24
    np.testing.assert_allclose(color[row, 0:5], [_BLUE] * 5, atol=1e-6)
    np.testing.assert_allclose(color[row, 5:8], [_RED] * 3, atol=1e-6)
    np.testing.assert_allclose(alpha[row, 0:8, 0], [1.0] * 8, atol=1e-6)
    np.testing.assert_allclose(color[16:20, 0], [_RED] * 3 + [_BLUE], atol=1e-6)


def test_a_faded_inset_stroke_matches_photoshop() -> None:
    """``layer_comps.psb``'s ``Фигура 5``: a 2 px inset stroke at 90% Normal.

    Photoshop's own save, over a yellow backdrop the band lets through: the
    preview's green and blue differ there, which a stroke over the layer's
    own paint -- a red fill under a gradient overlay -- could not give.
    """
    psd = PSDImage.open(full_name("layer_comps.psb"))
    layer = next(x for x in psd.descendants() if x.name == "Фигура 5")
    stroke = next(e for e in layer.effects if e.name == "Stroke")
    assert stroke.opacity == 90.0
    row, column = 163, 113
    preview = psd.topil()
    rendered = psd.composite(force=True, ignore_preview=True)
    assert preview is not None and rendered is not None
    expected = np.asarray(preview.convert("RGB"), dtype=np.float32)[row, column] / 255
    actual = np.asarray(rendered.convert("RGB"), dtype=np.float32)[row, column] / 255
    assert expected[1] - expected[2] > 0.08, "the backdrop shows through"
    np.testing.assert_allclose(actual, expected, atol=1.5 / 255)
