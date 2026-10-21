"""Layer effects on an adjustment layer, drawn from its mask.

``adjustments/exposure_rgb.psd`` is a 200 x 200 RGB document: an opaque
photograph under four exposure layers, each masked to a vertical band.
``Exposure 1`` is masked to the right quarter, (150, 0, 200, 200), and the
band is solid. The effects are forged onto it from ``layer_effects.psd``'s
descriptors, so what is drawn has a closed form.
"""

import copy

import numpy as np
import pytest

from psd_tools.api.adjustments import Exposure
from psd_tools.api.layers import Layer
from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import composite
from psd_tools.composite.composite import _effect_reach, _style_bbox
from psd_tools.constants import BlendMode, Tag
from psd_tools.psd.descriptor import Descriptor, Double, Enumerated
from psd_tools.terminology import Enum, Key, Klass, Type

from ..utils import full_name

_MASK = (150, 0, 200, 200)
_RED = (1.0, 0.0, 0.0)


def _exposure() -> tuple[PSDImage, Exposure]:
    psd = PSDImage.open(full_name("adjustments/exposure_rgb.psd"))
    layer = psd[3]
    assert isinstance(layer, Exposure) and layer.name == "Exposure 1"
    assert layer.bbox == (0, 0, 0, 0)
    assert layer.mask is not None and layer.mask.bbox == _MASK
    mask = layer.numpy("mask")
    assert mask is not None and np.all(mask == 1.0), "the fixture's mask is solid"
    return psd, layer


def _forge(layer: Layer, donor_name: str, klass: bytes) -> Descriptor:
    """Give ``layer`` one effect of ``layer_effects.psd``, Normal at full opacity."""
    donor = PSDImage.open(full_name("layer_effects.psd"))
    source = next(sub for sub in donor.descendants() if sub.name == donor_name)
    block = copy.deepcopy(
        source.tagged_blocks.get_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO)
    )
    layer.tagged_blocks.set_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO, block)
    desc = block[klass]
    desc[Key.Mode] = Enumerated(typeID=Type.BlendMode, enum=Enum.Normal)
    desc[Key.Opacity] = Double(100.0)
    return desc


def _paint(desc: Descriptor, rgb: tuple[float, float, float]) -> None:
    for key, value in zip((b"Rd  ", b"Grn ", b"Bl  "), rgb, strict=True):
        desc[Key.Color][key] = Double(255.0 * value)


def _red_overlay(layer: Layer) -> Descriptor:
    desc = _forge(layer, "Color Fill", Klass.SolidFill)
    _paint(desc, _RED)
    return desc


def _outside(shape: tuple[int, ...]) -> np.ndarray:
    outside = np.ones(shape[:2], dtype=bool)
    outside[_MASK[1] : _MASK[3], _MASK[0] : _MASK[2]] = False
    return outside


def _inside(canvas: np.ndarray) -> np.ndarray:
    return canvas[_MASK[1] : _MASK[3], _MASK[0] : _MASK[2]]


# -- Where the effect goes -------------------------------------------------- #


def test_a_color_overlay_on_an_adjustment_layer_covers_its_mask_and_nothing_else() -> (
    None
):
    """The mask is the adjustment layer's shape, so the overlay fills exactly it.

    A solid Normal overlay at full opacity replaces what is under it, and the
    adjustment is under it; outside the mask the canvas is what the
    adjustment alone left, to the float, alpha included.
    """
    psd, layer = _exposure()
    before, _, alpha_before = composite(psd)
    _red_overlay(layer)
    color, _, alpha = composite(psd)
    assert np.allclose(_inside(color), _RED, atol=1e-6)
    assert np.all(_inside(alpha) == 1.0)
    outside = _outside(color.shape)
    assert np.allclose(color[outside], before[outside], atol=1e-6)
    assert np.array_equal(alpha[outside], alpha_before[outside])


def test_without_an_effect_a_masked_adjustment_paints_nothing() -> None:
    """Alone over nothing, the adjustment adds no alpha; its overlay adds its mask's."""
    psd, layer = _exposure()

    def alone(sub: Layer) -> bool:
        return sub is layer

    _, _, alpha = composite(psd, layer_filter=alone)
    assert not alpha.any()
    _red_overlay(layer)
    color, _, alpha = composite(psd, layer_filter=alone)
    assert np.all(_inside(alpha) == 1.0)
    assert not alpha[_outside(alpha.shape)].any()
    assert np.allclose(_inside(color), _RED, atol=1e-6)


def test_an_overlay_on_an_adjustment_layer_blends_with_the_adjusted_canvas() -> None:
    """An effect's blend mode sees the adjustment, as Photoshop composites it.

    The layer has no paint of its own to blend into, so a Multiply overlay
    multiplies the adjusted canvas: red keeps the red channel and zeroes the
    others.
    """
    psd, layer = _exposure()
    before, _, _ = composite(psd)
    desc = _red_overlay(layer)
    desc[Key.Mode] = Enumerated(typeID=Type.BlendMode, enum=Enum.Multiply)
    color, _, _ = composite(psd)
    assert np.allclose(_inside(color), _inside(before) * _RED, atol=1e-6)


def _without(psd: PSDImage, layer: Layer) -> np.ndarray:
    layer.visible = False
    color, _, _ = composite(psd)
    layer.visible = True
    return color


def test_the_layer_opacity_fades_the_adjustment_and_its_effects_as_one() -> None:
    """At half opacity, half the canvas without the layer and half the overlay.

    As a layer's opacity fades it and its effects together: the solid
    overlay covers the adjustment, so none of the adjustment is left in the
    mix, where fading the two one after the other would leave a quarter.
    """
    psd, layer = _exposure()
    without = _without(psd, layer)
    layer.opacity = 128
    _red_overlay(layer)
    color, _, _ = composite(psd)
    share = 128 / 255
    expected = (1.0 - share) * _inside(without) + share * np.asarray(_RED)
    assert np.allclose(_inside(color), expected, atol=1e-6)


def test_a_blended_effect_is_faded_with_the_adjustment_it_sees() -> None:
    """A Multiply overlay multiplies the whole adjustment, then the layer opacity fades both."""
    psd, layer = _exposure()
    without = _without(psd, layer)
    adjusted, _, _ = composite(psd)
    layer.opacity = 128
    desc = _red_overlay(layer)
    desc[Key.Mode] = Enumerated(typeID=Type.BlendMode, enum=Enum.Multiply)
    color, _, _ = composite(psd)
    share = 128 / 255
    whole = _inside(adjusted) * _RED
    expected = _inside(without) + share * (whole - _inside(without))
    assert np.allclose(_inside(color), expected, atol=1e-6)


# -- Without a mask ---------------------------------------------------------- #


def _unmasked(size: float, style: bytes) -> tuple[PSDImage, Layer, np.ndarray]:
    """``Exposure 1`` alone over the photograph, its mask off, with a red stroke."""
    psd, layer = _exposure()
    for sub in psd:
        sub.visible = sub is psd[0] or sub is layer
    assert layer.mask is not None
    layer.mask.disabled = True
    before, _, _ = composite(psd)
    desc = _forge(layer, "Stroke", Klass.FrameFX)
    desc[Key.Style] = Enumerated(typeID=Type.FrameStyle, enum=style)
    desc[Key.SizeKey] = Double(size)
    _paint(desc, _RED)
    return psd, layer, before


def test_a_stroke_on_an_unmasked_adjustment_layer_follows_the_document_edge() -> None:
    """Without a mask the layer runs to the document's edge, and an inside stroke frames it."""
    psd, layer, before = _unmasked(5.0, Enum.InsetFrame)
    color, _, _ = composite(psd)
    frame = np.ones(color.shape[:2], dtype=bool)
    frame[5:-5, 5:-5] = False
    assert np.allclose(color[frame], _RED, atol=1e-6)
    assert np.allclose(color[~frame], before[~frame], atol=1e-6)


def test_an_outside_stroke_on_an_unmasked_adjustment_layer_falls_off_the_document() -> (
    None
):
    psd, layer, before = _unmasked(5.0, Enum.OutsetFrame)
    color, _, _ = composite(psd)
    assert np.allclose(color, before, atol=1e-6)


def test_an_inner_shadow_on_an_unmasked_adjustment_layer_falls_from_the_document_edge() -> (
    None
):
    psd, layer = _exposure()
    for sub in psd:
        sub.visible = sub is psd[0] or sub is layer
    assert layer.mask is not None
    layer.mask.disabled = True
    before, _, _ = composite(psd)
    desc = _forge(layer, "Inner Shadow", Klass.InnerShadow)
    _paint(desc, (0.0, 0.0, 0.0))
    desc[Key.UseGlobalAngle] = False
    desc[Key.LocalLightingAngle] = Double(180.0)
    desc[Key.Distance] = Double(4.0)
    desc[Key.Blur] = Double(0.0)
    desc[Key.ChokeMatte] = Double(0.0)
    color, _, _ = composite(psd)
    assert np.allclose(color[:, :4], 0.0, atol=1e-6)
    assert np.allclose(color[:, 4:], before[:, 4:], atol=1e-6)


def test_a_gradient_overlay_on_an_adjustment_layer_is_laid_out_on_its_mask() -> None:
    """The ramp spans the mask's box, not the layer's empty box or the canvas.

    Black to white at 0 degrees over the 50 px band: the ramp runs from the
    band's left edge to its right, the same on every row. Laid out on the
    canvas it would start at three quarters white.
    """
    psd, layer = _exposure()
    desc = _forge(layer, "Gradient Overlay", Enum.GradientFill)
    assert desc[Key.Type].enum == Enum.Linear
    assert desc[Key.Alignment] and desc[Key.Scale] == 100.0 and not desc[Key.Reverse]
    desc[Key.Angle] = Double(0.0)
    stops = desc[Key.Gradient][Key.Colors]
    assert [stop[Key.Location] for stop in stops] == [0, 4096]
    for stop, value in zip(stops, (0.0, 1.0), strict=True):
        _paint(stop, (value, value, value))
    color, _, _ = composite(psd)
    band = _inside(color)[:, :, 0]
    assert np.allclose(band, band[:1], atol=1e-6)
    ramp = band[0]
    assert ramp[0] == pytest.approx(0.0, abs=1 / 255)
    assert ramp[-1] == pytest.approx(1.0, abs=1 / 255)
    assert np.all(np.diff(ramp) > 0)
    assert ramp[24] <= 0.5 <= ramp[25]


def test_an_inner_shadow_on_an_adjustment_layer_is_cast_from_its_mask() -> None:
    """The mask is the silhouette the shadow falls into from its edge.

    Lit from the left, a hard black shadow 4 px deep falls into the band
    from its left edge; nothing changes outside the mask or deeper inside.
    """
    psd, layer = _exposure()
    before, _, _ = composite(psd)
    desc = _forge(layer, "Inner Shadow", Klass.InnerShadow)
    _paint(desc, (0.0, 0.0, 0.0))
    desc[Key.UseGlobalAngle] = False
    desc[Key.LocalLightingAngle] = Double(180.0)
    desc[Key.Distance] = Double(4.0)
    desc[Key.Blur] = Double(0.0)
    desc[Key.ChokeMatte] = Double(0.0)
    color, _, _ = composite(psd)
    left = _MASK[0]
    assert np.allclose(color[:, left : left + 4], 0.0, atol=1e-6)
    assert np.allclose(color[:, left + 4 :], before[:, left + 4 :], atol=1e-6)
    outside = _outside(color.shape)
    assert np.allclose(color[outside], before[outside], atol=1e-6)


# -- The boxes -------------------------------------------------------------- #


def test_the_style_box_of_an_adjustment_layer_is_its_mask() -> None:
    psd, layer = _exposure()
    assert _style_bbox(layer) == _MASK
    assert _style_bbox(psd[0]) == psd[0].bbox
    assert layer.mask is not None
    layer.mask.disabled = True
    assert _style_bbox(layer) == psd.viewbox


def test_a_mask_that_reveals_its_outside_gives_the_canvas_as_the_style_box() -> None:
    psd = PSDImage.open(full_name("adjustments/adjustment_backdrop_test.psd"))
    layer = next(sub for sub in psd.descendants() if sub.name == "Curves 1 copy")
    assert layer.mask is not None and layer.mask.background_color == 255
    assert layer.mask.bbox == (100, 0, 200, 200)
    assert _style_bbox(layer) == psd.viewbox


def test_a_styled_adjustment_layer_reaches_as_far_as_its_mask() -> None:
    """Only a styled one: an adjustment without a style has nothing to measure."""
    _, layer = _exposure()
    assert _effect_reach(layer) == (0, 0, 0, 0)
    _red_overlay(layer)
    assert _effect_reach(layer) == _MASK


def test_an_isolated_group_of_one_styled_adjustment_composites_on_the_mask() -> None:
    """A Normal group's canvas is measured from its children, which here have no box."""
    psd, layer = _exposure()
    before, _, alpha_before = composite(psd)
    _red_overlay(layer)
    group = psd.create_group([layer], name="Isolated")
    group.blend_mode = BlendMode.NORMAL
    color, _, alpha = composite(psd)
    assert np.allclose(_inside(color), _RED, atol=1e-6)
    assert np.all(_inside(alpha) == 1.0)
    outside = _outside(color.shape)
    assert np.allclose(color[outside], before[outside], atol=1e-6)
    assert np.array_equal(alpha[outside], alpha_before[outside])
