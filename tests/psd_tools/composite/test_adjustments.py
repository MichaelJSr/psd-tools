import colorsys
import copy
import importlib
import logging
import struct
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pytest
from PIL import Image

from psd_tools.api import adjustments
from psd_tools.api.layers import Group, Layer
from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import paint
from psd_tools.composite.adjustments import (
    _brightness_curve,
    _huesaturation,
    _lut_domain,
    _rows_from_rgb,
    apply_blackandwhite,
    apply_brightnesscontrast,
    apply_channelmixer,
    apply_colorbalance,
    apply_curves,
    apply_exposure,
    apply_gradientmap,
    apply_huesaturation,
    apply_levels,
    apply_posterize,
    apply_threshold,
    apply_vibrance,
)
from psd_tools.composite.blend import rgb2hsl
from psd_tools.composite.paint import _from_rgb
from psd_tools.constants import BlendMode, ColorMode, Tag
from psd_tools.psd.adjustments import ChannelMixer as ChannelMixerData
from psd_tools.psd.adjustments import ColorBalance as ColorBalanceData
from psd_tools.psd.adjustments import ColorStop, TransparencyStop
from psd_tools.psd.adjustments import Curves as CurvesData
from psd_tools.psd.adjustments import GradientMap as GradientMapData
from psd_tools.psd.adjustments import HueSaturation as HueSaturationData
from psd_tools.psd.descriptor import Descriptor, Integer

from ..utils import full_name
from .test_composite import check_composite_quality, check_icc_composite_quality

# The package's composite() shadows the submodule of the same name.
composite_module = importlib.import_module("psd_tools.composite.composite")

logger = logging.getLogger(__name__)


def _applied(out: np.ndarray | None) -> np.ndarray:
    """``out``, which an adjustment returns None for when it does not render."""
    assert out is not None
    return out


# Tolerances used for MSE computation over test images (shape (200, 200, C), values in [0,1])

LOOSE = 5e-4  # tolerance for approximate adjustment algorithms which can be improved (e.g. hue/saturation)
ACCURATE = 5e-5  # tolerance for accurate adjustment algorithms (e.g. curves)
STRICT = 5e-6  # tolerance for precise adjustment algorithms (e.g. threshold)


# Brightness & Contrast
brightnesscontrast_tests = {
    "brightnesscontrast": ACCURATE,
    "brightnesscontrast_legacy": LOOSE,
}
brightnesscontrast_colorspaces = ["rgb", "cmyk", "grayscale"]


@pytest.mark.parametrize("name", brightnesscontrast_tests.keys())
@pytest.mark.parametrize("colormode", brightnesscontrast_colorspaces)
def test_brightnesscontrast_composite_icc(name: str, colormode: str) -> None:
    filename = f"adjustments/{name}_{colormode}"
    if name.endswith("legacy"):
        pytest.xfail(
            "legacy adjustment layers recognized as PixelLayers with no bounding box"
        )
    check_icc_composite_quality(filename, brightnesscontrast_tests[name])


@pytest.mark.parametrize("name", brightnesscontrast_tests.keys())
@pytest.mark.parametrize("colormode", brightnesscontrast_colorspaces)
def test_brightnesscontrast_composite_error(name: str, colormode: str) -> None:
    filename = f"adjustments/{name}_{colormode}.psd"
    if name.endswith("legacy"):
        pytest.xfail(
            "legacy adjustment layers recognized as PixelLayers with no bounding box"
        )
    check_composite_quality(filename, brightnesscontrast_tests[name], False)


def _brightness_contrast_layer(
    brightness: int, contrast: int
) -> adjustments.BrightnessContrast:
    """The fixture's first non-legacy layer with the given settings."""
    psd = PSDImage.open(full_name("adjustments/brightnesscontrast_rgb.psd"))
    layer = next(
        layer
        for layer in psd.descendants()
        if isinstance(layer, adjustments.BrightnessContrast)
    )
    data = Descriptor()
    data[b"Brgh"] = Integer(brightness)
    data[b"Cntr"] = Integer(contrast)
    layer._data = data
    return layer


# Levels Photoshop exports through one non-legacy layer:
# (brightness, contrast, input level, output level).
_PHOTOSHOP_BRIGHTNESS_CONTRAST = [
    (-150, 0, 32, 12),
    (-150, 0, 128, 50),
    (-150, 0, 192, 79),
    (-33, 0, 64, 52),
    (-33, 0, 128, 104),
    (-33, 0, 192, 161),
    (-120, 70, 64, 18),
    (-120, 70, 128, 43),
    (-120, 70, 192, 82),
    (-51, -50, 64, 58),
    (-51, -50, 128, 103),
    (-51, -50, 192, 141),
    (-95, -50, 64, 45),
    (-95, -50, 128, 82),
    (0, -50, 64, 76),
    (0, -50, 192, 180),
    (4, 100, 64, 41),
    (4, 100, 128, 134),
    (-31, 47, 64, 42),
    (-31, 47, 192, 171),
    (23, 70, 64, 57),
    (23, 70, 192, 229),
    (150, -21, 32, 87),
    (150, -21, 96, 210),
]
# The curve is fitted, not exact: inputs up to 192 may be off by this many levels.
_BRIGHTNESS_CONTRAST_LEVELS = 2


@pytest.mark.parametrize(
    ("brightness", "contrast", "level", "expected"), _PHOTOSHOP_BRIGHTNESS_CONTRAST
)
def test_brightnesscontrast_matches_photoshop_levels(
    brightness: int, contrast: int, level: int, expected: int
) -> None:
    layer = _brightness_contrast_layer(brightness, contrast)
    image = np.full((1, 1, 3), level / 255, dtype=np.float32)
    out = _applied(apply_brightnesscontrast(image, ColorMode.RGB, layer))
    assert out * 255 == pytest.approx(
        np.full((1, 1, 3), expected), abs=_BRIGHTNESS_CONTRAST_LEVELS
    )


def test_brightnesscontrast_at_zero_leaves_the_image_unchanged() -> None:
    """Values between two levels are not truncated to the lower one."""
    layer = _brightness_contrast_layer(0, 0)
    image = np.linspace(0, 1, 997, dtype=np.float32).reshape(1, -1, 1)
    image = np.repeat(image, 3, axis=2)
    out = _applied(apply_brightnesscontrast(image, ColorMode.RGB, layer))
    assert np.allclose(out, image, atol=1e-6)


@pytest.mark.parametrize("brightness", [-150, -60, -1, 1, 60, 150])
def test_brightness_and_its_negative_are_inverse_curves(brightness: int) -> None:
    t = _lut_domain(256)
    there = _brightness_curve(brightness, t).astype(np.float32)
    assert _brightness_curve(-brightness, there) == pytest.approx(t, abs=1e-4)


@pytest.mark.parametrize("brightness", range(-150, 151, 25))
@pytest.mark.parametrize("contrast", [-50, 0, 50, 100])
def test_brightnesscontrast_keeps_black_white_and_order(
    brightness: int, contrast: int
) -> None:
    layer = _brightness_contrast_layer(brightness, contrast)
    ramp = np.repeat(_lut_domain(256).reshape(1, -1, 1), 3, axis=2)
    out = _applied(apply_brightnesscontrast(ramp, ColorMode.RGB, layer))[0, :, 0]
    assert out[0] == pytest.approx(0.0, abs=1e-6)
    assert out[-1] == pytest.approx(1.0, abs=1e-6)
    assert (np.diff(out) >= 0).all()


@pytest.mark.parametrize("brightness", [-1, 1])
def test_a_brightness_of_one_moves_no_level_by_much_more_than_one(
    brightness: int,
) -> None:
    """The shoulder flattens to the identity with the gain: no step at zero."""
    t = _lut_domain(256)
    assert _brightness_curve(brightness, t) == pytest.approx(t, abs=1.5 / 255)


# Levels
levels_tests = {
    "levels": ACCURATE,
}
levels_colorspaces = ["rgb", "cmyk", "grayscale"]


@pytest.mark.parametrize("name", levels_tests.keys())
@pytest.mark.parametrize("colormode", levels_colorspaces)
def test_levels_composite_icc(name: str, colormode: str) -> None:
    filename = f"adjustments/{name}_{colormode}"
    check_icc_composite_quality(filename, levels_tests[name])


@pytest.mark.parametrize("name", levels_tests.keys())
@pytest.mark.parametrize("colormode", levels_colorspaces)
def test_levels_composite_error(name: str, colormode: str) -> None:
    filename = f"adjustments/{name}_{colormode}.psd"
    check_composite_quality(filename, levels_tests[name], False)


# Curves
curves_tests = {
    "curves": ACCURATE,
}
curves_colorspaces = ["rgb", "cmyk", "grayscale"]


@pytest.mark.parametrize("name", curves_tests.keys())
@pytest.mark.parametrize("colormode", curves_colorspaces)
def test_curves_composite_icc(name: str, colormode: str) -> None:
    filename = f"adjustments/{name}_{colormode}"
    check_icc_composite_quality(filename, curves_tests[name])


@pytest.mark.parametrize("name", curves_tests.keys())
@pytest.mark.parametrize("colormode", curves_colorspaces)
def test_curves_composite_error(name: str, colormode: str) -> None:
    filename = f"adjustments/{name}_{colormode}.psd"
    check_composite_quality(filename, curves_tests[name], False)


def test_curves_without_extra_records_leaves_the_image_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = PSDImage.open(full_name("fill_adjustments.psd"))[6]
    assert isinstance(layer, adjustments.Curves)
    monkeypatch.setattr(layer, "_data", CurvesData(version=4))
    image = np.zeros((2, 2, 3), dtype=np.float32)
    assert apply_curves(image, ColorMode.RGB, layer) is image


@pytest.mark.parametrize(
    "index, apply",
    [
        (5, apply_levels),
        (7, apply_exposure),
        (8, apply_vibrance),
        (9, apply_huesaturation),
        (16, apply_posterize),
        (17, apply_threshold),
    ],
)
def test_an_absent_adjustment_block_leaves_the_image_alone(
    monkeypatch: pytest.MonkeyPatch, index: int, apply: Callable[..., np.ndarray]
) -> None:
    layer = PSDImage.open(full_name("fill_adjustments.psd"))[index]
    monkeypatch.setattr(layer, "_data", None)
    image = np.zeros((2, 2, 3), dtype=np.float32)
    assert apply(image, ColorMode.RGB, layer) is image


@pytest.mark.parametrize(
    "index, apply",
    [
        (10, apply_colorbalance),
        (11, apply_blackandwhite),
        (13, apply_channelmixer),
        (19, apply_gradientmap),
    ],
)
def test_an_absent_adjustment_block_is_not_rendered(
    monkeypatch: pytest.MonkeyPatch,
    index: int,
    apply: Callable[..., np.ndarray | None],
) -> None:
    layer = PSDImage.open(full_name("fill_adjustments.psd"))[index]
    monkeypatch.setattr(layer, "_data", None)
    image = np.zeros((2, 2, 3), dtype=np.float32)
    assert apply(image, ColorMode.RGB, layer) is None


@pytest.mark.parametrize(
    "index, apply",
    [
        (8, apply_vibrance),
        (10, apply_colorbalance),
        (11, apply_blackandwhite),
        (13, apply_channelmixer),
    ],
)
def test_an_rgb_only_adjustment_is_not_rendered_on_grayscale(
    index: int, apply: Callable[..., np.ndarray | None]
) -> None:
    layer = PSDImage.open(full_name("fill_adjustments.psd"))[index]
    image = np.full((2, 2, 1), 0.3, dtype=np.float32)
    assert apply(image, ColorMode.GRAYSCALE, layer) is None


def test_an_unrendered_adjustment_leaves_the_composite_alone_under_multiply() -> None:
    """A noise gradient map is not rendered, so it does not multiply the canvas by itself."""
    psd = PSDImage.open(full_name("fill_adjustments.psd"))
    layer = psd[19]
    assert isinstance(layer, adjustments.GradientMap) and not layer.visible
    expected = psd.composite(force=True)
    layer.visible = True
    layer.blend_mode = BlendMode.MULTIPLY
    layer._data = _gradient_map_data([(0, 50, _BLACK), (4096, 50, _WHITE)], mode=1)
    assert psd.composite(force=True).tobytes() == expected.tobytes()


def test_an_unrendered_adjustment_still_composites_its_clip_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Its clip layers go over the unadjusted content, as over a neutral one.

    ``adjustment_clipping.psd``'s ``Curves 3`` carries the clipped ``Curves
    4``. With ``Curves 3`` not rendered the document reads as with it neutral;
    hiding it hides ``Curves 4`` too, which reads differently.
    """
    psd = PSDImage.open(full_name("adjustments/adjustment_clipping.psd"))
    carrier = next(layer for layer in psd if layer.name == "Curves 3")
    assert [layer.name for layer in carrier.clip_layers] == ["Curves 4"]
    apply_curves = composite_module.ADJUSTMENT_FUNC["curves"]

    def made(result: Callable[[np.ndarray], np.ndarray | None]) -> Any:
        def apply(img: np.ndarray, colormode: ColorMode, layer: Layer) -> Any:
            if layer is carrier:
                return result(img)
            return apply_curves(img, colormode, layer)

        monkeypatch.setitem(composite_module.ADJUSTMENT_FUNC, "curves", apply)
        return psd.composite(force=True)

    neutral = made(lambda img: img)
    assert made(lambda img: None).tobytes() == neutral.tobytes()
    carrier.visible = False
    assert made(lambda img: None).tobytes() != neutral.tobytes()


def test_an_absent_adjustment_block_leaves_the_composite_alone_under_multiply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    psd = PSDImage.open(full_name("fill_adjustments.psd"))
    layer = psd[5]
    assert isinstance(layer, adjustments.Levels)
    layer.visible = False
    expected = psd.composite(force=True)
    layer.visible = True
    layer.blend_mode = BlendMode.MULTIPLY
    monkeypatch.setattr(layer, "_data", None)
    assert psd.composite(force=True).tobytes() == expected.tobytes()


def test_a_neutral_adjustment_still_blends_under_multiply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    psd = PSDImage.open(full_name("fill_adjustments.psd"))
    layer = psd[9]
    assert isinstance(layer, adjustments.HueSaturation)
    layer.visible = False
    hidden = psd.composite(force=True)
    layer.visible = True
    layer.blend_mode = BlendMode.MULTIPLY
    neutral = HueSaturationData(enable=0, colorization=(0, 0, 0), master=(0, 0, 0))
    monkeypatch.setattr(layer, "_data", neutral)
    assert psd.composite(force=True).tobytes() != hidden.tobytes()


# Exposure
exposure_tests = {
    "exposure": ACCURATE,
}
exposure_colorspaces = ["rgb", "grayscale"]


@pytest.mark.parametrize("name", exposure_tests.keys())
@pytest.mark.parametrize("colormode", exposure_colorspaces)
def test_exposure_composite_icc(name: str, colormode: str) -> None:
    filename = f"adjustments/{name}_{colormode}"
    check_icc_composite_quality(filename, exposure_tests[name])


@pytest.mark.parametrize("name", exposure_tests.keys())
@pytest.mark.parametrize("colormode", exposure_colorspaces)
def test_exposure_composite_error(name: str, colormode: str) -> None:
    filename = f"adjustments/{name}_{colormode}.psd"
    check_composite_quality(filename, exposure_tests[name], False)


# Hue/Saturation
huesaturation_tests = {
    "huesaturation": LOOSE,  # lower accuracy is due to how _apply_saturation works
    "huesaturation_saturation": ACCURATE,  # test saturation in isolation
    "huesaturation_lightness": ACCURATE,  # test lightness in isolation
    "huesaturation_colorize": ACCURATE,
}
huesaturation_colorspaces = ["rgb"]  # CMYK not implemented


@pytest.mark.parametrize("name", huesaturation_tests.keys())
@pytest.mark.parametrize("colormode", huesaturation_colorspaces)
def test_huesaturation_composite_icc(name: str, colormode: str) -> None:
    filename = f"adjustments/{name}_{colormode}"
    check_icc_composite_quality(filename, huesaturation_tests[name])


@pytest.mark.parametrize("name", huesaturation_tests.keys())
@pytest.mark.parametrize("colormode", huesaturation_colorspaces)
def test_huesaturation_composite_error(name: str, colormode: str) -> None:
    filename = f"adjustments/{name}_{colormode}.psd"
    check_composite_quality(filename, huesaturation_tests[name], False)


def test_huesaturation_near_white_stays_white() -> None:
    # One step below 1 in a channel rounds L to 1 while the channels differ.
    below_one = np.nextafter(np.float32(1), np.float32(0))
    img = np.array([[[1.0, 1.0, below_one]]], dtype=np.float32)
    assert rgb2hsl(img)[0, 0, 1] == 0
    master = (np.float32(0.1), np.float32(0.2), np.float32(0))
    np.testing.assert_allclose(_huesaturation(img, [], master), 1.0, atol=1e-6)


# Invert
invert_tests = {
    "invert": ACCURATE,
}
invert_colorspaces = ["rgb", "cmyk", "grayscale"]


@pytest.mark.parametrize("name", invert_tests.keys())
@pytest.mark.parametrize("colormode", invert_colorspaces)
def test_invert_composite_icc(name: str, colormode: str) -> None:
    filename = f"adjustments/{name}_{colormode}"
    check_icc_composite_quality(filename, invert_tests[name])


@pytest.mark.parametrize("name", invert_tests.keys())
@pytest.mark.parametrize("colormode", invert_colorspaces)
def test_invert_composite_error(name: str, colormode: str) -> None:
    filename = f"adjustments/{name}_{colormode}.psd"
    check_composite_quality(filename, invert_tests[name], False)


# Posterize
posterize_tests = {
    "posterize": ACCURATE,
    "posterize_16bits": ACCURATE,
}
posterize_colorspaces = ["rgb", "cmyk", "grayscale"]


@pytest.mark.parametrize("name", posterize_tests.keys())
@pytest.mark.parametrize("colormode", posterize_colorspaces)
def test_posterize_composite_icc(name: str, colormode: str) -> None:
    filename = f"adjustments/{name}_{colormode}"
    check_icc_composite_quality(filename, posterize_tests[name])


@pytest.mark.parametrize("name", posterize_tests.keys())
@pytest.mark.parametrize("colormode", posterize_colorspaces)
def test_posterize_composite_error(name: str, colormode: str) -> None:
    filename = f"adjustments/{name}_{colormode}.psd"
    check_composite_quality(filename, posterize_tests[name], False)


@pytest.mark.parametrize("colormode", ["rgb", "grayscale"])
def test_posterize_matches_photoshop_bytes(colormode: str) -> None:
    """Photoshop's 8-bit levels are whole bytes, so its export is matched exactly.

    The fixture's posterized edges are the only channels left to differ.
    """
    filename = f"adjustments/posterize_{colormode}"
    reference = np.asarray(Image.open(full_name(filename + ".png")))
    psd = PSDImage.open(full_name(filename + ".psd"))
    result = np.asarray(
        psd.composite(apply_icc=True, layer_filter=lambda layer: layer.is_visible())
    )
    assert np.mean(result == reference) > 0.999


def test_posterize_truncates_8_bit_levels_to_whole_bytes() -> None:
    psd = PSDImage.open(full_name("adjustments/posterize_rgb.psd"))
    layer = next(
        x
        for x in psd.descendants()
        if isinstance(x, adjustments.Posterize) and x.posterize == 13
    )
    ramp = np.linspace(0, 1, 256, dtype=np.float32).reshape(1, 256, 1)
    out = 255 * _applied(
        apply_posterize(np.repeat(ramp, 3, axis=2), ColorMode.RGB, layer)
    )
    exact = 255 * np.unique(np.floor(255 / 256 * ramp * 13)) / 12
    assert not np.array_equal(exact, np.floor(exact)), "13 levels are not whole bytes"
    np.testing.assert_allclose(np.unique(out), np.floor(exact), atol=1e-3)


# Threshold
threshold_tests = {
    "threshold": STRICT,
    "threshold_16bits": STRICT,
}
threshold_colorspaces = ["rgb", "grayscale"]  # CMYK not implemented


@pytest.mark.parametrize("name", threshold_tests.keys())
@pytest.mark.parametrize("colormode", threshold_colorspaces)
def test_threshold_composite_icc(name: str, colormode: str) -> None:
    filename = f"adjustments/{name}_{colormode}"
    check_icc_composite_quality(filename, threshold_tests[name])


@pytest.mark.parametrize("name", threshold_tests.keys())
@pytest.mark.parametrize("colormode", threshold_colorspaces)
def test_threshold_composite_error(name: str, colormode: str) -> None:
    filename = f"adjustments/{name}_{colormode}.psd"
    check_composite_quality(filename, threshold_tests[name], False)


# Nested composition
@pytest.mark.parametrize("number", ["1", "2", "3", "4", "5"])
def test_adjustment_nested_composition(number) -> None:
    filename = f"adjustments/adjustment_nested_composition_{number}"
    check_icc_composite_quality(filename, 0.0005)
    check_composite_quality(f"{filename}.psd", 0.0005, False)


# Clipping
def test_adjustment_clipping() -> None:
    filename = "adjustments/adjustment_clipping"
    check_icc_composite_quality(filename, 0.0005)
    check_composite_quality(f"{filename}.psd", 0.0005, False)


# Groups with a layer style
def test_an_adjustment_inside_a_styled_group_stops_at_the_group() -> None:
    """A styled pass-through group keeps its adjustments to itself.

    A layer style makes Photoshop composite the group on its own, so an
    adjustment inside it no longer reaches the layers under the group.
    ``adjustment_nested_composition_3.psd``: pass-through ``Group 6`` holds an
    unclipped ``Curves 9`` over ``Layer 15`` and ``Color Fill 1``, which fill
    the canvas under it; from row 150 down nothing but those two is drawn. The
    style is the inside stroke block of ``group-stroke-off-canvas.psd``, which
    cannot reach past the group's own box.
    """

    def under_the_group(styled: bool, curves_visible: bool) -> np.ndarray:
        psd = PSDImage.open(
            full_name("adjustments/adjustment_nested_composition_3.psd")
        )
        group = psd[4]
        assert group.name == "Group 6"
        assert group.blend_mode == BlendMode.PASS_THROUGH
        assert isinstance(group, Group)
        curves = group[0]
        assert isinstance(curves, adjustments.Curves)
        curves.visible = curves_visible
        if styled:
            donor = PSDImage.open(full_name("effects/group-stroke-off-canvas.psd"))[2]
            block = copy.deepcopy(
                donor.tagged_blocks.get_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO_V1)
            )
            group.tagged_blocks.set_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO_V1, block)
            assert group.effects.enabled
        image = psd.composite(force=True).convert("RGB")
        return np.asarray(image, dtype=np.int16)[150:]

    untouched = under_the_group(styled=False, curves_visible=False)
    passed_through = under_the_group(styled=False, curves_visible=True)
    assert np.abs(passed_through - untouched).max() > 1
    isolated = under_the_group(styled=True, curves_visible=True)
    assert np.abs(isolated - untouched).max() <= 1


def test_an_adjustment_nested_in_a_styled_group_still_reaches_its_content() -> None:
    """Isolation stops at the group's edge, not at a nested group's.

    ``Layer 14`` is switched on as the styled ``Group 6``'s content and
    ``Curves 9`` is moved into a pass-through group of its own above it; the
    curve must still change ``Layer 14``'s pixels while the rows under the
    group stay as they are.
    """

    def render(curves_visible: bool) -> tuple[np.ndarray, np.ndarray]:
        psd = PSDImage.open(
            full_name("adjustments/adjustment_nested_composition_3.psd")
        )
        group = psd[4]
        assert isinstance(group, Group)
        curves, content = group[0], group[1]
        assert isinstance(curves, adjustments.Curves)
        assert content.name == "Layer 14"
        content.visible = True
        curves.visible = curves_visible
        group.create_group([curves], name="Nested")
        donor = PSDImage.open(full_name("effects/group-stroke-off-canvas.psd"))[2]
        block = copy.deepcopy(
            donor.tagged_blocks.get_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO_V1)
        )
        group.tagged_blocks.set_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO_V1, block)
        image = np.asarray(psd.composite(force=True).convert("RGB"), dtype=np.int16)
        shape = content.numpy("shape")
        assert shape is not None
        covered = np.zeros(image.shape[:2], dtype=bool)
        left, top, right, bottom = content.bbox
        covered[top:bottom, left:right] = shape[..., 0] > 0.99
        return image, covered

    (without, covered), (with_curves, _) = render(False), render(True)
    assert not covered[190:].any()
    difference = np.abs(with_curves - without).max(axis=2)
    assert difference[covered].max() > 1
    assert difference[190:].max() <= 1


def test_an_adjustment_in_a_styled_group_reads_the_content_not_the_canvas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adjustment applies to the group's own content, at its own alpha.

    ``Layer 14`` is switched on at half opacity as the styled ``Group 6``'s
    only content, and ``Curves 9`` is moved into a pass-through group of its
    own above it. Photoshop adjusts the layer's own colour and composites it
    over the backdrop at the layer's alpha,
    ``a * f(C) + (1 - a) * B``, on every pixel the layer covers at all;
    adjusting the canvas the layer stands on gives ``f(a * C + (1 - a) * B)``,
    which a curve that is not a straight line tells apart. ``f`` is the curve
    under the adjustment's own mask. The style is faked through
    ``_has_enabled_effects`` so that no stroke lands on the pixels measured.
    """
    psd = PSDImage.open(full_name("adjustments/adjustment_nested_composition_3.psd"))
    group = psd[4]
    assert isinstance(group, Group) and group.blend_mode == BlendMode.PASS_THROUGH
    curves, content, other = group[0], group[1], group[3]
    assert isinstance(curves, adjustments.Curves) and content.name == "Layer 14"
    content.visible, content.opacity = True, 128
    other.visible = False
    group.create_group([curves], name="Nested")
    assert [sub.name for sub in group if sub.is_visible()] == ["Layer 14", "Nested"]
    monkeypatch.setattr(
        composite_module, "_has_enabled_effects", lambda layer: layer is group
    )

    group.visible = False
    backdrop, _, _ = composite_module.composite(psd)
    group.visible = True
    color, _, _ = composite_module.composite(psd)

    left, top, right, bottom = content.bbox
    own_color, own_shape = content.numpy("color"), content.numpy("shape")
    assert own_color is not None and own_shape is not None
    assert curves.mask is not None and curves.mask.bbox == psd.viewbox
    mask = curves.numpy("mask")
    assert mask is not None
    mask = mask[top:bottom, left:right]
    alpha = own_shape * (128 / 255)
    under = backdrop[top:bottom, left:right]

    def adjust(image: np.ndarray) -> np.ndarray:
        return image + mask * (apply_curves(image, ColorMode.RGB, curves) - image)

    expected = alpha * adjust(own_color) + (1.0 - alpha) * under
    flattened = adjust(alpha * own_color + (1.0 - alpha) * under)

    covered = own_shape[..., 0] > 0.0
    rendered = color[top:bottom, left:right]
    assert np.abs(rendered - expected)[covered].max() <= 1 / 255
    assert np.abs(flattened - expected)[own_shape[..., 0] > 0.99].max() > 4 / 255


def test_a_style_on_a_normal_group_changes_nothing_about_its_adjustments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Normal-mode group is composited on its own with or without a style.

    So the isolation rule for styled groups must not touch it: the partially
    covered pixels of ``Group 6``'s content would otherwise see a weaker
    ``Curves 9``. The style is faked through ``_has_enabled_effects`` so that
    nothing but the rule itself can change the render.
    """

    def render(passthrough: bool, styled: bool) -> np.ndarray:
        psd = PSDImage.open(
            full_name("adjustments/adjustment_nested_composition_3.psd")
        )
        group = psd[4]
        assert isinstance(group, Group)
        assert isinstance(group[0], adjustments.Curves)
        if not passthrough:
            group.blend_mode = BlendMode.NORMAL
        monkeypatch.setattr(
            composite_module, "_has_enabled_effects", lambda layer: styled
        )
        return np.asarray(psd.composite(force=True).convert("RGB"), dtype=np.int16)

    assert np.array_equal(render(False, styled=True), render(False, styled=False))
    assert not np.array_equal(render(True, styled=True), render(True, styled=False))


def test_a_normal_group_in_a_styled_group_adjusts_its_soft_edges_in_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Normal group nested in an isolating group keeps its own adjustments.

    ``Curves 9`` is moved above ``Layer 16``, whose edge is mostly partial
    coverage, into a Normal group inside ``Group 6``. Styling ``Group 6`` must
    not weaken the curve on those edge pixels. The style is faked as in the
    test above.
    """

    def render(styled: bool, curves_visible: bool = True) -> np.ndarray:
        psd = PSDImage.open(
            full_name("adjustments/adjustment_nested_composition_3.psd")
        )
        group = psd[4]
        assert isinstance(group, Group)
        curves, content = group[0], group[3]
        assert isinstance(curves, adjustments.Curves)
        assert content.name == "Layer 16"
        curves.visible = curves_visible
        group.create_group([content, curves], blend_mode=BlendMode.NORMAL)
        monkeypatch.setattr(
            composite_module, "_has_enabled_effects", lambda layer: styled
        )
        return np.asarray(psd.composite(force=True).convert("RGB"), dtype=np.int16)

    plain = render(styled=False)
    assert np.abs(plain - render(styled=False, curves_visible=False)).max() > 1
    assert np.abs(render(styled=True) - plain).max() <= 1


def test_a_normal_group_in_a_styled_group_renders_as_in_an_unstyled_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Normal group composites on its own, whether or not its parent isolates.

    ``Layer 16`` and ``Curves 9`` are moved into a Normal group inside the
    pass-through ``Group 6``. The new group's box, (54, 16, 199, 142), starts
    away from the parent's viewport origin, so an isolation it inherited would
    have to be laid out on its own viewport rather than its parent's; it is
    seeded as a fresh one instead.
    """

    def render(styled: bool, curves_visible: bool = True) -> np.ndarray:
        psd = PSDImage.open(
            full_name("adjustments/adjustment_nested_composition_3.psd")
        )
        group = psd[4]
        assert isinstance(group, Group) and group.blend_mode == BlendMode.PASS_THROUGH
        curves, content = group[0], group[3]
        assert isinstance(curves, adjustments.Curves) and content.name == "Layer 16"
        nested = group.create_group([content, curves], name="Nested")
        nested.blend_mode = BlendMode.NORMAL
        assert nested.bbox == (54, 16, 199, 142)
        assert [sub.name for sub in nested] == ["Layer 16", "Curves 9"]
        curves.visible = curves_visible
        monkeypatch.setattr(
            composite_module,
            "_has_enabled_effects",
            lambda layer: styled and layer is group,
        )
        color, _, _ = composite_module.composite(psd)
        return color

    styled = render(True)
    assert np.allclose(styled, render(False), atol=1e-6)
    assert not np.allclose(styled, render(True, curves_visible=False), atol=1 / 255)


# Channel mixer
_IDENTITY_MIX = [[100, 0, 0, 0, 0], [0, 100, 0, 0, 0], [0, 0, 100, 0, 0]]
_FOURTH_MIX = [0, 0, 0, 100, 0]


def _mixer_data(records: list[list[int]], monochrome: int = 0) -> ChannelMixerData:
    """A channel mixer block from ``[red, green, blue, fourth, constant]`` percent rows.

    The reader keeps the first row in ``data`` and the other three in ``unknown``.
    """
    flat = [value for record in records + [_FOURTH_MIX] for value in record]
    block = ChannelMixerData(
        monochrome=monochrome, unknown=struct.pack(">15h", *flat[5:])
    )
    block.data = flat[:5]
    return block


def _channel_mixer_layer() -> adjustments.ChannelMixer:
    layer = PSDImage.open(full_name("fill_adjustments.psd"))[13]
    assert isinstance(layer, adjustments.ChannelMixer)
    return layer


def test_channel_mixer_mixes_the_sources_and_adds_the_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Red takes half its blue away, blue gets a 20% constant, and both clip."""
    layer = _channel_mixer_layer()
    records = [[100, 0, -50, 0, 0], [0, 100, 0, 0, 0], [0, 0, 100, 0, 20]]
    monkeypatch.setattr(layer, "_data", _mixer_data(records))
    image = np.array([[[0.8, 0.5, 0.4], [0.1, 0.2, 0.9]]], dtype=np.float32)
    expected = [[[0.6, 0.5, 0.6], [0.0, 0.2, 1.0]]]
    out = _applied(apply_channelmixer(image, ColorMode.RGB, layer))
    assert out.dtype == np.float32
    assert np.allclose(out, expected, atol=1e-6)


def test_channel_mixer_monochrome_writes_one_mix_to_every_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Monochrome takes the first row as the gray mix and ignores the others."""
    layer = _channel_mixer_layer()
    records = [[30, 59, 11, 0, 0], [0, 0, 0, 0, 100], [0, 0, 0, 0, 100]]
    monkeypatch.setattr(layer, "_data", _mixer_data(records, monochrome=1))
    image = np.array([[[1.0, 0.0, 0.0], [0.2, 0.4, 0.6]]], dtype=np.float32)
    out = _applied(apply_channelmixer(image, ColorMode.RGB, layer))
    gray = image @ np.array([0.3, 0.59, 0.11], dtype=np.float32)
    assert np.allclose(out, gray[..., None], atol=1e-6)


def test_channel_mixer_with_a_short_block_is_not_rendered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _channel_mixer_layer()
    block = ChannelMixerData()
    block.data = [100, 0, 0, 0, 0]
    monkeypatch.setattr(layer, "_data", block)
    image = np.zeros((2, 2, 3), dtype=np.float32)
    assert apply_channelmixer(image, ColorMode.RGB, layer) is None


# Black & white
def _black_and_white_layer(**sliders: int) -> adjustments.BlackAndWhite:
    """The fixture's layer with the given sliders; the others keep their defaults."""
    layer = PSDImage.open(full_name("fill_adjustments.psd"))[11]
    assert isinstance(layer, adjustments.BlackAndWhite)
    data = Descriptor()
    for key, value in sliders.items():
        data[key.encode("ascii").ljust(4)] = Integer(value)
    layer._data = data
    return layer


def test_black_and_white_weights_a_pixel_by_its_primary_and_secondary_hue() -> None:
    """Each primary gets its slider; a hue between two gets both, by its position.

    The defaults are red 40, yellow 60, green 40, cyan 60, blue 20, magenta 80.
    """
    layer = _black_and_white_layer()
    image = np.array(
        [[[1, 0, 0], [1, 1, 0], [1, 0.5, 0], [0, 0, 1], [0, 1, 1], [0.5, 0.5, 0.5]]],
        dtype=np.float32,
    )
    expected = [0.4, 0.6, 0.5, 0.2, 0.6, 0.5]
    out = _applied(apply_blackandwhite(image, ColorMode.RGB, layer))
    assert np.allclose(out, np.array(expected)[None, :, None], atol=1e-6)


def test_black_and_white_reads_the_sliders_from_the_descriptor() -> None:
    layer = _black_and_white_layer(Rd=100, Bl=0, Mgnt=50)
    image = np.array([[[1, 0, 0], [0, 0, 1], [1, 0, 0.5]]], dtype=np.float32)
    # magenta 50 on the min..mid span, red 100 on the mid..max span
    expected = [1.0, 0.0, 0.5 * 0.5 + 0.5 * 1.0]
    out = _applied(apply_blackandwhite(image, ColorMode.RGB, layer))
    assert np.allclose(out, np.array(expected)[None, :, None], atol=1e-6)


# Gradient map
_LUM = np.array([0.3, 0.59, 0.11], dtype=np.float32)


def _gradient_map_data(
    stops: list[tuple[int, int, tuple[float, float, float]]],
    method: bytes = b"Gcls",
    reversed: int = 0,
    mode: int = 0,
    smoothness: int = 0,
) -> GradientMapData:
    """A gradient map block from ``(location, midpoint %, sRGB colour)`` stops.

    ``smoothness`` is in 4096ths, as the block stores it; at 0 every segment
    blends linearly, so the ramp reads off the output directly.
    """
    color_stops = [
        ColorStop(
            location=location,
            midpoint=midpoint,
            mode=0,
            color=tuple(round(value * 65535) for value in rgb) + (0,),
        )
        for location, midpoint, rgb in stops
    ]
    block = GradientMapData(
        version=3,
        method=method,
        is_reversed=reversed,
        interpolation=smoothness,
        mode=mode,
    )
    block.color_stops = color_stops
    block.transparency_stops = [
        TransparencyStop(0, 50, 255),
        TransparencyStop(4096, 50, 255),
    ]
    return block


def _gradient_map_layer(data: GradientMapData) -> adjustments.GradientMap:
    layer = PSDImage.open(full_name("fill_adjustments.psd"))[19]
    assert isinstance(layer, adjustments.GradientMap)
    layer._data = data
    return layer


_BLACK = (0.0, 0.0, 0.0)
_WHITE = (1.0, 1.0, 1.0)
_PRIMARIES = np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5]]], np.float32)
_RAMP = np.linspace(0.0, 1.0, 11, dtype=np.float32)[None, :, None].repeat(3, axis=2)


def test_gradient_map_maps_luminance_through_the_gradient() -> None:
    """A black-to-white map is the luminance itself, through an 8-bit table."""
    layer = _gradient_map_layer(
        _gradient_map_data([(0, 50, _BLACK), (4096, 50, _WHITE)])
    )
    luminance = _PRIMARIES @ _LUM
    out = _applied(apply_gradientmap(_PRIMARIES, ColorMode.RGB, layer))
    assert out.shape == _PRIMARIES.shape
    assert np.allclose(out, luminance[..., None], atol=1 / 255)


def test_gradient_map_reversed_runs_the_gradient_the_other_way() -> None:
    layer = _gradient_map_layer(
        _gradient_map_data([(0, 50, _BLACK), (4096, 50, _WHITE)], reversed=1)
    )
    luminance = _PRIMARIES @ _LUM
    out = _applied(apply_gradientmap(_PRIMARIES, ColorMode.RGB, layer))
    assert np.allclose(out, 1 - luminance[..., None], atol=1 / 255)


def test_gradient_map_holds_the_end_colours_outside_the_end_stops() -> None:
    """Luminance below the first stop or above the last takes that stop's colour."""
    red, blue = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    layer = _gradient_map_layer(_gradient_map_data([(1024, 50, red), (3072, 50, blue)]))
    image = np.array([[[0, 0, 0], [1, 1, 1], [0.5, 0.5, 0.5]]], dtype=np.float32)
    out = _applied(apply_gradientmap(image, ColorMode.RGB, layer))
    assert np.allclose(out[0, 0], red, atol=1 / 255)
    assert np.allclose(out[0, 1], blue, atol=1 / 255)
    assert np.allclose(out[0, 2], (0.5, 0.0, 0.5), atol=1 / 255)


def test_gradient_map_keeps_a_hard_edge_between_co_located_stops() -> None:
    """Two stops at one location end one segment and start the next.

    Just under the edge is the first stop's colour; from the edge on it is the
    second's. The duplicate at the start that Photoshop writes is harmless.
    """
    red, blue = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    layer = _gradient_map_layer(
        _gradient_map_data(
            [(0, 50, _BLACK), (0, 50, _BLACK), (2048, 50, red), (2048, 50, blue)]
            + [(4096, 50, _WHITE)]
        )
    )
    below, at = 0.5 - 2 / 255, 0.5
    image = np.array([[[below] * 3, [at] * 3, [0.75] * 3]], dtype=np.float32)
    out = _applied(apply_gradientmap(image, ColorMode.RGB, layer))
    assert np.allclose(out[0, 0], red, atol=3 / 255)
    assert np.allclose(out[0, 1], blue, atol=3 / 255)
    assert np.allclose(out[0, 2], (0.5, 0.5, 1.0), atol=1 / 255)


def test_gradient_map_midpoint_is_where_the_blend_reaches_one_half() -> None:
    """A midpoint of 25% on the end stop puts the half blend at a quarter."""
    layer = _gradient_map_layer(
        _gradient_map_data([(0, 50, _BLACK), (4096, 25, _WHITE)])
    )
    image = np.array([[[0.25, 0.25, 0.25], [0.625, 0.625, 0.625]]], dtype=np.float32)
    out = _applied(apply_gradientmap(image, ColorMode.RGB, layer))
    assert np.allclose(out[0, 0], 0.5, atol=1 / 255)
    assert np.allclose(out[0, 1], 0.75, atol=1 / 255)


@pytest.mark.parametrize("midpoint", [0, 100])
def test_gradient_map_ignores_a_midpoint_on_either_end_of_its_segment(
    midpoint: int,
) -> None:
    """At 0% or 100% the blend would be all at one end; it runs linearly instead."""
    image = np.repeat(np.linspace(0.0, 1.0, 9, dtype=np.float32)[None, :, None], 3, 2)
    linear = _gradient_map_layer(
        _gradient_map_data([(0, 50, _BLACK), (4096, 50, _WHITE)])
    )
    on_end = _gradient_map_layer(
        _gradient_map_data([(0, 50, _BLACK), (4096, midpoint, _WHITE)])
    )
    expected = _applied(apply_gradientmap(image, ColorMode.RGB, linear))
    out = _applied(apply_gradientmap(image, ColorMode.RGB, on_end))
    assert np.isfinite(out).all()
    assert np.allclose(out, expected, atol=1 / 255)
    assert np.allclose(out[0, :, 0], image[0, :, 0], atol=1 / 255)


def test_gradient_map_with_every_stop_at_one_location_steps_as_a_fill_does() -> None:
    """Below the location is the first stop, from it on the last, in map and fill alike."""
    red, blue = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    stops = [(2048, 50, red), (2048, 50, _BLACK), (2048, 50, blue)]
    layer = _gradient_map_layer(_gradient_map_data(stops))
    image = np.array([[[0.25] * 3, [0.5] * 3, [0.75] * 3]], dtype=np.float32)
    out = _applied(apply_gradientmap(image, ColorMode.RGB, layer))
    assert np.allclose(out[0], [red, blue, blue], atol=1 / 255)
    fill = paint._blend_stops(
        [0.5, 0.5, 0.5], [np.array(rgb) for _, _, rgb in stops], [0.5] * 3, 0.0
    )
    quarter = (len(fill) - 1) // 4
    assert np.allclose(fill[[quarter, 2 * quarter, 3 * quarter]], [red, blue, blue])


def test_gradient_map_perceptual_blends_in_oklab() -> None:
    """Halfway between black and white perceptually is Oklab L 0.5: linear 1/8."""
    layer = _gradient_map_layer(
        _gradient_map_data([(0, 50, _BLACK), (4096, 50, _WHITE)], method=b"Perc")
    )
    image = np.full((1, 1, 3), 0.5, dtype=np.float32)
    mid_gray = 1.055 * (1 / 8) ** (1 / 2.4) - 0.055
    out = _applied(apply_gradientmap(image, ColorMode.RGB, layer))
    assert np.allclose(out, mid_gray, atol=1 / 255)


def test_gradient_map_smoothness_eases_the_blend_towards_a_smoothstep() -> None:
    """At 100% the blend lands halfway between linear and a smoothstep, as a fill's."""
    gray = np.array([0.25, 0.5, 0.75], dtype=np.float32)
    image = np.repeat(gray[None, :, None], 3, axis=2)
    stops = [(0, 50, _BLACK), (4096, 50, _WHITE)]
    smooth = _gradient_map_layer(_gradient_map_data(stops, smoothness=4096))
    half = _gradient_map_layer(_gradient_map_data(stops, smoothness=2048))
    eased = gray + ((3 - 2 * gray) * gray * gray - gray) / 2
    out = _applied(apply_gradientmap(image, ColorMode.RGB, smooth))
    assert np.allclose(out[0, :, 0], eased, atol=1 / 255)
    assert float(out[0, 0, 0]) == pytest.approx(0.203125, abs=1 / 255)
    out = _applied(apply_gradientmap(image, ColorMode.RGB, half))
    assert np.allclose(out[0, :, 0], (gray + eased) / 2, atol=1 / 255)


def test_gradient_map_blends_its_stops_as_a_gradient_fill_does() -> None:
    """Smoothness and midpoints shape a map's ramp as they shape a fill's."""
    red = (1.0, 0.0, 0.0)
    stops = [(0, 50, _BLACK), (1024, 30, red), (4096, 70, _WHITE)]
    layer = _gradient_map_layer(_gradient_map_data(stops, smoothness=4096))
    fill = paint._blend_stops(
        [0.0, 0.25, 1.0], [np.array(rgb) for _, _, rgb in stops], [0.5, 0.3, 0.7], 1.0
    )
    greys = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    image = np.repeat(greys[None, :, None], 3, axis=2)
    out = _applied(apply_gradientmap(image, ColorMode.RGB, layer))
    samples = np.linspace(0.0, 1.0, len(fill))
    expected = np.stack(
        [np.interp(greys, samples, fill[:, channel]) for channel in range(3)], axis=1
    )
    assert np.allclose(out[0], expected, atol=1 / 255)
    # 100% smoothness bends the ramp away from the straight segments.
    linear = _gradient_map_layer(_gradient_map_data(stops))
    assert not np.allclose(
        _applied(apply_gradientmap(image, ColorMode.RGB, linear)), out, atol=2 / 255
    )


def test_gradient_map_writes_one_channel_on_a_grayscale_document() -> None:
    red = (1.0, 0.0, 0.0)
    layer = _gradient_map_layer(_gradient_map_data([(0, 50, red), (4096, 50, red)]))
    image = np.array([[[0.2], [0.9]]], dtype=np.float32)
    out = _applied(apply_gradientmap(image, ColorMode.GRAYSCALE, layer))
    assert out.shape == (1, 2, 1)
    assert np.allclose(out, 0.3, atol=1 / 255)


def test_gradient_map_clamps_luminance_outside_the_unit_range() -> None:
    """Unclipped 32-bit values take the end colours instead of indexing past the table."""
    red, blue = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    layer = _gradient_map_layer(_gradient_map_data([(0, 50, red), (4096, 50, blue)]))
    image = np.array([[[-0.5, -0.5, -0.5], [1.5, 1.5, 1.5]]], dtype=np.float32)
    out = _applied(apply_gradientmap(image, ColorMode.RGB, layer))
    assert np.allclose(out[0, 0], red, atol=1 / 255)
    assert np.allclose(out[0, 1], blue, atol=1 / 255)


def test_gradient_map_reads_a_gray_stop_as_ink() -> None:
    """A gray stop's component runs 0..10000 from white to black."""
    data = _gradient_map_data([(0, 50, _BLACK), (4096, 50, _WHITE)])
    for stop, ink in zip(data.color_stops, (10000, 2500), strict=True):
        stop.mode = 8
        stop.color = (ink, 0, 0, 0)
    layer = _gradient_map_layer(data)
    image = np.array([[[0, 0, 0], [1, 1, 1]]], dtype=np.float32)
    out = _applied(apply_gradientmap(image, ColorMode.RGB, layer))
    assert np.allclose(out[0, 0], 0.0, atol=1 / 255)
    assert np.allclose(out[0, 1], 0.75, atol=1 / 255)


def test_gradient_map_draws_stops_in_location_order() -> None:
    red, blue = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    listed = _gradient_map_layer(_gradient_map_data([(4096, 50, blue), (0, 50, red)]))
    ordered = _gradient_map_layer(_gradient_map_data([(0, 50, red), (4096, 50, blue)]))
    out = _applied(apply_gradientmap(_RAMP, ColorMode.RGB, listed))
    assert np.array_equal(
        out, _applied(apply_gradientmap(_RAMP, ColorMode.RGB, ordered))
    )


@pytest.mark.parametrize("colormode", [ColorMode.CMYK, ColorMode.GRAYSCALE])
def test_gradient_map_converts_its_table_as_a_fill_would(
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE],
) -> None:
    rgb = np.array(
        [[0, 0, 0], [1, 1, 1], [1, 0, 0], [0.2, 0.5, 0.9], [0.5, 0.5, 0.5]],
        dtype=np.float32,
    )
    expected = [_from_rgb(colormode, tuple(float(v) for v in row)) for row in rgb]
    assert np.allclose(_rows_from_rgb(colormode, rgb), expected, atol=1e-6)


def test_gradient_map_with_a_noise_gradient_is_not_rendered() -> None:
    layer = _gradient_map_layer(
        _gradient_map_data([(0, 50, _BLACK), (4096, 50, _WHITE)], mode=1)
    )
    image = np.zeros((2, 2, 3), dtype=np.float32)
    assert apply_gradientmap(image, ColorMode.RGB, layer) is None


# Colour balance
def _color_balance_layer(
    midtones: tuple[int, int, int] = (0, 0, 0),
    *,
    luminosity: bool,
    shadows: tuple[int, int, int] = (0, 0, 0),
    highlights: tuple[int, int, int] = (0, 0, 0),
) -> adjustments.ColorBalance:
    layer = PSDImage.open(full_name("fill_adjustments.psd"))[10]
    assert isinstance(layer, adjustments.ColorBalance)
    layer._data = ColorBalanceData(
        shadows=shadows, midtones=midtones, highlights=highlights, luminosity=luminosity
    )
    return layer


def test_color_balance_midtones_are_a_gamma_per_channel() -> None:
    """+100 toward red is the square root of red; the other channels stay."""
    layer = _color_balance_layer((100, 0, 0), luminosity=False)
    out = _applied(apply_colorbalance(_RAMP, ColorMode.RGB, layer))
    assert np.allclose(out[..., 0], np.sqrt(_RAMP[..., 0]), atol=1 / 255)
    assert np.allclose(out[..., 1:], _RAMP[..., 1:], atol=1 / 255)


def test_color_balance_preserving_luminosity_centres_the_amounts() -> None:
    """With the flag the three amounts lose their mean.

    So +100 red also darkens green and blue, and opposite shifts cancel out.
    """
    layer = _color_balance_layer((100, 0, 0), luminosity=True)
    out = _applied(apply_colorbalance(_RAMP, ColorMode.RGB, layer))
    assert np.allclose(out[..., 0], _RAMP[..., 0] ** (2 ** (-2 / 3)), atol=1 / 255)
    assert np.allclose(out[..., 1], _RAMP[..., 1] ** (2 ** (1 / 3)), atol=1 / 255)
    balanced = _color_balance_layer((100, -100, 0), luminosity=True)
    out = _applied(apply_colorbalance(_RAMP, ColorMode.RGB, balanced))
    assert np.allclose(out[..., 0], np.sqrt(_RAMP[..., 0]), atol=1 / 255)
    assert np.allclose(out[..., 1], _RAMP[..., 1] ** 2, atol=1 / 255)


def test_color_balance_shadows_and_highlights_work_their_own_end() -> None:
    """A shadow shift moves dark levels most, a highlight shift light ones.

    Neither moves black or white, in either direction, nor another channel.
    """
    dark, light = 3, 7
    for amount in (100, -100):
        for name, near, far in (("shadows", dark, light), ("highlights", light, dark)):
            layer = _color_balance_layer(luminosity=False, **{name: (amount, 0, 0)})
            out = _applied(apply_colorbalance(_RAMP, ColorMode.RGB, layer))
            shift = out[0, :, 0] - _RAMP[0, :, 0]
            assert np.allclose(shift[[0, -1]], 0.0, atol=1e-6)
            assert np.all(np.sign(shift[1:-1]) == np.sign(amount))
            assert abs(shift[near]) > abs(shift[far])
            assert np.allclose(out[..., 1:], _RAMP[..., 1:], atol=1 / 255)


@pytest.mark.parametrize("name", ["shadows", "highlights"])
def test_color_balance_preserving_luminosity_centres_every_range(name: str) -> None:
    preserved = _color_balance_layer(luminosity=True, **{name: (60, 0, 0)})
    centred = _color_balance_layer(luminosity=False, **{name: (40, -20, -20)})
    out = _applied(apply_colorbalance(_RAMP, ColorMode.RGB, preserved))
    assert np.allclose(
        out, _applied(apply_colorbalance(_RAMP, ColorMode.RGB, centred)), atol=1e-6
    )
    assert out[0, 5, 0] > _RAMP[0, 5, 0] > out[0, 5, 1]


# Per-channel exponents measured on a Photoshop render of
# midtones (100, -100, -16) with Preserve Luminosity.
_SILITHID_MIDTONES = (100, -100, -16)
_SILITHID_EXPONENTS = (0.46, 2.04, 1.10)
_SILITHID_TOLERANCE = 6 / 255


def test_color_balance_matches_the_exponents_measured_on_photoshop() -> None:
    layer = _color_balance_layer(_SILITHID_MIDTONES, luminosity=True)
    out = _applied(apply_colorbalance(_RAMP, ColorMode.RGB, layer))
    for channel, exponent in enumerate(_SILITHID_EXPONENTS):
        expected = _RAMP[..., channel] ** exponent
        assert np.allclose(out[..., channel], expected, atol=_SILITHID_TOLERANCE)


def test_color_balance_keeps_black_white_and_channel_order() -> None:
    layer = _color_balance_layer((-60, 30, 80), luminosity=True)
    out = _applied(apply_colorbalance(_RAMP, ColorMode.RGB, layer))
    assert np.allclose(out[0, 0], 0.0, atol=1 / 255)
    assert np.allclose(out[0, -1], 1.0, atol=1 / 255)
    assert np.all(np.diff(out, axis=1) >= -1e-6)


# Vibrance
def _vibrance_layer(vibrance: int, saturation: int) -> adjustments.Vibrance:
    layer = PSDImage.open(full_name("fill_adjustments.psd"))[8]
    assert isinstance(layer, adjustments.Vibrance)
    data = Descriptor()
    data[b"vibrance"] = Integer(vibrance)
    data[b"Strt"] = Integer(saturation)
    layer._data = data
    return layer


def _linear(value: np.ndarray) -> np.ndarray:
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def _linear_chroma(image: np.ndarray) -> np.ndarray:
    linear = _linear(image)
    return linear.max(axis=2) - linear.min(axis=2)


_PALE_BLUE = [0.55, 0.6, 0.7]
_DEEP_BLUE = [0.1, 0.2, 0.7]
_PALE_RED = [0.7, 0.6, 0.55]
_PALE_CYAN = [0.55, 0.7, 0.7]


def test_negative_vibrance_takes_more_chroma_from_a_pale_pixel() -> None:
    """Both pixels lose chroma and keep their brightest channel.

    The pale one keeps a smaller share of what it had.
    """
    image = np.array([[_PALE_BLUE, _DEEP_BLUE]], dtype=np.float32)
    out = _applied(apply_vibrance(image, ColorMode.RGB, _vibrance_layer(-100, 0)))
    kept = _linear_chroma(out) / _linear_chroma(image)
    assert np.all(kept < 1)
    assert kept[0, 0] < kept[0, 1]
    assert np.all(out.argmax(axis=2) == image.argmax(axis=2))


def test_a_weaker_negative_vibrance_removes_less() -> None:
    image = np.array([[_PALE_BLUE, _DEEP_BLUE]], dtype=np.float32)
    strong = _applied(apply_vibrance(image, ColorMode.RGB, _vibrance_layer(-100, 0)))
    weak = _applied(apply_vibrance(image, ColorMode.RGB, _vibrance_layer(-40, 0)))
    assert np.all(_linear_chroma(weak) > _linear_chroma(strong))
    assert np.all(_linear_chroma(weak) < _linear_chroma(image))


def test_positive_vibrance_boosts_pale_pixels_and_spares_skin_tones() -> None:
    """A pale pixel gains more than a saturated one.

    A red of the same saturation gains less than a cyan.
    """
    image = np.array([[_PALE_BLUE, _DEEP_BLUE, _PALE_RED, _PALE_CYAN]], np.float32)
    out = _applied(apply_vibrance(image, ColorMode.RGB, _vibrance_layer(100, 0)))
    gained = _linear_chroma(out) / _linear_chroma(image)
    assert np.all(gained > 1)
    assert gained[0, 0] > gained[0, 1]
    assert gained[0, 2] < gained[0, 3]


def test_saturation_scales_chroma_about_the_linear_luminance() -> None:
    """-100 is the Rec. 709 luminance gray.

    +50 is one and a half times the chroma around it.
    """
    image = np.array([[_DEEP_BLUE, _PALE_RED]], dtype=np.float32)
    luminance = _linear(image) @ np.array([0.2126, 0.7152, 0.0722])
    gray = _applied(apply_vibrance(image, ColorMode.RGB, _vibrance_layer(0, -100)))
    assert np.allclose(_linear(gray), luminance[..., None], atol=1 / 255)
    boosted = _applied(apply_vibrance(image, ColorMode.RGB, _vibrance_layer(0, 50)))
    expected = luminance[..., None] + (_linear(image) - luminance[..., None]) * 1.5
    assert np.allclose(_linear(boosted), np.clip(expected, 0, 1), atol=1 / 255)


# The fitted vibrance model, with its constants. Fitted on Photoshop renders
# of the layer over the same art; a changed constant fails this test.
_LINEAR_LUMINANCE = np.array([0.2126, 0.7152, 0.0722])
_BOOST, _BOOST_CURVE, _BOOST_TAIL, _BOOST_AMOUNT_CURVE = 2.44, 2.6, 0.25, 1.33
_BOOST_BRIGHT, _BOOST_DARK = 0.73, 0.023
_BOOST_GRAY, _BOOST_GRAY_SLOPE = 0.79, -0.29
_SKIN_WIDTH, _SKIN_CURVE = 0.058, 1.8
_WEIGHT_PALE, _WEIGHT_DROP, _WEIGHT_CURVE, _AMOUNT_CURVE = 1.01, 0.67, 2.04, 1.03
_GRAY_PALE, _GRAY_RISE, _GRAY_BRIGHT = -0.82, 1.58, 1.86
_GRAY_DARK, _GRAY_DARK_SCALE = 1.59, 0.015


def _fitted_vibrance(
    pixel: list[float], vibrance: float, saturation: float
) -> np.ndarray:
    encoded = np.array(pixel)
    hue = colorsys.rgb_to_hsv(*pixel)[0]
    linear = _linear(encoded)
    high, low = linear.max(), linear.min()
    chroma = high - low
    if vibrance > 0:
        sat = (encoded.max() - encoded.min()) / encoded.max()
        from_red = min(hue, 1 - hue) / _SKIN_WIDTH
        skin = np.exp(-0.5 * from_red**2) * (1 - sat) ** _SKIN_CURVE
        boost = _BOOST * ((1 - sat) ** _BOOST_CURVE + _BOOST_TAIL * sat * (1 - sat))
        boost *= (1 - _BOOST_BRIGHT * encoded.max()) * (1 - skin)
        boost *= 1 - np.exp(-high / _BOOST_DARK)
        gray = low + (_BOOST_GRAY + _BOOST_GRAY_SLOPE * sat) * chroma
        scale = 1 + vibrance**_BOOST_AMOUNT_CURVE * boost
        linear = gray + scale * (linear - gray)
    elif vibrance < 0:
        sat = chroma / high
        weight = (_WEIGHT_PALE - _WEIGHT_DROP * sat**_WEIGHT_CURVE) * (
            -vibrance
        ) ** _AMOUNT_CURVE
        position = (
            _GRAY_PALE
            + _GRAY_RISE * sat
            + _GRAY_BRIGHT * high * (1 - sat)
            + _GRAY_DARK * np.exp(-high / _GRAY_DARK_SCALE)
        )
        gray = low + min(position, 1) * chroma
        linear = linear + weight * (gray - linear)
    linear = np.clip(linear, 0, 1)
    luminance = linear @ _LINEAR_LUMINANCE
    linear = luminance + (linear - luminance) * (1 + saturation)
    linear = np.clip(linear, 0, 1)
    return np.where(
        linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055
    )


@pytest.mark.parametrize(
    ("pixel", "vibrance", "saturation"),
    [
        (_PALE_BLUE, -100, 0),
        (_DEEP_BLUE, -60, 0),
        ([0.9, 0.85, 0.8], -100, -30),
        ([0.2, 0.15, 0.1], -100, 0),
        ([0.05, 0.04, 0.03], -100, 0),
        (_PALE_CYAN, 100, 0),
        ([0.78, 0.94, 0.98], 100, 50),
        ([0.32, 0.33, 0.27], 100, 50),
        (_PALE_RED, 70, 40),
    ],
)
def test_vibrance_is_the_fitted_model(
    pixel: list[float], vibrance: int, saturation: int
) -> None:
    image = np.array([[pixel]], dtype=np.float32)
    out = _applied(
        apply_vibrance(image, ColorMode.RGB, _vibrance_layer(vibrance, saturation))
    )
    expected = _fitted_vibrance(pixel, vibrance / 100, saturation / 100)
    assert np.allclose(out[0, 0], expected, atol=0.5 / 255)


def test_positive_vibrance_boosts_a_dark_pale_pixel_more_than_a_bright_one() -> None:
    """A dark olive gains more chroma than a light cyan of similar saturation."""
    image = np.array([[[0.32, 0.33, 0.27], [0.8, 0.96, 1.0]]], dtype=np.float32)
    out = _applied(apply_vibrance(image, ColorMode.RGB, _vibrance_layer(100, 0)))
    gained = _linear_chroma(out) / _linear_chroma(image)
    assert gained[0, 0] > 1.5 * gained[0, 1]


def test_negative_vibrance_lifts_a_near_black_pixel_to_its_brightest_channel() -> None:
    """The other channels close more than half the gap; the brightest stays."""
    image = np.array([[[0.02, 0.01, 0.04]]], dtype=np.float32)
    out = _applied(apply_vibrance(image, ColorMode.RGB, _vibrance_layer(-100, 0)))
    assert out[0, 0].min() > (0.01 + 0.04) / 2
    assert out[0, 0, 2] == pytest.approx(0.04, abs=1e-6)


def test_negative_vibrance_keeps_a_bright_pale_pixel_bright() -> None:
    """A pale pixel near white keeps its luminance; a dark pale one loses some."""
    image = np.array([[[0.9, 0.85, 0.8], [0.25, 0.2, 0.15]]], dtype=np.float32)
    out = _applied(apply_vibrance(image, ColorMode.RGB, _vibrance_layer(-100, 0)))
    luminance_in = _linear(image) @ _LINEAR_LUMINANCE
    luminance_out = _linear(out) @ _LINEAR_LUMINANCE
    assert luminance_out[0, 0] > luminance_in[0, 0] * 0.98
    assert luminance_out[0, 1] < luminance_in[0, 1] * 0.9


def test_vibrance_at_rest_leaves_the_image_alone() -> None:
    image = np.array([[_DEEP_BLUE]], dtype=np.float32)
    assert apply_vibrance(image, ColorMode.RGB, _vibrance_layer(0, 0)) is image
