import copy
import logging
from collections.abc import Callable

import numpy as np
import pytest

from psd_tools.api import adjustments
from psd_tools.api.layers import Group
from psd_tools.api.psd_image import PSDImage
from psd_tools.composite.adjustments import (
    apply_curves,
    apply_exposure,
    apply_huesaturation,
    apply_levels,
    apply_posterize,
    apply_threshold,
)
from psd_tools.constants import BlendMode, ColorMode, Tag
from psd_tools.psd.adjustments import Curves as CurvesData
from psd_tools.psd.adjustments import HueSaturation as HueSaturationData

from ..utils import full_name
from .test_composite import check_composite_quality, check_icc_composite_quality

logger = logging.getLogger(__name__)

# Tolerances used for MSE computation over test images (shape (200, 200, C), values in [0,1])

LOOSE = 5e-4  # tolerance for approximate adjustment algorithms which can be improved (e.g. brightness/contrast)
ACCURATE = 5e-5  # tolerance for accurate adjustment algorithms (e.g. curves)
STRICT = 5e-6  # tolerance for precise adjustment algorithms (e.g. threshold)


# Brightness & Contrast
brightnesscontrast_tests = {
    "brightnesscontrast": LOOSE,
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
