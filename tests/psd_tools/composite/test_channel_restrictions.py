"""Blending Options > Channels: the colour channels a layer is left out of."""

import copy

import numpy as np
import pytest
from PIL import Image

from psd_tools.api.layers import Group, Layer
from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import composite
from psd_tools.constants import BlendMode, Tag
from psd_tools.psd.descriptor import Double
from psd_tools.psd.tagged_blocks import ChannelBlendingRestrictionsSetting
from psd_tools.terminology import Key, Klass

from ..utils import full_name

RED, GREEN, BLUE = 0, 1, 2
BACKDROP = (200, 100, 50)
# The preview is stored in whole bytes.
PREVIEW_ROUNDING = 1


def _restrict(layer: Layer, *channels: int) -> None:
    restricted = ChannelBlendingRestrictionsSetting()
    restricted.extend(channels)
    layer.tagged_blocks.set_data(Tag.CHANNEL_BLENDING_RESTRICTIONS_SETTING, restricted)


def _document(backdrop: bool = True) -> PSDImage:
    """A 4x4 document, over an opaque ``BACKDROP`` layer or over nothing."""
    psd = PSDImage.new(mode="RGB", size=(4, 4))
    if backdrop:
        psd.create_pixel_layer(Image.new("RGBA", (4, 4), (*BACKDROP, 255)), "Backdrop")
    return psd


def _white(parent: PSDImage | Group, alpha: int = 255) -> Layer:
    """White across the two left columns."""
    return parent.create_pixel_layer(Image.new("RGBA", (2, 4), (255, 255, 255, alpha)))


def _hide_nothing(group: Group) -> None:
    """Give ``group`` a style that draws nothing: an outer glow at 0%."""
    donor = PSDImage.open(full_name("layer_effects.psd"))
    source = next(sub for sub in donor.descendants() if sub.name == "Outer Glow")
    block = copy.deepcopy(
        source.tagged_blocks.get_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO)
    )
    block[Klass.OuterGlow][Key.Opacity] = Double(0.0)
    group.tagged_blocks.set_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO, block)


def test_photoshop_leaves_blue_out_of_the_inner_glow_layer() -> None:
    psd = PSDImage.open(full_name("layer_effects.psd"))
    layer = next(sub for sub in psd.descendants() if sub.name == "Inner Glow")
    assert list(
        layer.tagged_blocks.get_data(Tag.CHANNEL_BLENDING_RESTRICTIONS_SETTING)
    ) == [BLUE]
    x0, y0, x1, y1 = layer.bbox
    stored = psd.topil()
    assert stored is not None
    preview = np.asarray(stored.convert("RGB"), dtype=np.int16)[y0:y1, x0:x1]

    def render() -> np.ndarray:
        image = psd.composite(ignore_preview=True).convert("RGB")
        return np.asarray(image, dtype=np.int16)[y0:y1, x0:x1]

    restricted = render()
    _restrict(layer)
    unrestricted = render()

    assert np.abs(restricted - preview)[..., BLUE].max() <= PREVIEW_ROUNDING
    assert np.abs(unrestricted - preview)[..., BLUE].max() > PREVIEW_ROUNDING
    np.testing.assert_array_equal(
        restricted[..., [RED, GREEN]], unrestricted[..., [RED, GREEN]]
    )


@pytest.mark.parametrize("opacity", [255, 128])
def test_an_excluded_channel_keeps_an_opaque_backdrop(opacity: int) -> None:
    psd = _document()
    layer = _white(psd)
    layer.opacity = opacity
    _restrict(layer, RED)

    color, _, alpha = composite(psd)

    assert np.allclose(color[:, :, RED], BACKDROP[RED] / 255.0, atol=1e-6)
    assert np.all(color[:, :2, BLUE] > BACKDROP[BLUE] / 255.0)
    assert np.allclose(alpha, 1.0)


@pytest.mark.parametrize("coverage", [255, 128])
def test_an_excluded_channel_over_transparency_takes_no_colour(coverage: int) -> None:
    psd = _document(backdrop=False)
    _restrict(_white(psd, coverage), RED)

    color, _, alpha = composite(psd)

    assert np.allclose(alpha[:, :2], coverage / 255.0, atol=1e-6)
    assert np.all(color[:, :2, RED] == 0.0)
    assert np.allclose(color[:, :2, [GREEN, BLUE]], 1.0)


@pytest.mark.parametrize(
    ("blend_mode", "styled", "red"),
    [
        # A pass-through group shares the canvas, so the child keeps its red.
        (BlendMode.PASS_THROUGH, False, BACKDROP[RED]),
        # An isolated group's child stands on the group's own transparency.
        (BlendMode.NORMAL, False, 0),
        # So does a styled pass-through group's, which Photoshop renders as one
        # layer.
        (BlendMode.PASS_THROUGH, True, 0),
    ],
)
def test_a_child_keeps_the_channel_of_the_content_it_joins(
    blend_mode: BlendMode, styled: bool, red: int
) -> None:
    psd = _document()
    group = psd.create_group(name="Group", blend_mode=blend_mode)
    if styled:
        _hide_nothing(group)
    _restrict(_white(group), RED)

    color, _, _ = composite(psd)

    assert np.allclose(color[:, :2, RED], red / 255.0, atol=1e-6)
    assert np.allclose(color[:, :2, [GREEN, BLUE]], 1.0)
    assert np.allclose(color[:, 2:, RED], BACKDROP[RED] / 255.0, atol=1e-6)


@pytest.mark.parametrize("blend_mode", [BlendMode.PASS_THROUGH, BlendMode.NORMAL])
def test_a_restricted_group_leaves_the_channels_for_all_it_paints(
    blend_mode: BlendMode,
) -> None:
    psd = _document()
    group = psd.create_group(name="Group", blend_mode=blend_mode)
    _white(group)
    _restrict(group, GREEN, BLUE)

    color, _, _ = composite(psd)

    assert np.allclose(color[:, :2, RED], 1.0)
    expected = [BACKDROP[GREEN] / 255.0, BACKDROP[BLUE] / 255.0]
    assert np.allclose(color[:, :, [GREEN, BLUE]], expected, atol=1e-6)


def test_a_clip_layer_keeps_its_base_colour() -> None:
    psd = _document(backdrop=False)
    psd.create_pixel_layer(Image.new("RGBA", (4, 4), (*BACKDROP, 128)), "Base")
    clipped = _white(psd)
    clipped.clipping = True
    _restrict(clipped, RED)

    color, _, alpha = composite(psd)

    assert np.allclose(alpha, 128 / 255.0, atol=1e-6)
    assert np.allclose(color[:, :2, RED], BACKDROP[RED] / 255.0, atol=1e-6)
    assert np.allclose(color[:, :2, [GREEN, BLUE]], 1.0)


def test_an_adjustment_leaves_an_excluded_channel_as_it_was() -> None:
    psd = PSDImage.open(full_name("adjustments/invert_rgb.psd"))
    invert = psd[1]
    assert invert.kind == "invert"
    inverted, _, _ = composite(psd)
    plain, _, _ = composite(psd, layer_filter=lambda layer: layer is not invert)
    _restrict(invert, RED)

    color, _, _ = composite(psd)

    np.testing.assert_array_equal(color[..., RED], plain[..., RED])
    np.testing.assert_array_equal(
        color[..., [GREEN, BLUE]], inverted[..., [GREEN, BLUE]]
    )


def test_a_channel_the_canvas_lacks_is_ignored() -> None:
    psd = _document()
    layer = _white(psd)
    plain, _, _ = composite(psd)
    _restrict(layer, 3, 7)

    color, _, _ = composite(psd)

    np.testing.assert_array_equal(color, plain)
