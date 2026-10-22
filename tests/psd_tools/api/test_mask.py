import io
import logging
from typing import Any

import pytest
from PIL import Image

from psd_tools.api.psd_image import PSDImage
from psd_tools.constants import Tag
from psd_tools.psd.layer_and_mask import MaskData

from ..utils import full_name

logger = logging.getLogger(__name__)


@pytest.fixture
def layer_mask_data() -> PSDImage:
    return PSDImage.open(full_name("layer_mask_data.psd"))


@pytest.mark.parametrize("real", [True, False])
def test_layer_mask(layer_mask_data: PSDImage, real: bool) -> None:
    mask = layer_mask_data[4].mask
    assert mask is not None
    assert mask.real_flags is not None
    mask.real_flags.parameters_applied = real
    mask.background_color
    mask.bbox
    mask.size
    mask.disabled
    mask.flags
    mask.parameters
    mask.real_flags
    repr(mask)
    assert mask.topil()


def test_mask_topil_layer_sized_background_color() -> None:
    """Regression test for issue #389: layer_sized=True must apply background_color."""
    psdimage = PSDImage.new(mode="RGB", size=(100, 100))
    layer = psdimage.create_pixel_layer(Image.new("RGB", (100, 100)))
    # Create a 40x30 black mask patch
    layer.create_mask(Image.new("L", (40, 30), 0))

    # Simulate inverted mask: bbox at (10, 20)–(50, 50), background=255
    mask_data = layer._record.mask_data
    assert isinstance(mask_data, MaskData)
    mask_data.top = 20
    mask_data.left = 10
    mask_data.bottom = 50
    mask_data.right = 50
    mask_data.background_color = 255

    mask = layer.mask
    assert mask is not None

    # Raw topil() still returns bbox-sized image (backward compat)
    raw = mask.topil(layer_sized=False)
    assert raw is not None
    assert raw.size == (40, 30)

    # layer_sized=True returns full layer image with background fill
    img = mask.topil(layer_sized=True)
    assert img is not None
    assert img.size == layer.size  # (100, 100)
    assert img.getpixel((0, 0)) == 255  # outside bbox → background_color
    assert img.getpixel((99, 99)) == 255  # outside bbox → background_color
    assert img.getpixel((10, 20)) == 0  # top-left of pasted mask data


def test_mask_disabled_setter() -> None:
    psdimage = PSDImage.new(mode="RGB", size=(30, 30))
    layer = psdimage.create_pixel_layer(Image.new("RGB", (30, 30)))
    layer.create_mask(Image.new("L", (30, 30), 200))

    mask = layer.mask
    assert mask is not None
    assert not mask.disabled

    mask.disabled = True
    assert mask.disabled

    mask.disabled = False
    assert not mask.disabled


def test_mask_parameters_without_real_channel() -> None:
    """A 36-byte mask block with no REAL_USER_LAYER_MASK channel (#693)."""
    psdimage = PSDImage.open(full_name("mask-parameters-no-real-channel.psd"))
    layer = next(layer for layer in psdimage if layer.name == "heart")
    mask = layer.mask
    assert mask is not None
    assert mask.real_flags is None
    parameters = mask.parameters
    assert parameters is not None
    assert parameters.user_mask_density is None
    assert parameters.user_mask_feather == 3.0
    assert parameters.vector_mask_density == 191
    assert parameters.vector_mask_feather == 6.0


def _digit(name: str = "0") -> tuple[PSDImage, Any]:
    psd = PSDImage.open(full_name("layer-edit.psd"))
    return psd, next(x for x in psd.descendants() if x.name == name)


def test_mask_offset_moves_the_mask_and_keeps_its_size() -> None:
    psd, digit = _digit()
    mask = digit.mask
    assert mask.bbox == (4, 30, 44, 36)
    pixels = mask.topil()
    record = digit.bbox

    mask.offset = (10, 20)

    assert mask.bbox == (10, 20, 50, 26)
    assert digit.mask.bbox == (10, 20, 50, 26)
    assert digit.bbox == record
    assert psd.is_updated()
    moved = mask.topil()
    assert moved is not None and pixels is not None
    assert moved.tobytes() == pixels.tobytes()


def test_mask_left_and_top_setters() -> None:
    _, digit = _digit()
    digit.mask.left = 0
    assert digit.mask.bbox == (0, 30, 40, 36)
    digit.mask.top -= 5
    assert digit.mask.bbox == (0, 25, 40, 31)


def test_an_unmoved_mask_keeps_the_preview_current() -> None:
    psd, digit = _digit()
    digit.mask.offset = digit.mask.offset
    assert not psd.is_updated()


@pytest.mark.parametrize("value", [(1, 2, 3), (1.5, 2), 7, None])
def test_mask_offset_takes_two_integers(value: Any) -> None:
    psd, digit = _digit()
    with pytest.raises(ValueError, match="2 integers"):
        digit.mask.offset = value
    assert digit.mask.bbox == (4, 30, 44, 36)
    assert not psd.is_updated()


def test_a_moved_mask_round_trips_through_save() -> None:
    psd, digit = _digit("1")
    digit.mask.offset = (17, 40)
    buffer = io.BytesIO()
    psd.save(buffer, update_preview=False)
    buffer.seek(0)
    again = PSDImage.open(buffer)
    moved = next(x for x in again.descendants() if x.name == "1")
    assert moved.mask is not None
    assert moved.mask.bbox == (17, 40, 45, 46)
    before, after = digit.mask.topil(), moved.mask.topil()
    assert before is not None and after is not None
    assert after.tobytes() == before.tobytes()


def test_the_real_mask_moves_with_the_pixel_mask() -> None:
    psd = PSDImage.open(full_name("mask-density-layervectormask.psd"))
    layer = psd[1]
    for key in (Tag.VECTOR_MASK_SETTING1, Tag.VECTOR_MASK_SETTING2):
        if key in layer.tagged_blocks:
            del layer.tagged_blocks[key]
    mask = layer.mask
    assert mask is not None
    data = mask.data
    assert (data.left, data.top, data.right, data.bottom) == (15, -1, 33, 9)
    assert (data.real_left, data.real_top, data.real_right, data.real_bottom) == (
        0,
        0,
        32,
        8,
    )
    mask.offset = (14, 1)
    assert (data.left, data.top, data.right, data.bottom) == (14, 1, 32, 11)
    assert (data.real_left, data.real_top, data.real_right, data.real_bottom) == (
        -1,
        2,
        31,
        10,
    )


def test_a_mask_combined_with_a_vector_mask_is_not_moved() -> None:
    psd = PSDImage.open(full_name("mask-density-layervectormask.psd"))
    layer = psd[1]
    mask = layer.mask
    assert layer.has_vector_mask() and mask is not None
    with pytest.raises(ValueError, match="vector mask"):
        mask.offset = (0, 0)
    assert mask.bbox == (15, -1, 33, 9)
    assert not psd.is_updated()


@pytest.mark.parametrize("name", ["Ellipse 1", "Rounded Rectangle 1"])
def test_a_mask_drawn_from_a_vector_mask_is_not_moved(name: str) -> None:
    # Without real-mask fields the stored mask renders the vector path alone:
    # a shape layer's, and a pixel layer's with no user mask.
    psd = PSDImage.open(full_name("clipping-mask2.psd"))
    layer = next(x for x in psd.descendants() if x.name == name)
    mask = layer.mask
    assert layer.has_vector_mask() and mask is not None
    assert mask.data.real_flags is None
    bbox = mask.bbox
    with pytest.raises(ValueError, match="vector mask"):
        mask.left += 10
    assert mask.bbox == bbox
    assert not psd.is_updated()
