"""A clipping group: the clip layers paint on their base as if it were opaque."""

import numpy as np
import pytest
from PIL import Image

from psd_tools.api.layers import Layer
from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import composite
from psd_tools.constants import BlendMode

from ..utils import full_name

BASE = (240, 200, 40)
TEXTURE = (90, 140, 230)
# The base's alpha across the columns: opaque, then ever thinner.
BASE_ALPHAS = (255, 128, 64, 1)


def _clipping_group(blend_mode: BlendMode, opacity: int, base_kind: str) -> PSDImage:
    """``TEXTURE`` clipped to a ``BASE`` that fades out to the right.

    The base is a pixel layer whose transparency or mask fades, or a Normal
    group holding the first.
    """
    # An RGB document keeps a layer's alpha as its mask, an RGBA one as its
    # transparency.
    mode = "RGB" if base_kind == "mask" else "RGBA"
    psd = PSDImage.new(mode=mode, size=(len(BASE_ALPHAS), 2))
    base = np.zeros((2, len(BASE_ALPHAS), 4), dtype=np.uint8)
    base[..., :3] = BASE
    base[..., 3] = BASE_ALPHAS
    parent = psd.create_group(name="Base") if base_kind == "group" else psd
    parent.create_pixel_layer(Image.fromarray(base, "RGBA"), "Paint")
    clipped = psd.create_pixel_layer(
        Image.new("RGBA", (len(BASE_ALPHAS), 2), (*TEXTURE, 255)),
        "Texture",
        opacity=opacity,
        blend_mode=blend_mode,
    )
    clipped.clipping = True
    return psd


@pytest.mark.parametrize("base_kind", ["transparency", "mask", "group"])
@pytest.mark.parametrize(
    "blend_mode, opacity",
    [
        (BlendMode.MULTIPLY, 255),
        (BlendMode.COLOR, 255),
        (BlendMode.NORMAL, 128),
    ],
)
def test_a_clip_layer_blends_with_its_base_colour_whatever_its_alpha(
    blend_mode: BlendMode, opacity: int, base_kind: str
) -> None:
    """The base's alpha masks the group; it does not mix the clip layer in.

    A thin edge of the base is the same blend as its opaque middle, not the
    clip layer's own colour by the base's transparency.
    """
    color, _, alpha = composite(_clipping_group(blend_mode, opacity, base_kind))

    np.testing.assert_allclose(alpha[0, :, 0], np.array(BASE_ALPHAS) / 255.0, atol=1e-6)
    # Straight colour, over nothing: what the opaque column reads.
    np.testing.assert_allclose(
        color, np.broadcast_to(color[:, :1], color.shape), atol=1e-6
    )


def test_a_multiply_clip_layer_multiplies_into_a_thin_base() -> None:
    color, _, alpha = composite(
        _clipping_group(BlendMode.MULTIPLY, 255, "transparency")
    )

    expected = np.array(BASE) * np.array(TEXTURE) / 255.0**2
    assert alpha[0, 1, 0] == pytest.approx(128 / 255.0)
    np.testing.assert_allclose(color[0, 1], expected, atol=1e-6)


def _stroked_rectangle_with_a_clip_layer() -> PSDImage:
    """``stroke-center.psd``'s red rectangle, its centred blue stroke, a green clip."""
    psd = PSDImage.open(full_name("stroke-center.psd"))
    clipped = psd.create_pixel_layer(Image.new("RGBA", psd.size, (0, 255, 0, 255)))
    clipped.clipping = True
    return psd


def test_a_vector_stroke_is_drawn_over_its_layers_clip_layers() -> None:
    """Photoshop draws a shape's stroke over what clips to the shape."""
    psd = _stroked_rectangle_with_a_clip_layer()
    shape = psd[1]
    assert isinstance(shape, Layer) and shape.has_vector_mask()
    assert shape.stroke is not None and shape.stroke.enabled

    image = psd.composite(force=True, ignore_preview=True)

    assert image is not None
    pixels = np.asarray(image.convert("RGB"), dtype=int)
    assert pixels[50, 50] == pytest.approx([0, 255, 0], abs=2), "the clip layer"
    assert pixels[50, 32] == pytest.approx([0, 0, 255], abs=2), "inner half"
    assert pixels[50, 27] == pytest.approx([0, 0, 255], abs=2), "outer half"
    assert pixels[10, 10] == pytest.approx([255, 255, 255], abs=2), "outside"
