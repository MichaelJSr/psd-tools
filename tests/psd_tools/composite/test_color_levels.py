"""Color and Luminosity in an 8-bit RGB document: Photoshop 2023's integers."""

import functools

import numpy as np
import pytest
from PIL import Image

from psd_tools import PSDImage
from psd_tools.composite.blend import (
    color,
    color_levels,
    get_blend_func,
    luminosity,
    luminosity_levels,
)
from psd_tools.constants import BlendMode, ColorMode

# (backdrop, source, Photoshop 2023's 8-bit result), read off random-pixel
# probes where the rounded float formula lands a level away: no clip, then
# clipped under 0 (the clipped channel lands on 1), then over 255.
COLOR = [
    ((133, 126, 174), (222, 49, 64), (252, 79, 94)),
    ((79, 105, 190), (85, 171, 27), (63, 149, 5)),
    ((141, 3, 35), (115, 200, 219), (1, 66, 81)),
    ((144, 36, 151), (83, 172, 39), (40, 117, 1)),
    ((115, 38, 22), (7, 207, 226), (1, 83, 91)),
    ((137, 105, 172), (113, 77, 250), (128, 95, 255)),
    ((243, 171, 250), (224, 141, 62), (255, 189, 125)),
    ((0, 0, 0), (239, 107, 0), (0, 0, 0)),
]
LUMINOSITY = [
    ((29, 119, 222), (154, 44, 224), (23, 113, 216)),
    ((71, 175, 157), (151, 172, 133), (90, 194, 176)),
    ((170, 187, 2), (142, 40, 205), (93, 103, 1)),
    ((138, 248, 218), (5, 49, 128), (1, 65, 48)),
    ((121, 118, 201), (249, 199, 185), (209, 207, 255)),
    ((214, 5, 38), (187, 228, 223), (255, 198, 207)),
]


def _row(cases, i: int) -> np.ndarray:
    return np.array([[case[i] for case in cases]], dtype=np.uint8)


@pytest.mark.parametrize(
    ("blend", "cases"), [(color_levels, COLOR), (luminosity_levels, LUMINOSITY)]
)
def test_levels_match_photoshop(blend, cases) -> None:
    out = blend(_row(cases, 0) / np.float32(255), _row(cases, 1) / np.float32(255))
    assert out.dtype == np.float32
    assert np.rint(out[0] * 255).astype(int).tolist() == [list(c[2]) for c in cases]


def _picked(*args) -> object:
    blend = get_blend_func(*args)
    assert isinstance(blend, functools.partial)
    return blend.func


def test_only_eight_bit_rgb_takes_the_integers() -> None:
    assert _picked(BlendMode.COLOR, ColorMode.RGB, 8) is color_levels
    assert _picked(BlendMode.LUMINOSITY, ColorMode.RGB, 8) is luminosity_levels
    assert _picked(BlendMode.COLOR, ColorMode.RGB, 16) is color
    assert _picked(BlendMode.COLOR, ColorMode.RGB) is color
    assert _picked(BlendMode.LUMINOSITY, ColorMode.CMYK, 8) is luminosity


def test_color_layer_composites_in_levels() -> None:
    psd = PSDImage.new(mode="RGB", size=(len(COLOR), 1), depth=8)
    psd.create_pixel_layer(Image.fromarray(_row(COLOR, 0)), name="Backdrop")
    psd.create_pixel_layer(
        Image.fromarray(_row(COLOR, 1)), name="Color", blend_mode=BlendMode.COLOR
    )
    out = np.asarray(psd.composite(force=True).convert("RGB"))
    assert out[0].tolist() == [list(c[2]) for c in COLOR]
