"""A bevel's texture, drawn by the compositor: its bumps and where it tiles from.

The document is a 48 x 48 canvas with a grey 32 x 32 square carrying
``layer_effects.psd``'s inner bevel, narrowed to 2 px and textured with a
forged pattern: an 8 x 8 white tile with a black 4 x 4 block in its corner.
Inside the square, away from the bevel's own band, the texture is the only
thing that lights the surface, so its bumps can be read off row by row.
"""

import copy
import functools
import math
from typing import Any

import numpy as np
import pytest
from PIL import Image

from psd_tools.api.layers import Layer
from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import composite, paint
from psd_tools.constants import BlendMode, ColorMode, Tag
from psd_tools.psd.descriptor import Bool, Descriptor, Double, String
from psd_tools.psd.patterns import (
    Pattern,
    Patterns,
    VirtualMemoryArray,
    VirtualMemoryArrayList,
)
from psd_tools.psd.tagged_blocks import TaggedBlocks
from psd_tools.terminology import Enum, Key, Klass

from ..utils import full_name

pytest.importorskip("scipy")

_PATTERN_ID = "9d7c1e52-3b4a-4f6e-8a21-000000000b7e"
_PERIOD = 8
_SQUARE = 32
# Rows and columns of the square clear of its bevel's own band.
_INSIDE = 6

Box = tuple[int, int, int, int]


def _checker(mode: ColorMode = ColorMode.RGB) -> Pattern:
    """An 8 x 8 white tile with a black 4 x 4 block at its origin."""
    tile = np.full((_PERIOD, _PERIOD), 255, dtype=np.uint8)
    tile[: _PERIOD // 2, : _PERIOD // 2] = 0
    planes = []
    for _ in range(3 if mode == ColorMode.RGB else 1):
        plane = VirtualMemoryArray()
        plane.set_data((_PERIOD, _PERIOD), tile.tobytes(), 8)
        planes.append(plane)
    channels = planes + [VirtualMemoryArray() for _ in range(26 - len(planes))]
    return Pattern(
        image_mode=mode,
        point=(_PERIOD, _PERIOD),
        name="Checker\x00",
        pattern_id=_PATTERN_ID,
        data=VirtualMemoryArrayList(
            rectangle=(0, 0, _PERIOD, _PERIOD), channels=channels
        ),
    )


@functools.cache
def _effects_block() -> Descriptor:
    psd = PSDImage.open(full_name("layer_effects.psd"))
    layer = next(layer for layer in psd.descendants() if layer.name == "Bevel")
    return layer.tagged_blocks.get_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO)


def _texture(
    layer: Layer,
    *,
    angle: float = 90.0,
    depth: float = 400.0,
    inverted: bool = False,
    phase: tuple[float, float] = (0.0, 0.0),
    pattern_id: str = _PATTERN_ID,
) -> Descriptor:
    """Give ``layer`` the textured bevel; returns the bevel's descriptor."""
    block = copy.deepcopy(_effects_block())
    layer.tagged_blocks.set_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO, block)
    bevel = block[Klass.BevelEmboss]
    bevel[Key.Blur] = Double(2.0)
    bevel[Key.UseGlobalAngle] = Bool(False)
    bevel[Key.LocalLightingAngle] = Double(angle)
    bevel[b"useTexture"] = Bool(True)
    bevel[b"textureDepth"] = Double(depth)
    bevel[Key.InvertTexture] = Bool(inverted)
    bevel[Key.Alignment] = Bool(True)
    bevel[Key.Scale] = Double(100.0)
    pattern = Descriptor(classID=b"Ptrn")
    pattern[Key.Name] = String("Checker\x00")
    pattern[Key.ID] = String(pattern_id + "\x00")
    bevel[Enum.Pattern] = pattern
    point = Descriptor(classID=b"Pnt ")
    point[Key.Horizontal] = Double(phase[0])
    point[Key.Vertical] = Double(phase[1])
    bevel[b"phase"] = point
    return bevel


def _document(
    left: int = 8,
    top: int = 8,
    mode: str = "RGB",
    pattern_mode: ColorMode = ColorMode.RGB,
) -> tuple[PSDImage, Layer]:
    psd = PSDImage.new(mode, (48, 48))
    psd._record.layer_and_mask_information.tagged_blocks = TaggedBlocks()
    blocks = psd.tagged_blocks
    assert blocks is not None
    # ``ListElement`` carries an unbound item type, as in ``Patterns.read()``.
    patterns = Patterns([_checker(pattern_mode)])  # type: ignore[list-item]
    blocks.set_data(Tag.PATTERNS1, patterns)
    square = Image.new("RGBA", (_SQUARE, _SQUARE), (128, 128, 128, 255))
    layer = psd.create_pixel_layer(square, top=top, left=left)
    return psd, layer


def _rendered(psd: PSDImage, viewport: Box | None = None) -> np.ndarray:
    color, _, alpha = composite(psd, viewport=viewport)
    return np.concatenate((color, alpha), axis=2)


def _bumps(psd: PSDImage, bevel: Descriptor, box: Box) -> np.ndarray:
    """The textured render minus the untextured one, in grey, on ``box``."""
    textured = _rendered(psd)
    bevel[b"useTexture"] = Bool(False)
    plain = _rendered(psd)
    bevel[b"useTexture"] = Bool(True)
    left, top, right, bottom = box
    return (textured - plain)[top:bottom, left:right, 0]


def _inside(box: Box) -> Box:
    left, top, right, bottom = box
    return (left + _INSIDE, top + _INSIDE, right - _INSIDE, bottom - _INSIDE)


def _lit(
    psd: PSDImage, bevel: Descriptor, box: Box, origin: tuple[int, int], axis: int
) -> tuple[set[int], set[int]]:
    """Where the texture lights and shades a line across its blocks inside ``box``.

    ``axis`` 0 reads a column through the middle of the blocks, for a light
    from above, and 1 a row, for a light from the right. Returned are the
    positions brightened and darkened, modulo the tile, from ``origin``.
    """
    left, top, right, bottom = _inside(box)
    bumps = _bumps(psd, bevel, (left, top, right, bottom))
    if axis == 0:
        column = next(x for x in range(left, right) if (x - origin[0]) % _PERIOD == 1)
        line, start, offset = bumps[:, column - left], top, origin[1]
    else:
        row = next(y for y in range(top, bottom) if (y - origin[1]) % _PERIOD == 1)
        line, start, offset = bumps[row - top, :], left, origin[0]
    at = np.arange(start, start + line.size) - offset
    brighter = {int(i) % _PERIOD for i in at[line > 1e-3]}
    darker = {int(i) % _PERIOD for i in at[line < -1e-3]}
    return brighter, darker


def test_a_textures_dark_blocks_stand_up_and_catch_the_light() -> None:
    """Lit from above, a raised block's top edge is lit and its bottom shaded.

    The two rows astride each step take the light, and the inner bevel
    paints nothing outside the square.
    """
    psd, layer = _document()
    bevel = _texture(layer)
    assert _lit(psd, bevel, layer.bbox, (8, 8), axis=0) == ({7, 0}, {3, 4})
    bumps = _bumps(psd, bevel, psd.viewbox)
    outside = np.ones((48, 48), dtype=bool)
    outside[8:40, 8:40] = False
    assert not bumps[outside].any()


def test_lit_from_the_right_a_blocks_right_edge_takes_the_light() -> None:
    psd, layer = _document()
    bevel = _texture(layer, angle=0.0)
    assert _lit(psd, bevel, layer.bbox, (8, 8), axis=1) == ({3, 4}, {7, 0})


@pytest.mark.parametrize(
    ("point", "phase", "origin"),
    [
        ((8.0, 8.0), (0.0, 0.0), (8, 8)),
        ((10.4, 5.6), (0.0, 0.0), (10, 6)),
        ((8.0, 8.0), (3.0, -2.0), (11, 6)),
        ((-300.0, 75.0), (1.0, 1.0), (-299, 76)),
    ],
)
def test_a_layers_texture_tiles_from_its_reference_point_and_phase(
    point: tuple[float, float], phase: tuple[float, float], origin: tuple[int, int]
) -> None:
    psd, layer = _document()
    layer.reference_point = point
    bevel = _texture(layer, phase=phase)
    assert _lit(psd, bevel, layer.bbox, origin, axis=0) == ({7, 0}, {3, 4})
    bevel[Key.LocalLightingAngle] = Double(0.0)
    assert _lit(psd, bevel, layer.bbox, origin, axis=1) == ({3, 4}, {7, 0})


@pytest.mark.parametrize(
    ("blend_mode", "origin"),
    [(BlendMode.PASS_THROUGH, (0, 0)), (BlendMode.NORMAL, (13, 11))],
)
def test_a_groups_texture_tiles_from_its_layout_box(
    blend_mode: BlendMode,
    origin: tuple[int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pass-through group's from the document's origin, another from its box.

    The group's own reference point is not read, as for a pattern overlay.
    """
    psd, layer = _document(left=13, top=11)
    group = psd.create_group([layer], blend_mode=blend_mode)
    group.tagged_blocks.set_data(Tag.REFERENCE_POINT, [5.0, 3.0])
    bevel = _texture(group)
    assert group.bbox == (13, 11, 45, 43)

    origins: list[tuple[int, int] | None] = []
    draw = paint.draw_pattern_fill

    def recording(*args: Any, **kwargs: Any) -> Any:
        if args[2] is bevel:
            origins.append(kwargs.get("origin"))
        return draw(*args, **kwargs)

    monkeypatch.setattr(paint, "draw_pattern_fill", recording)
    assert _lit(psd, bevel, group.bbox, origin, axis=0) == ({7, 0}, {3, 4})
    assert origins and set(origins) == {origin}


def test_a_missing_texture_pattern_draws_the_untextured_bevel() -> None:
    psd, layer = _document()
    bevel = _texture(layer, pattern_id="00000000-0000-0000-0000-000000000000")
    textured = _rendered(psd)
    bevel[b"useTexture"] = Bool(False)
    assert np.array_equal(textured, _rendered(psd))


def test_a_texture_deeper_than_photoshop_writes_is_drawn_at_the_deepest() -> None:
    psd, layer = _document()
    bevel = _texture(layer, depth=-1e300)
    forged = _rendered(psd)
    assert np.isfinite(forged).all()
    bevel[b"textureDepth"] = Double(-1000.0)
    assert np.array_equal(forged, _rendered(psd))


def test_an_unreadable_texture_depth_drops_the_bevel() -> None:
    """As any bevel parameter no arithmetic survives does."""
    psd, layer = _document()
    bevel = _texture(layer, depth=math.nan)
    result = _rendered(psd)
    bevel[Key.Enabled] = Bool(False)
    assert np.array_equal(result, _rendered(psd))


@pytest.mark.parametrize("mode", ["CMYK", "L"])
def test_a_texture_outside_rgb_draws_the_untextured_bevel(mode: str) -> None:
    """Grayscale is textured as RGB is; another mode is not."""
    psd, layer = _document(mode=mode)
    bevel = _texture(layer)
    textured = _rendered(psd)
    bevel[b"useTexture"] = Bool(False)
    plain = _rendered(psd)
    assert np.isfinite(textured).all()
    assert np.array_equal(textured, plain) == (mode == "CMYK")


@pytest.mark.parametrize("mode", ["RGB", "L"])
def test_a_grayscale_pattern_textures_as_an_rgb_one_does(mode: str) -> None:
    psd, layer = _document(mode=mode, pattern_mode=ColorMode.GRAYSCALE)
    bevel = _texture(layer)
    assert _lit(psd, bevel, layer.bbox, (8, 8), axis=0) == ({7, 0}, {3, 4})


def test_an_inverted_texture_is_one_of_negative_depth() -> None:
    """Either turns the bumps over: the blocks sink, lit along their bottom edges."""
    psd, layer = _document()
    bevel = _texture(layer, inverted=True)
    inverted = _rendered(psd)
    assert _lit(psd, bevel, layer.bbox, (8, 8), axis=0) == ({3, 4}, {7, 0})
    bevel[Key.InvertTexture] = Bool(False)
    raised = _rendered(psd)
    bevel[b"textureDepth"] = Double(-400.0)
    sunk = _rendered(psd)
    assert np.allclose(inverted, sunk, atol=1e-5)
    assert not np.allclose(inverted, raised, atol=1e-2)


@pytest.mark.parametrize(
    "viewport",
    [(10, 10, 30, 30), (20, 17, 27, 45), (0, 0, 13, 13), (15, 30, 48, 48)],
)
def test_a_cut_through_a_texture_renders_what_the_whole_does(viewport: Box) -> None:
    """The tiling is anchored in the document, not on the viewport asked for."""
    psd, layer = _document()
    layer.reference_point = (10.0, 5.0)
    _texture(layer, phase=(3.0, 1.0))
    whole = _rendered(psd)
    left, top, right, bottom = viewport
    assert np.array_equal(_rendered(psd, viewport), whole[top:bottom, left:right])
