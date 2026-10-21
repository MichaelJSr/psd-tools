"""Several instances of one effect kind on a layer: which one is on top.

Photoshop lists the instance it shows on top first, so the compositor paints
them in reverse file order. The layer forged here is
``double-stroke-effects.psd``'s rectangle, whose effects block carries a list
for every kind that can repeat.
"""

import copy

import numpy as np
import pytest

from psd_tools.api.layers import Layer
from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import composite
from psd_tools.composite.composite import _readable, _stroke_reach
from psd_tools.constants import Tag
from psd_tools.psd.descriptor import Descriptor, Double, Enumerated, List
from psd_tools.terminology import Enum, Key, Klass

from ..utils import full_name

pytest.importorskip("scipy")

_RED = (1.0, 0.0, 0.0)
_BLUE = (0.0, 0.0, 1.0)

# The rectangle's path spans rows 3-27 and columns 4-27 of a 32x32 canvas.
_INSIDE = (slice(3, 28), slice(4, 28))
_BOTTOM = 28
_COLUMN = 16


def _rectangle() -> tuple[PSDImage, Layer]:
    psd = PSDImage.open(full_name("effects/double-stroke-effects.psd"))
    layer = next(sub for sub in psd.descendants() if sub.name == "Rectangle 1")
    assert layer.bbox == (2, 1, 30, 29)
    return psd, layer


def _color(rgb: tuple[float, float, float]) -> Descriptor:
    color = Descriptor(classID=Klass.RGBColor.value)
    color[Key.Red] = Double(255.0 * rgb[0])
    color[Key.Green] = Double(255.0 * rgb[1])
    color[Key.Blue] = Double(255.0 * rgb[2])
    return color


def _instances(layer: Layer, key: bytes, count: int) -> list[Descriptor]:
    """The first ``count`` instances of ``key``'s list, enabled and present.

    Edits the effects block itself, which ``layer.effects`` re-reads.
    """
    data = layer.tagged_blocks.get_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO_V0)
    assert isinstance(data, Descriptor)
    items = data[key]
    assert isinstance(items, List)
    while len(items) < count:
        items.append(copy.deepcopy(items[0]))
    for item in items:
        item[b"enab"] = True
        item[b"present"] = True
    return list(items)[:count]


def _strokes(
    layer: Layer, *rings: tuple[float, tuple[float, float, float], float]
) -> None:
    """Make the layer's strokes outset rings of (size, colour, opacity)."""
    for desc, (size, rgb, opacity) in zip(
        _instances(layer, b"frameFXMulti", len(rings)), rings, strict=True
    ):
        desc[Key.Style] = Enumerated(typeID=b"FStl", enum=Enum.OutsetFrame)
        desc[Key.SizeKey] = Double(size)
        desc[Key.Color] = _color(rgb)
        desc[b"Opct"] = Double(opacity)


def _over(
    t: float, top: tuple[float, float, float], under: tuple[float, float, float]
) -> tuple[float, float, float]:
    """``top`` at opacity ``t`` over an opaque ``under``."""
    r, g, b = (t * a + (1.0 - t) * u for a, u in zip(top, under, strict=True))
    return r, g, b


def _below(psd: PSDImage) -> np.ndarray:
    """The three rows under the rectangle, in the middle column."""
    color = composite(psd)[0]
    return color[_BOTTOM : _BOTTOM + 3, _COLUMN]


@pytest.mark.parametrize(
    ("first", "second"),
    [(_RED, _BLUE), (_BLUE, _RED)],
    ids=["red-first", "blue-first"],
)
def test_the_first_listed_stroke_is_on_top(
    first: tuple[float, float, float], second: tuple[float, float, float]
) -> None:
    """A 1 px ring listed over a 3 px ring shows on the 3 px one.

    Hard edges, so the rows under the rectangle are exactly one paint or the
    other: the first listed ring on the first row, the second ring past it.
    """
    psd, layer = _rectangle()
    _strokes(layer, (1.0, first, 100.0), (3.0, second, 100.0))
    np.testing.assert_allclose(_below(psd), [first, second, second], atol=1e-6)


def test_a_wider_first_listed_stroke_hides_the_one_under_it() -> None:
    psd, layer = _rectangle()
    _strokes(layer, (3.0, _BLUE, 100.0), (1.0, _RED, 100.0))
    np.testing.assert_allclose(_below(psd), [_BLUE] * 3, atol=1e-6)


@pytest.mark.parametrize("opacity", [25.0, 50.0, 75.0])
def test_a_faded_first_listed_stroke_is_composited_over_the_second(
    opacity: float,
) -> None:
    """``t * first + (1 - t) * second`` where the two rings overlap."""
    psd, layer = _rectangle()
    _strokes(layer, (1.0, _RED, opacity), (3.0, _BLUE, 100.0))
    mixed = _over(opacity / 100.0, _RED, _BLUE)
    np.testing.assert_allclose(_below(psd), [mixed, _BLUE, _BLUE], atol=1e-6)


@pytest.mark.parametrize(
    "sizes", [(1.0, 3.0), (3.0, 1.0)], ids=["wide-second", "wide-first"]
)
def test_the_stroke_reach_covers_every_instance(sizes: tuple[float, float]) -> None:
    _, layer = _rectangle()
    _strokes(layer, (sizes[0], _RED, 100.0), (sizes[1], _BLUE, 100.0))
    assert len(_readable(layer, "stroke")) == 2
    assert _stroke_reach(layer)[3] >= layer.bbox[3] + 3


def _overlays(layer: Layer, *paints: tuple[tuple[float, float, float], float]) -> None:
    """Make the layer's colour overlays (colour, opacity), strokes off."""
    for desc in _instances(layer, b"frameFXMulti", 2):
        desc[b"enab"] = False
    for desc, (rgb, opacity) in zip(
        _instances(layer, b"solidFillMulti", len(paints)), paints, strict=True
    ):
        desc[Key.Color] = _color(rgb)
        desc[b"Opct"] = Double(opacity)


@pytest.mark.parametrize(
    ("first", "second"),
    [(_RED, _BLUE), (_BLUE, _RED)],
    ids=["red-first", "blue-first"],
)
def test_the_first_listed_overlay_is_on_top(
    first: tuple[float, float, float], second: tuple[float, float, float]
) -> None:
    psd, layer = _rectangle()
    _overlays(layer, (first, 100.0), (second, 100.0))
    center = composite(psd)[0][15, _COLUMN]
    np.testing.assert_allclose(center, first, atol=1e-6)


@pytest.mark.parametrize("opacity", [25.0, 50.0, 75.0])
def test_a_faded_first_listed_overlay_is_composited_over_the_second(
    opacity: float,
) -> None:
    """``t * first + (1 - t) * second`` over the whole layer."""
    psd, layer = _rectangle()
    _overlays(layer, (_RED, opacity), (_BLUE, 100.0), (_RED, 100.0))
    mixed = _over(opacity / 100.0, _RED, _BLUE)
    inside = composite(psd)[0][_INSIDE]
    np.testing.assert_allclose(inside, np.broadcast_to(mixed, inside.shape), atol=1e-6)


@pytest.mark.parametrize("opacity", [25.0, 50.0, 75.0])
def test_a_faded_first_listed_drop_shadow_is_composited_over_the_second(
    opacity: float,
) -> None:
    """Two hard shadows cast 2 px down: ``t * first + (1 - t) * second``."""
    psd, layer = _rectangle()
    for desc in _instances(layer, b"frameFXMulti", 2):
        desc[b"enab"] = False
    paints = ((_RED, opacity), (_BLUE, 100.0))
    for desc, (rgb, value) in zip(
        _instances(layer, b"dropShadowMulti", 2), paints, strict=True
    ):
        desc[Key.Color] = _color(rgb)
        desc[b"Opct"] = Double(value)
        desc[b"Md  "] = Enumerated(typeID=b"BlnM", enum=b"Nrml")
        desc[b"uglg"] = False
        desc[b"lagl"] = Double(90.0)
        desc[b"Dstn"] = Double(2.0)
        desc[b"blur"] = Double(0.0)
        desc[b"Ckmt"] = Double(0.0)
    mixed = _over(opacity / 100.0, _RED, _BLUE)
    np.testing.assert_allclose(_below(psd), [mixed, mixed, (1.0, 1.0, 1.0)], atol=1e-6)
