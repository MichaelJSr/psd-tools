"""Several instances of one effect kind on a layer: which one is on top.

Photoshop lists the instance it shows on top first, so the compositor paints
them in reverse file order. The layer forged here is
``double-stroke-effects.psd``'s rectangle, whose effects block carries a list
for every kind that can repeat.
"""

import copy
import math

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
_GREY = (0.5, 0.5, 0.5)

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
    layer: Layer,
    *rings: tuple[float, tuple[float, float, float], float],
    style: Enum = Enum.OutsetFrame,
) -> None:
    """Make the layer's strokes rings of (size, colour, opacity), outset by default."""
    for desc, (size, rgb, opacity) in zip(
        _instances(layer, b"frameFXMulti", len(rings)), rings, strict=True
    ):
        desc[Key.Style] = Enumerated(typeID=b"FStl", enum=style)
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


def _backdrop(psd: PSDImage) -> np.ndarray:
    """The same rows as :py:func:`_below`, with the rectangle left out."""
    color = composite(psd, layer_filter=lambda layer: layer.name != "Rectangle 1")[0]
    return color[_BOTTOM : _BOTTOM + 3, _COLUMN]


@pytest.mark.parametrize("opacity", [0.0, 25.0, 50.0, 75.0])
def test_a_faded_first_listed_stroke_shows_the_backdrop_not_the_second(
    opacity: float,
) -> None:
    """The second ring is cut from under the first, whatever the first's opacity.

    So ``t * first + (1 - t) * backdrop`` on the first row, and the second
    ring only past it. Photoshop shows this at 0 %; the other opacities are
    the same rule extrapolated, with no Photoshop sample behind them.
    """
    psd, layer = _rectangle()
    backdrop = _backdrop(psd)
    _strokes(layer, (1.0, _RED, opacity), (3.0, _BLUE, 100.0))
    mixed = _over(opacity / 100.0, _RED, tuple(backdrop[0]))
    np.testing.assert_allclose(_below(psd), [mixed, _BLUE, _BLUE], atol=1e-6)


def test_a_wide_faded_first_listed_stroke_hides_the_narrow_one_under_it() -> None:
    psd, layer = _rectangle()
    backdrop = _backdrop(psd)
    _strokes(layer, (3.0, _RED, 0.0), (1.0, _BLUE, 100.0))
    np.testing.assert_allclose(_below(psd), backdrop, atol=1e-6)


@pytest.mark.parametrize("opacity", [100.0, 50.0])
def test_rings_that_meet_within_a_pixel_fill_it_between_them(opacity: float) -> None:
    """A 1.5 px ring over a 3 px one: the second row is half of each.

    The two bands add rather than composite one over the other, so no
    backdrop shows between them; a faded first ring shows it only through
    its own half. The 50 % case extrapolates the cut rule to a faded ring,
    which no Photoshop sample checks.
    """
    psd, layer = _rectangle()
    backdrop = _backdrop(psd)
    _strokes(layer, (1.5, _RED, opacity), (3.0, _BLUE, 100.0))
    first = _over(opacity / 100.0, _RED, tuple(backdrop[1]))
    half = _over(0.5, first, _BLUE)
    np.testing.assert_allclose(_below(psd)[1], half, atol=1e-6)


@pytest.mark.parametrize("layer_opacity", [255, 128])
@pytest.mark.parametrize("opacity", [100.0, 50.0])
def test_bands_keep_their_stacking_order_across_blend_modes(
    opacity: float, layer_opacity: int
) -> None:
    """Normal 1.5 px over Multiply 3 px over Normal 5 px: the second row.

    Only adjacent bands in one blend mode are added into one source, so the
    two Normal bands stay either side of the Multiply one and the first
    ring's outer half-pixel goes over the Multiply band, not under it. The
    bands of different modes still add on the pixel they share: half the
    first ring and half the multiplied backdrop, with the backdrop showing
    only through the first ring's own half when it is faded. The layer's
    opacity fades the two halves together.
    """
    psd, layer = _rectangle()
    backdrop = _backdrop(psd)
    layer.opacity = layer_opacity
    _strokes(layer, (1.5, _RED, opacity), (3.0, _GREY, 100.0), (5.0, _BLUE, 100.0))
    _instances(layer, b"frameFXMulti", 3)[1][Key.Mode] = Enumerated(
        typeID=b"BlnM", enum=Enum.Multiply
    )
    first = _over(opacity / 100.0, _RED, tuple(backdrop[1]))
    multiplied = tuple(backdrop[1] * _GREY)
    shared = _over(0.5, first, multiplied)
    expected = _over(layer_opacity / 255.0, shared, tuple(backdrop[1]))
    np.testing.assert_allclose(_below(psd)[1], expected, atol=1e-6)


@pytest.mark.parametrize("opacity", [100.0, 50.0])
def test_a_faded_normal_inset_ring_replaces_the_layer_over_a_screen_one(
    opacity: float,
) -> None:
    """Normal 1.5 px over Screen 3 px, inset: the second row inside.

    Half the pixel is the Normal ring, which replaces the layer, so the
    backdrop shows through its fade; the other half is the screened layer,
    with none of the plain layer left between them.
    """
    psd, layer = _rectangle()
    row = _BOTTOM - 2
    color = composite(psd, layer_filter=lambda sub: sub.name != "Rectangle 1")[0]
    backdrop = tuple(color[row, _COLUMN])
    _strokes(layer, (1.5, _RED, opacity), (3.0, _GREY, 100.0), style=Enum.InsetFrame)
    _instances(layer, b"frameFXMulti", 2)[1][Key.Mode] = Enumerated(
        typeID=b"BlnM", enum=Enum.Screen
    )
    color = composite(psd)[0]
    paint = color[_INSIDE][12, 12]
    first = _over(opacity / 100.0, _RED, backdrop)
    screened = tuple(1.0 - (1.0 - paint) * (1.0 - np.array(_GREY)))
    np.testing.assert_allclose(
        color[row, _COLUMN], _over(0.5, first, screened), atol=1e-6
    )


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
    # An 8-bit document holds the shadow's opacity at a level (_faded_at_8_bits).
    mixed = _over(math.floor(opacity * 255.0 / 100.0 + 0.5) / 255.0, _RED, _BLUE)
    np.testing.assert_allclose(_below(psd), [mixed, mixed, (1.0, 1.0, 1.0)], atol=1e-6)
