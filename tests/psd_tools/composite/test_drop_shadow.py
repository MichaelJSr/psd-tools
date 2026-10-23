"""The drop shadow layer effect, drawn and composited.

Four documents in the fixture corpus carry a drop shadow Photoshop rendered
into its own preview -- ``layer_effects.psd``, ``masks.psb``,
``layer_comps.psb`` and ``layer_params.psb`` -- so the fidelity tests below
measure against those rather than against a shadow authored for the purpose.
The combinations the corpus does not pair with a shadow (a faded layer, fill
opacity, the knockout switched off, a shadow on a group or on a pixel layer
with a vector mask) are forged, by editing or copying ``masks.psb``'s.
"""

import copy
import logging
import math
import sys
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
from PIL import Image

from psd_tools.api.effects import DropShadow, Stroke
from psd_tools.api.layers import Group, Layer
from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import composite, effects, utils
from psd_tools.composite.composite import (
    Compositor,
    _beneath_reach,
    _content_bbox,
    _covers_beneath,
    _effect_reach,
    paste,
)
from psd_tools.constants import BevelStyle, BlendMode, Resource, Tag
from psd_tools.psd.base import ByteElement
from psd_tools.psd.descriptor import (
    Bool,
    Descriptor,
    Double,
    Enumerated,
    List,
    String,
)
from psd_tools.terminology import Enum, Key, Klass, Type

from ..utils import full_name
from .test_composite import _feathered_stroke_layer, _mse

pytest.importorskip("scipy")

logger = logging.getLogger(__name__)


def _disc(size: int, radius: float) -> np.ndarray:
    """A hard-edged disc centred on a ``size`` x ``size`` canvas."""
    centre = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    disc = (xx - centre) ** 2 + (yy - centre) ** 2 <= radius**2
    return disc.astype(np.float32)[:, :, None]


def _centroid(coverage: np.ndarray) -> tuple[float, float]:
    weights = coverage[:, :, 0]
    yy, xx = np.mgrid[0 : weights.shape[0], 0 : weights.shape[1]]
    total = float(weights.sum())
    return float((xx * weights).sum()) / total, float((yy * weights).sum()) / total


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (90.0, (0.0, 10.0)),
        (0.0, (-10.0, 0.0)),
        (180.0, (10.0, 0.0)),
        (-90.0, (0.0, -10.0)),
    ],
)
def test_a_drop_shadow_is_cast_away_from_the_light(
    angle: float, expected: tuple[float, float]
) -> None:
    """The shadow falls away from the light, down the image for 90 degrees.

    Photoshop's angle is where the light comes from, so the shadow falls the
    other way, and image rows run down: a light at 90 degrees is overhead and
    casts the shadow straight down. Unblurred, so the silhouette moves whole
    and its centroid moves by exactly the distance.
    """
    shape = _disc(81, 12.0)
    shadow = effects.draw_drop_shadow(
        (0, 0, 81, 81), shape, distance=10.0, angle=angle, size=0.0, spread=0.0
    )
    x, y = _centroid(shadow)
    assert (x - 40.0, y - 40.0) == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize(
    ("distance", "angle", "moved"),
    [
        (4.0, 125.0, (2, 3)),
        (10.0, 128.0, (6, 8)),
        (4.0, -10.0, (-4, -1)),
        (4.0, 30.0, (-3, 2)),
        (5.0, 60.0, (-3, 4)),
        (7.0, 120.0, (4, 6)),
    ],
    ids=[
        "down-right",
        "rounded-up",
        "up-left",
        "a-hair-under-two",
        "half-left",
        "half-right",
    ],
)
def test_a_drop_shadow_moves_the_silhouette_by_whole_pixels(
    distance: float, angle: float, moved: tuple[int, int]
) -> None:
    """Photoshop rounds each axis of the offset to the nearest pixel.

    Unblurred, the shadow is the silhouette moved whole, with no edge
    interpolated across a fraction of a pixel. ``sin(30)`` is a hair under
    a half, so that case pins rounding against flooring the offset; the
    last two fall a half pixel off an axis, a hair either side in floating
    point, and are rounded away from zero.
    """
    shape = _disc(41, 8.0)
    shadow = effects.draw_drop_shadow(
        (0, 0, 41, 41), shape, distance=distance, angle=angle, size=0.0, spread=0.0
    )
    dx, dy = moved
    assert np.array_equal(shadow, np.roll(shape, (dy, dx), axis=(0, 1)))


def test_a_drop_shadow_is_cast_by_whole_pixels_as_photoshops_preview_shows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``layer_params.psb``'s shape casts its shadow 30 px from -170 degrees.

    That is 29.54 px across and 5.21 up. Cast there, or with each axis cut
    toward zero, the shadow lies further from Photoshop's preview than cast
    at the nearest whole pixels.
    """

    def error() -> float:
        psd = PSDImage.open(full_name("layer_params.psb"))
        shape = next(x for x in psd.descendants() if x.name.startswith("Фигура"))
        left, top, right, bottom = shape.bbox
        return _error(psd, (left - 40, top - 40, right + 40, bottom + 40))

    nearest = error()
    exact = effects._cast_offset
    monkeypatch.setattr(effects, "_shadow_offset", exact)
    assert nearest < error()

    def truncated(distance: float, angle: float) -> tuple[float, float]:
        dx, dy = exact(distance, angle)
        return float(math.trunc(dx)), float(math.trunc(dy))

    monkeypatch.setattr(effects, "_shadow_offset", truncated)
    assert nearest < error()


def test_spread_grows_the_silhouette_and_takes_the_blur_it_replaces() -> None:
    """Spread is Photoshop's "expands the matte before blurring".

    The descriptor stores it under the key an inner shadow calls choke, which
    shrinks. At 100% the whole of ``size`` goes to the growth and none to the
    blur, so the shadow is the silhouette grown by ``size`` with a hard edge.
    At 0% the same reach is all blur.
    """
    shape = _disc(81, 12.0)
    hard = effects.draw_drop_shadow(
        (0, 0, 81, 81), shape, distance=0.0, angle=0.0, size=6.0, spread=100.0
    )
    soft = effects.draw_drop_shadow(
        (0, 0, 81, 81), shape, distance=0.0, angle=0.0, size=6.0, spread=0.0
    )
    row = hard[40, 40:, 0]
    # Covered out to the disc's radius plus the size, and not a pixel beyond.
    assert np.all(row[: 12 + 6 + 1] == 1.0), row
    assert np.all(row[12 + 6 + 1 :] == 0.0), row
    # The soft shadow ramps across the same reach instead.
    ramp = soft[40, 40 + 12 - 3 : 40 + 12 + 6, 0]
    assert np.all(np.diff(ramp) < 0), ramp
    assert np.all((ramp > 0.0) & (ramp < 1.0)), ramp


def test_a_drop_shadow_blur_treats_the_canvas_edge_as_empty() -> None:
    """Nothing past the canvas casts a shadow.

    ``gaussian_filter`` reflects at the edge by default, which would read a
    silhouette that touches the edge as continuing past it. Half the kernel
    falls on empty canvas at the edge pixel here, so it is about half covered.
    """
    shape = np.zeros((40, 40, 1), dtype=np.float32)
    shape[:, :10] = 1.0
    shadow = effects.draw_drop_shadow(
        (0, 0, 40, 40), shape, distance=0.0, angle=0.0, size=9.0, spread=0.0
    )
    assert 0.5 < float(shadow[20, 0, 0]) < 0.6


def test_a_drop_shadow_asks_for_scipy_by_name_when_it_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shadow needs scipy the way a stroke does, and says so the same way."""
    monkeypatch.setattr(effects, "HAS_SCIPY", False)
    with pytest.raises(ImportError, match="Drop shadow effects require: scipy"):
        effects.draw_drop_shadow(
            (0, 0, 8, 8),
            np.ones((8, 8, 1), dtype=np.float32),
            distance=1.0,
            angle=90.0,
            size=2.0,
            spread=0.0,
        )


def test_shadow_geometry_is_where_the_files_values_are_checked() -> None:
    """One value that is not finite, the angle included, is an unreadable shadow."""
    _, card = _masks_card()
    desc = _shadow_descriptor(card)
    assert effects.shadow_geometry(desc, 90.0) == (2.0, 90.0, 50.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        effects.shadow_geometry(desc, float("nan"))
    desc[Key.ChokeMatte] = Double(float("inf"))
    with pytest.raises(ValueError, match="finite"):
        effects.shadow_geometry(desc, 90.0)


def test_drop_shadow_bbox_holds_the_layer_and_the_cast_shadow() -> None:
    """The canvas grows by the size on every side and by the offset on one."""
    bbox = (10, 20, 30, 40)
    assert effects.drop_shadow_bbox(bbox, 5.0, 90.0, 3.0) == (5, 15, 35, 50)
    assert effects.drop_shadow_bbox(bbox, 5.0, 0.0, 3.0) == (0, 15, 35, 45)
    assert effects.drop_shadow_bbox((0, 0, 0, 0), 5.0, 90.0, 3.0) == (0, 0, 0, 0)


@pytest.mark.parametrize("angle", [0.0, 90.0, 180.0, -90.0, 120.0])
def test_drop_shadow_bbox_does_not_depend_on_where_the_layer_is(angle: float) -> None:
    """The same layer moved is the same canvas moved.

    ``cos(90)`` is 6e-17 rather than 0, which floors to a different column
    against 0 than against 10 unless the offset is rounded first.
    """
    here = effects.drop_shadow_bbox((0, 0, 20, 20), 5.0, angle, 3.0)
    there = effects.drop_shadow_bbox((10, 10, 30, 30), 5.0, angle, 3.0)
    assert there == tuple(value + 10 for value in here)


def test_the_shadow_canvas_holds_the_whole_shadow() -> None:
    """Nothing the shadow casts falls past the canvas it is drawn on.

    Drawn on the canvas :py:func:`drop_shadow_bbox` asks for and again with
    room to spare, the two agree, and the roomy one has nothing outside the
    first. A gaussian left to its default reach would run on past ``size``,
    and the canvas would cut it off in a visible rectangle.
    """
    bbox = (0, 0, 40, 40)
    shape = np.ones((40, 40, 1), dtype=np.float32)
    for spread in (0.0, 30.0):
        canvas = effects.drop_shadow_bbox(bbox, 7.0, 120.0, 12.0)
        roomy = (canvas[0] - 20, canvas[1] - 20, canvas[2] + 20, canvas[3] + 20)
        tight = effects.draw_drop_shadow(
            canvas,
            paste(canvas, bbox, shape),
            distance=7.0,
            angle=120.0,
            size=12.0,
            spread=spread,
        )
        wide = effects.draw_drop_shadow(
            roomy,
            paste(roomy, bbox, shape),
            distance=7.0,
            angle=120.0,
            size=12.0,
            spread=spread,
        )
        assert np.allclose(paste(roomy, canvas, tight), wide, atol=1e-6)


def _feathered_disc(size: int, radius: float, feather: float) -> np.ndarray:
    """:py:func:`_disc` fading from full coverage ``feather`` pixels inside its rim.

    The fade is quantised to 8-bit steps, as a layer's pixels are, so the rim
    pixels keep a small coverage of their own and the disc covers exactly
    what the hard one does.
    """
    centre = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    distance = np.hypot(xx - centre, yy - centre)
    fade = np.clip((radius + 0.5 - distance) / feather, 0.0, 1.0)
    quantised = np.rint(fade * 255.0) / 255.0
    return (_disc(size, radius) * quantised[:, :, None]).astype(np.float32)


def _grown_and_blurred(
    shape: np.ndarray, pixels: int, size: float, sigma_per_size: float
) -> np.ndarray:
    """The closed form of a spread shadow cast nowhere.

    The covered pixels, grown by ``pixels`` from where their coverage ends
    (a covered pixel's rim lies its own coverage past its inner side, so a
    full one ends half a pixel past its centre), and then blurred by what
    is left of ``size``.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter  # type: ignore[import-untyped]  # noqa: PLC0415

    alpha = shape[:, :, 0]
    covered = alpha >= effects._SPREAD_FLOOR
    outside, nearest = distance_transform_edt(~covered, return_indices=True)
    rim = alpha[nearest[0], nearest[1]]
    matte = np.where(covered, 1.0, np.clip(pixels + rim - outside, 0.0, 1.0))
    radius = size - pixels
    sigma = radius * sigma_per_size
    return gaussian_filter(
        matte.astype(np.float32), sigma, mode="constant", truncate=radius / sigma
    )


def test_a_feathered_disc_spread_casts_the_shadow_of_a_hard_one() -> None:
    """Spread grows the coverage, not its levels: a feather comes out hard.

    Photoshop's spread counts every pixel the silhouette covers at all as
    covered, so a disc feathered over its outer four pixels casts, at 60%
    spread, the shadow of the hard disc with the same rim: opaque out to
    the grown edge, then the blur. Fitted against Photoshop's exports of
    feathered layers; a dilation that kept the feather's own levels draws
    a shadow both narrower and lighter.
    """
    hard = _disc(81, 12.0)
    soft = _feathered_disc(81, 12.0, 4.0)
    assert np.array_equal(soft > 0.0, hard > 0.0)
    assert 0.0 < float(soft[40, 52, 0]) < 0.3, "the rim is not feathered"
    settings: dict[str, Any] = {
        "distance": 0.0,
        "angle": 0.0,
        "size": 10.0,
        "spread": 60.0,
    }
    shadow = effects.draw_drop_shadow((0, 0, 81, 81), soft, **settings)
    pixels = effects._spread_pixels(10.0, 60.0)
    assert pixels == 6
    expected = _grown_and_blurred(soft, pixels, 10.0, effects._SHADOW_SIGMA_PER_SIZE)
    assert np.allclose(shadow[:, :, 0], expected, atol=1e-6)
    # Opaque over the feather and past the rim, and ending where the hard
    # disc's shadow ends, to within the rim pixels' own coverage.
    solid = effects.draw_drop_shadow((0, 0, 81, 81), hard, **settings)
    for matte in (shadow, solid):
        assert np.all(matte[40, 40:53, 0] > 0.99), matte[40, 40:60, 0]
        assert np.all(matte[40, 53:56, 0] > 0.9), matte[40, 40:60, 0]
    assert abs(_half_coverage(shadow) - _half_coverage(solid)) < 1.0
    # Without spread the feather shows: the blur of a fading rim is lighter.
    unspread: dict[str, Any] = {**settings, "spread": 0.0}
    faded = effects.draw_drop_shadow((0, 0, 81, 81), soft, **unspread)
    full = effects.draw_drop_shadow((0, 0, 81, 81), hard, **unspread)
    assert float(faded[40, 50, 0]) < float(full[40, 50, 0]) - 0.1


def _half_coverage(shadow: np.ndarray) -> float:
    """Where a shadow's middle row falls through half coverage, rightwards."""
    row = shadow[40, 40:, 0]
    index = int(np.argmax(row < 0.5))
    above, below = float(row[index - 1]), float(row[index])
    return 40 + index - 1 + (above - 0.5) / (above - below)


@pytest.mark.parametrize("rim", [(1.0,), (0.2,), (0.5, 0.2)])
def test_the_spread_edge_follows_the_rim_pixels_own_coverage(
    rim: tuple[float, ...],
) -> None:
    """A grown edge lies the spread past where the outermost rim pixel ends.

    A fully covered rim pixel ends half a pixel past its centre; one a
    fifth covered ends a fifth of the way across it. Unblurred, the pixel
    the grown edge crosses carries just that share, and the pixels between
    are opaque whatever the feather behind the rim held: grown level by
    level, a two-pixel feather would leave them part covered.
    """
    square = np.zeros((41, 41, 1), dtype=np.float32)
    square[10:31, 10:31] = 1.0
    shape = square.copy()
    shape[10:31, 31 - len(rim) : 31, 0] = rim
    grown = effects.draw_drop_shadow(
        (0, 0, 41, 41), shape, distance=0.0, angle=0.0, size=3.0, spread=100.0
    )
    row = grown[20, 30:36, 0]
    assert np.allclose(row, [1.0, 1.0, 1.0, rim[-1], 0.0, 0.0], atol=1e-6), row


def test_spread_grows_the_coverage_not_its_level() -> None:
    """A half-covered silhouette casts an opaque spread shadow.

    Spread reads the silhouette as covered or not, so a square at half
    coverage grows into an opaque matte, where without spread the blur of
    it never passes a half. This follows from the rule the feathered edges
    fit; no export shows a flat half-covered layer on its own.
    """
    shape = np.zeros((41, 41, 1), dtype=np.float32)
    shape[15:26, 15:26] = 0.5
    grown = effects.draw_drop_shadow(
        (0, 0, 41, 41), shape, distance=0.0, angle=0.0, size=6.0, spread=100.0
    )
    assert float(grown[20, 20, 0]) == 1.0
    assert float(grown[20, 28, 0]) == 1.0
    assert float(grown[20, 35, 0]) == 0.0
    blurred = effects.draw_drop_shadow(
        (0, 0, 41, 41), shape, distance=0.0, angle=0.0, size=6.0, spread=0.0
    )
    assert float(blurred.max()) <= 0.5 + 1e-6


@pytest.mark.parametrize(
    ("size", "spread", "pixels"),
    [(5.0, 7.0, 0), (5.0, 9.0, 0), (5.0, 10.0, 1), (13.0, 59.0, 8), (5.0, 50.0, 3)],
)
def test_spread_is_taken_in_whole_pixels(
    size: float, spread: float, pixels: int
) -> None:
    """Photoshop dilates by whole pixels, rounding a half pixel up.

    Fitted against its exports: a feathered layer under a spread of a third
    of a pixel keeps its feather, which rules out rounding up, and
    ``layer_params.psb``'s 2.5 px spread renders as 3 (the fidelity bound
    below pins it), which rules out rounding down.
    """
    assert effects._spread_pixels(size, spread) == pixels


def test_a_spread_that_rounds_to_nothing_leaves_a_feather_to_the_blur() -> None:
    soft = _feathered_disc(81, 12.0, 4.0)
    settings: dict[str, Any] = {"distance": 0.0, "angle": 0.0, "size": 5.0}
    faint = effects.draw_drop_shadow((0, 0, 81, 81), soft, spread=7.0, **settings)
    unspread = effects.draw_drop_shadow((0, 0, 81, 81), soft, spread=0.0, **settings)
    assert np.array_equal(faint, unspread)
    hardened = effects.draw_drop_shadow((0, 0, 81, 81), soft, spread=10.0, **settings)
    assert float(hardened[40, 50, 0]) > float(unspread[40, 50, 0]) + 0.1


def test_a_fringe_fainter_than_a_matte_can_hold_is_not_spread() -> None:
    """Spread reads the silhouette at 8 bits, as Photoshop's matte holds it.

    A composited group's alpha can fall below a 255th and stay above zero
    where a child's faint effect trails off; Photoshop's matte holds no such
    coverage, and spread grows nothing from it.
    """
    disc = _disc(81, 12.0)
    fringed = np.maximum(disc, _disc(81, 20.0) * (0.4 / 255.0)).astype(np.float32)
    settings: dict[str, Any] = {
        "distance": 0.0,
        "angle": 0.0,
        "size": 6.0,
        "spread": 100.0,
    }
    assert np.array_equal(
        effects.draw_drop_shadow((0, 0, 81, 81), fringed, **settings),
        effects.draw_drop_shadow((0, 0, 81, 81), disc, **settings),
    )
    held = np.maximum(disc, _disc(81, 20.0) * (1.0 / 255.0)).astype(np.float32)
    grown = effects.draw_drop_shadow((0, 0, 81, 81), held, **settings)
    assert float(grown[40, 40 + 20 + 3, 0]) == 1.0


def test_the_complement_of_an_almost_opaque_pixel_is_spread() -> None:
    """A choke reads a pixel one level short of opaque as uncovered.

    Inner effects spread the complement of the silhouette, and the
    complement of 254/255, taken in float32, lands a hair under a 255th;
    it still counts as covered, so a square at that level is choked all the
    way through, where an opaque one keeps a clear centre. The rule applied
    to the complement; no export pins it on its own.
    """
    settings: dict[str, Any] = {
        "distance": 0.0,
        "angle": 0.0,
        "size": 4.0,
        "choke": 100.0,
    }
    almost = np.zeros((41, 41, 1), dtype=np.float32)
    almost[10:31, 10:31] = 254.0 / 255.0
    opaque = np.zeros((41, 41, 1), dtype=np.float32)
    opaque[10:31, 10:31] = 1.0
    choked = effects.draw_inner_shadow((0, 0, 41, 41), almost, **settings)
    banded = effects.draw_inner_shadow((0, 0, 41, 41), opaque, **settings)
    assert float(choked[20, 20, 0]) == pytest.approx(254.0 / 255.0, abs=1e-6)
    assert float(banded[20, 20, 0]) == 0.0
    assert float(banded[20, 12, 0]) == 1.0


# -- Compositing ------------------------------------------------------------ #


def _masks_card(name: str = "Rounded Rectangle 3") -> tuple[PSDImage, Layer]:
    """``masks.psb`` and one of its two cards.

    Each card is a 262 x 287 shape layer casting a black Multiply shadow of
    size 50 at 10% opacity, 2 px straight down, with the layer knocking it
    out -- Photoshop's defaults, apart from the size.
    """
    psd = PSDImage.open(full_name("masks.psb"))
    return psd, next(layer for layer in psd if layer.name == name)


def _around(layer: Layer, margin: int = 16) -> tuple[int, int, int, int]:
    left, top, right, bottom = layer.bbox
    return (left - margin, top - margin, right + margin, bottom + margin)


def _alone(
    layer: Layer, viewport: tuple[int, int, int, int], force: bool = False
) -> np.ndarray:
    """The layer's alpha, composited alone over nothing on ``viewport``."""
    height, width = viewport[3] - viewport[1], viewport[2] - viewport[0]
    compositor = Compositor(
        viewport,
        np.ones((height, width, 3), dtype=np.float32),
        np.zeros((height, width, 1), dtype=np.float32),
        force=force,
    )
    compositor.apply(layer)
    return compositor.finish()[2]


def _forge_shadow(layer: Layer, angle: float = 90.0, **values: float) -> Descriptor:
    """Give ``layer`` a copy of the first card's effects, its shadow edited.

    The shadow is lit from ``angle`` rather than from the document's global
    light, which differs between documents. ``values`` maps descriptor keys,
    by their :py:class:`Key` names, to the number to put there. Returns the
    shadow's descriptor.
    """
    _, card = _masks_card()
    block = copy.deepcopy(
        card.tagged_blocks.get_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO)
    )
    layer.tagged_blocks.set_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO, block)
    shadow = block[Klass.DropShadow]
    shadow[Key.UseGlobalAngle] = False
    shadow[Key.LocalLightingAngle] = Double(angle)
    for name, value in values.items():
        shadow[getattr(Key, name)] = Double(value)
    return shadow


def _shadow_descriptor(layer: Layer) -> Descriptor:
    return next(iter(layer.effects.find("dropshadow"))).descriptor


def test_layer_opacity_fades_the_drop_shadow_with_the_layer() -> None:
    """A half-opaque layer casts a half-opaque shadow.

    The shadow is composited apart from the layer's source, as an outset
    stroke's band is, so it carries the layer opacity itself: nothing later
    applies it where the layer covers nothing.
    """
    psd, layer = _masks_card()
    viewport = _around(layer)
    full = _alone(layer, viewport)
    layer.opacity = 128
    faded = _alone(layer, viewport)

    # Four pixels left of the card, where only the shadow paints.
    y, x = (layer.bbox[1] + layer.bbox[3]) // 2 - viewport[1], 16 - 4
    assert float(full[y, x, 0]) > 0.01, "no shadow to fade"
    assert float(faded[y, x, 0]) == pytest.approx(
        float(full[y, x, 0]) * 128 / 255.0, rel=1e-4
    )


def test_fill_opacity_leaves_the_drop_shadow_and_the_knockout_alone() -> None:
    """Fill opacity fades the layer's own paint and not its effects.

    At fill 0 the card paints nothing, and its shadow is still there at full
    strength beside it. Under it the knockout, which cuts by the layer's
    coverage rather than by its paint, leaves nothing at all: that is the
    "shadow only" setup Photoshop's knockout switch exists for.
    """
    psd, layer = _masks_card()
    viewport = _around(layer)
    full = _alone(layer, viewport)
    layer.tagged_blocks.set_data(Tag.BLEND_FILL_OPACITY, ByteElement(0))
    faded = _alone(layer, viewport)

    y = (layer.bbox[1] + layer.bbox[3]) // 2 - viewport[1]
    beside, under = 16 - 4, (layer.bbox[2] - layer.bbox[0]) // 2 + 16
    assert float(faded[y, beside, 0]) == pytest.approx(float(full[y, beside, 0]))
    assert float(faded[y, under, 0]) == 0.0


def test_a_drop_shadow_lies_beneath_the_layer_not_beside_it() -> None:
    """A shadow is attenuated by the layer over it, not scaled up to fill in.

    An outset stroke's band goes on at ``t / (1 - covered)`` so that it adds
    to the layer's coverage. A shadow is under the layer instead, so with the
    knockout off the layer composites over it like over any backdrop: at fill
    ``f`` the alpha is ``f + a0 * (1 - f)``, where ``a0`` is the shadow alone,
    read off the same card at fill 0. The band's arithmetic gives ``f + a0``
    wherever the shadow is partial.

    Read just inside the card's left edge, where the card covers the pixel
    fully and its shadow, cast 2 px down, is still ramping up.
    """
    psd, layer = _masks_card()
    _shadow_descriptor(layer)[b"layerConceals"] = False
    viewport = _around(layer)
    y = (layer.bbox[1] + layer.bbox[3]) // 2 - viewport[1]
    inside = slice(16 + 2, 16 + 20)

    layer.tagged_blocks.set_data(Tag.BLEND_FILL_OPACITY, ByteElement(0))
    shadow = _alone(layer, viewport)[y, inside, 0]
    fill = 64
    layer.tagged_blocks.set_data(Tag.BLEND_FILL_OPACITY, ByteElement(fill))
    alpha = _alone(layer, viewport)[y, inside, 0]

    assert np.all(np.diff(shadow) > 0), "the shadow is not partial here"
    f = fill / 255.0
    assert np.allclose(alpha, f + shadow * (1.0 - f), atol=1e-6)


def _soft_square(keep_stroke: bool = False) -> tuple[Layer, Descriptor]:
    """``feathered-stroke.psd``'s square casting a hard shadow straight under it.

    The square fades across a 6 px mask ramp. The shadow is black, opaque and
    spread to a band reaching 3 px past the square's box, so it covers the
    whole ramp in full. With ``keep_stroke`` the square keeps its 6 px outset stroke,
    which covers what the ramp leaves of each pixel. Returns the square and
    the shadow's descriptor.
    """
    _, layer = _feathered_stroke_layer("Outset 6")
    block = layer.tagged_blocks.get_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO)
    stroke = block[b"FrFX"]
    _, card = _masks_card()
    donor = card.tagged_blocks.get_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO)
    shadow = copy.deepcopy(donor[Klass.DropShadow])
    block.clear()
    block.update(copy.deepcopy(donor))
    block[Klass.DropShadow] = shadow
    if keep_stroke:
        block[b"FrFX"] = stroke
    shadow[Key.UseGlobalAngle] = False
    shadow[Key.LocalLightingAngle] = Double(90.0)
    for key, value in (
        (Key.Blur, 4.0),
        (Key.ChokeMatte, 100.0),
        (Key.Distance, 0.0),
        (Key.Opacity, 100.0),
    ):
        shadow[key] = Double(value)
    return layer, shadow


def _ramp_row(layer: Layer) -> tuple[np.ndarray, np.ndarray, int]:
    """The ramp's coverage, the row through it and where the ramp starts."""
    viewport = _around(layer, 8)
    mask = layer.numpy("mask")
    assert mask is not None
    ramp = mask[mask.shape[0] // 2, :6, 0]
    assert np.all((ramp > 0.0) & (ramp < 1.0)), ramp
    y = (layer.bbox[1] + layer.bbox[3]) // 2 - viewport[1]
    x0 = layer.bbox[0] - viewport[0]
    return ramp, _alone(layer, viewport)[y, :, 0], x0


def test_at_full_fill_the_knockout_leaves_a_soft_edge_its_shadow() -> None:
    """A Normal layer at full fill hides its shadow itself, so nothing is cut.

    Under the ramp the square paints ``alpha`` of each pixel and the shadow
    shows through the rest, whole: the row is opaque, as it is with the
    knockout off. Cut by ``alpha`` as well, the shadow would leave
    ``alpha + (1 - alpha) ** 2`` there and the ramp translucent.
    """
    layer, shadow = _soft_square()
    viewport = _around(layer, 8)
    ramp, row, x0 = _ramp_row(layer)
    assert np.allclose(row[x0 - 3 : x0 + 16], 1.0, atol=1e-6), row
    knocked_out = _over_grey(layer, viewport)
    shadow[b"layerConceals"] = False
    assert np.allclose(knocked_out, _over_grey(layer, viewport), atol=1e-6)


@pytest.mark.parametrize("opacity", [255, 128])
@pytest.mark.parametrize("fill", [0, 128])
def test_the_knockout_leaves_the_shadow_what_the_layers_shape_leaves(
    fill: int, opacity: int
) -> None:
    """Faded by fill opacity, a layer shows the backdrop through itself.

    The shadow shows at ``1 - alpha`` once the layer is on, the backdrop
    through the rest of what the layer's paint leaves, so with the shadow
    opaque under the ramp the alpha there is ``o * (alpha * f + 1 - alpha)``.
    At fill 0 that is the shadow cut by the layer's coverage.
    """
    layer, _ = _soft_square()
    layer.tagged_blocks.set_data(Tag.BLEND_FILL_OPACITY, ByteElement(fill))
    layer.opacity = opacity
    ramp, row, x0 = _ramp_row(layer)
    f, o = fill / 255.0, opacity / 255.0
    expected = o * (ramp * f + 1.0 - ramp)
    assert np.allclose(row[x0 : x0 + 6], expected, atol=1e-5), row[x0 : x0 + 6]


def test_a_layer_in_another_mode_cuts_its_shadow_under_its_whole_shape() -> None:
    """Only a Normal layer's paint hides its shadow, so only its cut is eased.

    A Multiply layer lets what lies beneath show through its paint, so its
    shadow is cut by ``alpha`` and the layer covers ``alpha`` of what that
    leaves: ``alpha + (1 - alpha) ** 2`` over the ramp, at full fill.
    """
    layer, _ = _soft_square()
    layer.blend_mode = BlendMode.MULTIPLY
    ramp, row, x0 = _ramp_row(layer)
    expected = ramp + (1.0 - ramp) ** 2
    assert np.allclose(row[x0 : x0 + 6], expected, atol=1e-5), row[x0 : x0 + 6]


def test_a_faint_stroke_beside_a_soft_edge_shows_the_backdrop_not_the_shadow() -> None:
    """The stroke's paint hides the shadow as the layer's does, and no more.

    Over the ramp the square and its outset stroke together claim every
    pixel whole, so nothing of the shadow may show there. At 50% the stroke
    paints half of its share, and the backdrop shows through the other half.
    """
    layer, _ = _soft_square(keep_stroke=True)
    stroke = cast(Stroke, next(iter(layer.effects.find("stroke"))))
    stroke.descriptor[Key.Opacity] = Double(50.0)
    assert stroke.opacity == 50.0
    ramp, row, x0 = _ramp_row(layer)
    expected = ramp + 0.5 * (1.0 - ramp)
    assert np.allclose(row[x0 : x0 + 6], expected, atol=1e-5), row[x0 : x0 + 6]


@pytest.mark.parametrize(
    ("mode", "opacity"),
    [(Enum.Normal, 100.0), (Enum.Normal, 50.0), (Enum.Multiply, 100.0)],
    ids=["normal", "normal at 50%", "multiply"],
)
def test_a_strokes_antialiased_rim_hides_the_shadow_at_its_opacity(
    mode: bytes, opacity: float
) -> None:
    """Beside the square the stroke alone claims ``b`` of a pixel.

    A Normal stroke at opacity ``o`` hides ``o * b`` of the shadow once on,
    so the cut leaves the shadow at ``(1 - b) / (1 - o * b)`` and it shows at
    ``1 - b``. A Multiply stroke hides none of it and the cut takes ``b``.
    The shadow is spread past the rim, where it is opaque.
    """
    layer, shadow = _soft_square(keep_stroke=True)
    shadow[Key.Blur] = Double(10.0)
    stroke = cast(Stroke, next(iter(layer.effects.find("stroke"))))
    stroke.descriptor[Key.Mode] = Enumerated(Type.BlendMode, mode)
    stroke.descriptor[Key.Opacity] = Double(opacity)
    viewport = _around(layer, 12)
    left, top, right, bottom = layer.bbox
    beside = np.ones((viewport[3] - viewport[1], viewport[2] - viewport[0]), bool)
    beside[
        top - viewport[1] : bottom - viewport[1],
        left - viewport[0] : right - viewport[0],
    ] = False

    alpha = _alone(layer, viewport)[:, :, 0]
    shadow[Key.Enabled] = False
    o = opacity / 100.0
    band = _alone(layer, viewport)[:, :, 0] / o
    rim = beside & (band > 0.0) & (band < 1.0)
    assert rim.any()
    shadow[Key.Enabled] = True
    shadow[b"layerConceals"] = False
    assert np.allclose(_alone(layer, viewport)[:, :, 0][rim], 1.0, atol=1e-6)
    b = band[rim]
    p = o if mode == Enum.Normal else 0.0
    expected = o * b + (1.0 - o * b) * (1.0 - b) / (1.0 - p * b)
    assert np.allclose(alpha[rim], expected, atol=1e-5), alpha[rim]


# The card's one stroke, inside, Normal and black, and switched off.
_STROKES = b"frameFXMulti"
_RED = {Key.Red: 255.0}
_OUTSIDE_MULTIPLY = {Key.Style: Enum.OutsetFrame, Key.Mode: Enum.Multiply}


@pytest.mark.parametrize(
    ("kind", "values", "covers"),
    [
        (Klass.DropShadow, {}, True),
        (Klass.DropShadow, {Key.Color: _RED}, False),
        (Klass.DropShadow, {Key.Color: _RED, Key.Mode: None}, False),
        (Klass.DropShadow, {Key.Mode: Enum.Screen}, False),
        (Klass.DropShadow, {Key.Mode: Enum.Normal, Key.Color: _RED}, True),
        (Klass.OuterGlow, {}, True),
        (Klass.OuterGlow, {Key.Color: {Key.Blue: 0.0}}, False),
        (Klass.OuterGlow, {Key.Color: {Key.Blue: 0.0}, Key.Opacity: 0.0}, True),
        (Klass.OuterGlow, {Key.Gradient: Descriptor()}, False),
        (_STROKES, {Key.Style: Enum.OutsetFrame, Key.Color: _RED}, True),
        (_STROKES, _OUTSIDE_MULTIPLY, True),
        (_STROKES, {**_OUTSIDE_MULTIPLY, Key.Color: _RED}, False),
        (
            _STROKES,
            {**_OUTSIDE_MULTIPLY, Key.Style: Enum.CenteredFrame, Key.Color: _RED},
            False,
        ),
        (_STROKES, {Key.Mode: Enum.Multiply, Key.Color: _RED}, True),
        (
            _STROKES,
            {
                **_OUTSIDE_MULTIPLY,
                Key.PaintType: Enum.GradientFill,
                Key.Gradient: Descriptor(),
            },
            False,
        ),
        (
            _STROKES,
            {
                **_OUTSIDE_MULTIPLY,
                Key.PaintType: Enum.Pattern,
                Key.Pattern: Descriptor(),
            },
            False,
        ),
    ],
    ids=[
        "black multiply",
        "red multiply",
        "red with no mode",
        "screen shadow",
        "normal",
        "white screen",
        "yellow screen",
        "at 0%",
        "gradient glow",
        "outside normal stroke",
        "outside black multiply stroke",
        "outside red multiply stroke",
        "centre red multiply stroke",
        "inside red multiply stroke",
        "outside gradient multiply stroke",
        "outside pattern multiply stroke",
    ],
)
def test_a_pass_through_group_covers_as_the_effects_beside_its_children_do(
    kind: bytes, values: dict[bytes, Any], covers: bool
) -> None:
    """A child's shadow, glow or outer stroke lands on the backdrop in its own mode.

    It is part of the group's alpha, so the group keeps ``1 - alpha`` of what
    lies beneath only where it does too: in Normal, or as a colour its mode
    paints whatever it lands on -- black multiplied, white screened. A drop
    shadow with no mode multiplies. A gradient or pattern is not taken to
    cover. A child effect at 0%, or a stroke inside the child, paints nothing
    beside it and has no say. ``values`` maps keys to what to put there, or
    to None to remove one, and a dict to the colour channels to set.
    """
    psd, layer = _masks_card()
    group = psd.create_group([layer])
    assert group.blend_mode == BlendMode.PASS_THROUGH
    block = layer.tagged_blocks.get_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO)
    effect = block[kind][0] if kind == _STROKES else block[kind]
    effect[Key.Enabled] = Bool(True)
    effect[b"present"] = Bool(True)
    for key, value in values.items():
        if value is None:
            del effect[key]
        elif isinstance(value, dict):
            for channel, level in value.items():
                effect[key][channel] = Double(level)
        elif isinstance(value, float):
            effect[key] = Double(value)
        elif isinstance(value, Enum):
            effect[key] = Enumerated(effect[key].typeID, value)
        else:
            effect[key] = value
    if kind != Klass.DropShadow:
        block[Klass.DropShadow][Key.Enabled] = Bool(False)
    assert sum(effect.enabled for effect in layer.effects) == 1
    assert _covers_beneath(layer, None)
    assert _covers_beneath(group, None) is covers


@pytest.mark.parametrize(
    ("style", "values", "covers"),
    [
        (BevelStyle.OUTER_BEVEL, {}, True),
        (BevelStyle.OUTER_BEVEL, {Key.HighlightColor: {Key.Blue: 0.0}}, False),
        (
            BevelStyle.OUTER_BEVEL,
            {Key.HighlightColor: {Key.Blue: 0.0}, Key.HighlightOpacity: 0.0},
            True,
        ),
        (BevelStyle.INNER_BEVEL, {Key.HighlightColor: {Key.Blue: 0.0}}, True),
        (BevelStyle.EMBOSS, {Key.ShadowColor: {Key.Red: 255.0}}, False),
        (
            BevelStyle.PILLOW_EMBOSS,
            {Key.ShadowColor: {Key.Red: 255.0}, Key.ShadowMode: Enum.Normal},
            True,
        ),
    ],
    ids=[
        "outer default",
        "outer yellow highlight",
        "outer at 0%",
        "inner",
        "emboss red shadow",
        "pillow normal shadow",
    ],
)
def test_a_pass_through_group_covers_as_the_bevels_beside_its_children_do(
    style: BevelStyle, values: dict[bytes, Any], covers: bool
) -> None:
    """A bevel that reaches outside its layer paints there in its own modes.

    Its Screen highlight and Multiply shadow cover as white and black do,
    or in Normal; an inner bevel paints over the layer and has no say.
    """
    psd, layer = _masks_card()
    group = psd.create_group([layer])
    block = layer.tagged_blocks.get_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO)
    block[Klass.DropShadow][Key.Enabled] = Bool(False)
    bevel = block[Klass.BevelEmboss]
    bevel[Key.Enabled] = Bool(True)
    bevel[b"present"] = Bool(True)
    bevel[Key.BevelStyle] = Enumerated(bevel[Key.BevelStyle].typeID, style.value)
    for key, value in values.items():
        if isinstance(value, dict):
            for channel, level in value.items():
                bevel[key][channel] = Double(level)
        elif key == Key.ShadowMode:
            bevel[key] = Enumerated(Type.BlendMode, value)
        else:
            bevel[key] = Double(value)
    assert _covers_beneath(layer, None)
    assert _covers_beneath(group, None) is covers


def test_a_drop_shadow_reaches_a_viewport_its_layer_misses() -> None:
    """The shadow is cast from the whole layer, wherever the viewport is.

    The strip below the card is clear of the card's own box, so the cull has
    to measure the shadow's reach to keep the card at all, and the
    silhouette it casts from lies wholly outside the strip. The strip has to
    come out as the same rows of a render that includes the card.
    """
    psd, layer = _masks_card()
    left, top, right, bottom = layer.bbox
    strip = (left, bottom + 4, right, bottom + 24)
    assert utils.intersect(strip, layer.bbox) == (0, 0, 0, 0)
    assert utils.intersect(strip, _effect_reach(layer)) != (0, 0, 0, 0)

    alpha = _alone(layer, strip)
    wide = _alone(layer, (left, top, right, bottom + 24))
    assert float(alpha.max()) > 0.01, "the strip lost the shadow"
    assert np.allclose(alpha, wide[bottom + 4 - top :], atol=1e-6)


def test_a_group_canvas_makes_room_for_a_child_shadow() -> None:
    """A group's content box counts the shadows its children cast.

    It is the canvas an isolated group composites its children on (#808).
    ``masks.psb``'s header band casts a 100 px shadow past the group that
    holds it, which the group's own box, a union of its children's boxes,
    does not reach.
    """
    psd = PSDImage.open(full_name("masks.psb"))
    header = next(layer for layer in psd if layer.name == "Header")
    band = next(layer for layer in psd.descendants() if layer.name == "Rectangle 2")
    assert band.parent is header
    reach = _beneath_reach(band)
    assert reach[3] > header.bbox[3]
    assert _content_bbox(header)[3] >= reach[3]


def _header(blend_mode: BlendMode) -> tuple[Layer, Layer]:
    """``masks.psb``'s header group, and the band inside it."""
    psd = PSDImage.open(full_name("masks.psb"))
    header = next(layer for layer in psd if layer.name == "Header")
    header.blend_mode = blend_mode
    band = next(layer for layer in psd.descendants() if layer.name == "Rectangle 2")
    return header, band


def _forge_header_effects(layer: Layer) -> Descriptor:
    """Give ``layer`` a copy of the header band's effects, and return the block.

    The band carries a drop shadow and a faint 2 px outside stroke, which is
    the pairing the knockout and a group's stroke are measured with.
    """
    _, band = _header(BlendMode.PASS_THROUGH)
    block = copy.deepcopy(
        band.tagged_blocks.get_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO)
    )
    layer.tagged_blocks.set_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO, block)
    return block


@pytest.mark.parametrize("blend_mode", [BlendMode.PASS_THROUGH, BlendMode.NORMAL])
def test_a_group_casts_its_shadow_from_its_childrens_effects_too(
    blend_mode: BlendMode,
) -> None:
    """A group's shadow is drawn on a canvas that holds its whole silhouette.

    The band's own shadow, made hard and cast 40 px down, runs past the
    group's box, which is a union of its children's own boxes. The group's
    shadow, cast 20 px further, has to reach 20 px past the band's; grown
    from the group's box alone, the silhouette would be cut off before that
    and the group's shadow with it.
    """
    header, band = _header(blend_mode)
    band_shadow = _shadow_descriptor(band)
    for key, value in ((Key.Blur, 0.0), (Key.Distance, 40.0), (Key.Opacity, 100.0)):
        band_shadow[key] = Double(value)
    viewport = (0, 0, 640, 200)
    # The band ends at 102 and its shadow at 142, so 150 is the group's alone.
    y, x = 150, 320
    assert float(_alone(header, viewport)[y, x, 0]) == 0.0

    _forge_shadow(header, Blur=0.0, Distance=20.0, Opacity=100.0)
    assert float(_alone(header, viewport)[y, x, 0]) == pytest.approx(1.0)


@pytest.mark.parametrize("blend_mode", [BlendMode.PASS_THROUGH, BlendMode.NORMAL])
def test_a_groups_stroke_follows_the_shadow_its_child_casts_past_it(
    blend_mode: BlendMode,
) -> None:
    """A group's stroke is drawn on a box grown from what the group paints.

    The band's shadow, made hard and cast 40 px down, runs 40 px past the
    group's box, and an outside stroke on the group has to follow the
    shadow's lower edge out there. Grown from the group's own box, the
    stroke's canvas would end a few pixels below it, and the band along that
    edge would never be drawn.
    """
    header, band = _header(blend_mode)
    band_shadow = _shadow_descriptor(band)
    for key, value in ((Key.Blur, 0.0), (Key.Distance, 40.0), (Key.Opacity, 100.0)):
        band_shadow[key] = Double(value)
    block = _forge_header_effects(header)
    block[Klass.DropShadow][Key.Enabled] = False
    stroke = block[Klass.FrameFX]
    stroke[Key.Opacity] = Double(100.0)
    stroke[Key.SizeKey] = Double(4.0)
    stroke[Key.Style] = Enumerated(Type.FrameFill, Enum.OutsetFrame)

    viewport = (0, 0, 640, 200)
    stroke[Key.Enabled] = False
    without = _alone(header, viewport)[:, 320, 0]
    stroke[Key.Enabled] = True
    alpha = _alone(header, viewport)[:, 320, 0]
    # The first clear row below the band's shadow, well past the group's box.
    edge = header.bbox[3] + int(np.argmax(without[header.bbox[3] :] == 0.0))
    assert edge > header.bbox[3] + 8 and float(without[edge - 1]) == 1.0
    assert np.all(alpha[edge : edge + 4] == pytest.approx(1.0, abs=1e-2))
    assert float(alpha[edge + 6]) == 0.0


def test_the_knockout_cuts_the_shadow_under_the_layers_stroke_too() -> None:
    """The knockout cuts the shadow under the layer's stroke band as well.

    The header band has a faint 2 px outside stroke, which its box holds as
    its last two rows. Its shadow, made hard and cast 20 px down, would lie
    under the stroke along the bottom edge; Photoshop cuts it away there, so
    the stroke shows the backdrop through itself and those rows come out as
    they do with the shadow off, while below the stroke the shadow is whole.
    """
    _, band = _header(BlendMode.PASS_THROUGH)
    stroke = cast(Stroke, next(iter(band.effects.find("stroke"))))
    assert stroke.enabled and stroke.opacity < 100.0 and stroke.size == 2.0
    viewport = (0, 0, 640, 200)
    bottom, x = band.bbox[3] - 2, 320

    shadow = _shadow_descriptor(band)
    shadow[Key.Enabled] = False
    without = _alone(band, viewport)
    shadow[Key.Enabled] = True
    for key, value in ((Key.Blur, 0.0), (Key.Distance, 20.0), (Key.Opacity, 100.0)):
        shadow[key] = Double(value)
    alpha = _alone(band, viewport)

    assert float(without[bottom - 1, x, 0]) == 1.0, "the band ends elsewhere"
    assert 0.0 < float(without[bottom, x, 0]) < 1.0, "no faint stroke here"
    assert float(without[bottom + 2, x, 0]) == 0.0, "the stroke runs on"
    assert np.allclose(alpha[bottom : bottom + 2, x], without[bottom : bottom + 2, x])
    assert float(alpha[bottom + 2, x, 0]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("style", "at_edge", "beyond_edge"),
    [
        (Enum.OutsetFrame, 1.0, 1.0),
        (Enum.CenteredFrame, 1.0, 0.5),
        (Enum.InsetFrame, 0.5, 0.0),
    ],
)
def test_the_knockout_adds_the_stroke_band_to_the_silhouette(
    monkeypatch: pytest.MonkeyPatch, style: bytes, at_edge: float, beyond_edge: float
) -> None:
    """Only the part of the band outside the boundary adds to the silhouette.

    An outset band is complementary to the coverage it is traced from, so a
    pixel the edge half covers is half layer and half stroke, and the shadow
    under it is cut away whole; a union of the two would leave half of it
    showing through every antialiased edge. An inset band paints over the
    layer, and on that pixel equals its coverage: added it would cut the
    shadow from under the uncovered half too, so the knockout there stays at
    the coverage. A centred band is half each.

    The layer is a 21 px square whose last column is half covered, so the
    boundary the band is traced from cuts that column's pixels in half.
    """
    psd = PSDImage.new("RGB", (41, 41))
    square = Image.new("RGBA", (21, 21), (255, 255, 255, 255))
    for y in range(21):
        square.putpixel((20, y), (255, 255, 255, 128))
    layer = psd.create_pixel_layer(square, top=10, left=10)
    assert layer.bbox == (10, 10, 31, 31)
    _forge_header_effects(layer)
    stroke = cast(Stroke, next(iter(layer.effects.find("stroke"))))
    assert stroke.size == 2.0 and stroke.opacity < 100.0
    stroke.descriptor[Key.Style] = Enumerated(Type.FrameFill, style)

    viewport = psd.viewbox
    compositor = Compositor(
        viewport,
        np.ones((41, 41, 3), dtype=np.float32),
        np.zeros((41, 41, 1), dtype=np.float32),
    )
    silhouette = compositor._cast_shape(layer, viewport)
    assert float(silhouette[20, 30, 0]) == pytest.approx(128 / 255.0)
    knockout = compositor._knockout_shape(layer, viewport, silhouette)
    assert float(knockout[20, 20, 0]) == 1.0
    assert float(knockout[20, 30, 0]) == pytest.approx(at_edge, abs=1e-2)
    assert float(knockout[20, 31, 0]) == pytest.approx(beyond_edge, abs=1e-2)
    assert float(knockout[20, 34, 0]) == 0.0


@pytest.mark.parametrize("blend_mode", [BlendMode.PASS_THROUGH, BlendMode.NORMAL])
def test_a_groups_stroke_and_its_knockout_follow_the_groups_alpha(
    blend_mode: BlendMode,
) -> None:
    """A group's stroke is traced from its alpha, and the knockout cuts under it.

    A half-opaque child states a boundary at every one of its pixels, so the
    group's outside stroke fills the remainder ``1 - c`` of each, as it does
    across a pixel layer's partial alpha. The knockout cuts the shadow from
    under the whole band whatever the stroke's opacity, so with it on none of
    the shadow is left under the child, and with it off the shadow shows
    there.
    """
    psd = PSDImage.new("RGB", (60, 60))
    child = psd.create_pixel_layer(
        Image.new("RGBA", (20, 20), (255, 255, 255, 255)), top=20, left=20
    )
    child.opacity = 128
    group = psd.create_group([child])
    group.blend_mode = blend_mode
    block = _forge_header_effects(group)
    shadow = block[Klass.DropShadow]
    shadow[Key.UseGlobalAngle] = False
    for key, value in ((Key.Blur, 0.0), (Key.Distance, 0.0), (Key.Opacity, 100.0)):
        shadow[key] = Double(value)
    assert cast(
        DropShadow, next(iter(group.effects.find("dropshadow")))
    ).layer_knocks_out
    stroke = _strokes(group)[0]
    assert 0.0 < stroke.opacity < 100.0, "a faint stroke, so the knockout shows"

    c = 128 / 255.0
    painted = c + (1.0 - c) * stroke.opacity / 100.0
    assert float(_alone(group, psd.viewbox)[30, 30, 0]) == pytest.approx(
        painted, abs=1e-5
    )
    shadow[b"layerConceals"] = False
    assert float(_alone(group, psd.viewbox)[30, 30, 0]) > painted + 0.1
    shadow[Key.Enabled] = False
    assert float(_alone(group, psd.viewbox)[30, 30, 0]) == pytest.approx(
        painted, abs=1e-5
    )


def test_a_groups_knockout_traces_its_stroke_from_the_groups_alpha() -> None:
    """The band is traced from the alpha the group's stroke is drawn around.

    The icons cover their pixels fully at 40% opacity, so their alpha states
    a boundary at every one of those pixels: the outset band fills the rest of
    each, and the knockout is whole across the icons.
    """
    from scipy.ndimage import binary_erosion  # type: ignore[import-untyped]  # noqa: PLC0415

    social = _social(BlendMode.NORMAL)
    _, band = _header(BlendMode.PASS_THROUGH)
    social.tagged_blocks.set_data(
        Tag.OBJECT_BASED_EFFECTS_LAYER_INFO,
        copy.deepcopy(band.tagged_blocks.get_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO)),
    )
    assert _strokes(social), "no stroke to trace"
    viewport = _SOCIAL_VIEWPORT
    height, width = viewport[3] - viewport[1], viewport[2] - viewport[0]
    compositor = Compositor(
        viewport,
        np.ones((height, width, 3), dtype=np.float32),
        np.zeros((height, width, 1), dtype=np.float32),
    )
    silhouette = compositor._cast_shape(social, viewport)
    knockout = compositor._knockout_shape(social, viewport, silhouette)

    # Three pixels into an icon, past the reach of the band around it.
    icons = np.isclose(silhouette[:, :, 0], 102 / 255.0, atol=1e-6)
    inside = binary_erosion(icons, iterations=3)
    assert inside.any(), "no icon interior"
    assert np.allclose(knockout[inside], 1.0, atol=1e-5)


def _strokes(layer: Layer) -> list[Stroke]:
    return [cast(Stroke, effect) for effect in layer.effects.find("stroke")]


def test_a_stroke_too_large_to_draw_does_not_cut_the_shadow() -> None:
    """A stroke the document's budget refuses is not drawn, and cuts nothing.

    The budget holds the shadow's canvas and not the stroke's, so the layer
    renders as it does with the stroke switched off.
    """
    psd = PSDImage.open(full_name("masks.psb"))
    band = next(layer for layer in psd.descendants() if layer.name == "Rectangle 2")
    stroke = _strokes(band)[0]
    stroke.descriptor[Key.SizeKey] = Double(100.0)
    shadow = _shadow_descriptor(band)
    for key, value in ((Key.Blur, 0.0), (Key.Distance, 20.0), (Key.Opacity, 100.0)):
        shadow[key] = Double(value)
    geometry = effects.shadow_geometry(shadow, _drop_shadow(band).angle)
    grown = effects.drop_shadow_bbox(
        band.bbox, geometry.distance, geometry.angle, geometry.size
    )
    budget = (grown[2] - grown[0]) * (grown[3] - grown[1]) * effects._SHADOW_BYTES
    with pytest.raises(ValueError):
        effects.stroke_bbox(band.bbox, stroke.descriptor, budget)
    psd.max_alloc_bytes = budget
    viewport = (0, 0, 640, 300)

    alpha = _alone(band, viewport)
    stroke.descriptor[Key.Enabled] = False
    assert np.array_equal(alpha, _alone(band, viewport))
    # The shadow is drawn, just past the band's bottom edge.
    assert float(alpha[band.bbox[3] + 4, 320, 0]) == pytest.approx(1.0)


def test_a_pass_through_group_keeps_its_shadow_beneath_it() -> None:
    """The shadow of a pass-through group does not paint over the group.

    A pass-through group is composited before what goes on beside it, which
    would put its shadow over it; the shadow goes on before the group is
    resolved instead, and the group is resolved over it. With the knockout
    off, the band's middle has to come out as it does with no group shadow at
    all, and below the band the shadow has to be there.
    """

    def render(with_shadow: bool) -> np.ndarray:
        header, _ = _header(BlendMode.PASS_THROUGH)
        if with_shadow:
            shadow = _forge_shadow(header, Blur=10.0, Distance=30.0, Opacity=100.0)
            shadow[b"layerConceals"] = False
        viewport = (0, 0, 640, 260)
        height, width = viewport[3], viewport[2]
        compositor = Compositor(
            viewport,
            np.ones((height, width, 3), dtype=np.float32),
            np.ones((height, width, 1), dtype=np.float32),
        )
        compositor.apply(header)
        return compositor.result_over_backdrop()

    shadowed, plain = render(True), render(False)
    assert np.allclose(shadowed[50, 320], plain[50, 320], atol=1e-6)
    assert not np.allclose(shadowed[120, 320], plain[120, 320], atol=1e-2)


# The social icons in ``masks.psb``'s footer: a pass-through group of four
# shape layers, each at layer opacity 102.
_SOCIAL_VIEWPORT = (150, 960, 490, 1100)


def _social(blend_mode: BlendMode) -> Group:
    psd = PSDImage.open(full_name("masks.psb"))
    social = next(layer for layer in psd.descendants() if layer.name == "Social")
    social.blend_mode = blend_mode
    return cast(Group, social)


def _composited(
    group: Layer, viewport: tuple[int, int, int, int] = _SOCIAL_VIEWPORT
) -> np.ndarray:
    """``group`` over an opaque white backdrop, colour and alpha together."""
    height, width = viewport[3] - viewport[1], viewport[2] - viewport[0]
    compositor = Compositor(
        viewport,
        np.ones((height, width, 3), dtype=np.float32),
        np.ones((height, width, 1), dtype=np.float32),
    )
    compositor.apply(group)
    color, _, alpha = compositor.finish()
    return np.concatenate((compositor.result_over_backdrop(), alpha), axis=2)


def test_a_pass_through_group_casts_the_shadow_a_normal_one_does() -> None:
    """Over contents that blend normally, pass-through changes nothing.

    The icons are 40% opaque, so their shadow shows through them wherever it
    lies under them. Held back from what the group covers after the fact, it
    would come out at ``a + s * (1 - a) ** 2`` there rather than
    ``a + s * (1 - a)``, and over the icons' colour rather than under it.
    """
    rendered = []
    for blend_mode in (BlendMode.PASS_THROUGH, BlendMode.NORMAL):
        social = _social(blend_mode)
        shadow = _forge_shadow(social, Blur=3.0, Distance=4.0, Opacity=100.0)
        shadow[b"layerConceals"] = False
        rendered.append(_composited(social))
    passthrough, normal = rendered
    assert not np.allclose(passthrough, _composited(_social(BlendMode.NORMAL)))
    assert np.allclose(passthrough, normal, atol=1e-5)


def test_a_faded_pass_through_group_is_resolved_over_its_shadow() -> None:
    """A pass-through group's contents blend with its shadow under them.

    At fill 128 the group is not re-applied by interpolation, but its
    contents are still resolved over the canvas beneath it, which has to
    hold the shadow by then. The icons are white and set to Multiply, so
    over their own shadow they change nothing, and what is left at an icon
    is the shadow alone: 40% black over white.
    """
    social = _social(BlendMode.PASS_THROUGH)
    social.tagged_blocks.set_data(Tag.BLEND_FILL_OPACITY, ByteElement(128))
    for icon in social:
        icon.blend_mode = BlendMode.MULTIPLY
    shadow = _forge_shadow(social, Blur=0.0, Distance=0.0, Opacity=100.0)
    shadow[b"layerConceals"] = False
    contents = _alone(_social(BlendMode.PASS_THROUGH), _SOCIAL_VIEWPORT)[:, :, 0]
    rendered = _composited(social)

    icons = np.argwhere(np.isclose(contents, 102 / 255.0, atol=1e-6))
    assert len(icons), "no fully covered icon pixel"
    y, x = icons[len(icons) // 2]
    assert np.allclose(rendered[y, x, :3], 1.0 - 102 / 255.0, atol=1e-5)


@pytest.mark.parametrize("knockout", [1, 2], ids=["shallow", "deep"])
def test_a_group_shadow_does_not_depend_on_the_viewport(knockout: int) -> None:
    """The silhouette is read the same way whatever the viewport holds.

    One icon knocks out at fill 0. Resolved over the canvas, a knockout
    takes alpha from the backdrop it reaches, so a silhouette read off the
    resolved group would cast a shadow from the knocked-out icon in a full
    render and none in a render of the shadow alone. Read afresh over
    nothing, the strip below the icons comes out the same in both.
    """
    social = _social(BlendMode.PASS_THROUGH)
    icon = next(layer for layer in social if layer.name == "facebook")
    icon.tagged_blocks.set_data(Tag.KNOCKOUT_SETTING, ByteElement(knockout))
    icon.tagged_blocks.set_data(Tag.BLEND_FILL_OPACITY, ByteElement(0))
    _forge_shadow(social, Blur=0.0, Distance=60.0, Opacity=100.0)
    full = _composited(social)
    left, top, right, bottom = _SOCIAL_VIEWPORT
    strip = (left, 1030, right, bottom)
    assert np.allclose(_composited(social, strip), full[1030 - top :], atol=1e-6)


def test_nested_shadowed_groups_are_resolved_once_each() -> None:
    """Shadows on nested groups cost linearly in the nesting, not exponentially.

    A group's silhouette is its contents composited over nothing, once per
    box and pass, so the one layer at the bottom of eight groups that each
    cast a shadow is resolved once for the render and once for each group's
    silhouette. Resolving a pass-through group again over its own shadow
    would double that at every level.
    """
    psd = PSDImage.new("RGB", (80, 80))
    leaf = psd.create_pixel_layer(
        Image.new("RGBA", (20, 20), (255, 255, 255, 255)), top=30, left=30
    )
    node: Layer = leaf
    for depth in range(8):
        node = psd.create_group([node], name="Group %d" % depth)
        _forge_shadow(node, Blur=0.0, Distance=3.0, Opacity=100.0)

    resolved = 0
    resolve = Compositor._resolve_source

    def counting(compositor: Compositor, layer: Layer) -> Any:
        nonlocal resolved
        resolved += layer is leaf
        return resolve(compositor, layer)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Compositor, "_resolve_source", counting)
        _alone(node, psd.viewbox)
    assert resolved == 8 + 1


@pytest.mark.parametrize("blend_mode", [BlendMode.PASS_THROUGH, BlendMode.NORMAL])
def test_a_group_casts_its_shadow_from_how_opaque_its_contents_are(
    blend_mode: BlendMode,
) -> None:
    """A group's silhouette is its contents' alpha, not their coverage.

    The icons cover their pixels fully at 40% opacity, so their hard shadow,
    cast clear of them, is 40% opaque too, and an icon faded to nothing
    casts nothing. Read off coverage, every icon would cast an opaque
    shadow, the invisible one included.
    """
    social = _social(blend_mode)
    faded = next(layer for layer in social if layer.name == "facebook")
    faded.opacity = 0
    viewport = _SOCIAL_VIEWPORT
    contents = _alone(social, viewport)[:, :, 0]
    _forge_shadow(social, Blur=0.0, Distance=60.0, Opacity=100.0)
    alpha = _alone(social, viewport)[:, :, 0]

    icons = np.argwhere(np.isclose(contents, 102 / 255.0, atol=1e-6))
    assert len(icons), "no fully covered icon pixel"
    for y, x in icons[:: max(1, len(icons) // 50)]:
        assert float(alpha[y + 60, x]) == pytest.approx(102 / 255.0, abs=1e-5)
    # Under where the faded icon would have cast its shadow.
    left, top, right, bottom = faded.bbox
    below = alpha[
        top + 60 - viewport[1] : bottom + 60 - viewport[1],
        left - viewport[0] : right - viewport[0],
    ]
    assert float(below.max()) == 0.0


@pytest.mark.parametrize("force", [False, True])
def test_a_masked_pixel_layer_casts_its_shadow_from_its_pixels(force: bool) -> None:
    """The silhouette is what the layer covers, not the outline of its mask.

    ``masks/2.psd``'s third ellipse is a pixel layer under a vector mask that
    spans the whole canvas. A stroke traces that mask under ``force``; a
    shadow cast from it would cover the whole canvas and be knocked out under
    all of it, leaving nothing. Both force modes cover the same pixels, so
    they have to cast the same shadow.
    """
    psd = PSDImage.open(full_name("masks/2.psd"))
    layer = next(layer for layer in psd if layer.name == "ellipse3")
    assert layer.bbox == (8, 16, 56, 64)
    # Lit from the right, so the shadow falls 6 px to the left.
    _forge_shadow(layer, angle=0.0, Blur=0.0, Distance=6.0, Opacity=100.0)
    alpha = _alone(layer, psd.viewbox, force=force)
    reference = _alone(layer, psd.viewbox)
    assert np.allclose(alpha, reference, atol=1e-6)
    # Left of the ellipse, where only its shadow can be: opaque, faded by the
    # layer's own opacity.
    assert float(alpha[40, 4, 0]) == pytest.approx(layer.opacity / 255.0)


def _stroked_rectangle(alignment: bytes, fill_enabled: bool) -> Layer:
    """``stroke.psd``'s rectangle, its 1.47 px vector stroke moved and its fill set.

    The shadow forged on it is lit from the right and hard, so it falls 6 px
    to the left of the rectangle, where only the shadow can paint.
    """
    psd = PSDImage.open(full_name("stroke.psd"))
    layer = next(x for x in psd.descendants() if x.name == "Rectangle 1")
    stroke = layer.tagged_blocks.get_data(Tag.VECTOR_STROKE_DATA)
    stroke[b"strokeStyleLineAlignment"] = Enumerated(
        b"strokeStyleLineAlignment", alignment
    )
    stroke[b"fillEnabled"] = Bool(fill_enabled)
    layer.tagged_blocks.set_data(Tag.VECTOR_STROKE_DATA, stroke)
    _forge_shadow(layer, angle=0.0, Blur=0.0, Distance=6.0, Opacity=100.0)
    return layer


def test_a_shape_layer_with_its_fill_off_casts_its_shadow_from_its_stroke() -> None:
    """The silhouette is what the layer paints, not the fill under it.

    With the fill switched off and the layer redrawn from its path, the
    layer is its inner vector stroke, a ring, and its shadow is that ring
    moved: hollow inside. Cast from the fill it would be a solid block.
    """
    layer = _stroked_rectangle(b"strokeStyleAlignInside", fill_enabled=False)
    viewport = _around(layer)
    alpha = _alone(layer, viewport, force=True)
    _shadow_descriptor(layer)[Key.Enabled] = False
    ring = _alone(layer, viewport, force=True)

    y = (layer.bbox[1] + layer.bbox[3]) // 2 - viewport[1]
    left = int(np.argmax(ring[y, :, 0] > 0.0))
    assert float(ring[y, left, 0]) > 0.5 and float(ring[y, left + 4, 0]) == 0.0, (
        "not a ring"
    )
    # Left of the ring only the shadow paints: the ring, moved 6 px.
    assert np.allclose(alpha[:, left - 6 : left], ring[:, left : left + 6], atol=1e-6)


def test_a_shape_layers_shadow_reaches_as_far_as_its_outer_vector_stroke() -> None:
    """A redrawn outer stroke casts a shadow along with the fill it outlines.

    The stroke reaches past the path, so the shadow of the stroked rectangle
    starts a pixel further out than the fill's would.
    """
    layer = _stroked_rectangle(b"strokeStyleAlignOutside", fill_enabled=True)
    viewport = _around(layer)
    alpha = _alone(layer, viewport, force=True)
    _shadow_descriptor(layer)[Key.Enabled] = False
    stroked = _alone(layer, viewport, force=True)
    layer.tagged_blocks.get_data(Tag.VECTOR_STROKE_DATA)[b"strokeEnabled"] = Bool(False)
    filled = _alone(layer, viewport, force=True)

    y = (layer.bbox[1] + layer.bbox[3]) // 2 - viewport[1]
    left = int(np.argmax(stroked[y, :, 0] > 0.0))
    assert left < int(np.argmax(filled[y, :, 0] > 0.0)), (
        "the stroke does not reach past the path"
    )
    # Left of the stroke only the shadow paints: the stroked layer, moved 6 px.
    assert np.allclose(
        alpha[:, left - 6 : left], stroked[:, left : left + 6], atol=1e-6
    )


@pytest.mark.parametrize("fill_enabled", [False, True], ids=["no fill", "fill"])
@pytest.mark.parametrize(
    "alignment",
    [b"strokeStyleAlignInside", b"strokeStyleAlignCenter", b"strokeStyleAlignOutside"],
    ids=["inside", "centre", "outside"],
)
def test_a_redrawn_shape_casts_what_it_is_resolved_to(
    fill_enabled: bool, alignment: bytes
) -> None:
    """Under ``force`` the silhouette is the alpha the layer resolves to, before its opacity.

    That is the fill and the stroke as the layer composites them: nothing
    inside the path with the fill off, and the stroke wherever it is drawn.
    """
    layer = _stroked_rectangle(alignment, fill_enabled)
    _shadow_descriptor(layer)[Key.Enabled] = False
    viewport = _around(layer)
    height, width = viewport[3] - viewport[1], viewport[2] - viewport[0]
    compositor = Compositor(
        viewport,
        np.ones((height, width, 3), dtype=np.float32),
        np.zeros((height, width, 1), dtype=np.float32),
        force=True,
    )
    source = compositor._resolve_source(layer)
    silhouette = compositor._cast_shape(layer, viewport)
    assert np.allclose(silhouette, source.alpha / source.opacity, atol=1e-6)
    assert float(silhouette[height // 2, width // 2, 0]) == float(fill_enabled)
    assert silhouette.max() > 0.0, "the stroke casts"


@pytest.mark.parametrize(
    "filename",
    [
        "stroke-center.psd",
        "stroke-center-no-fill.psd",
        "stroke-outer.psd",
        "stroke-outer-no-fill.psd",
    ],
)
def test_a_redrawn_vector_stroke_casts_the_shadow_its_pixels_do(filename: str) -> None:
    """The silhouette is what the layer paints, its vector stroke included.

    Each file is a 10 px stroke on a 40 x 40 rectangle, fill on and off,
    authored in Photoshop, so its stored pixels hold the stroke and the fill
    state. Redrawn under ``force``, the layer casts the shadow they do.
    """
    pytest.importorskip("aggdraw")
    psd = PSDImage.open(full_name(filename))
    layer = psd[1]
    _forge_shadow(layer, angle=0.0, Blur=0.0, Distance=6.0, Opacity=100.0)
    stored = _alone(layer, psd.viewbox)
    redrawn = _alone(layer, psd.viewbox, force=True)
    assert np.abs(redrawn - stored).max() <= 1 / 255 + 1e-6


@pytest.mark.parametrize(
    "name", ["stroke-outer-no-fill.psd", "stroke-center-no-fill.psd"]
)
def test_a_shape_with_its_fill_off_casts_its_shadow_from_its_stroke(name: str) -> None:
    """Redrawn from its path, an outline-only shape casts what it paints.

    Each rectangle has its fill switched off and only its vector stroke on,
    so it casts a ring: the shadow shows through the hollow middle, which
    is not knocked out, and nothing is cast from the path's interior. The
    stored pixels hold the same ring, so both force modes cast the same shadow.
    """
    psd = PSDImage.open(full_name(name))
    layer = next(layer for layer in psd if layer.name == "Rectangle 1")
    assert utils.is_fill_disabled(layer)
    _forge_shadow(layer, Blur=0.0, Distance=6.0, Opacity=100.0)
    alpha = _alone(layer, psd.viewbox, force=True)
    # The redrawn ring's antialiasing is the stored one's to an 8-bit level.
    assert np.allclose(alpha, _alone(layer, psd.viewbox), atol=1 / 255)
    # Inside the hollow, just under the top of the ring, which casts there.
    assert layer.stroke is not None
    top = layer.bbox[1] + int(layer.stroke.line_width)
    assert float(alpha[top + 2, 50, 0]) == pytest.approx(1.0)
    assert float(alpha[50, 50, 0]) == 0.0


def test_a_shadow_canvas_out_of_proportion_to_its_layer_is_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The distance is the file's, so the canvas it grows is held in proportion.

    Cast this far, the card's shadow needs a canvas past what the growth
    guard allows a layer and viewport this size, under any budget.
    """
    psd, layer = _masks_card()
    psd.max_alloc_bytes = "unlimited"
    _forge_shadow(layer, angle=45.0, Distance=8000.0)
    with caplog.at_level(logging.DEBUG, logger="psd_tools.composite.composite"):
        _alone(layer, _around(layer))
    assert "Cannot measure a drop shadow" in caplog.text
    assert "allowed for it" in caplog.text


@pytest.mark.parametrize("force", [False, True])
def test_a_stroke_only_shape_casts_its_shadow_from_its_stroke(force: bool) -> None:
    """A shape with its fill off casts from the stroke it paints, not its path.

    ``stroke-outer-no-fill.psd`` is a 10 px outer stroke round an unfilled
    40 x 40 rectangle. Its shadow, hard and cast 10 px left with the
    knockout off, falls where the stroke's left side would, and nothing
    falls in the empty middle. Redrawn under ``force``, the shape casts what
    its stored pixels do.
    """
    psd = PSDImage.open(full_name("stroke-outer-no-fill.psd"))
    layer = next(layer for layer in psd if layer.name == "Rectangle 1")
    shadow = _forge_shadow(layer, angle=0.0, Blur=0.0, Distance=10.0, Opacity=100.0)
    shadow[b"layerConceals"] = Bool(False)
    alpha = _alone(layer, psd.viewbox, force=force)
    assert float(alpha[50, 50, 0]) == 0.0
    assert float(alpha[50, 14, 0]) == pytest.approx(1.0)


def _setting(key: bytes, value: Any) -> Callable[[Descriptor], Any]:
    return lambda desc: desc.__setitem__(key, value)


# Ways a drop shadow's descriptor can defeat the reads that draw it: a number
# that will not parse, one that parses to something no arithmetic survives,
# and a colour that cannot be read. NaN and infinity raise nowhere on their
# own, and let through they took every pixel of the render with them.
_UNDRAWABLE: dict[str, Callable[[Descriptor], Any]] = {
    "size": _setting(Key.Blur, "wide"),
    "infinite-size": _setting(Key.Blur, Double(float("inf"))),
    "infinite-distance": _setting(Key.Distance, Double(float("inf"))),
    "nan-angle": _setting(Key.LocalLightingAngle, Double(float("nan"))),
    "nan-spread": _setting(Key.ChokeMatte, Double(float("nan"))),
    "nan-opacity": _setting(Key.Opacity, Double(float("nan"))),
    "infinite-opacity": _setting(Key.Opacity, Double(float("inf"))),
    "no-color": lambda desc: desc.pop(Key.Color),
    "color-class": lambda desc: setattr(desc[Key.Color], "classID", b"XXXX"),
}


def _rendered(mutate: Callable[[Descriptor], Any]) -> np.ndarray:
    psd = PSDImage.open(full_name("layer_comps.psb"))
    for layer in psd.descendants():
        for effect in list(layer.effects.find("dropshadow")):
            mutate(effect.descriptor)
    color, _, alpha = composite(psd)
    return np.concatenate((color, alpha), axis=2)


@pytest.mark.parametrize("defect", sorted(_UNDRAWABLE))
def test_an_unreadable_drop_shadow_is_dropped_and_the_document_renders(
    defect: str,
) -> None:
    """A descriptor that cannot be read costs its shadow, not the render.

    Asserted against the same document with the shadow switched off, so a
    shadow drawn at some invented size or colour would not pass. The angle
    is read off the descriptor only where the shadow does not use the global
    light, which ``layer_comps.psb``'s do not.
    """
    disabled = _rendered(_setting(Key.Enabled, False))
    assert not np.array_equal(_rendered(lambda desc: None), disabled)
    rendered = _rendered(_UNDRAWABLE[defect])
    assert np.isfinite(rendered).all()
    assert np.array_equal(rendered, disabled)


def test_without_scipy_a_drop_shadow_is_skipped_and_the_document_renders(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A missing scipy costs the shadow, with a warning, not the render."""
    psd, layer = _masks_card()
    viewport = _around(layer)
    shadow = _shadow_descriptor(layer)
    shadow[Key.Enabled] = False
    disabled = _alone(layer, viewport)
    shadow[Key.Enabled] = True
    monkeypatch.setattr(
        sys.modules["psd_tools.composite.composite"], "HAS_SCIPY", False
    )
    monkeypatch.setattr(effects, "HAS_SCIPY", False)
    with caplog.at_level(logging.WARNING, logger="psd_tools.composite.composite"):
        rendered = _alone(layer, viewport)
    assert np.array_equal(rendered, disabled)
    assert "install scipy" in caplog.text


def _over_grey(layer: Layer, viewport: tuple[int, int, int, int]) -> np.ndarray:
    """The layer's colour, composited alone over an opaque mid grey."""
    height, width = viewport[3] - viewport[1], viewport[2] - viewport[0]
    compositor = Compositor(
        viewport,
        np.full((height, width, 3), 0.5, dtype=np.float32),
        np.ones((height, width, 1), dtype=np.float32),
    )
    compositor.apply(layer)
    return compositor.finish()[0]


def test_a_drop_shadow_with_no_blend_mode_multiplies() -> None:
    """Photoshop's default for a drop shadow is Multiply, as an inner shadow's is.

    A red shadow tells the two apart over grey: multiplied it darkens the
    grey, painted Normal it covers it.
    """
    psd, layer = _masks_card()
    viewport = _around(layer)
    shadow = _shadow_descriptor(layer)
    for key, value in ((Key.Blur, 0.0), (Key.Distance, 20.0), (Key.Opacity, 100.0)):
        shadow[key] = Double(value)
    shadow[Key.Color][Key.Red] = Double(255.0)
    multiplied = _over_grey(layer, viewport)
    shadow[Key.Mode] = Enumerated(Type.BlendMode, Enum.Normal)
    normal = _over_grey(layer, viewport)
    del shadow[Key.Mode]
    unstated = _over_grey(layer, viewport)
    assert not np.allclose(multiplied, normal)
    assert np.array_equal(unstated, multiplied)


# -- Fidelity against Photoshop's own render ------------------------------- #

# Each case is a document, the region around its drop shadows (None for the
# whole document), the bound the render is held to, and a floor under the
# error of the same render with the shadows switched off -- which is what
# shows the region is measuring a shadow at all. The bounds are what the
# ``force=False`` render these tests make scores, with room for a pixel:
#
# - layer_effects.psd: the region stops short of the layer below.
# - layer_params.psb: what is left is the outer glow on the same layer, drawn
#   to a fit of its own. The bound is what pins spread: leaving it out,
#   shrinking the silhouette by it, or rounding its half pixel down instead
#   of up scores above it.
# - layer_comps.psb: the rest lies along the shapes' edges and in the
#   gradient overlays.
# - masks.psb: the header band carries a stroke, and its bound is what pins
#   the knockout under the stroke's band; cutting under the silhouette alone
#   scores above it.
_PREVIEWS = [
    ("layer_effects.psd", (60, 215, 620, 296), 1e-5, 1e-3),
    ("masks.psb", None, 1e-6, 1.5e-4),
    ("layer_params.psb", (130, 230, 470, 560), 2.3e-4, 2e-2),
    ("layer_comps.psb", None, 1.6e-3, 3e-3),
]


def _error(
    psd: PSDImage, region: tuple[int, int, int, int] | None, force: bool = False
) -> float:
    reference = psd.numpy()
    color, _, alpha = composite(psd, force=force)
    result = color
    if reference.shape[2] > color.shape[2]:
        result = np.concatenate((color, alpha), axis=2)
    if region is not None:
        left, top, right, bottom = region
        reference = reference[top:bottom, left:right]
        result = result[top:bottom, left:right]
    return float(_mse(reference, result))


@pytest.mark.parametrize(
    ("filename", "region", "bound", "without"),
    _PREVIEWS,
    ids=[case[0] for case in _PREVIEWS],
)
def test_drop_shadows_render_to_photoshops_preview(
    filename: str,
    region: tuple[int, int, int, int] | None,
    bound: float,
    without: float,
) -> None:
    """Every drop shadow in the corpus, against Photoshop's own render of it."""
    psd = PSDImage.open(full_name(filename))
    error = _error(psd, region)

    unshadowed = PSDImage.open(full_name(filename))
    for layer in unshadowed.descendants():
        for effect in list(layer.effects.find("dropshadow")):
            effect.descriptor[Key.Enabled] = False
    assert _error(unshadowed, region) >= without, "the region has no shadow"
    assert error <= bound


def test_drop_shadows_render_to_photoshops_preview_when_forced() -> None:
    """``force=True`` redraws masks.psb's cards from their paths."""
    assert _error(PSDImage.open(full_name("masks.psb")), None, force=True) <= 1.5e-5


# The gaussian width per pixel of size that best matches Photoshop's previews,
# fitted on the shadows the fixtures above cast. A shadow a few pixels wide
# cannot tell widths apart; the fit is set by the wide ones.
_FITTED_SIGMA_PER_SIZE = 0.45


def test_the_blur_width_is_the_fitted_constant() -> None:
    """The blur's width is a fit, so a change to it is a measured change.

    Photoshop's blur is not a plain gaussian; a gaussian of this width stands
    in for it. The fidelity cases above hold the fit to Photoshop's previews;
    this pins the number itself so that it is not retuned on one fixture.
    """
    assert effects._SHADOW_SIGMA_PER_SIZE == _FITTED_SIGMA_PER_SIZE


# -- Contour, noise, global light and the allocation guard ------------------ #


def _drop_shadow(layer: Layer) -> DropShadow:
    return cast(DropShadow, next(iter(layer.effects.find("dropshadow"))))


def _contour(points: list[tuple[float, float]]) -> Descriptor:
    """A contour descriptor through ``points`` on Photoshop's 0-255 grid."""
    desc = Descriptor(classID=b"ShpC")
    curve = List()
    for x, y in points:
        point = Descriptor(classID=b"CrPt")
        point[Key.Horizontal] = Double(x)
        point[Key.Vertical] = Double(y)
        curve.append(point)
    desc[Key.Name] = String("forged")
    desc[Key.Curve] = curve
    return desc


def test_photoshops_linear_contour_is_the_identity() -> None:
    """The default contour, "Linear" through two corners, is skipped as ``None``.

    So is a missing contour. Every shadow in the fixture corpus but one
    carries this contour.
    """
    _, card = _masks_card()
    shadow = _drop_shadow(card)
    assert shadow.contour[Key.Name].value.rstrip("\x00") == "Linear"
    assert effects.contour_table(shadow.contour) is None
    assert effects.contour_table(None) is None
    assert effects.contour_table(_contour([(0.0, 0.0), (255.0, 255.0)])) is None
    assert effects.contour_table(_contour([(0.0, 0.0)])) is None


def test_a_contour_table_passes_through_its_points_in_order() -> None:
    """``layer_params.psb``'s caption carries Photoshop's nine-point "Gaussian".

    The table has an entry per grid value, so each point of the curve lands
    on an entry and the curve has to pass through it; between points it may
    not turn back, which is what the monotone spline guarantees.
    """
    psd = PSDImage.open(full_name("layer_params.psb"))
    caption = next(layer for layer in psd.descendants() if layer.name == "Test caption")
    shadow = _drop_shadow(caption)
    points = [
        (float(point[Key.Horizontal]), float(point[Key.Vertical]))
        for point in shadow.contour[Key.Curve]
    ]
    assert len(points) == 9
    table = effects.contour_table(shadow.contour)
    assert table is not None
    assert table.shape == (effects._CONTOUR_STEPS,)
    for x, y in points:
        assert float(table[int(x)]) == pytest.approx(y / 255.0, abs=1e-6)
    assert np.all(np.diff(table) >= 0.0)


def test_a_curved_contour_asks_for_scipy_by_name_when_it_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The spline through three or more points needs scipy, and says so."""
    monkeypatch.setattr(effects, "HAS_SCIPY", False)
    with pytest.raises(ImportError, match="Layer effect contours require: scipy"):
        effects.contour_table(_contour([(0.0, 0.0), (128.0, 64.0), (255.0, 255.0)]))
    # Two points need no spline.
    assert effects.contour_table(_contour([(0.0, 51.0), (255.0, 255.0)])) is not None


def test_a_two_point_contour_is_a_straight_line() -> None:
    """Two points make a line, and a step holds its values past its ends."""
    table = effects.contour_table(_contour([(0.0, 51.0), (255.0, 255.0)]))
    assert table is not None
    assert np.allclose(table, np.linspace(51.0 / 255.0, 1.0, 256), atol=1e-6)
    table = effects.contour_table(_contour([(64.0, 0.0), (128.0, 255.0)]))
    assert table is not None
    assert np.all(table[:65] == 0.0)
    assert np.all(table[128:] == 1.0)
    assert np.all(np.diff(table[64:129]) > 0.0)


def test_a_corner_point_turns_the_contour_sharply() -> None:
    """A point marked as a corner ends one run of the curve and starts the next.

    Photoshop's "Cone" peaks at a corner, so each side is a straight line,
    where a spline through all three points rounds the peak off. A run of
    three or more points beside a corner is still curved, and needs scipy.
    """
    cone = _contour([(0.0, 0.0), (128.0, 255.0), (255.0, 0.0)])
    rounded = effects.contour_table(cone)
    cone[Key.Curve][1][Key.Continuity] = Bool(False)
    sharp = effects.contour_table(cone)
    assert rounded is not None and sharp is not None
    grid = np.arange(effects._CONTOUR_STEPS)
    assert np.allclose(sharp, np.interp(grid, [0, 128, 255], [0.0, 1.0, 0.0]))
    assert float(rounded[64]) > float(sharp[64]) + 0.1

    shoulder = _contour([(0.0, 0.0), (64.0, 32.0), (128.0, 255.0), (255.0, 0.0)])
    shoulder[Key.Curve][2][Key.Continuity] = Bool(False)
    table = effects.contour_table(shoulder)
    assert table is not None
    assert float(table[64]) == pytest.approx(32.0 / 255.0, abs=1e-6)
    assert not np.allclose(table[:65], np.linspace(0.0, 32.0 / 255.0, 65))
    assert np.allclose(table[128:], np.interp(grid[128:], [128, 255], [1.0, 0.0]))


def test_a_contour_reshapes_the_falloff_of_a_drop_shadow() -> None:
    """The contour maps the cast matte's coverage, after the blur and offset.

    Inverted, it paints the shadow where the blurred matte is clear and
    clears it where the matte is full, over the whole canvas the shadow is
    drawn on.
    """
    shape = _disc(81, 12.0)
    plain = effects.draw_drop_shadow(
        (0, 0, 81, 81), shape, distance=3.0, angle=90.0, size=6.0, spread=0.0
    )
    inverted = effects.contour_table(_contour([(0.0, 255.0), (255.0, 0.0)]))
    flipped = effects.draw_drop_shadow(
        (0, 0, 81, 81),
        shape,
        distance=3.0,
        angle=90.0,
        size=6.0,
        spread=0.0,
        contour=inverted,
        anti_aliased=True,
    )
    assert np.allclose(flipped, 1.0 - plain, atol=1e-6)


def test_anti_aliasing_reads_the_contour_between_its_entries() -> None:
    """Without anti-aliasing the table is read at the nearest entry.

    A contour that steps from 0 to 1 between two entries paints a coverage
    that falls between them as one or the other, the way an 8-bit matte
    would; anti-aliased, it paints the value between the two.
    """
    steps = effects._CONTOUR_STEPS
    table = np.where(np.arange(steps) >= steps // 2, 1.0, 0.0).astype(np.float32)
    alpha = np.full((1, 1), (steps // 2 - 0.5) / (steps - 1), dtype=np.float32)
    hard = float(effects._contoured(alpha, table, anti_aliased=False)[0, 0])
    assert hard in (0.0, 1.0)
    smooth = float(effects._contoured(alpha, table, anti_aliased=True)[0, 0])
    assert 0.0 < smooth < 1.0


def test_a_shadows_contour_is_applied_when_it_is_composited() -> None:
    """A contour that lifts zero coverage paints the shadow's whole canvas.

    The card's shadow is made hard and cast nowhere, so beside the card it
    covers nothing; given a flat contour, the pixel just beside the card,
    which is on the canvas the shadow is drawn on, is covered in full.
    """
    _, layer = _masks_card()
    _forge_shadow(layer, Blur=0.0, Distance=0.0, Opacity=100.0)
    viewport = _around(layer)
    y, x = (layer.bbox[1] + layer.bbox[3]) // 2 - viewport[1], 16 - 1
    plain = _alone(layer, viewport)
    _shadow_descriptor(layer)[Key.TransferSpec] = _contour(
        [(0.0, 255.0), (255.0, 255.0)]
    )
    flat = _alone(layer, viewport)
    assert float(plain[y, x, 0]) == 0.0
    assert float(flat[y, x, 0]) == pytest.approx(1.0)


def test_the_caption_contour_brings_layer_params_closer_to_photoshop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Photoshop's preview of the contoured caption shadow is matched better.

    The caption's region also holds a bevel that is not drawn, so the error
    does not reach the bounds the plain shadows are held to; the contour has
    to improve on the identity table, and the shadow on no shadow at all.
    """
    filename = "layer_params.psb"
    psd = PSDImage.open(full_name(filename))
    caption = next(layer for layer in psd.descendants() if layer.name == "Test caption")
    region = _around(caption, 30)
    contoured = _error(psd, region)

    module = sys.modules["psd_tools.composite.composite"]
    monkeypatch.setattr(module, "contour_table", lambda desc: None)
    identity = _error(PSDImage.open(full_name(filename)), region)

    unshadowed = PSDImage.open(full_name(filename))
    caption = next(
        layer for layer in unshadowed.descendants() if layer.name == "Test caption"
    )
    _shadow_descriptor(caption)[Key.Enabled] = False
    without = _error(unshadowed, region)
    assert contoured < identity < without


def test_noise_is_read_but_not_applied(caplog: pytest.LogCaptureFixture) -> None:
    """A noisy shadow renders as the same shadow without noise, and says so."""
    psd, layer = _masks_card()
    viewport = _around(layer)
    plain = _alone(layer, viewport)
    _shadow_descriptor(layer)[Key.Noise] = Double(50.0)
    assert _drop_shadow(layer).noise == 50.0
    with caplog.at_level(logging.DEBUG, logger="psd_tools.composite.composite"):
        noisy = _alone(layer, viewport)
    assert "noise" in caplog.text
    assert np.array_equal(noisy, plain)


def test_a_shadow_lit_by_the_global_light_follows_the_documents_angle() -> None:
    """Under global light the angle is the document's, read when it is drawn.

    The forged shadow is hard and cast 20 px, so where it falls is read off
    single pixels: below the card with the light overhead, left of it with
    the light from the right. The reach the cull and the group canvas
    measure moves with it.
    """
    psd, layer = _masks_card()
    shadow = _forge_shadow(layer, Blur=0.0, Distance=20.0, Opacity=100.0)
    shadow[Key.UseGlobalAngle] = True
    effect = _drop_shadow(layer)
    assert effect.use_global_light
    assert effect.angle == 90
    viewport = _around(layer, 30)
    left, top, right, bottom = layer.bbox
    y, x = (top + bottom) // 2 - viewport[1], left - 10 - viewport[0]
    below_y, below_x = bottom + 10 - viewport[1], (left + right) // 2 - viewport[0]

    lit_from_above = _alone(layer, viewport)
    assert float(lit_from_above[below_y, below_x, 0]) == pytest.approx(1.0)
    assert float(lit_from_above[y, x, 0]) == 0.0
    reach_above = _effect_reach(layer)

    psd.image_resources[Resource.GLOBAL_ANGLE].data.value = 0
    assert effect.angle == 0
    lit_from_right = _alone(layer, viewport)
    assert float(lit_from_right[y, x, 0]) == pytest.approx(1.0)
    assert float(lit_from_right[below_y, below_x, 0]) == 0.0
    reach_right = _effect_reach(layer)
    assert reach_right[0] == reach_above[0] - 20
    assert reach_right[3] == reach_above[3] - 20


def test_a_shadow_with_its_own_angle_ignores_the_global_light() -> None:
    """With global light off the descriptor's angle is used, whatever the document says."""
    psd, layer = _masks_card()
    _forge_shadow(layer, angle=90.0, Blur=0.0, Distance=20.0, Opacity=100.0)
    psd.image_resources[Resource.GLOBAL_ANGLE].data.value = 0
    viewport = _around(layer, 30)
    left, top, right, bottom = layer.bbox
    alpha = _alone(layer, viewport)
    below = alpha[bottom + 10 - viewport[1], (left + right) // 2 - viewport[0], 0]
    beside = alpha[(top + bottom) // 2 - viewport[1], left - 10 - viewport[0], 0]
    assert float(below) == pytest.approx(1.0)
    assert float(beside) == 0.0


def test_drop_shadow_bbox_refuses_a_canvas_over_the_allocation_budget() -> None:
    """The grown box is checked against the budget before it is allocated.

    The size and distance are read unvalidated from the file, so a shadow of
    Photoshop's largest size on a small layer is still bounded by what the
    caller lets it allocate, as a stroke's box is.
    """
    bbox = (0, 0, 20, 20)
    grown = effects.drop_shadow_bbox(
        bbox, 0.0, 90.0, 250.0, max_alloc_bytes="unlimited"
    )
    width, height = grown[2] - grown[0], grown[3] - grown[1]
    budget = width * height * effects._SHADOW_BYTES
    assert (
        effects.drop_shadow_bbox(bbox, 0.0, 90.0, 250.0, max_alloc_bytes=budget)
        == grown
    )
    with pytest.raises(ValueError):
        effects.drop_shadow_bbox(bbox, 0.0, 90.0, 250.0, max_alloc_bytes=budget - 1)
    # An empty box casts nothing and is not measured.
    empty = (0, 0, 0, 0)
    assert effects.drop_shadow_bbox(empty, 0.0, 90.0, 250.0, max_alloc_bytes=1) == empty


def test_a_drop_shadow_over_the_documents_budget_is_dropped_not_drawn() -> None:
    """A shadow whose canvas the document's budget refuses costs the shadow only.

    The budget is set just under what the shadow's canvas costs and above
    what the layer's own pixels do, so the layer renders as it does with the
    shadow switched off rather than raising.
    """
    psd, layer = _masks_card()
    viewport = _around(layer)
    shadow = _shadow_descriptor(layer)
    shadow[Key.Enabled] = False
    disabled = _alone(layer, viewport)
    shadow[Key.Enabled] = True
    assert not np.array_equal(_alone(layer, viewport), disabled)

    geometry = effects.shadow_geometry(shadow, 90.0)
    grown = effects.drop_shadow_bbox(
        layer.bbox, geometry.distance, geometry.angle, geometry.size
    )
    width, height = grown[2] - grown[0], grown[3] - grown[1]
    psd.max_alloc_bytes = width * height * effects._SHADOW_BYTES - 1
    assert np.array_equal(_alone(layer, viewport), disabled)


def test_without_scipy_the_shadows_are_left_out_and_the_document_renders(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A document with drop shadows still composites where scipy is missing.

    The shadows are left out, as they were before they were drawn at all,
    and the pass says so once, not once per layer: ``masks.psb`` carries
    three.
    """
    psd = PSDImage.open(full_name("masks.psb"))
    shadows = [
        effect
        for layer in psd.descendants()
        for effect in layer.effects.find("dropshadow")
    ]
    assert len(shadows) == 3
    for effect in shadows:
        effect.descriptor[Key.Enabled] = False
    color, _, alpha = composite(psd)
    for effect in shadows:
        effect.descriptor[Key.Enabled] = True

    module = sys.modules["psd_tools.composite.composite"]
    monkeypatch.setattr(module, "HAS_SCIPY", False)
    with caplog.at_level(logging.WARNING, logger="psd_tools.composite.composite"):
        without_color, _, without_alpha = composite(psd)
    assert np.array_equal(without_color, color)
    assert np.array_equal(without_alpha, alpha)
    warned = [record for record in caplog.records if "scipy" in record.getMessage()]
    assert len(warned) == 1 and warned[0].levelno == logging.WARNING
