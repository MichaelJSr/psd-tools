"""The outer glow layer effect, drawn and composited.

Two documents in the fixture corpus carry an outer glow Photoshop rendered
into its own preview: ``layer_effects.psd``, a soft white Screen glow on a type
layer over a flat background, and ``layer_params.psb``, a gradient glow blended
Hard Light with a contour and a range of its own, on a pixel layer that also
casts a drop shadow. The fidelity tests measure against those. Everything the
corpus does not pair with a glow is forged from ``layer_effects.psd``'s, or
drawn on a disc.
"""

import copy
import logging
import math
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

from psd_tools.api.effects import OuterGlow
from psd_tools.api.layers import Layer
from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import composite, effects, paint, utils
from psd_tools.composite.composite import (
    Compositor,
    _beneath_reach,
    _content_bbox,
    _effect_reach,
)
from psd_tools.constants import BlendMode, ColorMode, GlowTechnique, Tag
from psd_tools.psd.base import ByteElement
from psd_tools.psd.descriptor import Descriptor, Double, Enumerated, List
from psd_tools.terminology import Enum, Key, Klass, Type

from ..utils import full_name
from .test_composite import _feathered_stroke_layer, _mse
from .test_drop_shadow import _alone, _around, _contour, _disc, _masks_card

logger = logging.getLogger(__name__)

_CANVAS = (0, 0, 81, 81)
_RADIUS = 12.0


def _glow(shape: np.ndarray, **params: Any) -> np.ndarray:
    """A solid glow's coverage, with the keyword defaults of a plain one."""
    settings: dict[str, Any] = {"size": 6.0, "spread": 0.0, "range_": 100.0}
    settings.update(params)
    return effects._beside(effects.glow_falloff(shape, **settings), shape)


def _ring(glow: np.ndarray) -> np.ndarray:
    """The glow along a row from the disc's centre outward."""
    return glow[40, 40:, 0]


# -- The primitive --------------------------------------------------------- #


@pytest.mark.parametrize("precise", [False, True], ids=["soft", "precise"])
@pytest.mark.parametrize("spread", [0.0, 100.0])
def test_an_outer_glow_never_lies_under_its_silhouette(
    precise: bool, spread: float
) -> None:
    """The glow is cut away under the silhouette, by its coverage.

    Photoshop never shows an outer glow through its layer, so there is no
    knockout switch: a pixel the silhouette covers fully gets none of it,
    and one it half covers gets at most half.
    """
    disc = _disc(81, _RADIUS)
    glow = _glow(disc, spread=spread, precise=precise)
    assert glow.shape == disc.shape
    assert float(glow[disc > 0.0].max()) == 0.0
    assert float(glow.max()) > 0.0, "nothing drawn beside the disc"

    faint = disc * 0.5
    glow = _glow(faint, spread=spread, precise=precise)
    assert float(glow[disc > 0.0].max()) <= 0.5 + 1e-6


@pytest.mark.parametrize("precise", [False, True], ids=["soft", "precise"])
def test_a_glow_of_size_zero_draws_nothing(precise: bool) -> None:
    """A size of zero is no glow at all, whatever the spread and range say."""
    glow = _glow(
        _disc(81, _RADIUS), size=0.0, spread=100.0, range_=1.0, precise=precise
    )
    assert not glow.any()


@pytest.mark.parametrize("precise", [False, True], ids=["soft", "precise"])
def test_spread_at_full_makes_a_hard_band_of_the_size(precise: bool) -> None:
    """At 100% spread the whole size goes to growth and none to the falloff.

    Covered from the disc's edge out to the size, and not a pixel beyond,
    with either technique.
    """
    glow = _glow(_disc(81, _RADIUS), size=6.0, spread=100.0, precise=precise)
    row = _ring(glow)
    inside = int(_RADIUS) + 1
    assert np.all(row[:inside] == 0.0), row
    assert np.all(row[inside : inside + 6] == 1.0), row
    assert np.all(row[inside + 6 :] == 0.0), row


def test_the_precise_technique_ramps_straight_with_the_distance() -> None:
    """Precise is a straight ramp from the edge to nothing at the size.

    The pixel centres sit half a pixel past the disc's last covered pixel,
    so the ramp reads one minus that distance over the size.
    """
    glow = _glow(_disc(81, _RADIUS), size=6.0, precise=True)
    row = _ring(glow)[int(_RADIUS) + 1 :]
    expected = [1.0 - (step + 0.5) / 6.0 for step in range(6)]
    assert np.allclose(row[:6], expected, atol=1e-6), row[:6]
    assert np.all(row[6:] == 0.0)


def test_a_soft_glow_is_a_drop_shadow_cast_nowhere() -> None:
    """Beside the silhouette the soft matte is the drop shadow's at distance zero.

    The two share their spread and blur, so what differs is only what the
    glow cuts away under the silhouette and what the range does to the
    rest; at 100% the range does nothing.
    """
    disc = _disc(81, _RADIUS)
    glow = _glow(disc, size=6.0, spread=30.0)
    shadow = effects.draw_drop_shadow(
        _CANVAS, disc, distance=0.0, angle=0.0, size=6.0, spread=30.0
    )
    assert np.allclose(glow, shadow * (1.0 - disc), atol=1e-6)


def _soft_disc(size: int, radius: float, hard_radius: float) -> np.ndarray:
    """A disc fading from opaque at ``hard_radius`` to faint at ``radius``.

    Every pixel of the hard disc of ``radius`` keeps some coverage, so the
    two discs have the same extent, and none fades below what the coverage
    gain reads as whole.
    """
    centre = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    distance = np.hypot(xx - centre, yy - centre)
    ramp = np.clip((radius + 1.0 - distance) / (radius + 1.0 - hard_radius), 0.0, 1.0)
    soft = (_disc(size, radius)[:, :, 0] * ramp).astype(np.float32)[:, :, None]
    assert float(soft[soft > 0.0].min()) >= 1.0 / _FITTED_COVERAGE_GAIN
    return soft


# What a glow is cast from, fitted on Photoshop's renders of layers carrying
# glows: with no spread the blur of the opacity; spread, the blur of the
# opacity scaled by the gain and held to one, which reads the layer's
# coverage, with a hard shape's antialiased edge, up to the reach wide, kept
# as it is. The switch is the half pixel, where the spread rounds to a whole
# one.
_FITTED_COVERAGE_GAIN = 20.0
_FITTED_SPREAD_PIXELS = 0.5
_FITTED_EDGE_REACH = 2
_COVERAGE_SIZE = 32.0
_COVERAGE_SPREAD = 10.0


def test_the_coverage_switch_and_gain_are_the_fitted_constants() -> None:
    """A fit, so a change to either is a measured change.

    Just under the half pixel the opacity is cast as it is; at it, scaled by
    the gain and held to one. The probed pixels lie beyond the edge reach of
    the one opaque pixel, so none of them reads as an edge.
    """
    alpha = np.zeros((1, 8), dtype=np.float32)
    alpha[0, 1:4] = [0.5 / _FITTED_COVERAGE_GAIN, 1.0 / _FITTED_COVERAGE_GAIN, 0.3]
    alpha[0, 7] = 1.0
    under = 100.0 * (_FITTED_SPREAD_PIXELS - 1e-3) / 100.0
    assert np.array_equal(effects._glow_coverage(alpha, 100.0, under), alpha)
    at = effects._glow_coverage(alpha, 100.0, _FITTED_SPREAD_PIXELS)
    assert np.allclose(at, [[0.0, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
    assert np.array_equal(
        effects._glow_coverage(alpha, 1.0, 100.0 * _FITTED_SPREAD_PIXELS), at
    )


@pytest.mark.parametrize("precise", [False, True], ids=["soft", "precise"])
def test_a_spread_glow_on_a_soft_edged_disc_is_the_hard_discs(precise: bool) -> None:
    """A spread glow is cast from the silhouette's coverage, not its opacity.

    Outside the ramp the soft disc's glow is the hard disc's, pixel for
    pixel; over the ramp it is the hard disc's falloff cut by what the ramp
    leaves. Cast from the opacity it would be a fraction of that, since most
    of the soft disc is faint.
    """
    hard = _disc(121, _RADIUS)
    soft = _soft_disc(121, _RADIUS, 4.0)
    partial = (soft > 0.0) & (soft < 1.0)
    assert partial.sum() > 0.8 * (hard > 0.0).sum(), "the disc is not soft"

    falloff = effects.glow_falloff(
        hard,
        size=_COVERAGE_SIZE,
        spread=_COVERAGE_SPREAD,
        range_=100.0,
        precise=precise,
    )
    glow = _glow(soft, size=_COVERAGE_SIZE, spread=_COVERAGE_SPREAD, precise=precise)
    assert np.allclose(glow, falloff[:, :, None] * (1.0 - soft), atol=1e-6)
    outside = hard[:, :, 0] == 0.0
    hard_glow = _glow(
        hard, size=_COVERAGE_SIZE, spread=_COVERAGE_SPREAD, precise=precise
    )
    assert np.array_equal(glow[outside], hard_glow[outside])
    if not precise:
        from_opacity = effects._soft_matte(
            soft[:, :, 0], _COVERAGE_SIZE, _COVERAGE_SPREAD, 0.45
        )
        assert float(glow[outside].max()) > 1.5 * float(from_opacity[outside].max())


def test_a_glow_with_no_spread_on_a_soft_edged_disc_is_cast_from_its_opacity() -> None:
    """Without a spread the soft disc glows by its opacity, and so fainter.

    The same holds of a spread that rounds to no pixel at all. The precise
    technique is left out: it ramps from the silhouette's outline whatever
    the opacity.
    """
    hard = _disc(81, _RADIUS)
    soft = _soft_disc(81, _RADIUS, 4.0)
    outside = hard[:, :, 0] == 0.0
    for spread in (0.0, 100.0 * (_FITTED_SPREAD_PIXELS - 1e-3) / _COVERAGE_SIZE):
        faint = _glow(soft, size=_COVERAGE_SIZE, spread=spread)[outside]
        bright = _glow(hard, size=_COVERAGE_SIZE, spread=spread)[outside]
        assert np.all(faint <= bright + 1e-6)
        assert float((bright - faint).max()) > 0.05
        matte = effects._soft_matte(soft[:, :, 0], _COVERAGE_SIZE, spread, 0.45)
        assert np.allclose(faint, effects._beside(matte, soft)[outside], atol=1e-6)


def test_the_coverage_keeps_an_antialiased_edge_and_gains_the_rest() -> None:
    """A partial pixel between clear and opaque keeps its value; any other is gained.

    A hard shape's antialiased edge is left where it lies, so its glow
    starts where the edge does, whether the edge is a pixel wide or the two
    a resampled shape's is, and so is the edge of a shape painted a step
    short of solid. A feather's pixels border other partial pixels and a
    faint patch borders nothing opaque: those cast whole above the gain's
    reach and in proportion below it.
    """
    alpha = np.zeros((9, 9), dtype=np.float32)
    alpha[:, :4] = 1.0
    alpha[:, 4] = 0.3  # an antialiased edge: opaque on one side, clear on the other
    alpha[1, 7] = 0.5 / _FITTED_COVERAGE_GAIN  # a faint, isolated pixel
    alpha[6:8, 6:9] = 0.1  # a faint patch with no opaque neighbour
    coverage = effects._glow_coverage(alpha, _COVERAGE_SIZE, _COVERAGE_SPREAD)
    assert np.all(coverage[:, :4] == 1.0)
    assert np.allclose(coverage[:, 4], 0.3)
    assert coverage[1, 7] == pytest.approx(0.5)
    assert np.all(coverage[6:8, 6:9] == 1.0)
    assert np.all(coverage[alpha == 0.0] == 0.0)

    short = alpha.copy()
    short[:, :4] = 254.0 / 255.0
    assert np.allclose(
        effects._glow_coverage(short, _COVERAGE_SIZE, _COVERAGE_SPREAD)[:, 4], 0.3
    )

    feather = np.zeros((5, 12), dtype=np.float32)
    feather[:, :6] = np.linspace(1.0, 0.1, 6)[None, :]
    assert np.all(
        effects._glow_coverage(feather, _COVERAGE_SIZE, _COVERAGE_SPREAD)[:, :6] == 1.0
    )
    assert np.array_equal(effects._glow_coverage(feather, _COVERAGE_SIZE, 0.0), feather)


def test_an_antialiased_edge_is_kept_as_far_as_the_fitted_reach() -> None:
    """A band of partial pixels two wide is an edge; one three wide is not.

    A resampled shape's edge is antialiased over two pixels, and both keep
    their value. Of a band one wider, only the middle pixel lies within the
    reach of both the clear side and the opaque one; the outer two are
    gained like any other.
    """
    reach = _FITTED_EDGE_REACH
    band = np.zeros((3, 12), dtype=np.float32)
    band[:, :4] = 1.0
    band[:, 4 : 4 + reach] = np.linspace(0.8, 0.2, reach)[None, :]
    coverage = effects._glow_coverage(band, _COVERAGE_SIZE, _COVERAGE_SPREAD)
    assert np.allclose(coverage[:, 4 : 4 + reach], band[:, 4 : 4 + reach])

    wide = np.zeros((3, 12), dtype=np.float32)
    wide[:, :4] = 1.0
    wide[:, 4 : 5 + reach] = np.linspace(0.8, 0.2, reach + 1)[None, :]
    coverage = effects._glow_coverage(wide, _COVERAGE_SIZE, _COVERAGE_SPREAD)
    assert np.all(coverage[:, 4] == 1.0)
    assert np.allclose(coverage[:, 5], wide[:, 5])
    assert np.all(coverage[:, 6] == 1.0)


# The width of the gaussian that stands in for Photoshop's blur, per pixel of
# size, and the floor the range is held above, both fitted on Photoshop's own
# renders of soft glows: the fixture previews below for the blur, and for the
# floor a corpus of documents with glows at ranges from 1% to 100%, since no
# fixture carries a range far from 100%.
_FITTED_SIGMA_PER_SIZE = 0.45
_FITTED_RANGE_FLOOR = 15.0


def test_the_blur_width_and_the_range_floor_are_the_fitted_constants() -> None:
    """Both are fits, so a change to either is a measured change.

    The fidelity cases below hold the blur to Photoshop's previews; this
    pins the numbers themselves so they are not retuned on one fixture, and
    is all that holds the range floor, which the fixtures cannot measure.
    """
    assert effects._GLOW_SIGMA_PER_SIZE == _FITTED_SIGMA_PER_SIZE
    assert effects._GLOW_RANGE_FLOOR == _FITTED_RANGE_FLOOR


def _squeeze(range_: float) -> float:
    """What the range scales the matte by: the inverse of the range and floor together."""
    return math.hypot(100.0, _FITTED_RANGE_FLOOR) / math.hypot(
        range_, _FITTED_RANGE_FLOOR
    )


@pytest.mark.parametrize("precise", [False, True], ids=["soft", "precise"])
def test_the_range_squeezes_the_falloff_toward_the_silhouette(precise: bool) -> None:
    """A range below 100% scales the matte up and holds it to one.

    At 100% the matte is painted as it is. At 50% it is scaled by a little
    under two, so the part of the falloff nearest the silhouette is painted
    full and the rest at nearly twice its coverage. The disc is hard, so the
    cut under it changes nothing and the two renders are related exactly.
    """
    disc = _disc(81, _RADIUS)
    whole = _glow(disc, range_=100.0, precise=precise)
    half = _glow(disc, range_=50.0, precise=precise)
    assert np.allclose(half, np.clip(whole * _squeeze(50.0), 0.0, 1.0), atol=1e-6)
    assert np.all(half >= whole - 1e-6)
    assert float((half - whole).max()) > 0.1


def test_a_small_range_draws_a_soft_halo_not_a_hard_band() -> None:
    """The floor keeps a 1% range from squeezing the falloff to a step.

    Read as the inverse of the slider alone, 1% would scale the matte a
    hundredfold and paint a hard band out to where the blur dies away;
    Photoshop paints a halo that still fades. The floor bounds the scale, so
    the far half of the falloff keeps coverages strictly between the two.
    """
    glow = _glow(_disc(81, 12.0), size=12.0, range_=1.0)
    tail = _ring(glow)[int(_RADIUS) + 1 + 6 : int(_RADIUS) + 1 + 12]
    assert np.all((tail > 0.0) & (tail < 1.0)), tail
    assert np.all(np.diff(tail) < 0.0), tail
    assert _squeeze(1.0) == pytest.approx(
        math.hypot(100.0, 15.0) / math.hypot(1.0, 15.0)
    )


def test_the_range_is_held_to_the_dialogs_percentages() -> None:
    """A range outside 1 to 100 draws as the nearest end of the slider.

    The glow divides by the range, so a zero is held off before it gets
    there; ``glow_geometry`` holds the file's value the same way.
    """
    disc = _disc(81, _RADIUS)
    assert np.array_equal(_glow(disc, range_=0.0), _glow(disc, range_=1.0))
    assert np.array_equal(_glow(disc, range_=500.0), _glow(disc, range_=100.0))
    desc = Descriptor(classID=Klass.OuterGlow.value)
    desc[Key.Blur] = Double(3.0)
    desc[Key.InputRange] = Double(0.0)
    assert effects.glow_geometry(desc) == (3.0, 0.0, 1.0)
    desc[Key.InputRange] = Double(500.0)
    assert effects.glow_geometry(desc) == (3.0, 0.0, 100.0)
    del desc[Key.InputRange]
    assert effects.glow_geometry(desc) == (3.0, 0.0, effects._DEFAULT_GLOW_RANGE)


@pytest.mark.parametrize(
    "value", [Double(float("inf")), Double(float("nan")), "wide"], ids=str
)
def test_glow_geometry_refuses_a_size_it_cannot_draw(value: Any) -> None:
    """A size that will not parse, or parses to no number, is unreadable."""
    desc = Descriptor(classID=Klass.OuterGlow.value)
    desc[Key.Blur] = value
    with pytest.raises((ValueError, TypeError)):
        effects.glow_geometry(desc)


def test_a_contour_maps_the_glow_before_it_is_cut_under_the_silhouette() -> None:
    """A contour that lifts zero coverage paints the whole canvas, bar the disc.

    The cut under the silhouette comes last, so even a flat contour shows
    nothing through the layer.
    """
    disc = _disc(81, _RADIUS)
    flat = effects.contour_table(_contour([(0.0, 255.0), (255.0, 255.0)]))
    glow = _glow(disc, contour=flat, anti_aliased=True)
    assert np.all(glow[disc == 0.0] == 1.0)
    assert np.all(glow[disc > 0.0] == 0.0)


def test_an_outer_glow_asks_for_scipy_by_name_when_it_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A glow needs scipy the way a shadow does, and says so the same way."""
    monkeypatch.setattr(effects, "HAS_SCIPY", False)
    with pytest.raises(ImportError, match="Outer glow effects require: scipy"):
        _glow(_disc(81, _RADIUS))


def _sized(size: float) -> Descriptor:
    desc = Descriptor(classID=Klass.OuterGlow.value)
    desc[Key.Blur] = Double(size)
    return desc


def test_outer_glow_bbox_grows_the_box_by_the_size_on_every_side() -> None:
    """A shadow's canvas for a shadow cast nowhere, under the same budget."""
    bbox = (10, 20, 30, 40)
    assert effects.outer_glow_bbox(bbox, _sized(3.0)) == effects.drop_shadow_bbox(
        bbox, 0.0, 0.0, 3.0
    )
    assert effects.outer_glow_bbox(bbox, _sized(3.0)) == (5, 15, 35, 45)
    assert effects.outer_glow_bbox((0, 0, 0, 0), _sized(3.0)) == (0, 0, 0, 0)

    desc = _sized(250.0)
    grown = effects.outer_glow_bbox(bbox, desc, max_alloc_bytes="unlimited")
    width, height = grown[2] - grown[0], grown[3] - grown[1]
    budget = width * height * effects._SHADOW_BYTES
    assert effects.outer_glow_bbox(bbox, desc, max_alloc_bytes=budget) == grown
    with pytest.raises(ValueError):
        effects.outer_glow_bbox(bbox, desc, max_alloc_bytes=budget - 1)


def test_a_gradient_glows_canvas_is_budgeted_for_its_paint() -> None:
    """A gradient paint costs what a gradient fill does, past the falloff's cost."""
    bbox = (10, 20, 30, 40)
    desc = _sized(250.0)
    desc[Key.Gradient] = Descriptor(classID=Klass.Gradient.value)
    grown = effects.outer_glow_bbox(bbox, desc, max_alloc_bytes="unlimited")
    width, height = grown[2] - grown[0], grown[3] - grown[1]
    assert paint.GRADIENT_FILL_BYTES > effects._SHADOW_BYTES
    budget = width * height * paint.GRADIENT_FILL_BYTES
    assert effects.outer_glow_bbox(bbox, desc, max_alloc_bytes=budget) == grown
    with pytest.raises(ValueError):
        effects.outer_glow_bbox(bbox, desc, max_alloc_bytes=budget - 1)


# -- Paint ----------------------------------------------------------------- #


def _donor() -> tuple[PSDImage, Layer]:
    """``layer_effects.psd`` and the type layer that carries its outer glow.

    A soft white glow of size 41 at 35% opacity, blended Screen, with no
    spread and Photoshop's default 50% range, over a flat grey background.
    """
    psd = PSDImage.open(full_name("layer_effects.psd"))
    return psd, next(layer for layer in psd.descendants() if layer.name == "Outer Glow")


def _glow_descriptor(layer: Layer) -> Descriptor:
    return next(iter(layer.effects.find("outerglow"))).descriptor


def _outer_glow(layer: Layer) -> OuterGlow:
    return cast(OuterGlow, next(iter(layer.effects.find("outerglow"))))


def _paint(
    desc: Descriptor, falloff: np.ndarray, shape: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``glow_paint`` on a canvas the size of ``falloff``, beside nothing by default."""
    if shape is None:
        shape = np.zeros(falloff.shape + (1,), dtype=np.float32)
    viewport = (0, 0, falloff.shape[1], falloff.shape[0])
    return effects.glow_paint(viewport, ColorMode.RGB, desc, shape, falloff)


def test_a_solid_glow_is_painted_in_its_colour_over_its_falloff() -> None:
    """The colour is the descriptor's, the coverage the falloff cut under the shape."""
    _, donor = _donor()
    falloff = np.linspace(0.0, 1.0, 9, dtype=np.float32).reshape(3, 3)
    shape = np.zeros((3, 3, 1), dtype=np.float32)
    shape[1, 1, 0] = 0.5
    color, painted = _paint(_glow_descriptor(donor), falloff, shape=shape)
    assert color.shape == (3, 3, 3)
    assert np.all(color == 1.0)
    assert painted.shape == (3, 3, 1)
    assert np.allclose(painted[:, :, 0], falloff * (1.0 - shape[:, :, 0]), atol=1e-6)


def test_the_falloff_is_not_cut_under_the_silhouette() -> None:
    """``glow_falloff`` leaves the cut to the paint, which places a gradient by it."""
    disc = _disc(81, _RADIUS)
    falloff = effects.glow_falloff(disc, size=6.0, spread=0.0, range_=100.0)
    assert falloff.shape == (81, 81)
    assert float(falloff[disc[:, :, 0] > 0.0].min()) > 0.0


def _gradient_glow() -> tuple[Descriptor, list[tuple[float, ...]], list[float]]:
    """``layer_params.psb``'s gradient glow, with its colour and opacity stops.

    Three colour stops and three transparency stops, the last two of each
    at the same positions; the gradient fades from opaque to a tenth.
    """
    psd = PSDImage.open(full_name("layer_params.psb"))
    assert psd.color_mode == ColorMode.RGB
    layer = next(
        layer for layer in psd.descendants() if layer.name.startswith("Фигура")
    )
    desc = _glow_descriptor(layer)
    gradient = desc[Key.Gradient]
    colors = [
        tuple(
            float(stop[Key.Color][key]) / 255.0 for key in (b"Rd  ", b"Grn ", b"Bl  ")
        )
        for stop in gradient[Key.Colors]
    ]
    opacities = [
        float(stop[Key.Opacity]) / 100.0 for stop in gradient[Key.Transparency]
    ]
    assert len(colors) == 3 and len(opacities) == 3
    assert opacities[0] == 1.0 and opacities[-1] < 0.2
    return desc, colors, opacities


def test_a_gradient_glow_runs_from_its_first_stop_at_the_edge_to_its_last() -> None:
    """The gradient is laid along the falloff, its start where the coverage is full.

    So the opaque end of Photoshop's default glow gradient sits against the
    layer. Its transparency stops hold the coverage under them, and reversing the
    gradient turns both around.
    """
    desc, colors, opacities = _gradient_glow()
    falloff = np.array([[1.0, 0.0]], dtype=np.float32)
    color, painted = _paint(desc, falloff)
    assert color.shape == (1, 2, 3)
    assert np.allclose(color[0, 0], colors[0], atol=1e-6)
    assert np.allclose(color[0, 1], colors[-1], atol=1e-6)
    assert float(painted[0, 0, 0]) == pytest.approx(opacities[0])
    assert float(painted[0, 1, 0]) == 0.0

    desc[Key.Reverse] = True
    color, _ = _paint(desc, falloff)
    assert np.allclose(color[0, 0], colors[-1], atol=1e-6)
    assert np.allclose(color[0, 1], colors[0], atol=1e-6)


def test_a_gradients_transparency_holds_the_coverage_under_it() -> None:
    """The coverage is the falloff where the transparency read there is above it.

    With every transparency stop at 40%, the coverage follows the falloff
    up to 0.4 and stays there; the gradient does not scale the falloff
    down, which would leave a glow at 0.16 where Photoshop paints 0.4.
    """
    desc, _, _ = _gradient_glow()
    for stop in desc[Key.Gradient][Key.Transparency]:
        stop[Key.Opacity] = Double(40.0)
    falloff = np.array([[0.2, 0.6, 1.0]], dtype=np.float32)
    _, painted = _paint(desc, falloff)
    assert np.allclose(painted[0, :, 0], [0.2, 0.4, 0.4], atol=1e-6)


def test_a_gradient_opaque_at_both_ends_paints_what_a_solid_colour_would() -> None:
    """With no transparency to hold it, the coverage is the falloff itself."""
    desc, _, _ = _gradient_glow()
    for stop in desc[Key.Gradient][Key.Transparency]:
        stop[Key.Opacity] = Double(100.0)
    disc = _disc(81, _RADIUS)
    falloff = effects.glow_falloff(disc, size=6.0, spread=0.0, range_=100.0)
    _, painted = _paint(desc, falloff, disc)
    assert np.allclose(painted, _glow(disc), atol=1e-6)


def test_a_gradient_is_placed_by_the_falloff_and_not_by_the_cut() -> None:
    """A half-covered edge pixel takes the colour of its falloff, not half of it.

    The cut under the silhouette scales the coverage after the gradient is
    read, so an anti-aliased edge keeps the gradient's start against the
    layer instead of sliding it along.
    """
    desc, colors, _ = _gradient_glow()
    falloff = np.ones((1, 2), dtype=np.float32)
    shape = np.array([[[0.5], [0.0]]], dtype=np.float32)
    color, painted = _paint(desc, falloff, shape=shape)
    assert np.allclose(color[0, 0], colors[0], atol=1e-6)
    assert np.allclose(color[0, 1], colors[0], atol=1e-6)
    assert float(painted[0, 0, 0]) == pytest.approx(0.5 * float(painted[0, 1, 0]))


# -- Compositing ----------------------------------------------------------- #


def _forge_glow(layer: Layer, **values: float) -> Descriptor:
    """Give ``layer`` a copy of the donor's effects, its glow edited.

    ``values`` maps descriptor keys, by their :py:class:`Key` names, to the
    number to put there. Returns the glow's descriptor. The donor's other
    effects are all switched off, so the layer carries the glow alone.
    """
    _, donor = _donor()
    block = copy.deepcopy(
        donor.tagged_blocks.get_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO)
    )
    layer.tagged_blocks.set_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO, block)
    glow = block[Klass.OuterGlow]
    for name, value in values.items():
        glow[getattr(Key, name)] = Double(value)
    return glow


def _hard_glow(layer: Layer, size: float = 4.0) -> Descriptor:
    """A fully opaque glow spread to a hard band of ``size`` pixels."""
    return _forge_glow(layer, Blur=size, ChokeMatte=100.0, Opacity=100.0)


def test_layer_opacity_fades_the_glow_with_the_layer_and_never_shows_under_it() -> None:
    """A half-opaque layer draws a half-opaque glow and nothing beneath itself.

    The glow is composited apart from the layer's source, so it carries the
    layer opacity itself. Under the layer it is cut away whatever the layer's
    opacity, so the card's middle is the card's own alpha and no more.
    """
    _, layer = _masks_card()
    _hard_glow(layer)
    viewport = _around(layer)
    full = _alone(layer, viewport)
    layer.opacity = 128
    faded = _alone(layer, viewport)

    y = (layer.bbox[1] + layer.bbox[3]) // 2 - viewport[1]
    beside, under = 16 - 2, (layer.bbox[2] - layer.bbox[0]) // 2 + 16
    assert float(full[y, beside, 0]) == pytest.approx(1.0)
    assert float(faded[y, beside, 0]) == pytest.approx(128 / 255.0, rel=1e-4)
    assert float(faded[y, under, 0]) == pytest.approx(128 / 255.0, rel=1e-4)


def test_fill_opacity_leaves_the_glow_alone_and_the_layer_still_hides_it() -> None:
    """Fill opacity fades the layer's own paint and not its effects.

    At fill 0 the card paints nothing, its glow is still whole beside it,
    and under it there is nothing at all: the glow is cut by the layer's
    coverage rather than by its paint.
    """
    _, layer = _masks_card()
    _hard_glow(layer)
    viewport = _around(layer)
    full = _alone(layer, viewport)
    layer.tagged_blocks.set_data(Tag.BLEND_FILL_OPACITY, ByteElement(0))
    faded = _alone(layer, viewport)

    y = (layer.bbox[1] + layer.bbox[3]) // 2 - viewport[1]
    beside, under = 16 - 2, (layer.bbox[2] - layer.bbox[0]) // 2 + 16
    assert float(faded[y, beside, 0]) == pytest.approx(float(full[y, beside, 0]))
    assert float(faded[y, under, 0]) == 0.0


def test_a_layer_over_its_glow_leaves_it_what_its_coverage_leaves() -> None:
    """Under a partial pixel the glow shows for the part the layer does not paint.

    ``feathered-stroke.psd``'s square fades across a 6 px mask ramp. Its hard
    glow, spread and so cast from the square's coverage, reaches past the
    viewport on every side, and over the ramp the glow and the square
    together cover every pixel whole: the glow has ``1 - alpha`` of each.
    Cut by ``1 - alpha`` and then composited under the layer as well, it
    would have ``(1 - alpha)`` squared and leave the ramp translucent. At
    fill opacity zero the square paints nothing, and the glow has the plain
    ``1 - alpha``.
    """
    psd, layer = _feathered_stroke_layer("Outset 6")
    _hard_glow(layer, size=_COVERAGE_SIZE)
    viewport = _around(layer, 8)
    mask = layer.numpy("mask")
    assert mask is not None
    ramp = mask[mask.shape[0] // 2, :, 0]
    assert np.all((ramp[:6] > 0.0) & (ramp[:6] < 1.0)), ramp

    alpha = _alone(layer, viewport)
    y = (layer.bbox[1] + layer.bbox[3]) // 2 - viewport[1]
    x0 = layer.bbox[0] - viewport[0]
    row = alpha[y, :, 0]
    assert np.allclose(row, 1.0, atol=1e-6), row

    layer.tagged_blocks.set_data(Tag.BLEND_FILL_OPACITY, ByteElement(0))
    unpainted = _alone(layer, viewport)[y, :, 0]
    assert np.allclose(unpainted[x0 : x0 + 6], 1.0 - ramp[:6], atol=1e-6)
    assert np.allclose(unpainted[x0 - 4 : x0], 1.0, atol=1e-6)


def test_a_layer_in_another_mode_leaves_its_glow_the_complement_of_its_alpha() -> None:
    """Only a Normal layer covers ``alpha`` of its glow, so only its glow is boosted.

    A Soft Light layer lets what lies beneath show through its paint, so its
    glow is cut by ``1 - alpha`` alone and the square over it covers
    ``alpha`` of what that leaves: ``alpha + (1 - alpha) ** 2`` over the ramp.
    """
    psd, layer = _feathered_stroke_layer("Outset 6")
    _hard_glow(layer, size=_COVERAGE_SIZE)
    layer.blend_mode = BlendMode.SOFT_LIGHT
    viewport = _around(layer, 8)
    mask = layer.numpy("mask")
    assert mask is not None
    ramp = mask[mask.shape[0] // 2, :6, 0]

    y = (layer.bbox[1] + layer.bbox[3]) // 2 - viewport[1]
    x0 = layer.bbox[0] - viewport[0]
    row = _alone(layer, viewport)[y, :, 0]
    assert np.allclose(row[x0 : x0 + 6], ramp + (1.0 - ramp) ** 2, atol=1e-6), row
    assert np.allclose(row[x0 - 4 : x0], 1.0, atol=1e-6)


def test_an_adjustment_layers_glow_is_cut_by_its_mask_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An adjustment layer paints nothing over its glow, so the cut is ``1 - mask``.

    ``exposure_rgb.psd``'s first exposure is masked to the right of the
    document. Given a hard glow and a mask ramping up over 20 px, the glow
    alone over nothing has exactly what the ramp leaves of each pixel, and
    nothing under the solid mask, rather than the full glow a layer that
    painted over it would be given to cover.
    """
    psd = PSDImage.open(full_name("adjustments/exposure_rgb.psd"))
    layer = next(each for each in psd.descendants() if each.name == "Exposure 1")
    _hard_glow(layer, size=_COVERAGE_SIZE)
    viewport = (100, 0, 200, 200)
    read_mask = Compositor._get_mask

    def ramped(
        self: Compositor,
        target: Layer,
        box: tuple[int, int, int, int] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if target is not layer:
            return read_mask(self, target, box, *args, **kwargs)
        left, top, right, bottom = self.viewport if box is None else box
        columns = np.arange(left, right, dtype=np.float32)
        ramp = np.clip((columns - 150.0) / 20.0, 0.0, 1.0)
        return np.broadcast_to(ramp[None, :, None], (bottom - top, right - left, 1))

    monkeypatch.setattr(Compositor, "_get_mask", ramped)
    alpha = _alone(layer, viewport)[100, :, 0]
    ramp = np.clip((np.arange(100) - 50) / 20.0, 0.0, 1.0)
    assert np.all(alpha[:10] == 0.0)  # beyond the glow's size
    assert np.allclose(alpha[30:50], 1.0, atol=1e-6)
    assert np.allclose(alpha[50:70], 1.0 - ramp[50:70], atol=1e-6), alpha[50:70]
    assert np.all(alpha[70:] == 0.0)


def _over_grey(layer: Layer, viewport: tuple[int, int, int, int]) -> np.ndarray:
    """The layer composited over an opaque mid-grey backdrop."""
    height, width = viewport[3] - viewport[1], viewport[2] - viewport[0]
    compositor = Compositor(
        viewport,
        np.full((height, width, 3), 0.5, dtype=np.float32),
        np.ones((height, width, 1), dtype=np.float32),
    )
    compositor.apply(layer)
    return compositor.result_over_backdrop()


def _blue(glow: Descriptor) -> None:
    for key, value in ((b"Rd  ", 0.0), (b"Grn ", 0.0), (b"Bl  ", 255.0)):
        glow[Key.Color][key] = Double(value)


def test_the_blend_mode_is_the_descriptors_and_defaults_as_the_api_does() -> None:
    """A glow is blended as ``OuterGlow.blend_mode`` reads it: Normal when unsaid.

    A hard blue glow over grey: Screen leaves the grey in the red and green
    channels and Normal replaces them, so the pixel beside the card tells
    the two apart.
    """
    _, layer = _masks_card()
    glow = _hard_glow(layer)
    _blue(glow)
    viewport = _around(layer)
    y, x = (layer.bbox[1] + layer.bbox[3]) // 2 - viewport[1], 16 - 2

    assert glow[Key.Mode].enum == Enum.Screen
    assert np.allclose(_over_grey(layer, viewport)[y, x], (0.5, 0.5, 1.0), atol=1e-6)

    glow[Key.Mode].enum = Enum.Normal
    normal = _over_grey(layer, viewport)[y, x]
    assert np.allclose(normal, (0.0, 0.0, 1.0), atol=1e-6)

    del glow[Key.Mode]
    assert _outer_glow(layer).blend_mode is BlendMode.NORMAL
    assert np.allclose(_over_grey(layer, viewport)[y, x], normal, atol=1e-6)


def test_a_glow_goes_over_the_drop_shadow_of_the_same_layer() -> None:
    """Beneath the layer the glow lies above the shadow, as Photoshop stacks them.

    The card's own shadow is made hard, black and cast nowhere, and a hard
    white Screen glow added beside it. Over the shadow the glow screens the
    black to white; under it the shadow would multiply the white to black.
    """
    _, layer = _masks_card()
    block = layer.tagged_blocks.get_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO)
    shadow = block[Klass.DropShadow]
    for key, value in (
        (Key.Distance, 0.0),
        (Key.Blur, 4.0),
        (Key.ChokeMatte, 100.0),
        (Key.Opacity, 100.0),
    ):
        shadow[key] = Double(value)
    _, donor = _donor()
    block[Klass.OuterGlow] = copy.deepcopy(_glow_descriptor(donor))
    glow = _glow_descriptor(layer)
    for key, value in ((Key.Blur, 4.0), (Key.ChokeMatte, 100.0), (Key.Opacity, 100.0)):
        glow[key] = Double(value)
    assert [effect.name for effect in layer.effects] == ["DropShadow", "OuterGlow"]

    viewport = _around(layer)
    y, x = (layer.bbox[1] + layer.bbox[3]) // 2 - viewport[1], 16 - 2
    assert np.allclose(_over_grey(layer, viewport)[y, x], 1.0, atol=1e-6)


def test_a_glow_reaches_a_viewport_its_layer_misses() -> None:
    """The glow is drawn from the whole layer, wherever the viewport is.

    The strip beside the card is clear of the card's own box, so the cull
    has to measure the glow's reach to keep the card at all, and the
    silhouette it is drawn from lies wholly outside the strip.
    """
    _, layer = _masks_card()
    _forge_glow(layer, Blur=20.0, Opacity=100.0)
    left, top, right, bottom = layer.bbox
    strip = (left, bottom + 4, right, bottom + 24)
    assert utils.intersect(strip, layer.bbox) == (0, 0, 0, 0)
    assert utils.intersect(strip, _effect_reach(layer)) != (0, 0, 0, 0)

    alpha = _alone(layer, strip)
    wide = _alone(layer, (left, top, right, bottom + 24))
    assert float(alpha.max()) > 0.01, "the strip lost the glow"
    assert np.allclose(alpha, wide[bottom + 4 - top :], atol=1e-6)


def test_a_group_canvas_makes_room_for_a_child_glow() -> None:
    """A group's content box counts the glows its children draw.

    It is the canvas an isolated group composites its children on (#808).
    ``masks.psb``'s header band, given a 100 px glow, reaches past the group
    that holds it, which the group's own box does not.
    """
    psd = PSDImage.open(full_name("masks.psb"))
    header = next(layer for layer in psd if layer.name == "Header")
    band = next(layer for layer in psd.descendants() if layer.name == "Rectangle 2")
    assert band.parent is header
    _forge_glow(band, Blur=100.0)
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


@pytest.mark.parametrize("blend_mode", [BlendMode.PASS_THROUGH, BlendMode.NORMAL])
def test_a_group_draws_its_glow_from_its_composited_contents(
    blend_mode: BlendMode,
) -> None:
    """A group's glow is drawn from the alpha of its children, composited.

    With the band's own effects off, the group covers the band and nothing
    else; a hard 20 px glow on the group then covers the pixel 10 px below
    the band in full, and under the band the group's alpha is the band's.
    """
    header, band = _header(blend_mode)
    for effect in list(band.effects):
        effect.descriptor[Key.Enabled] = False
    viewport = (0, 0, 640, 200)
    y, x = band.bbox[3] + 10, 320
    inside = (band.bbox[1] + band.bbox[3]) // 2
    before = _alone(header, viewport)
    assert float(before[y, x, 0]) == 0.0
    assert float(before[inside, x, 0]) == 1.0

    _hard_glow(header, size=20.0)
    after = _alone(header, viewport)
    assert float(after[y, x, 0]) == pytest.approx(1.0)
    assert float(after[inside, x, 0]) == 1.0


def test_a_pass_through_group_keeps_its_glow_beneath_it() -> None:
    """The glow of a pass-through group does not paint over the group.

    The glow goes on before the group is resolved, and the group is resolved
    over it: the band's middle comes out as it does with no glow, and below
    the band the glow is there.
    """

    def render(with_glow: bool) -> np.ndarray:
        header, _ = _header(BlendMode.PASS_THROUGH)
        if with_glow:
            _forge_glow(header, Blur=30.0, Opacity=100.0)
        return _over_grey(header, (0, 0, 640, 260))

    glowing, plain = render(True), render(False)
    assert np.allclose(glowing[50, 320], plain[50, 320], atol=1e-6)
    assert not np.allclose(glowing[115, 320], plain[115, 320], atol=1e-2)


def test_the_technique_is_read_and_an_unknown_one_is_soft() -> None:
    """Precise draws a different glow from softer; a technique the file invents draws softer."""
    _, layer = _masks_card()
    glow = _forge_glow(layer, Blur=12.0, Opacity=100.0)
    viewport = _around(layer, 20)
    assert glow[Key.GlowTechnique].enum == Enum.SoftMatte
    soft = _alone(layer, viewport)
    glow[Key.GlowTechnique].enum = Enum.PreciseMatte
    assert _outer_glow(layer).glow_type is GlowTechnique.PRECISE_MATTE
    precise = _alone(layer, viewport)
    assert not np.allclose(soft, precise, atol=1e-2)
    glow[Key.GlowTechnique].enum = b"XXXX"
    assert _outer_glow(layer).glow_type is None
    assert np.array_equal(_alone(layer, viewport), soft)


def test_noise_and_jitter_are_read_but_not_applied(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A noisy, jittered glow renders as the plain one, and says so."""
    _, layer = _masks_card()
    glow = _forge_glow(layer)
    viewport = _around(layer)
    plain = _alone(layer, viewport)
    glow[Key.Noise] = Double(50.0)
    glow[Key.ShadingNoise] = Double(50.0)
    assert _outer_glow(layer).noise == 50.0
    assert _outer_glow(layer).quality_jitter == 50.0
    with caplog.at_level(logging.DEBUG, logger="psd_tools.composite.composite"):
        noisy = _alone(layer, viewport)
    assert "noise" in caplog.text and "jitter" in caplog.text
    assert np.array_equal(noisy, plain)


def test_a_glow_over_the_documents_budget_is_dropped_not_drawn() -> None:
    """A glow whose canvas the document's budget refuses costs the glow only.

    A budget that holds a solid glow's canvas refuses the same canvas once
    the glow is a gradient, whose paint costs more.
    """
    psd, layer = _masks_card()
    glow = _forge_glow(layer)
    viewport = _around(layer)
    glow[Key.Enabled] = False
    disabled = _alone(layer, viewport)
    glow[Key.Enabled] = True
    assert not np.array_equal(_alone(layer, viewport), disabled)

    grown = effects.outer_glow_bbox(layer.bbox, glow)
    width, height = grown[2] - grown[0], grown[3] - grown[1]
    psd.max_alloc_bytes = width * height * effects._SHADOW_BYTES - 1
    assert np.array_equal(_alone(layer, viewport), disabled)

    psd.max_alloc_bytes = width * height * effects._SHADOW_BYTES
    assert not np.array_equal(_alone(layer, viewport), disabled)
    glow[Key.Gradient] = copy.deepcopy(_gradient_glow()[0][Key.Gradient])
    assert np.array_equal(_alone(layer, viewport), disabled)


def _setting(key: bytes, value: Any) -> Callable[[Descriptor], Any]:
    return lambda desc: desc.__setitem__(key, value)


def _empty_gradient(desc: Descriptor) -> None:
    gradient = Descriptor(classID=Klass.Gradient.value)
    gradient[Type.GradientForm] = Enumerated(
        typeID=Type.GradientForm.value, enum=Enum.CustomStops.value
    )
    gradient[Key.Colors] = List()
    desc[Key.Gradient] = gradient


# Ways a glow's descriptor can defeat the reads that draw it, as a shadow's
# can, plus a gradient with nothing to paint.
_UNDRAWABLE: dict[str, Callable[[Descriptor], Any]] = {
    "size": _setting(Key.Blur, "wide"),
    "infinite-size": _setting(Key.Blur, Double(float("inf"))),
    "nan-spread": _setting(Key.ChokeMatte, Double(float("nan"))),
    "nan-range": _setting(Key.InputRange, Double(float("nan"))),
    "nan-opacity": _setting(Key.Opacity, Double(float("nan"))),
    "infinite-opacity": _setting(Key.Opacity, Double(float("inf"))),
    "no-color": lambda desc: desc.pop(Key.Color),
    "color-class": lambda desc: setattr(desc[Key.Color], "classID", b"XXXX"),
    "empty-gradient": _empty_gradient,
}


def _rendered(mutate: Callable[[Descriptor], Any]) -> np.ndarray:
    psd, donor = _donor()
    mutate(_glow_descriptor(donor))
    color, _, alpha = composite(psd)
    return np.concatenate((color, alpha), axis=2)


@pytest.mark.parametrize("defect", sorted(_UNDRAWABLE))
def test_an_unreadable_outer_glow_is_dropped_and_the_document_renders(
    defect: str,
) -> None:
    """A descriptor that cannot be read costs its glow, not the render.

    Asserted against the same document with the glow switched off, so a
    glow drawn at some invented size or colour would not pass.
    """
    disabled = _rendered(_setting(Key.Enabled, False))
    assert not np.array_equal(_rendered(lambda desc: None), disabled)
    rendered = _rendered(_UNDRAWABLE[defect])
    assert np.isfinite(rendered).all()
    assert np.array_equal(rendered, disabled)


@pytest.mark.parametrize(
    ("stops", "key"),
    [
        (Key.Transparency, Key.Opacity),
        (Key.Transparency, Key.Location),
        (Key.Colors, Key.Location),
    ],
    ids=["opacity", "transparency-location", "colour-location"],
)
def test_a_glow_whose_gradient_stop_is_nan_still_renders_finite(
    stops: bytes, key: bytes
) -> None:
    """The stop is held in range where it is read, so the glow is still drawn."""

    def nan_stop(desc: Descriptor) -> None:
        gradient = copy.deepcopy(_gradient_glow()[0][Key.Gradient])
        gradient[stops][-1][key] = Double(float("nan"))
        desc[Key.Gradient] = gradient

    rendered = _rendered(nan_stop)
    assert np.isfinite(rendered).all()
    assert not np.array_equal(rendered, _rendered(_setting(Key.Enabled, False)))


# -- Fidelity against Photoshop's own render ------------------------------- #

# Each case is a document, the region around its outer glow, the bound the
# render is held to, and a floor under the error of the same render with the
# glow switched off, which is what shows the region is measuring a glow at
# all. The bounds are what the render scores, with room for a pixel:
#
# - layer_effects.psd: the rows beside and below the glowing text, clear of
#   the emboss above it, whose outer shadow is not drawn.
# - layer_params.psb: the glowing shape and its drop shadow; what is left is
#   the blur that stands in for Photoshop's. The bound is what pins the
#   gradient's transparency as a cap on the falloff: scaling the falloff by
#   it scores above it.
_PREVIEWS = [
    ("layer_effects.psd", (56, 98, 541, 164), 1e-5, 4e-3),
    ("layer_params.psb", (130, 230, 470, 560), 3.2e-4, 5e-3),
]

# How much closer to the preview the glow has to bring its region than
# leaving it out does.
_IMPROVEMENT = 4.0


def _error(
    psd: PSDImage, region: tuple[int, int, int, int], force: bool = False
) -> float:
    reference = psd.numpy()
    color, _, alpha = composite(psd, force=force)
    result = color
    if reference.shape[2] > color.shape[2]:
        result = np.concatenate((color, alpha), axis=2)
    left, top, right, bottom = region
    return float(
        _mse(reference[top:bottom, left:right], result[top:bottom, left:right])
    )


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize(
    ("filename", "region", "bound", "without"),
    _PREVIEWS,
    ids=[case[0] for case in _PREVIEWS],
)
def test_outer_glows_render_to_photoshops_preview(
    filename: str,
    region: tuple[int, int, int, int],
    bound: float,
    without: float,
    force: bool,
) -> None:
    """Every outer glow in the corpus, against Photoshop's own render of it.

    The region has to measure a glow, and drawing it has to bring the region
    measurably closer to the preview than leaving it out.
    """
    error = _error(PSDImage.open(full_name(filename)), region, force)

    unglowed = PSDImage.open(full_name(filename))
    for layer in unglowed.descendants():
        for effect in list(layer.effects.find("outerglow")):
            effect.descriptor[Key.Enabled] = False
    floor = _error(unglowed, region, force)
    assert floor >= without, "the region has no glow"
    assert error <= bound
    assert error * _IMPROVEMENT < floor
