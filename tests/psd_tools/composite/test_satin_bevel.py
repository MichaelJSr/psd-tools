"""The satin and bevel/emboss layer effects, drawn and composited.

``layer_effects.psd`` carries an inner bevel, an emboss and a satin that
Photoshop rendered into its preview, ``layer_comps.psb`` an inner bevel on a
large disc, and ``layer_params.psb`` a pillow emboss lit by the document's
global altitude, so the fidelity tests measure against those. The geometry the
corpus does not isolate is checked on forged silhouettes, where the result has
a closed form.
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from psd_tools.api.effects import BevelEmboss
from psd_tools.api.layers import Layer
from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import composite, effects
from psd_tools.constants import BevelStyle, BevelTechnique
from psd_tools.psd.descriptor import Descriptor, Double, Enumerated
from psd_tools.terminology import Key

from ..utils import full_name
from .test_composite import _mse

_SQUARE = (0, 0, 40, 40)


def _square() -> np.ndarray:
    """A hard-edged 20 x 20 square in the middle of a 40 x 40 canvas."""
    shape = np.zeros((40, 40, 1), dtype=np.float32)
    shape[10:30, 10:30] = 1.0
    return shape


def _disc(size: int, radius: float) -> np.ndarray:
    """A hard-edged disc centred on a ``size`` x ``size`` canvas."""
    centre = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    disc = (xx - centre) ** 2 + (yy - centre) ** 2 <= radius**2
    return disc.astype(np.float32)[:, :, None]


def _satin(shape: np.ndarray, **kwargs: Any) -> np.ndarray:
    params: dict[str, Any] = dict(distance=3.0, angle=0.0, size=0.0, inverted=False)
    params.update(kwargs)
    height, width = shape.shape[:2]
    return effects.draw_satin((0, 0, width, height), shape, **params)


def _bevel(shape: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
    params: dict[str, Any] = dict(
        style=BevelStyle.INNER_BEVEL,
        technique=BevelTechnique.SOFT_MATTE,
        up=True,
        depth=100.0,
        size=8.0,
        soften=0.0,
        angle=90.0,
        altitude=30.0,
    )
    params.update(kwargs)
    height, width = shape.shape[:2]
    return effects.draw_bevel((0, 0, width, height), shape, **params)


# -- Satin ------------------------------------------------------------------ #


def test_an_unblurred_satin_lights_where_one_copy_passes_the_other() -> None:
    """The copies, moved 3 px either way, disagree within 3 px of an edge.

    Lit from 0 degrees the copies move along the rows, so the satin covers
    the three columns inside the square's left and right edges and nothing
    between them or along the top and bottom.
    """
    satin = _satin(_square())[:, :, 0]
    expected = np.zeros((40, 40), dtype=np.float32)
    expected[10:30, 10:13] = 1.0
    expected[10:30, 27:30] = 1.0
    assert np.allclose(satin, expected, atol=1e-6)


def test_a_satin_at_no_distance_is_empty_or_the_whole_layer() -> None:
    """Copies that do not move agree everywhere; inverted, that is the layer."""
    shape = _disc(41, 12.0)
    assert float(_satin(shape, distance=0.0, size=6.0).max()) == 0.0
    inverted = _satin(shape, distance=0.0, size=6.0, inverted=True)
    assert np.allclose(inverted, shape, atol=1e-6)


@pytest.mark.parametrize("angle", [0.0, 30.0, 90.0, 135.0])
def test_a_satin_is_the_same_for_opposite_angles(angle: float) -> None:
    """The copies move both ways, so turning the angle round swaps them."""
    shape = np.zeros((48, 48, 1), dtype=np.float32)
    shape[10:38, 12:30] = 1.0
    shape[28:38, 12:40] = 1.0
    satin = _satin(shape, distance=5.0, size=6.0, angle=angle)
    turned = _satin(shape, distance=5.0, size=6.0, angle=angle + 180.0)
    assert satin.any()
    assert np.allclose(satin, turned, atol=1e-5)


def test_a_satin_mirrors_with_its_layer() -> None:
    """A mirrored layer under a mirrored angle gets the mirrored satin."""
    shape = np.zeros((48, 48, 1), dtype=np.float32)
    shape[10:38, 12:30] = 1.0
    shape[28:38, 12:40] = 1.0
    satin = _satin(shape, distance=5.0, size=6.0, angle=30.0)
    mirrored = _satin(shape[:, ::-1], distance=5.0, size=6.0, angle=150.0)
    assert np.allclose(satin[:, ::-1], mirrored, atol=1e-5)


def test_a_satin_stays_inside_its_layer() -> None:
    """Inverted, a satin is lit away from the edges, but never past them."""
    shape = _disc(41, 12.0)
    satin = _satin(shape, distance=4.0, size=6.0, angle=60.0, inverted=True)
    assert float(satin[shape == 0.0].max()) == 0.0
    assert float(satin.max()) > 0.5


# -- Bevel ------------------------------------------------------------------ #

_STYLES = [
    BevelStyle.INNER_BEVEL,
    BevelStyle.OUTER_BEVEL,
    BevelStyle.EMBOSS,
    BevelStyle.PILLOW_EMBOSS,
]
_TECHNIQUES = [
    BevelTechnique.SOFT_MATTE,
    BevelTechnique.PRECISE_MATTE,
    BevelTechnique.SLOPE_LIMIT_MATTE,
]


@pytest.mark.parametrize("style", _STYLES)
@pytest.mark.parametrize("technique", _TECHNIQUES)
def test_a_bevel_at_no_depth_paints_nothing(
    style: BevelStyle, technique: BevelTechnique
) -> None:
    """A surface with no slope is lit as the flat layer is: no light, no shade."""
    highlight, shadow = _bevel(
        _disc(41, 12.0), style=style, technique=technique, depth=0.0, soften=3.0
    )
    assert float(highlight.max()) < 1e-6
    assert float(shadow.max()) < 1e-6


def _lean(amount: np.ndarray) -> tuple[float, float]:
    """Where ``amount`` lies, as a unit vector from the canvas centre."""
    weights = amount[:, :, 0]
    yy, xx = np.mgrid[0 : weights.shape[0], 0 : weights.shape[1]]
    centre = (weights.shape[1] - 1) / 2.0, (weights.shape[0] - 1) / 2.0
    total = float(weights.sum())
    dx = float((xx * weights).sum()) / total - centre[0]
    dy = float((yy * weights).sum()) / total - centre[1]
    length = math.hypot(dx, dy)
    return dx / length, dy / length


@pytest.mark.parametrize("style", _STYLES)
@pytest.mark.parametrize("angle", [0.0, 90.0, 150.0, -45.0])
def test_a_bevel_is_lit_on_the_side_facing_the_light(
    style: BevelStyle, angle: float
) -> None:
    """The highlight leans toward the light and the shadow away from it.

    The angle is where the light comes from, counter-clockwise from the
    positive x axis, and image rows run down. A pillow emboss sinks the edge,
    so its outer slope faces the other way; the disc's inside is what decides
    where its highlight lies, which is why the canvas is masked to it here.
    """
    shape = _disc(61, 16.0)
    highlight, shadow = _bevel(shape, style=style, angle=angle)
    if style == BevelStyle.PILLOW_EMBOSS:
        highlight, shadow = highlight * shape, shadow * shape
    toward = math.cos(math.radians(angle)), -math.sin(math.radians(angle))
    lit, shaded = _lean(highlight), _lean(shadow)
    assert lit[0] * toward[0] + lit[1] * toward[1] > 0.99
    assert shaded[0] * toward[0] + shaded[1] * toward[1] < -0.99


@pytest.mark.parametrize("technique", _TECHNIQUES)
def test_a_bevel_turns_round_with_its_light(technique: BevelTechnique) -> None:
    """A disc lit from the opposite side is the same disc turned half round."""
    shape = _disc(61, 16.0)
    highlight, shadow = _bevel(shape, technique=technique, angle=30.0, soften=2.0)
    turned = _bevel(shape, technique=technique, angle=210.0, soften=2.0)
    assert np.allclose(highlight, turned[0][::-1, ::-1], atol=1e-5)
    assert np.allclose(shadow, turned[1][::-1, ::-1], atol=1e-5)


@pytest.mark.parametrize("style", _STYLES)
def test_a_bevel_stamped_down_is_one_lit_from_the_other_side(
    style: BevelStyle,
) -> None:
    """Turning the surface over is turning the light round, the altitude kept."""
    shape = np.zeros((48, 48, 1), dtype=np.float32)
    shape[10:38, 12:30] = 1.0
    shape[28:38, 12:40] = 1.0
    down = _bevel(shape, style=style, up=False, angle=40.0)
    turned = _bevel(shape, style=style, up=True, angle=220.0)
    assert np.allclose(down[0], turned[0], atol=1e-5)
    assert np.allclose(down[1], turned[1], atol=1e-5)


def test_a_chisel_hard_bevel_lights_its_faces_flat() -> None:
    """Each face of a hard chisel is a plane sloping one pixel per pixel.

    Lit from overhead (90 degrees) at altitude ``a``, the top face's normal
    leans toward the light, so it is lit at ``(cos a + sin a) / sqrt 2``;
    the bottom face leans away and is lit at ``(sin a - cos a) / sqrt 2``,
    below zero at 30 degrees, so it takes the whole shadow. Read on the
    middle column, clear of the corners' mitres, and inside the face, clear
    of the edge and the plateau.
    """
    shape = np.zeros((60, 60, 1), dtype=np.float32)
    shape[10:50, 10:50] = 1.0
    altitude = math.radians(30.0)
    highlight, shadow = _bevel(
        shape, technique=BevelTechnique.PRECISE_MATTE, size=8.0, angle=90.0
    )
    top = (math.cos(altitude) + math.sin(altitude)) / math.sqrt(2.0)
    flat = math.sin(altitude)
    expected = (top - flat) / (1.0 - flat)
    assert np.allclose(highlight[12:16, 30, 0], expected, atol=1e-5)
    assert np.allclose(shadow[12:16, 30, 0], 0.0)
    assert np.allclose(shadow[44:48, 30, 0], 1.0, atol=1e-5)
    assert np.allclose(highlight[44:48, 30, 0], 0.0)
    # The plateau inside the band is flat, and lit as the flat layer is.
    assert float(highlight[24:36, 24:36].max()) == 0.0
    assert float(shadow[24:36, 24:36].max()) == 0.0


def test_soften_blurs_a_bevels_lighting() -> None:
    """Softened, the chisel's top face is no longer one flat value."""
    shape = np.zeros((60, 60, 1), dtype=np.float32)
    shape[10:50, 10:50] = 1.0
    hard, _ = _bevel(shape, technique=BevelTechnique.PRECISE_MATTE, size=8.0)
    soft, _ = _bevel(shape, technique=BevelTechnique.PRECISE_MATTE, size=8.0, soften=6)
    assert float(np.ptp(hard[11:17, 30, 0])) < 1e-5
    assert float(np.ptp(soft[11:17, 30, 0])) > 0.05


def test_an_unreadable_bevel_parameter_raises() -> None:
    """NaN would pass through every filter and take the canvas with it."""
    with pytest.raises(ValueError):
        _bevel(_square(), depth=float("nan"))
    with pytest.raises(ValueError):
        _satin(_square(), angle=float("inf"))


# -- In the compositor ------------------------------------------------------ #


def _layer(psd: PSDImage, name: str) -> Layer:
    return next(layer for layer in psd.descendants() if layer.name == name)


def _bevel_of(layer: Layer) -> BevelEmboss:
    effect = next(iter(layer.effects.find("bevelemboss")))
    assert isinstance(effect, BevelEmboss)
    return effect


def _rendered(
    psd: PSDImage, viewport: tuple[int, int, int, int] | None = None
) -> np.ndarray:
    color, _, alpha = composite(psd, viewport=viewport)
    return np.concatenate((color, alpha), axis=2)


def test_an_inner_bevel_paints_inside_the_layer_and_an_outer_one_beside_it() -> None:
    """``layer_effects.psd``'s inner bevel, then the same bevel made outer.

    Compared with the bevel switched off: the inner one changes no pixel
    the layer does not cover, the outer one no pixel it covers whole.
    """
    psd = PSDImage.open(full_name("layer_effects.psd"))
    layer = _layer(psd, "Bevel")
    assert layer.bbox == (99, 33, 289, 80)
    viewport = (70, 5, 320, 110)
    shape = np.zeros((105, 250), dtype=np.float32)
    left, top, right, bottom = layer.bbox
    own = layer.numpy("shape")
    assert own is not None
    shape[top - 5 : bottom - 5, left - 70 : right - 70] = own[:, :, 0]
    bevel = _bevel_of(layer)
    inner = _rendered(psd, viewport)
    bevel.descriptor[Key.BevelStyle] = Enumerated(
        typeID=Key.BevelStyle, enum=BevelStyle.OUTER_BEVEL.value
    )
    outer = _rendered(psd, viewport)
    bevel.descriptor[Key.Enabled] = False
    plain = _rendered(psd, viewport)

    changed_inner = np.abs(inner - plain).max(axis=2) > 1e-6
    changed_outer = np.abs(outer - plain).max(axis=2) > 1e-6
    assert changed_inner.any() and changed_outer.any()
    assert not changed_inner[shape == 0.0].any()
    assert not changed_outer[shape == 1.0].any()


@pytest.fixture(scope="module")
def whole() -> np.ndarray:
    return _rendered(PSDImage.open(full_name("layer_effects.psd")))


@pytest.mark.parametrize(
    "viewport",
    [
        (305, 10, 623, 105),
        (400, 40, 450, 70),
        # Clear of the emboss layer's own box, inside what it paints beside it.
        (305, 10, 329, 34),
        (383, 560, 470, 590),
    ],
)
def test_a_cut_through_a_bevel_or_a_satin_renders_what_the_whole_does(
    viewport: tuple[int, int, int, int], whole: np.ndarray
) -> None:
    """The silhouette is read past the viewport, so the edge is not the layer's."""
    psd = PSDImage.open(full_name("layer_effects.psd"))
    left, top, right, bottom = viewport
    assert np.array_equal(_rendered(psd, viewport), whole[top:bottom, left:right])


def _setting(key: bytes, value: Any) -> Callable[[Descriptor], Any]:
    return lambda desc: desc.__setitem__(key, value)


# A number that will not parse, one that parses to something no arithmetic
# survives, and a colour that cannot be read: each costs the effect. Rendered
# on a box around the layers that carry the effect.
_AROUND = {"bevelemboss": (60, 0, 640, 110), "satin": (370, 550, 580, 620)}
_UNDRAWABLE: dict[str, tuple[str, Callable[[Descriptor], Any]]] = {
    "bevel-size": ("bevelemboss", _setting(Key.Blur, "wide")),
    "bevel-nan-depth": ("bevelemboss", _setting(Key.StrengthRatio, Double(math.nan))),
    "bevel-inf-soften": ("bevelemboss", _setting(Key.Softness, Double(math.inf))),
    "bevel-nan-opacity": (
        "bevelemboss",
        _setting(Key.HighlightOpacity, Double(math.nan)),
    ),
    "bevel-no-color": ("bevelemboss", lambda desc: desc.pop(Key.ShadowColor)),
    "satin-inf-distance": ("satin", _setting(Key.Distance, Double(math.inf))),
    "satin-nan-angle": ("satin", _setting(Key.LocalLightingAngle, Double(math.nan))),
    "satin-no-color": ("satin", lambda desc: desc.pop(Key.Color)),
}


@pytest.mark.parametrize("defect", sorted(_UNDRAWABLE))
def test_an_unreadable_effect_is_dropped_and_the_document_renders(
    defect: str,
) -> None:
    """Asserted against the same document with the effect switched off."""
    kind, mutate = _UNDRAWABLE[defect]

    def rendered(mutate: Callable[[Descriptor], Any]) -> np.ndarray:
        psd = PSDImage.open(full_name("layer_effects.psd"))
        for layer in psd.descendants():
            for effect in list(layer.effects.find(kind)):
                mutate(effect.descriptor)
        return _rendered(psd, _AROUND[kind])

    disabled = rendered(_setting(Key.Enabled, False))
    assert not np.array_equal(rendered(lambda desc: None), disabled)
    result = rendered(mutate)
    assert np.isfinite(result).all()
    assert np.array_equal(result, disabled)


# -- Fidelity against Photoshop's own render ------------------------------- #

# Each case is a document, the region around one effect, the effect's kind,
# the bound the render is held to, and a floor under the error of the same
# render with that kind switched off, which shows the region measures it:
#
# - layer_comps.psb: the disc's drop shadow and its edges are the rest.
# - layer_params.psb: the stroke's gradient and the outer glow are the rest.
#   The bound pins two things: the emboss shades the stroke, so it goes on
#   above it, and the altitude is the document's global one, 40 degrees,
#   not the 30 the descriptor holds; either undone scores above it.
_PREVIEWS = [
    ("layer_effects.psd", (99, 33, 289, 80), "bevelemboss", 1.5e-4, 5e-3),
    ("layer_effects.psd", (305, 10, 623, 105), "bevelemboss", 6e-4, 1e-2),
    ("layer_effects.psd", (383, 560, 563, 608), "satin", 1e-5, 5e-2),
    ("layer_comps.psb", (17, 20, 180, 183), "bevelemboss", 2.2e-3, 5e-3),
    ("layer_params.psb", (117, 66, 484, 139), "bevelemboss", 1.25e-2, 3e-2),
]


def _error(psd: PSDImage, region: tuple[int, int, int, int]) -> float:
    """The error on ``region``, rendered on it alone: see the cut test above."""
    left, top, right, bottom = region
    reference = psd.numpy()[top:bottom, left:right]
    result = _rendered(psd, region)[:, :, : reference.shape[2]]
    return float(_mse(reference, result))


@pytest.mark.parametrize(
    ("filename", "region", "kind", "bound", "without"),
    _PREVIEWS,
    ids=["%s-%s-%d" % (case[0], case[2], case[1][0]) for case in _PREVIEWS],
)
def test_satin_and_bevel_render_to_photoshops_preview(
    filename: str,
    region: tuple[int, int, int, int],
    kind: str,
    bound: float,
    without: float,
) -> None:
    psd = PSDImage.open(full_name(filename))
    error = _error(psd, region)
    for layer in psd.descendants():
        for effect in list(layer.effects.find(kind)):
            effect.descriptor[Key.Enabled] = False
    assert _error(psd, region) >= without, "the region has no %s" % kind
    assert error <= bound


# The fitted constants: the gaussian widths per pixel of size of the satin's
# blur, of the blur that rounds a smooth bevel and of soften, and the slope
# of a bevel's faces at 100% depth. The cases above hold them to Photoshop's
# previews; the cards in the TTSCardMaker corpus with a soften and a 979%
# depth agree with the same values.
_FITTED = {
    "_SATIN_SIGMA_PER_SIZE": 0.45,
    "_BEVEL_SIGMA_PER_SIZE": 0.45,
    "_SOFTEN_SIGMA_PER_SIZE": 0.45,
    "_BEVEL_SLOPE_PER_DEPTH": 1.0,
}


def test_the_fitted_constants_are_pinned() -> None:
    for name, value in _FITTED.items():
        assert getattr(effects, name) == value, name


@pytest.mark.parametrize(
    ("name", "region", "kind"),
    [
        ("_SATIN_SIGMA_PER_SIZE", (383, 560, 563, 608), "satin"),
        ("_BEVEL_SIGMA_PER_SIZE", (99, 33, 289, 80), "bevelemboss"),
        ("_BEVEL_SLOPE_PER_DEPTH", (99, 33, 289, 80), "bevelemboss"),
    ],
)
def test_the_fit_is_a_minimum(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    region: tuple[int, int, int, int],
    kind: str,
) -> None:
    """A fifth either side of each fitted value scores worse on its fixture."""
    psd = PSDImage.open(full_name("layer_effects.psd"))
    fitted = _error(psd, region)
    for factor in (0.8, 1.2):
        monkeypatch.setattr(effects, name, _FITTED[name] * factor)
        assert _error(psd, region) > fitted, factor
