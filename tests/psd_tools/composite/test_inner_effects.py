"""The inner shadow and inner glow layer effects, drawn and composited.

``layer_effects.psd`` carries one of each that Photoshop rendered into its
preview, on type layers, and those are what the fidelity tests measure. The
rest forge the effect onto ``effects/outside-stroke.psd``'s ``Rect``, a hard
16 x 16 blue square at (8, 8) on a 32 x 32 canvas, copying the descriptor
from ``layer_effects.psd`` and editing it, so what is drawn has a closed form.
"""

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from psd_tools.api.layers import Layer
from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import composite, effects
from psd_tools.constants import Resource, Tag
from psd_tools.psd.descriptor import Descriptor, Double, Enumerated
from psd_tools.terminology import Enum, Key, Klass, Type

from ..utils import full_name
from .test_composite import _mse
from .test_drop_shadow import _contour

pytest.importorskip("scipy")


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


_BOX = (0, 0, 81, 81)


def _shadow(shape: np.ndarray, **values: float) -> np.ndarray:
    params: dict[str, float] = {"distance": 0.0, "angle": 90.0, "size": 0.0}
    params.update(values)
    params.setdefault("choke", 0.0)
    return effects.draw_inner_shadow(
        _BOX,
        shape,
        distance=params["distance"],
        angle=params["angle"],
        size=params["size"],
        choke=params["choke"],
    )


# -- The drawers ------------------------------------------------------------ #


def test_an_inner_shadow_stays_inside_the_silhouette() -> None:
    """Nothing is cast outside the layer, and nowhere more than it covers."""
    shape = _disc(81, 20.0)
    shadow = _shadow(shape, distance=6.0, size=8.0, choke=30.0)
    assert shadow.shape == shape.shape
    assert np.all(shadow[shape == 0.0] == 0.0)
    assert np.all(shadow <= shape)
    assert shadow.any()


@pytest.mark.parametrize(
    ("angle", "sign"),
    [(90.0, (0, -1)), (0.0, (1, 0)), (180.0, (-1, 0)), (-90.0, (0, 1))],
)
def test_an_inner_shadow_falls_on_the_side_the_light_comes_from(
    angle: float, sign: tuple[int, int]
) -> None:
    """The rim nearest the light casts its shadow into the layer.

    The outside moves away from the light, as a drop shadow does, so it
    enters the layer on the light's side: the top for a light overhead at 90
    degrees, since image rows run down.
    """
    shadow = _shadow(_disc(81, 20.0), distance=8.0, angle=angle, size=4.0)
    x, y = _centroid(shadow)
    for offset, expected in zip((x - 40.0, y - 40.0), sign, strict=True):
        if expected == 0:
            assert offset == pytest.approx(0.0, abs=1e-3)
        else:
            assert offset * expected > 5.0


def test_an_unblurred_inner_shadow_is_the_outside_moved_in() -> None:
    """With no size, the shadow is the complement of the layer, shifted, cut to it."""
    shape = _disc(81, 20.0)
    shadow = _shadow(shape, distance=5.0, angle=90.0)
    moved = np.ones_like(shape)
    moved[5:] = 1.0 - shape[:-5]
    assert np.array_equal(shadow, shape * moved)


def test_an_inner_shadow_with_no_distance_or_size_casts_nothing() -> None:
    shadow = _shadow(_disc(81, 20.0))
    assert not shadow.any()


def test_a_full_choke_is_a_hard_band_as_deep_as_the_size() -> None:
    """Choke shrinks the silhouette before the blur, which takes what is left.

    At 100% all of the size goes to the choke, so the shadow is a hard band
    reaching ``size`` pixels in from the edge and nothing past it: along a
    straight edge, the last ``size`` columns of a square.
    """
    shape = np.zeros((81, 81, 1), dtype=np.float32)
    shape[20:61, 20:61] = 1.0
    size = 4
    row = _shadow(shape, size=float(size), choke=100.0)[40, 40:61, 0]
    assert np.all(row[-size:] == 1.0), row
    assert np.all(row[:-size] == 0.0), row


def test_choke_deepens_the_shadow_at_the_same_size() -> None:
    shape = _disc(81, 20.0)
    totals = [
        float(_shadow(shape, size=8.0, choke=choke).sum())
        for choke in (0.0, 50.0, 100.0)
    ]
    assert totals == sorted(totals)
    assert totals[0] < totals[2]


def test_the_canvas_past_an_inner_shadow_counts_as_outside() -> None:
    """A layer that fills its canvas still has an edge there to cast from.

    The blur reads past the canvas as uncovered, so the edge pixel takes
    about half the shadow and the centre none.
    """
    shape = np.ones((40, 40, 1), dtype=np.float32)
    shadow = effects.draw_inner_shadow(
        (0, 0, 40, 40), shape, distance=0.0, angle=0.0, size=9.0, choke=0.0
    )
    assert 0.4 < float(shadow[20, 0, 0]) < 0.6
    assert float(shadow[20, 20, 0]) == 0.0


def _glow(shape: np.ndarray, **values: Any) -> np.ndarray:
    params: dict[str, Any] = {"size": 8.0, "choke": 0.0, "glow_range": 100.0}
    params.update(values)
    return effects.draw_inner_glow(_BOX, shape, **params)


def test_an_edge_glow_is_full_at_the_edge_and_fades_inward() -> None:
    shape = _disc(81, 20.0)
    glow = _glow(shape)
    assert np.all(glow <= shape)
    row = glow[40, 40:61, 0]
    assert np.all(np.diff(row) >= 0.0), row
    assert float(row[0]) == 0.0
    assert float(row[20]) > 0.5


@pytest.mark.parametrize("precise", [False, True])
def test_a_center_glow_is_the_complement_of_the_edge_glow(precise: bool) -> None:
    shape = _disc(81, 20.0)
    edge = _glow(shape, precise=precise)
    center = _glow(shape, precise=precise, center=True)
    assert np.allclose(edge + center, shape, atol=1e-6)


def test_a_glows_range_spreads_its_fade_over_that_share() -> None:
    """Range is the share of the fade the contour is laid over.

    At 50% the matte reaches full strength halfway along the fade, which is
    twice the full-range matte held at one. Dividing by the range is the
    reading the card renders bear out, rather than one derived from Photoshop.
    """
    shape = _disc(81, 20.0)
    full = _glow(shape, glow_range=100.0)
    half = _glow(shape, glow_range=50.0)
    assert np.allclose(half, np.minimum(2.0 * full, 1.0) * shape, atol=1e-6)


def test_a_precise_glow_fades_linearly_with_the_distance_to_the_edge() -> None:
    """The precise technique ramps along the distance field, not a blur.

    A hard edge at column 10: the first column inside lies half a pixel from
    it, and the ramp falls by one size-th per pixel from there. Measured
    short of the far canvas edge, which counts as an edge of its own.
    """
    shape = np.zeros((20, 40, 1), dtype=np.float32)
    shape[:, 10:] = 1.0
    size = 8.0
    glow = effects.draw_inner_glow(
        (0, 0, 40, 20), shape, size=size, choke=0.0, precise=True, glow_range=100.0
    )
    depth = np.arange(20) + 0.5
    expected = np.clip((size - depth) / size, 0.0, 1.0)
    assert np.allclose(glow[10, 10:30, 0], expected, atol=1e-6)
    assert not glow[:, :10].any()


def test_the_canvas_past_a_precise_glow_counts_as_outside() -> None:
    """A layer that fills its canvas has an edge there under either technique.

    The first pixel inside a hard edge lies half a pixel from it, so the
    precise ramp starts there, and the centre is out of its reach.
    """
    shape = np.ones((40, 40, 1), dtype=np.float32)
    size = 9.0
    glow = effects.draw_inner_glow(
        (0, 0, 40, 40), shape, size=size, choke=0.0, precise=True, glow_range=100.0
    )
    assert float(glow[20, 0, 0]) == pytest.approx((size - 0.5) / size, abs=1e-6)
    assert float(glow[20, 20, 0]) == 0.0


def test_an_inverted_contour_turns_an_edge_glow_into_a_center_one() -> None:
    table = effects.contour_table(_contour([(0.0, 255.0), (255.0, 0.0)]))
    assert table is not None
    shape = _disc(81, 20.0)
    inverted = _glow(shape, contour=table, anti_aliased=True)
    assert np.allclose(inverted, _glow(shape, center=True), atol=1e-5)


@pytest.mark.parametrize(
    "draw",
    [
        lambda shape: _shadow(shape, distance=float("nan")),
        lambda shape: _shadow(shape, size=float("inf")),
        lambda shape: _glow(shape, choke=float("nan")),
        lambda shape: _glow(shape, glow_range=float("inf")),
    ],
    ids=["shadow-distance", "shadow-size", "glow-choke", "glow-range"],
)
def test_a_parameter_that_is_not_finite_raises(
    draw: Callable[[np.ndarray], np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        draw(_disc(81, 20.0))


@pytest.mark.parametrize(
    ("what", "message"),
    [
        (None, "Drop shadow"),
        ("Inner shadow", "Inner shadow"),
        ("Inner glow", "Inner glow"),
    ],
    ids=["default", "shadow", "glow"],
)
def test_shadow_geometry_names_the_effect_whose_value_is_not_finite(
    what: str | None, message: str
) -> None:
    desc = Descriptor()
    desc[Key.Blur] = Double(float("nan"))
    with pytest.raises(ValueError, match=message + " parameters must be finite"):
        if what is None:
            effects.shadow_geometry(desc)
        else:
            effects.shadow_geometry(desc, what)


@pytest.mark.parametrize(
    ("draw", "name"),
    [(_shadow, "Inner shadow"), (_glow, "Inner glow")],
    ids=["shadow", "glow"],
)
def test_an_inner_effect_asks_for_scipy_by_name_when_it_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    draw: Callable[[np.ndarray], np.ndarray],
    name: str,
) -> None:
    monkeypatch.setattr(effects, "HAS_SCIPY", False)
    with pytest.raises(ImportError, match="%s effects require: scipy" % name):
        draw(_disc(81, 20.0))


# The blur the inner effects take is the drop shadow's, at the same fitted
# width per pixel of size.
_FITTED_SIGMA_PER_SIZE = 0.45


def test_the_inner_effects_blur_at_the_drop_shadows_fitted_width() -> None:
    """The blur is shared with the drop shadow, so a change to it moves both.

    The glows on the card renders score best at the drop shadow's width,
    with the range dividing the matte; neither is retuned for them.
    """
    assert effects._SHADOW_SIGMA_PER_SIZE == _FITTED_SIGMA_PER_SIZE
    shape = _disc(81, 20.0)
    blurred = effects._cast_matte(
        1.0 - shape[:, :, 0],
        distance=0.0,
        angle=0.0,
        size=8.0,
        spread=0.0,
        outside=1.0,
        what="Inner glow",
    )
    assert np.array_equal(_glow(shape)[:, :, 0], blurred * shape[:, :, 0])


# -- Compositing ------------------------------------------------------------ #

# Where the blue square lies on the canvas.
_SQUARE = (8, 8, 24, 24)
_DONORS = {Klass.InnerShadow: "Inner Shadow", Klass.InnerGlow: "Inner Glow"}


def _square() -> tuple[PSDImage, Layer]:
    psd = PSDImage.open(full_name("effects/outside-stroke.psd"))
    return psd, psd[0]


def _forge(
    layer: Layer, klass: Klass, angle: float = 90.0, **values: float
) -> Descriptor:
    """Give ``layer`` ``layer_effects.psd``'s inner effect of one kind, edited.

    It is made a solid Normal effect at full opacity, lit from ``angle``
    rather than the global light; ``values`` maps :py:class:`Key` names to
    numbers to put in the descriptor. Returns the effect's descriptor.
    """
    donor = PSDImage.open(full_name("layer_effects.psd"))
    source = next(sub for sub in donor.descendants() if sub.name == _DONORS[klass])
    block = copy.deepcopy(
        source.tagged_blocks.get_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO)
    )
    layer.tagged_blocks.set_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO, block)
    desc = block[klass]
    desc[Key.Mode] = Enumerated(typeID=Type.BlendMode, enum=Enum.Normal)
    desc[Key.Opacity] = Double(100.0)
    desc[Key.UseGlobalAngle] = False
    desc[Key.LocalLightingAngle] = Double(angle)
    for name, value in values.items():
        desc[getattr(Key, name)] = Double(value)
    return desc


def _render(
    psd: PSDImage, viewport: tuple[int, int, int, int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    color, _, alpha = composite(psd, viewport=viewport)
    return color, alpha


def _shaded(color: np.ndarray) -> np.ndarray:
    """How far a black shadow took the square's blue down, per pixel."""
    left, top, right, bottom = _SQUARE
    return 1.0 - color[top:bottom, left:right, 2]


def test_an_inner_shadow_paints_inside_the_layer_only() -> None:
    """The layer's alpha is untouched, and so is every pixel it does not cover."""
    psd, layer = _square()
    desc = _forge(layer, Klass.InnerShadow, Distance=4.0, Blur=3.0)
    color, alpha = _render(psd)
    desc[Key.Enabled] = False
    plain_color, plain_alpha = _render(psd)
    assert np.array_equal(alpha, plain_alpha)
    outside = plain_alpha[:, :, 0] == 0.0
    assert np.array_equal(color[outside], plain_color[outside])
    assert not np.array_equal(color, plain_color)


@pytest.mark.parametrize(
    ("angle", "band"),
    [(90.0, (slice(0, 4), slice(None))), (0.0, (slice(None), slice(12, 16)))],
    ids=["overhead", "from-the-right"],
)
def test_an_unblurred_inner_shadow_shades_the_rim_facing_the_light(
    angle: float, band: tuple[slice, slice]
) -> None:
    """A hard shadow cast 4 px shades exactly the 4 px band on the light's side."""
    psd, layer = _square()
    _forge(layer, Klass.InnerShadow, angle=angle, Distance=4.0, Blur=0.0)
    shaded = _shaded(_render(psd)[0])
    expected = np.zeros_like(shaded)
    expected[band] = 1.0
    assert np.allclose(shaded, expected, atol=1e-6)


def test_an_inner_shadow_follows_the_global_light() -> None:
    """Under global light the angle is the document's, read when it is drawn."""
    psd, layer = _square()
    desc = _forge(layer, Klass.InnerShadow, angle=0.0, Distance=4.0, Blur=0.0)
    desc[Key.UseGlobalAngle] = True
    psd.image_resources[Resource.GLOBAL_ANGLE].data.value = 90
    overhead = _shaded(_render(psd)[0])
    assert np.allclose(overhead[:4], 1.0) and not overhead[4:].any()
    psd.image_resources[Resource.GLOBAL_ANGLE].data.value = 0
    from_the_right = _shaded(_render(psd)[0])
    assert np.allclose(from_the_right[:, 12:], 1.0)
    assert not from_the_right[:, :12].any()


def test_a_full_choke_glows_a_hard_ring_inside_the_edge() -> None:
    """A white glow at full choke whitens the 3 px inside the edge, and no more."""
    psd, layer = _square()
    _forge(layer, Klass.InnerGlow, ChokeMatte=100.0, Blur=3.0)
    color, _ = _render(psd)
    left, top, right, bottom = _SQUARE
    middle = (top + bottom) // 2
    row = color[middle, left:right]
    assert np.allclose(row[:3], 1.0) and np.allclose(row[-3:], 1.0)
    assert np.allclose(row[3:-3], (0.0, 0.0, 1.0))


@pytest.mark.parametrize(
    ("klass", "default"),
    [(Klass.InnerShadow, Enum.Multiply), (Klass.InnerGlow, Enum.Screen)],
    ids=["shadow", "glow"],
)
def test_an_inner_effect_with_no_blend_mode_takes_photoshops_default(
    klass: Klass, default: Enum
) -> None:
    psd, layer = _square()
    desc = _forge(layer, klass, Distance=4.0, Blur=4.0)
    desc[Key.Mode] = Enumerated(typeID=Type.BlendMode, enum=default)
    explicit, _ = _render(psd)
    desc.pop(Key.Mode)
    assert np.array_equal(_render(psd)[0], explicit)


def test_an_inner_glow_lies_over_a_color_overlay() -> None:
    """Photoshop stacks the inner glow above the overlays.

    A red overlay over the whole square and a white glow at full choke: the
    ring is the glow's white, the middle the overlay's red.
    """
    psd, layer = _square()
    _forge(layer, Klass.InnerGlow, ChokeMatte=100.0, Blur=3.0)
    donor = PSDImage.open(full_name("layer_effects.psd"))
    fill = next(sub for sub in donor.descendants() if sub.name == "Color Fill")
    overlay = copy.deepcopy(
        fill.tagged_blocks.get_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO)[
            Klass.SolidFill
        ]
    )
    overlay[Key.Mode] = Enumerated(typeID=Type.BlendMode, enum=Enum.Normal)
    overlay[Key.Opacity] = Double(100.0)
    red = overlay[Key.Color]
    for key, value in ((b"Rd  ", 255.0), (b"Grn ", 0.0), (b"Bl  ", 0.0)):
        red[key] = Double(value)
    block = layer.tagged_blocks.get_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO)
    block[Klass.SolidFill] = overlay
    color, _ = _render(psd)
    middle = (_SQUARE[1] + _SQUARE[3]) // 2
    assert np.allclose(color[middle, _SQUARE[0]], 1.0)
    assert np.allclose(color[middle, middle], (1.0, 0.0, 0.0))


def test_an_inner_effect_does_not_depend_on_the_viewport() -> None:
    """A viewport that cuts the layer does not lend it an edge (#804).

    The silhouette is read on the effect's own box, so the cut is not
    mistaken for the layer's boundary and glowed from.
    """
    psd, layer = _square()
    _forge(layer, Klass.InnerGlow, Blur=6.0)
    whole, _ = _render(psd)
    half, _ = _render(psd, (0, 0, 16, 32))
    assert np.allclose(half, whole[:, :16], atol=1e-6)


def _setting(key: bytes, value: Any) -> Callable[[Descriptor], Any]:
    return lambda desc: desc.__setitem__(key, value)


# Ways an inner effect's descriptor can defeat the reads that draw it.
_UNDRAWABLE: dict[str, Callable[[Descriptor], Any]] = {
    "size": _setting(Key.Blur, "wide"),
    "infinite-size": _setting(Key.Blur, Double(float("inf"))),
    "nan-distance": _setting(Key.Distance, Double(float("nan"))),
    "nan-choke": _setting(Key.ChokeMatte, Double(float("nan"))),
    "nan-opacity": _setting(Key.Opacity, Double(float("nan"))),
    "no-color": lambda desc: desc.pop(Key.Color),
}


def _gradient_glow(desc: Descriptor) -> None:
    """A glow painted with a gradient rather than a colour, which is not drawn."""
    desc.pop(Key.Color)
    desc[Key.Gradient] = Descriptor()


_UNDRAWABLE_GLOW: dict[str, Callable[[Descriptor], Any]] = {
    "nan-range": _setting(Key.InputRange, Double(float("nan"))),
    "gradient": _gradient_glow,
}
_DEFECTS = [(Klass.InnerShadow, defect) for defect in sorted(_UNDRAWABLE)] + [
    (Klass.InnerGlow, defect) for defect in sorted({**_UNDRAWABLE, **_UNDRAWABLE_GLOW})
]


@pytest.mark.parametrize(("klass", "defect"), _DEFECTS)
def test_an_unreadable_inner_effect_is_dropped_and_the_document_renders(
    klass: Klass, defect: str
) -> None:
    psd, layer = _square()
    desc = _forge(layer, klass, Distance=4.0, Blur=4.0)
    drawn, _ = _render(psd)
    desc[Key.Enabled] = False
    disabled, _ = _render(psd)
    assert not np.array_equal(drawn, disabled)
    desc[Key.Enabled] = True
    {**_UNDRAWABLE, **_UNDRAWABLE_GLOW}[defect](desc)
    rendered, _ = _render(psd)
    assert np.isfinite(rendered).all()
    assert np.array_equal(rendered, disabled)


@pytest.mark.parametrize("klass", [Klass.InnerShadow, Klass.InnerGlow])
def test_an_inner_effect_over_the_documents_budget_is_dropped_not_drawn(
    klass: Klass,
) -> None:
    """An effect whose canvas the budget refuses costs the effect only.

    The budget is set just under what the effect's canvas costs and above
    what the layer's own pixels do, so the layer renders as it does with
    the effect switched off rather than raising.
    """
    psd, layer = _square()
    desc = _forge(layer, klass, Distance=4.0, Blur=4.0)
    desc[Key.Enabled] = False
    disabled, _ = _render(psd)
    desc[Key.Enabled] = True
    assert not np.array_equal(_render(psd)[0], disabled)

    _, size, _ = effects.shadow_geometry(desc)
    grown = effects.drop_shadow_bbox(layer.bbox, 0.0, 0.0, size)
    width, height = grown[2] - grown[0], grown[3] - grown[1]
    psd.max_alloc_bytes = width * height * effects._SHADOW_BYTES - 1
    assert np.array_equal(_render(psd)[0], disabled)


# -- Fidelity against Photoshop's own render ------------------------------- #


def _preview_error(
    name: str, kind: str, enabled: bool, channels: slice = slice(None)
) -> float:
    psd = PSDImage.open(full_name("layer_effects.psd"))
    layer = next(sub for sub in psd.descendants() if sub.name == name)
    if not enabled:
        for effect in list(layer.effects.find(kind)):
            effect.descriptor[Key.Enabled] = False
    reference = psd.numpy()
    color, _, _ = composite(psd)
    left, top, right, bottom = layer.bbox
    region = (slice(top - 4, bottom + 4), slice(left - 4, right + 4), channels)
    return float(_mse(reference[region], color[region]))


def test_an_inner_shadow_renders_to_photoshops_preview() -> None:
    """``layer_effects.psd``'s text with a black Multiply inner shadow."""
    assert _preview_error("Inner Shadow", "innershadow", False) >= 5e-3
    assert _preview_error("Inner Shadow", "innershadow", True) <= 1e-5


def test_an_inner_glow_renders_to_photoshops_preview() -> None:
    """``layer_effects.psd``'s text with a white Screen inner glow.

    Measured on red and green alone: the layer's blending options exclude
    blue, which this module does not apply.
    """
    red_green = slice(0, 2)
    assert _preview_error("Inner Glow", "innerglow", False, red_green) >= 5e-2
    assert _preview_error("Inner Glow", "innerglow", True, red_green) <= 3e-4
