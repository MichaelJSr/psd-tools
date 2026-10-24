"""The space a type layer's anti-aliased edge blends in.

Three fixtures carry a type layer Photoshop composited under its "Blend Text
Colors Using Gamma" preference into their preview -- ``text.psb``,
``masks2.psb`` and ``artboard.psd`` -- and ``layer_effects.psd`` adds one
with each effect. Four carry type layers Photoshop composited straight, as
stored pixels: ``masks.psb`` (fonts substituted), ``text.psd`` (text data
without ``bounds``) and the 32-bit ``pen-text.psd`` and ``pen-text.psb``.
``text-kept-raster.psd`` carries one of each with nothing in the file to tell
them apart. The cases the corpus does not pair with a type layer (a styled
group around one, a blend mode) are forged.
"""

import copy
import importlib
import io
import warnings
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from PIL import Image

from psd_tools.api.layers import Group, Layer, PixelLayer, TypeLayer
from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import composite
from psd_tools.composite.blend import get_blend_func
from psd_tools.composite.composite import Compositor, _blends_in_text_space
from psd_tools.composite.text_blend import (
    TEXT_BLEND_GAMMA,
    from_text_space,
    to_text_space,
)
from psd_tools.composite.widen import make_widen
from psd_tools.constants import BlendMode, ColorMode, Compression, Tag
from psd_tools.psd.descriptor import Double, Enumerated
from psd_tools.terminology import Enum, Key

from ..utils import full_name

# Photoshop's default for the preference, which is what its previews fit.
_PHOTOSHOP_DEFAULT_GAMMA = 1.45


def _render(psd: PSDImage, text_gamma: float | None) -> np.ndarray:
    """The document composited from its layers, text blended in ``text_gamma``."""
    left, top, right, bottom = psd.viewbox
    height, width = bottom - top, right - left
    compositor = Compositor(
        psd.viewbox,
        np.ones((height, width, 3), dtype=np.float32),
        np.zeros((height, width, 1), dtype=np.float32),
        force=True,
        widen=make_widen(psd),
        color_mode=psd.color_mode,
        text_gamma=text_gamma,
    )
    for layer in psd:
        compositor.apply(layer)
    return compositor.finish()[0]


def _type_layers(psd: PSDImage) -> Iterator[TypeLayer]:
    for layer in psd.descendants():
        if isinstance(layer, TypeLayer) and layer.is_visible():
            yield layer


def _edges(psd: PSDImage, layer: Layer) -> tuple[slice, slice, np.ndarray]:
    """The layer's box on the canvas and, within it, where a glyph edge is."""
    left, top, right, bottom = layer.bbox
    shape = layer.numpy("shape")
    assert shape is not None
    coverage = shape[:, :, 0]
    # Clipped to the canvas, as the artboard's text runs past it.
    coverage = coverage[: psd.height - top, : psd.width - left]
    rows, columns = (
        slice(top, min(bottom, psd.height)),
        slice(left, min(right, psd.width)),
    )
    return rows, columns, (coverage > 0.05) & (coverage < 0.9)


def _edge_error(psd: PSDImage, rendered: np.ndarray, layer: Layer) -> float:
    """Mean absolute error on the layer's glyph edges, in 8-bit steps."""
    rows, columns, edges = _edges(psd, layer)
    reference = psd.numpy()[rows, columns, :3]
    return float(np.abs(rendered[rows, columns] - reference)[edges].mean() * 255)


# -- The space -------------------------------------------------------------- #


def test_the_text_space_round_trips() -> None:
    ramp = np.linspace(0.0, 1.0, 1001)
    assert from_text_space(to_text_space(ramp)) == pytest.approx(ramp, abs=1e-9)
    assert to_text_space(np.array([0.0, 1.0])) == pytest.approx([0.0, 1.0])
    assert np.all(np.diff(to_text_space(ramp)) > 0)


def test_the_gamma_is_photoshops_default() -> None:
    """The constant is a fit, so a change to it is a measured change."""
    assert TEXT_BLEND_GAMMA == _PHOTOSHOP_DEFAULT_GAMMA


# -- Against Photoshop's previews ------------------------------------------- #

# A glyph edge differs from the 8-bit preview by rounding alone.
_EDGE_BOUND = 1.0


@pytest.mark.parametrize("filename", ["text.psb", "masks2.psb", "artboard.psd"])
def test_a_type_layer_blends_to_photoshops_preview(filename: str) -> None:
    psd = PSDImage.open(full_name(filename))
    layer = next(_type_layers(psd))
    assert _edge_error(psd, _render(psd, TEXT_BLEND_GAMMA), layer) <= _EDGE_BOUND
    assert _edge_error(psd, _render(psd, None), layer) > 20.0


def test_the_fitted_gamma_is_the_best_of_its_neighbours() -> None:
    psd = PSDImage.open(full_name("text.psb"))
    layer = next(_type_layers(psd))
    candidates = [1.0, 1.3, 1.4, TEXT_BLEND_GAMMA, 1.5, 1.6, 1.8]
    errors = {g: _edge_error(psd, _render(psd, g), layer) for g in candidates}
    assert min(errors, key=errors.__getitem__) == TEXT_BLEND_GAMMA


def test_pixels_the_text_leaves_alone_are_untouched() -> None:
    """The space is a round trip elsewhere, and must return every bit."""
    psd = PSDImage.open(full_name("text.psb"))
    layer = next(_type_layers(psd))
    left, top, right, bottom = layer.bbox
    outside = np.ones(psd.viewbox[3::-2][::-1], dtype=bool)[: psd.height, : psd.width]
    outside[top:bottom, left:right] = False
    straight = _render(psd, None)
    blended = _render(psd, TEXT_BLEND_GAMMA)
    assert np.array_equal(straight[outside], blended[outside])
    assert not np.array_equal(
        straight[top:bottom, left:right], blended[top:bottom, left:right]
    )


# -- Which layers Photoshop blends in the gamma ----------------------------- #


def _edge_errors(psd: PSDImage, text_gamma: float | None) -> list[float]:
    """The edge error of every visible type layer, composited through the API."""
    color, _, _ = composite(psd, force=True, text_gamma=text_gamma)
    return [_edge_error(psd, color, layer) for layer in _type_layers(psd)]


def test_a_substituted_font_keeps_its_stored_raster() -> None:
    """A flagged layer mixes straight; the same layer unflagged mixes in the gamma.

    ``masks.psb`` carries ``substitutesUsed`` on every type layer, its twin
    ``masks.psd`` on none.
    """
    substituted = PSDImage.open(full_name("masks.psb"))
    assert all(
        layer._data.text_data.get(b"substitutesUsed")
        for layer in _type_layers(substituted)
    )
    default, _, _ = composite(substituted, force=True)
    straight, _, _ = composite(substituted, force=True, text_gamma=None)
    assert np.array_equal(default, straight)
    assert max(_edge_errors(substituted, TEXT_BLEND_GAMMA)) <= _EDGE_BOUND

    rendered = PSDImage.open(full_name("masks.psd"))
    assert not any(
        layer._data.text_data.get(b"substitutesUsed")
        for layer in _type_layers(rendered)
    )
    blended = _edge_errors(rendered, TEXT_BLEND_GAMMA)
    mixed = _edge_errors(rendered, None)
    assert len(blended) == 6
    assert all(b < m - 2.0 for b, m in zip(blended, mixed, strict=True))


def test_text_data_without_bounds_keeps_its_stored_raster() -> None:
    """``text.psd`` lacks the ``bounds`` key its ``text.psb`` twin carries."""
    stale = PSDImage.open(full_name("text.psd"))
    assert b"bounds" not in next(_type_layers(stale))._data.text_data
    assert _edge_errors(stale, TEXT_BLEND_GAMMA) == _edge_errors(stale, None)
    assert max(_edge_errors(stale, TEXT_BLEND_GAMMA)) <= _EDGE_BOUND

    current = PSDImage.open(full_name("text.psb"))
    assert b"bounds" in next(_type_layers(current))._data.text_data
    assert max(_edge_errors(current, TEXT_BLEND_GAMMA)) <= _EDGE_BOUND
    assert min(_edge_errors(current, None)) > 20.0


@pytest.mark.parametrize("filename", ["pen-text.psd", "pen-text.psb"])
def test_a_32_bit_document_mixes_straight(filename: str) -> None:
    """Linear channels are mixed as they are, ``bounds`` or not."""
    psd = PSDImage.open(full_name(filename))
    assert psd.depth == 32
    assert _edge_errors(psd, TEXT_BLEND_GAMMA) == _edge_errors(psd, None)
    assert max(_edge_errors(psd, TEXT_BLEND_GAMMA)) <= _EDGE_BOUND


def test_only_an_rgb_document_blends_in_the_gamma() -> None:
    """The space is built on the sRGB curve, which no other mode's channels follow."""
    psd = PSDImage.open(full_name("text.psb"))
    layer = next(_type_layers(psd))
    assert _blends_in_text_space(layer, ColorMode.RGB)
    assert _blends_in_text_space(layer, None)
    assert not _blends_in_text_space(layer, ColorMode.CMYK)
    assert not _blends_in_text_space(layer, ColorMode.GRAYSCALE)
    pixels = next(
        layer
        for layer in PSDImage.open(full_name("masks.psd")).descendants()
        if not isinstance(layer, TypeLayer)
    )
    assert not _blends_in_text_space(pixels, ColorMode.RGB)


def test_text_gamma_reaches_every_composite_method() -> None:
    """``None`` on the document, a layer and an artboard restores the straight mix."""
    psd = PSDImage.open(full_name("text.psb"))
    assert not np.array_equal(
        np.asarray(psd.composite(force=True)),
        np.asarray(psd.composite(force=True, text_gamma=None)),
    )
    # A layer composites over a transparent backdrop, so the difference needs
    # something of the layer's own under the glyph edge: its stroke band.
    _, layer = _layer_effects("Stroke")
    default = layer.composite(force=True)
    straight = layer.composite(force=True, text_gamma=None)
    assert default is not None and straight is not None
    assert not np.array_equal(np.asarray(default), np.asarray(straight))

    artboard = PSDImage.open(full_name("artboard.psd"))[0]
    default = artboard.composite(force=True)
    straight = artboard.composite(force=True, text_gamma=None)
    assert default is not None and straight is not None
    assert not np.array_equal(np.asarray(default), np.asarray(straight))


# -- Effects ---------------------------------------------------------------- #


def _layer_effects(name: str) -> tuple[PSDImage, TypeLayer]:
    psd = PSDImage.open(full_name("layer_effects.psd"))
    layer = next(layer for layer in _type_layers(psd) if layer.name == name)
    return psd, layer


@pytest.mark.parametrize("name", ["Stroke", "Color Fill", "Gradient Overlay"])
def test_the_layers_own_effects_lie_beneath_the_blend(name: str) -> None:
    """The text's edge mixes against its own stroke band, overlaid fill and all.

    The band goes on first and the fill blends over it in the text space; an
    overlay recolours the fill before that blend rather than after it.
    """
    psd, layer = _layer_effects(name)
    assert _edge_error(psd, _render(psd, TEXT_BLEND_GAMMA), layer) <= _EDGE_BOUND
    assert _edge_error(psd, _render(psd, None), layer) > 3.0


def _style_from(donor: str) -> Tag:
    """The effects block of ``layer_effects.psd``'s layer called ``donor``."""
    _, layer = _layer_effects(donor)
    return copy.deepcopy(
        layer.tagged_blocks.get_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO)
    )


def test_a_styled_group_blends_its_text_straight() -> None:
    """A styled group's text never meets the backdrop; a plain one's does.

    Photoshop renders a group that carries a layer style as one layer and
    puts the style on that; a pass-through group without one hands its text
    through to the backdrop.
    """
    psd = PSDImage.open(full_name("text.psb"))
    layer = next(_type_layers(psd))
    rows, columns, edges = _edges(psd, layer)
    group = Group.new(psd, "forged")
    group.blend_mode = BlendMode.PASS_THROUGH
    layer.move_to_group(group)

    through = _render(psd, TEXT_BLEND_GAMMA)[rows, columns][edges]
    straight = _render(psd, None)[rows, columns][edges]
    assert not np.allclose(through, straight, atol=4 / 255)

    group.tagged_blocks.set_data(
        Tag.OBJECT_BASED_EFFECTS_LAYER_INFO, _style_from("Drop Shadow")
    )
    styled = _render(psd, TEXT_BLEND_GAMMA)[rows, columns][edges]
    assert styled == pytest.approx(_render(psd, None)[rows, columns][edges], abs=1e-6)


def test_text_in_a_styled_group_still_meets_its_own_style() -> None:
    """The layer's own stroke is painted into the group before its fill is.

    Where the band is opaque the fill meets the same thing with the group's
    style as without it, so the edge comes out the same; without the gamma
    it does not.
    """
    psd = PSDImage.open(full_name("text.psb"))
    layer = next(_type_layers(psd))
    rows, columns, edges = _edges(psd, layer)
    layer.tagged_blocks.set_data(
        Tag.OBJECT_BASED_EFFECTS_LAYER_INFO, _style_from("Stroke")
    )
    alone = _render(psd, TEXT_BLEND_GAMMA)[rows, columns][edges]
    straight = _render(psd, None)[rows, columns][edges]

    group = Group.new(psd, "forged")
    group.blend_mode = BlendMode.PASS_THROUGH
    layer.move_to_group(group)
    group.tagged_blocks.set_data(
        Tag.OBJECT_BASED_EFFECTS_LAYER_INFO, _style_from("Drop Shadow")
    )
    grouped = _render(psd, TEXT_BLEND_GAMMA)[rows, columns][edges]
    assert grouped == pytest.approx(alone, abs=1 / 255)
    assert np.abs(grouped - straight).mean() > 4 / 255


def _rings(layer: Layer, *rings: tuple[float, float, float]) -> None:
    """Give the layer outset stroke rings of (size, grey, opacity), first on top.

    The list comes from ``double-stroke-effects.psd``'s rectangle, the one
    fixture whose effects block carries a stroke list.
    """
    donor = PSDImage.open(full_name("effects/double-stroke-effects.psd"))
    rectangle = next(sub for sub in donor.descendants() if sub.name == "Rectangle 1")
    data = copy.deepcopy(
        rectangle.tagged_blocks.get_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO_V0)
    )
    strokes = data[b"frameFXMulti"]
    while len(strokes) < len(rings):
        strokes.append(copy.deepcopy(strokes[0]))
    for desc in strokes[len(rings) :]:
        desc[b"enab"] = False
    for desc, (size, grey, opacity) in zip(strokes[: len(rings)], rings, strict=True):
        desc[b"enab"] = True
        desc[b"present"] = True
        desc[Key.Style] = Enumerated(typeID=b"FStl", enum=Enum.OutsetFrame)
        desc[Key.SizeKey] = Double(size)
        desc[b"Opct"] = Double(opacity)
        for channel in (Key.Red, Key.Green, Key.Blue):
            desc[Key.Color][channel] = Double(255.0 * grey)
    layer.tagged_blocks.set_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO_V0, data)


def test_a_faint_stroke_over_a_wide_one_leaves_the_text_on_the_backdrop() -> None:
    """A 1 px ring at 0 % listed over a 6 px ring cuts the wide ring back.

    The glyph edge then meets the backdrop, in the gamma, as it does with no
    style at all, where the wide ring alone would put its grey there. An
    opaque ring on top reaches the edge as the wide ring alone does.
    """
    psd = PSDImage.open(full_name("text.psb"))
    layer = next(_type_layers(psd))
    assert not layer.has_effects()
    rows, columns, edges = _edges(psd, layer)
    plain = _render(psd, TEXT_BLEND_GAMMA)[rows, columns][edges]

    wide = (6.0, 0.5, 100.0)
    _rings(layer, wide)
    ringed = _render(psd, TEXT_BLEND_GAMMA)[rows, columns][edges]
    assert np.abs(ringed - plain).max() > 0.1

    _rings(layer, (1.0, 0.0, 0.0), wide)
    cut = _render(psd, TEXT_BLEND_GAMMA)[rows, columns][edges]
    assert cut == pytest.approx(plain, abs=1 / 255)

    _rings(layer, (1.0, 0.0, 100.0), (6.0, 1.0, 100.0))
    ringed = _render(psd, TEXT_BLEND_GAMMA)[rows, columns][edges]
    assert np.abs(ringed - plain).mean() > 4 / 255


# -- The mix ---------------------------------------------------------------- #


def _mixed_pixel(
    channels: int,
    color_mode: ColorMode,
    backdrop: float,
    source: float,
    coverage: float,
    blend_mode: BlendMode,
) -> np.ndarray:
    """One pixel of ``backdrop`` after a type layer covers ``coverage`` of it.

    ``source`` is the layer's value on every one of the ``channels``.
    """
    compositor = Compositor(
        (0, 0, 1, 1),
        np.full((1, 1, channels), backdrop, dtype=np.float32),
        np.ones((1, 1, 1), dtype=np.float32),
        color_mode=color_mode,
    )
    shape = np.full((1, 1, 1), coverage, dtype=np.float32)
    compositor._apply_source(
        np.full((1, 1, channels), source, dtype=np.float32),
        shape,
        shape,
        blend_mode,
        text_gamma=TEXT_BLEND_GAMMA,
    )
    return compositor.result_over_backdrop()[0, 0]


def _expected(
    backdrop: float, source: float, coverage: float, blend_mode: BlendMode
) -> float:
    blended = get_blend_func(blend_mode, ColorMode.RGB)(
        np.array([backdrop]), np.array([source])
    )
    mixed = coverage * to_text_space(blended) + (1 - coverage) * to_text_space(
        np.array([backdrop])
    )
    return float(from_text_space(mixed)[0])


@pytest.mark.parametrize("blend_mode", [BlendMode.NORMAL, BlendMode.MULTIPLY])
def test_the_mix_is_by_coverage_in_the_text_space(blend_mode: BlendMode) -> None:
    """The blend mode sees the document's values; the mix by coverage does not."""
    pixel = _mixed_pixel(3, ColorMode.RGB, 0.8, 0.3, 0.5, blend_mode)
    expected = _expected(0.8, 0.3, 0.5, blend_mode)
    assert pixel == pytest.approx(np.full(3, expected), abs=1e-6)
    straight = 0.5 * 0.3 * (1 if blend_mode == BlendMode.NORMAL else 0.8) + 0.5 * 0.8
    assert abs(expected - straight) > 0.02


def _covered_pixel(
    backdrop: float, source: float, coverage: float, share: float
) -> float:
    """One grey pixel after ``share`` of a type layer's ``coverage`` paints it.

    The layer shows ``source`` mixed straight over the backdrop by ``share``
    within its coverage, as an interior effect over a layer at fill 0 does.
    """
    compositor = Compositor(
        (0, 0, 1, 1),
        np.full((1, 1, 3), backdrop, dtype=np.float32),
        np.ones((1, 1, 1), dtype=np.float32),
        color_mode=ColorMode.RGB,
    )
    shape = np.full((1, 1, 1), coverage, dtype=np.float32)
    inside = np.full((1, 1, 3), backdrop + share * (source - backdrop), np.float32)
    compositor._apply_source(
        np.full((1, 1, 3), source, dtype=np.float32),
        shape,
        shape * share,
        BlendMode.NORMAL,
        text_gamma=TEXT_BLEND_GAMMA,
        text_inside=(shape, inside),
    )
    return float(compositor.result_over_backdrop()[0, 0, 0])


def test_only_the_coverage_mixes_in_the_text_space() -> None:
    """What the layer shows within its coverage mixes by that coverage alone."""
    straight = 0.3 + 0.4 * (0.9 - 0.3)
    expected = from_text_space(
        0.5 * to_text_space(np.array([0.3])) + 0.5 * to_text_space(np.array([straight]))
    )[0]
    assert _covered_pixel(0.3, 0.9, 0.5, 0.4) == pytest.approx(expected, abs=1e-6)
    assert _covered_pixel(0.3, 0.9, 1.0, 0.4) == pytest.approx(straight, abs=1e-6)
    whole = _expected(0.3, 0.9, 0.5, BlendMode.NORMAL)
    assert _covered_pixel(0.3, 0.9, 0.5, 1.0) == pytest.approx(whole, abs=1e-6)


# -- A layer below full fill: ``effects/text-fill-window.psd`` -------------- #

# Photoshop 2023, one row 110 px tall per layer, each a type layer across a
# brown and a light backdrop: at opacity 50 and at fill 50, then at fill 0
# under the cards' Name glow (inner, Normal 90%, size 9, range 38%), the
# same at opacity 50, a colour overlay at 50%, fill 50 under the glow, the
# glow over a knocked-out drop shadow, the shadow alone, a rasterized copy of
# the glow and shadow, and fill 50 under both.
_FILL_WINDOW = "effects/text-fill-window.psd"
_ROW_HEIGHT = 110
_SHADOWED = (
    "fill 0 inner glow shadow",
    "fill 0 shadow",
    "pixels fill 0 inner glow shadow",
    "fill 50 inner glow shadow",
)


def _row_errors(psd: PSDImage) -> dict[str, float]:
    """The worst level of each row's composite against Photoshop's."""
    reference = psd.numpy()[..., :3]
    error = np.abs(composite(psd)[0] - reference).max(axis=-1)
    rows = [layer for layer in psd if layer.name not in ("Background", "backdrop")]
    assert len(rows) == 10
    return {
        layer.name: float(
            error[index * _ROW_HEIGHT : (index + 1) * _ROW_HEIGHT].max() * 255
        )
        for index, layer in enumerate(rows)
    }


def test_a_faded_layers_interior_mixes_straight_inside_its_window() -> None:
    """Every row is within 2 levels of Photoshop's composite.

    The opacity and fill rows fade the paint in the text gamma; an interior
    effect at fill below 100% goes on straight over the backdrop the
    layer's shape uncovers, and only that shape and the opacity mix in the
    gamma, over its shadow where it knocks one out.
    """
    errors = _row_errors(PSDImage.open(full_name(_FILL_WINDOW)))
    assert {name: error for name, error in errors.items() if error >= 2.0} == {}


def test_a_shadow_cut_under_the_shape_misses_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cut under the shape with the layer put on over the cut, each shadow row misses."""
    monkeypatch.setattr(Compositor, "_knocks_out_window", lambda self, layer: False)
    errors = _row_errors(PSDImage.open(full_name(_FILL_WINDOW)))
    assert all(errors[name] > 4.0 for name in _SHADOWED), errors
    assert all(
        error < 2.0 for name, error in errors.items() if name not in _SHADOWED
    ), errors


# -- Layers Photoshop kept as pixels ---------------------------------------- #

# ``text-kept-raster.psd``: a card's name and Ability, each over its own opaque
# stroke, with the card's preview wherever either has a glyph edge. Photoshop
# composited the name straight: the session that saved it re-laid the name
# out without redrawing its raster. The Ability, redrawn, it blended in the
# gamma. Their records differ in nothing that says so.
_KEPT_RASTER = "text-kept-raster.psd"


def _kept_raster() -> tuple[PSDImage, TypeLayer, TypeLayer]:
    psd = PSDImage.open(full_name(_KEPT_RASTER))
    layers = {layer.name: layer for layer in _type_layers(psd)}
    return psd, layers["Suffocation"], layers["Ability"]


def _composite_errors(psd: PSDImage, **kwargs: object) -> list[float]:
    """Edge errors of the name and the Ability, composited through the API."""
    color, _, _ = composite(psd, force=True, **kwargs)  # type: ignore[arg-type]
    _, name, ability = _kept_raster()
    return [_edge_error(psd, color, layer) for layer in (name, ability)]


def test_a_kept_raster_is_read_off_the_preview() -> None:
    psd, _, _ = _kept_raster()
    name, ability = _composite_errors(psd)
    assert name <= _EDGE_BOUND and ability <= _EDGE_BOUND
    name, ability = _composite_errors(psd, text_preview=False)
    assert name > 5.0
    assert ability <= _EDGE_BOUND


def test_a_layer_can_be_pinned_by_id() -> None:
    """A pin overrides ``text_gamma`` and the preview alike."""
    psd, name, ability = _kept_raster()
    straight = {name.layer_id: None}
    assert (
        max(_composite_errors(psd, text_preview=False, layer_text_gamma=straight))
        <= _EDGE_BOUND
    )
    pinned = _composite_errors(psd, layer_text_gamma={name.layer_id: TEXT_BLEND_GAMMA})
    assert pinned[0] > 5.0
    spoiled = _composite_errors(psd, layer_text_gamma={ability.layer_id: None})
    assert spoiled[0] <= _EDGE_BOUND and spoiled[1] > 5.0
    assert _composite_errors(
        psd, text_gamma=None, layer_text_gamma={ability.layer_id: TEXT_BLEND_GAMMA}
    ) == pytest.approx(_composite_errors(psd))


def _with_preview(psd: PSDImage, preview: np.ndarray) -> PSDImage:
    """The document saved and reopened with ``preview`` as its stored composite."""
    buffer = io.BytesIO()
    image = Image.fromarray(np.round(preview * 255).astype(np.uint8), "RGB")
    psd.save(buffer, preview=image)
    buffer.seek(0)
    return PSDImage.open(buffer)


def test_a_preview_that_shows_neither_mix_keeps_the_gamma() -> None:
    """Straight must fit the preview, not merely miss it by less."""
    psd, _, _ = _kept_raster()
    gamma, _, _ = composite(psd, force=True, text_preview=False)
    darker = _with_preview(psd, np.clip(gamma[..., :3] - 24 / 255, 0.0, 1.0))
    expected, _, _ = composite(darker, force=True, text_preview=False)
    color, _, _ = composite(darker, force=True)
    assert np.array_equal(color, expected)


def test_a_viewport_reads_the_preview_where_it_stands() -> None:
    """A region rendered alone, as an edit is, finds the same choice."""
    psd, name, _ = _kept_raster()
    left, top, right, bottom = name.bbox
    region = (left - 8, top - 8, right + 8, bottom + 8)
    whole = np.asarray(psd.composite(force=True, apply_icc=False))
    alone = np.asarray(psd.composite(viewport=region, force=True, apply_icc=False))
    assert np.array_equal(alone, whole[top - 8 : bottom + 8, left - 8 : right + 8])
    default = psd.composite(
        viewport=region, force=True, apply_icc=False, text_preview=False
    )
    assert not np.array_equal(alone, np.asarray(default))


def test_a_layers_composite_reads_no_preview() -> None:
    """The preview is the document's; a layer composited alone is not it."""
    _psd, name, _ = _kept_raster()
    alone = name.composite(force=True)
    assert alone is not None
    color, _, _ = composite(name, force=True, text_preview=False)
    expected, _, _ = composite(name, force=True)
    assert np.array_equal(color, expected)


def test_an_edge_painted_over_is_settled_on_the_finished_canvas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A veil over the name changes its edge after it goes on.

    Where the name goes on, the edge only leans straight; the composite with
    it straight is what the preview shows once the veil is on too.
    """
    psd, name, _ = _kept_raster()
    left, top, right, bottom = name.bbox
    veil = Image.new("RGBA", (right - left, bottom - top), (128, 128, 128, 96))
    psd.append(PixelLayer.frompil(veil, psd, "veil", top, left))
    straight, _, _ = composite(
        psd, force=True, text_preview=False, layer_text_gamma={name.layer_id: None}
    )
    veiled = _with_preview(psd, straight[..., :3])

    compositing = importlib.import_module("psd_tools.composite.composite")
    settled: list[bool] = []
    settles = compositing._settles_straight

    def counting(*args: Any) -> bool:
        settled.append(settles(*args))
        return settled[-1]

    monkeypatch.setattr(compositing, "_settles_straight", counting)
    color, _, _ = composite(veiled, force=True)
    expected, _, _ = composite(
        veiled, force=True, text_preview=False, layer_text_gamma={name.layer_id: None}
    )
    assert settled == [True]
    assert np.array_equal(color, expected)


def _settles(seen: list[int], gamma: int = 120, straight: int = 100) -> bool:
    """``_settles_straight`` on a row of edge pixels the preview shows as ``seen``.

    The finished canvas holds ``gamma`` with the layer in the gamma and
    ``straight`` with it straight, in 8-bit steps.
    """
    count = len(seen)
    compositing = importlib.import_module("psd_tools.composite.composite")
    reference = np.repeat(np.array(seen, dtype=np.uint8)[None, :, None], 3, axis=2)
    return compositing._settles_straight(
        np.full((1, count, 3), gamma / 255, dtype=np.float32),
        np.full((1, count, 3), straight / 255, dtype=np.float32),
        np.ones((1, count, 1), dtype=np.float32),
        (0, 0, count, 1),
        reference,
        (np.zeros(count, dtype=int), np.arange(count)),
    )


def test_straight_takes_a_clear_majority_a_close_fit_and_enough_pixels() -> None:
    assert _settles([100] * 20)
    assert not _settles([100] * 14 + [120] * 6)
    assert not _settles([103] * 20)
    assert not _settles([100] * 10)
    # Pixels where the two mixes differ by rounding alone say nothing.
    assert not _settles([100] * 20, gamma=102)


def test_only_the_layers_that_settle_go_on_straight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both layers lean straight where they go on; only the name settles.

    The leaning edges are tried over their own box, not the document.
    """
    psd, name, ability = _kept_raster()
    compositing = importlib.import_module("psd_tools.composite.composite")
    evidence = compositing._edge_evidence

    def leaning(*args: Any) -> Any:
        found = evidence(*args)
        return None if found is None else (1.0, 5.0 / 255.0, *found[2:])

    tried: list[tuple[int, ...]] = []

    def name_only(first: np.ndarray, second: np.ndarray, *args: Any) -> bool:
        tried.append(first.shape[:2])
        rows = args[-1][0]
        return bool(name.bbox[1] <= rows.min() and rows.max() < name.bbox[3])

    monkeypatch.setattr(compositing, "_edge_evidence", leaning)
    monkeypatch.setattr(compositing, "_settles_straight", name_only)
    color, _, _ = composite(psd, force=True)
    assert len(tried) == 2
    assert all(shape != (psd.height, psd.width) for shape in tried)
    expected, _, _ = composite(
        psd,
        force=True,
        layer_text_gamma={name.layer_id: None, ability.layer_id: TEXT_BLEND_GAMMA},
    )
    assert np.array_equal(color, expected)
    both, _, _ = composite(
        psd, force=True, layer_text_gamma={name.layer_id: None, ability.layer_id: None}
    )
    assert not np.array_equal(color, both)


@pytest.mark.parametrize(
    "filename", [_KEPT_RASTER, "16bit5x5.psd", "blend-modes/darken.psd"]
)
def test_the_preview_is_read_as_topil_reads_it(filename: str) -> None:
    """8-bit at any depth, with the transparency Photoshop stored."""
    psd = PSDImage.open(full_name(filename))
    compositing = importlib.import_module("psd_tools.composite.composite")
    reference = compositing._text_reference(psd)
    preview = psd.topil(apply_icc=False)
    assert preview is not None and reference is not None
    assert reference.dtype == np.uint8
    assert np.array_equal(reference, np.asarray(preview))


def test_the_preview_is_read_once_until_it_changes() -> None:
    psd, _, _ = _kept_raster()
    compositing = importlib.import_module("psd_tools.composite.composite")
    first = compositing._text_reference(psd)
    assert compositing._text_reference(psd) is first
    gray = Image.new("RGB", (psd.width, psd.height), (128, 128, 128))
    psd.save(io.BytesIO(), preview=gray)
    second = compositing._text_reference(psd)
    assert second is not first and second is not None
    assert np.all(second == 128)


def test_a_preview_that_does_not_decode_is_not_read() -> None:
    """Without a word: a decode warning is process-wide, so none is raised."""
    psd, _, _ = _kept_raster()
    image_data = psd._record.image_data
    image_data.compression = Compression.ZIP
    image_data.data = b"not a deflate stream"
    compositing = importlib.import_module("psd_tools.composite.composite")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert compositing._text_reference(psd) is None
    color, _, _ = composite(psd, force=True)
    expected, _, _ = composite(psd, force=True, text_preview=False)
    assert np.array_equal(color, expected)
