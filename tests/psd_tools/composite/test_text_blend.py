"""The space a type layer's anti-aliased edge blends in.

Three fixtures carry a type layer Photoshop composited under its "Blend Text
Colors Using Gamma" preference into their preview -- ``text.psb``,
``masks2.psb`` and ``artboard.psd`` -- and ``layer_effects.psd`` adds one
with each effect. The cases the corpus does not pair with a type layer (a
styled group around one, a blend mode, a grey or CMYK canvas) are forged.
"""

import copy
from collections.abc import Iterator

import numpy as np
import pytest

from psd_tools.api.layers import Group, Layer, TypeLayer
from psd_tools.api.psd_image import PSDImage
from psd_tools.composite.blend import get_blend_func
from psd_tools.composite.composite import Compositor
from psd_tools.composite.text_blend import (
    TEXT_BLEND_GAMMA,
    from_text_space,
    to_text_space,
)
from psd_tools.composite.widen import make_widen
from psd_tools.constants import BlendMode, ColorMode, Tag

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


@pytest.mark.parametrize(
    ("channels", "color_mode"),
    [(3, ColorMode.RGB), (1, ColorMode.GRAYSCALE), (4, ColorMode.CMYK)],
)
@pytest.mark.parametrize("blend_mode", [BlendMode.NORMAL, BlendMode.MULTIPLY])
def test_the_mix_is_by_coverage_in_the_text_space(
    channels: int, color_mode: ColorMode, blend_mode: BlendMode
) -> None:
    """Every channel of every mode takes the same mix.

    The blend mode still sees the document's values; only the mix by
    coverage is in the text space.
    """
    pixel = _mixed_pixel(channels, color_mode, 0.8, 0.3, 0.5, blend_mode)
    expected = _expected(0.8, 0.3, 0.5, blend_mode)
    assert pixel == pytest.approx(np.full(channels, expected), abs=1e-6)
    straight = 0.5 * 0.3 * (1 if blend_mode == BlendMode.NORMAL else 0.8) + 0.5 * 0.8
    assert abs(expected - straight) > 0.02
