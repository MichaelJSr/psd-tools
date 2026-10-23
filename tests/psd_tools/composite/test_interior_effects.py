"""Interior effects on a layer not blended as a group.

With "Blend Interior Effects as Group" off, Photoshop's default, a layer's
paint goes on in its own mode at full coverage, each interior effect goes on
that in the effect's own mode, and the layer's coverage and opacity draw the
result back towards the backdrop once (#846).

The pixel fixtures are built here: an opaque grey backdrop under a 2 x 1
layer whose left pixel is opaque and whose right one is half transparent.
The overlays are forged from ``layer_effects.psd``'s ``Color Fill``.
"""

import copy

import numpy as np
import pytest
from PIL import Image

from psd_tools.api.layers import Layer, TypeLayer
from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import composite
from psd_tools.composite.blend import darker_color, screen
from psd_tools.composite.composite import Compositor
from psd_tools.composite.text_blend import TEXT_BLEND_GAMMA
from psd_tools.composite.widen import make_widen
from psd_tools.constants import BlendMode, Tag
from psd_tools.psd.descriptor import Descriptor, Double, Enumerated
from psd_tools.terminology import Enum, Key, Klass, Type

from ..utils import full_name

# 8-bit colours, as the layers store them.
_BACKDROP = (128, 128, 128)
_PAINT = (51, 51, 153)
# Darker than the paint screened onto the backdrop, lighter than the paint.
_OVERLAY = (0, 128, 0)
# The right pixel's alpha.
_HALF = 128 / 255.0


def _rgb(values: tuple[int, int, int]) -> np.ndarray:
    return np.array(values, dtype=np.float32).reshape(1, 1, 3) / 255.0


def _overlay(layer: Layer, mode: bytes, rgb: tuple[int, int, int]) -> None:
    """Give ``layer`` ``layer_effects.psd``'s colour overlay, in ``mode``."""
    donor = PSDImage.open(full_name("layer_effects.psd"))
    source = next(sub for sub in donor.descendants() if sub.name == "Color Fill")
    block = copy.deepcopy(
        source.tagged_blocks.get_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO)
    )
    layer.tagged_blocks.set_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO, block)
    desc: Descriptor = block[Klass.SolidFill]
    desc[Key.Mode] = Enumerated(typeID=Type.BlendMode, enum=mode)
    desc[Key.Opacity] = Double(100.0)
    for key, value in zip((b"Rd  ", b"Grn ", b"Bl  "), rgb, strict=True):
        desc[Key.Color][key] = Double(float(value))


def _document(blend_mode: BlendMode, opacity: int = 255) -> tuple[PSDImage, Layer]:
    psd = PSDImage.new(mode="RGB", size=(2, 1))
    psd.create_pixel_layer(Image.new("RGB", (2, 1), _BACKDROP), name="backdrop")
    image = Image.new("RGBA", (2, 1))
    image.putpixel((0, 0), (*_PAINT, 255))
    image.putpixel((1, 0), (*_PAINT, 128))
    layer = psd.create_pixel_layer(
        image, name="layer", opacity=opacity, blend_mode=blend_mode
    )
    return psd, layer


def _faded(full: np.ndarray, share: float) -> np.ndarray:
    return _rgb(_BACKDROP) + share * (full - _rgb(_BACKDROP))


def test_an_overlay_goes_on_the_layer_composited_in_its_mode() -> None:
    """A Darker Color overlay on a Screen layer compares with the screened canvas.

    The screened canvas is lighter than the overlay, so the overlay wins
    wherever the layer has paint, and the layer's alpha and opacity draw
    that back towards the backdrop once. Blended as a group, the overlay
    would compare with the paint instead, lose, and leave the screen.
    """
    opacity = 69
    psd, layer = _document(BlendMode.SCREEN, opacity)
    _overlay(layer, b"darkerColor", _OVERLAY)
    color, _, alpha = composite(psd)

    screened = screen(_rgb(_BACKDROP), _rgb(_PAINT))
    full = darker_color(screened, _rgb(_OVERLAY))
    assert np.allclose(full, _rgb(_OVERLAY)), "the overlay has to win"
    share = opacity / 255.0
    assert np.allclose(color[0, 0], _faded(full, share)[0, 0], atol=1e-5)
    assert np.allclose(color[0, 1], _faded(full, _HALF * share)[0, 0], atol=1e-5)
    assert np.all(alpha == 1.0)

    layer.tagged_blocks.set_data(Tag.BLEND_INTERIOR_ELEMENTS, 1)
    as_group, _, _ = composite(psd)
    assert np.allclose(as_group[0, 0], _faded(screened, share)[0, 0], atol=1e-5)


def test_fill_opacity_fades_the_paint_and_not_the_overlay() -> None:
    """At fill 0 a Multiply overlay multiplies what lies beneath the layer.

    The paint goes on at no strength, so the overlay meets the backdrop
    through it; blended into the layer it would meet nothing and go on as
    if it were Normal.
    """
    psd, layer = _document(BlendMode.NORMAL)
    layer.fill_opacity = 0
    _overlay(layer, b"multiply", _OVERLAY)
    color, _, _ = composite(psd)

    full = _rgb(_BACKDROP) * _rgb(_OVERLAY)
    assert np.allclose(color[0, 0], full[0, 0], atol=1e-5)
    assert np.allclose(color[0, 1], _faded(full, _HALF)[0, 0], atol=1e-5)


def test_a_normal_layer_at_full_fill_composites_as_before() -> None:
    """Under Normal at full fill the two ways agree, and the layer keeps its path."""
    psd, layer = _document(BlendMode.NORMAL, 128)
    _overlay(layer, b"multiply", _OVERLAY)
    color, _, _ = composite(psd)
    full = _rgb(_PAINT) * _rgb(_OVERLAY)
    assert np.allclose(color[0, 0], _faded(full, 128 / 255.0)[0, 0], atol=1e-5)


@pytest.mark.parametrize("blend_mode", [BlendMode.MULTIPLY, BlendMode.SCREEN])
def test_a_groups_overlay_goes_on_over_the_group_in_its_mode(
    blend_mode: BlendMode,
) -> None:
    """A group composited as one layer takes the same law as a layer.

    A Normal overlay at full opacity covers what the group's mode made of
    its contents, so the group renders as a Normal one does.
    """
    renders = {}
    for mode in (BlendMode.NORMAL, blend_mode):
        psd, layer = _document(BlendMode.NORMAL)
        group = psd.create_group([layer], name="group", blend_mode=mode)
        _overlay(group, b"normal", _OVERLAY)
        renders[mode], _, _ = composite(psd)
    assert np.allclose(renders[BlendMode.NORMAL][0, 0], _rgb(_OVERLAY)[0, 0])
    assert np.allclose(renders[blend_mode], renders[BlendMode.NORMAL], atol=1e-6)


# -- A type layer ----------------------------------------------------------- #


def _render(psd: PSDImage) -> np.ndarray:
    """The document from its layers, text in the gamma, no preview read."""
    left, top, right, bottom = psd.viewbox
    height, width = bottom - top, right - left
    compositor = Compositor(
        psd.viewbox,
        np.ones((height, width, 3), dtype=np.float32),
        np.zeros((height, width, 1), dtype=np.float32),
        force=True,
        widen=make_widen(psd),
        color_mode=psd.color_mode,
        text_gamma=TEXT_BLEND_GAMMA,
    )
    for layer in psd:
        compositor.apply(layer)
    return compositor.finish()[0]


@pytest.mark.parametrize("blend_mode", [BlendMode.LINEAR_DODGE, BlendMode.MULTIPLY])
def test_a_full_normal_overlay_hides_a_type_layers_mode_to_its_edge(
    blend_mode: BlendMode,
) -> None:
    """The overlay covers the paint whole, so the mode leaves no trace.

    ``layer_effects.psd``'s ``Color Fill`` text carries a Normal overlay at
    full opacity. In another mode it renders as it does in Normal, glyph
    edges and all: the canvas goes on in the text gamma once, by the glyphs'
    coverage, after the overlay, whatever mode put the paint under it.
    """
    psd = PSDImage.open(full_name("layer_effects.psd"))
    layer = next(sub for sub in psd.descendants() if sub.name == "Color Fill")
    assert isinstance(layer, TypeLayer)
    normal = _render(psd)

    layer.blend_mode = blend_mode
    apart = _render(psd)
    left, top, right, bottom = layer.bbox
    box = (slice(top, bottom), slice(left, right))
    shape = layer.numpy("shape")
    assert shape is not None
    edges = (shape[..., 0] > 0.05) & (shape[..., 0] < 0.9)
    assert edges.sum() > 100
    assert np.allclose(apart[box], normal[box], atol=1e-5)

    layer.tagged_blocks.set_data(Tag.BLEND_INTERIOR_ELEMENTS, 1)
    as_group = _render(psd)
    assert not np.allclose(as_group[box], normal[box], atol=4 / 255)


def _style_over_overlay(donor_name: str) -> Descriptor:
    """``layer_effects.psd``'s style on ``donor_name``, its Normal overlay on."""
    donor = PSDImage.open(full_name("layer_effects.psd"))
    styled = next(sub for sub in donor.descendants() if sub.name == donor_name)
    block = copy.deepcopy(
        styled.tagged_blocks.get_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO)
    )
    overlay = block[b"solidFillMulti"][0]
    overlay[b"enab"] = overlay[b"present"] = True
    overlay[Key.Mode] = Enumerated(typeID=Type.BlendMode, enum=b"normal")
    overlay[Key.Opacity] = Double(100.0)
    return block


@pytest.mark.parametrize("opacity", [255, 128])
@pytest.mark.parametrize("donor_name", ["Stroke", "Inner Glow"])
def test_an_effect_at_the_edge_meets_the_layer_as_on_a_normal_one(
    donor_name: str, opacity: int
) -> None:
    """An effect at the layer's edge goes on there as on a Normal layer.

    The band outside goes on beside the layer's finished coverage, at the
    layer opacity, and an inner glow covers its share of the layer's region.
    ``layer_effects.psd``'s outside stroke or inner glow, with the layer's
    Normal overlay switched on under it, on a soft-edged disc: the overlay
    covers the paint whole, so a Screen disc renders as a Normal one does,
    where the edge cuts the effect included.
    """
    block = _style_over_overlay(donor_name)

    size, scale = 40, 4
    fine = (np.arange(size * scale) + 0.5) / scale - size / 2
    inside = (fine[:, None] ** 2 + fine[None, :] ** 2) <= 10.0**2
    coverage = inside.reshape(size, scale, size, scale).mean(axis=(1, 3))
    disc = np.zeros((size, size, 4), dtype=np.float32)
    disc[..., :3] = _rgb(_PAINT)
    disc[..., 3] = coverage

    renders = {}
    for blend_mode in (BlendMode.NORMAL, BlendMode.SCREEN):
        psd = PSDImage.new(mode="RGB", size=(size, size))
        psd.create_pixel_layer(Image.new("RGB", (size, size), _BACKDROP), "backdrop")
        layer = psd.create_pixel_layer_from_array(
            disc, "disc", blend_mode=blend_mode, opacity=opacity
        )
        layer.tagged_blocks.set_data(
            Tag.OBJECT_BASED_EFFECTS_LAYER_INFO, copy.deepcopy(block)
        )
        renders[blend_mode], _, _ = composite(psd)
    edge = (coverage > 0.0) & (coverage < 1.0)
    assert edge.sum() > 20
    assert not np.allclose(renders[BlendMode.NORMAL], _rgb(_BACKDROP), atol=0.1)
    assert np.allclose(renders[BlendMode.SCREEN], renders[BlendMode.NORMAL], atol=1e-5)


@pytest.mark.parametrize("style", [Enum.InsetFrame, Enum.CenteredFrame])
def test_a_normal_stroke_knocks_the_layer_out_under_it(style: bytes) -> None:
    """A stroke is no interior effect: inside the layer it knocks it out.

    ``layer_effects.psd``'s white stroke at half opacity, over the layer's
    Normal overlay, on a square at half opacity. Within the band the stroke
    shows the backdrop through it, not the overlay, whether the layer is at
    full fill, at fill 0 or in Screen.
    """
    block = _style_over_overlay("Stroke")
    stroke = block[b"FrFX"]
    stroke[Key.Style] = Enumerated(typeID=Type.FrameStyle, enum=style)
    stroke[Key.Opacity] = Double(50.0)
    opacity = 128
    square = np.zeros((40, 40, 4), dtype=np.float32)
    square[10:30, 10:30, :3] = _rgb(_PAINT)
    square[10:30, 10:30, 3] = 1.0

    renders = []
    for blend_mode, fill in (
        (BlendMode.NORMAL, 255),
        (BlendMode.NORMAL, 0),
        (BlendMode.SCREEN, 255),
    ):
        psd = PSDImage.new(mode="RGB", size=(40, 40))
        psd.create_pixel_layer(Image.new("RGB", (40, 40), _BACKDROP), "backdrop")
        layer = psd.create_pixel_layer_from_array(
            square, "square", blend_mode=blend_mode, opacity=opacity
        )
        layer.fill_opacity = fill
        layer.tagged_blocks.set_data(
            Tag.OBJECT_BASED_EFFECTS_LAYER_INFO, copy.deepcopy(block)
        )
        renders.append(composite(psd)[0])

    band = 0.5 * _rgb((255, 255, 255)) + 0.5 * _rgb(_BACKDROP)
    expected = _faded(band, opacity / 255.0)[0, 0]
    for render in renders:
        assert np.allclose(render[20, 11], expected, atol=1e-5)
        assert np.allclose(render, renders[0], atol=1e-5)
