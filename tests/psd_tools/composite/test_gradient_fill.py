"""The ramp ``paint.draw_gradient_fill()`` lays out and the colours it blends.

A gradient's descriptor says where its ramp runs -- ``Angl``, ``Scl ``,
``Ofst``, ``Algn`` -- and how its stops blend -- ``Mdpn`` and ``gs99``. The
closed forms below are for a black-to-white gradient, whose rendered grey is
the ramp position itself, on boxes whose aspect tells the width from the
height.
"""

import math

import numpy as np
import pytest

from psd_tools import PSDImage
from psd_tools.composite import paint
from psd_tools.composite.composite import _draw_gradient_overlay, composite
from psd_tools.constants import ColorMode, GradientInterpolation, Tag
from psd_tools.psd.descriptor import (
    Bool,
    Descriptor,
    Double,
    Enumerated,
    Integer,
    List,
    UnitFloat,
)
from psd_tools.terminology import Enum, Key, Klass, Type, Unit

from ..utils import full_name

BLACK = (0.0, 0.0, 0.0)
WHITE = (255.0, 255.0, 255.0)


def _stop(
    color: tuple[float, float, float], location: int, midpoint: int = 50
) -> Descriptor:
    rgb = Descriptor(classID=Klass.RGBColor.value)
    rgb[Key.Red], rgb[Key.Green], rgb[Key.Blue] = (Double(c) for c in color)
    stop = Descriptor(classID=b"Clrt")
    stop[Key.Color] = rgb
    stop[Key.Type] = Enumerated(
        typeID=Type.ColorStopType.value, enum=Enum.UserStop.value
    )
    stop[Key.Location] = Integer(location)
    stop[Key.Midpoint] = Integer(midpoint)
    return stop


def _opacity_stop(opacity: float, location: int, midpoint: int = 50) -> Descriptor:
    stop = Descriptor(classID=b"TrnS")
    stop[Key.Opacity] = UnitFloat(opacity, Unit.Percent)
    stop[Key.Location] = Integer(location)
    stop[Key.Midpoint] = Integer(midpoint)
    return stop


def _gradient(
    angle: float = 0.0,
    kind: bytes = Enum.Linear.value,
    scale: float = 100.0,
    stops: list[Descriptor] | None = None,
    transparency: list[Descriptor] | None = None,
    offset: tuple[float, float] | None = None,
    method: bytes | None = None,
    reverse: bool = False,
    smoothness: float | None = 0.0,
) -> Descriptor:
    """A ``gradientLayer`` descriptor, black to white unless ``stops`` says.

    Its blend is linear unless ``smoothness`` says, so the ramp's geometry
    reads off the grey directly; ``None`` leaves ``Intr`` out.
    """
    colors = List()
    colors.extend(stops or [_stop(BLACK, 0), _stop(WHITE, 4096)])
    grad = Descriptor(classID=b"Grdn")
    grad[Type.GradientForm] = Enumerated(
        typeID=Type.GradientForm.value, enum=Enum.CustomStops.value
    )
    if smoothness is not None:
        grad[Key.Interpolation] = Double(smoothness * 4096.0)
    grad[Key.Colors] = colors
    if transparency is not None:
        alphas = List()
        alphas.extend(transparency)
        grad[Key.Transparency] = alphas
    desc = Descriptor(classID=b"gradientLayer")
    desc[Key.Gradient] = grad
    desc[Key.Type] = Enumerated(typeID=Type.GradientType.value, enum=kind)
    desc[Key.Angle] = UnitFloat(angle, Unit.Angle)
    desc[Key.Scale] = UnitFloat(scale, Unit.Percent)
    desc[Key.Reverse] = Bool(reverse)
    if offset is not None:
        point = Descriptor(classID=Klass.Point.value)
        point[Key.Horizontal] = UnitFloat(offset[0], Unit.Percent)
        point[Key.Vertical] = UnitFloat(offset[1], Unit.Percent)
        desc[Key.Offset] = point
    if method is not None:
        desc[b"gs99"] = Enumerated(
            typeID=b"gradientInterpolationMethodType", enum=method
        )
    return desc


def _grey(desc: Descriptor, width: int, height: int) -> np.ndarray:
    """The red plane of ``desc`` drawn on a ``width`` x ``height`` box."""
    color, _ = paint.draw_gradient_fill((0, 0, width, height), ColorMode.RGB, desc)
    assert color is not None and color.shape == (height, width, 3)
    return color[..., 0]


def test_a_vertical_ramp_spans_the_height_of_a_wide_box() -> None:
    """The first stop sits on the bottom row and the last on the top row."""
    grey = _grey(_gradient(angle=90.0), 800, 80)
    assert np.allclose(grey, grey[:, :1]), "every column is the same ramp"
    assert np.allclose(grey[:, 0], 1.0 - np.arange(80) / 79, atol=1e-5)


@pytest.mark.parametrize(
    "angle", [0.0, 90.0, -90.0, 180.0, 270.0, 90.00000000000001, 89.99999, -88.92]
)
def test_an_axis_ramp_reaches_both_stops_whatever_the_aspect(angle: float) -> None:
    grey = _grey(_gradient(angle=angle), 100, 20)
    assert grey.min() == pytest.approx(0.0, abs=1 / 255)
    assert grey.max() == pytest.approx(1.0, abs=1 / 255)


def test_a_diagonal_ramp_spans_the_chord_through_the_centre() -> None:
    """A 135 degree ramp on a 300x100 box runs from the top edge to the bottom.

    So it is as long as the height lets a 45 degree line be, not as long as
    the box's projection onto that direction.
    """
    width, height, angle = 300, 100, 135.0
    theta = math.radians(angle)
    chord = min(width / abs(math.cos(theta)), height / abs(math.sin(theta)))
    assert chord == pytest.approx(100 * math.sqrt(2))
    rows, columns = np.mgrid[0:height, 0:width]
    along = (columns - (width - 1) / 2) * math.cos(theta) - (
        rows - (height - 1) / 2
    ) * math.sin(theta)
    expected = np.clip(0.5 + along / chord, 0.0, 1.0)
    assert np.allclose(
        _grey(_gradient(angle=angle), width, height), expected, atol=0.01
    )


def test_a_radial_ramp_ends_half_a_chord_from_the_centre() -> None:
    grey = _grey(_gradient(angle=90.0, kind=Enum.Radial.value), 100, 100)
    assert grey[50, 50] < 0.03, "the first stop at the centre"
    assert grey[0, 50] == pytest.approx(1.0, abs=0.02), "the last stop on the edge"
    assert grey[0, 0] == 1.0, "clipped past the corner"


def test_scale_shortens_the_ramp_around_the_centre() -> None:
    grey = _grey(_gradient(angle=90.0, scale=50.0), 100, 20)[:, 0]
    assert (grey[:5] == 1.0).all() and (grey[15:] == 0.0).all()
    assert (np.diff(grey[4:16]) < 0).all(), "the ramp lives in the middle half"
    reverse = _grey(_gradient(angle=90.0, scale=50.0, reverse=True), 100, 20)[:, 0]
    assert np.allclose(reverse, 1.0 - grey)


@pytest.mark.parametrize(
    ("angle", "offset", "centre"),
    [
        (90.0, (0.0, 25.0), 74.5),
        (90.0, (0.0, -25.0), 24.5),
        (0.0, (25.0, 0.0), 74.5),
        (0.0, (-25.0, 0.0), 24.5),
    ],
)
def test_offset_moves_the_centre_by_a_share_of_the_box(
    angle: float, offset: tuple[float, float], centre: float
) -> None:
    grey = _grey(_gradient(angle=angle, offset=offset), 100, 100)
    line = grey[:, 0] if angle == 90.0 else grey[0, :]
    halfway = np.flatnonzero(np.isclose(line, 0.5, atol=1 / 198))
    assert set(halfway.tolist()) == {math.floor(centre), math.ceil(centre)}


def test_the_midpoint_is_read_from_the_stop_that_ends_the_segment() -> None:
    """``Mdpn`` 25 on the last stop puts the halfway grey a quarter of the way in.

    On the first stop it names a segment that does not exist.
    """
    plain = _grey(_gradient(), 100, 1)[0]
    bent = _grey(_gradient(stops=[_stop(BLACK, 0), _stop(WHITE, 4096, 25)]), 100, 1)[0]
    unbent = _grey(_gradient(stops=[_stop(BLACK, 0, 25), _stop(WHITE, 4096)]), 100, 1)[
        0
    ]
    assert np.allclose(bent, _bent_at(0.25, np.arange(100) / 99), atol=1e-5)
    assert np.array_equal(unbent, plain)


def _bent_at(midpoint: float, along: np.ndarray) -> np.ndarray:
    """A segment's blend, linear to 0.5 at ``midpoint`` and linear on to 1."""
    return np.where(
        along <= midpoint,
        0.5 * along / midpoint,
        0.5 + 0.5 * (along - midpoint) / (1.0 - midpoint),
    )


def _smoothed(along: np.ndarray) -> np.ndarray:
    """The blend at 100% smoothness: halfway from linear to a smoothstep."""
    return along + ((3.0 - 2.0 * along) * along * along - along) / 2


@pytest.mark.parametrize("smoothness", [1.0, None], ids=["100%", "default"])
def test_full_smoothness_eases_the_blend_halfway_to_a_smoothstep(
    smoothness: float | None,
) -> None:
    along = np.arange(101) / 100
    grey = _grey(_gradient(smoothness=smoothness), 101, 1)[0]
    assert np.allclose(grey, _smoothed(along), atol=1e-5)
    assert grey[25] == pytest.approx(0.203125) and grey[50] == 0.5
    half = _grey(_gradient(smoothness=0.5), 101, 1)[0]
    assert np.allclose(half, (along + _smoothed(along)) / 2, atol=1e-5)


def test_a_smooth_ramp_matches_photoshops_own_pixels() -> None:
    """``gradient-fill.psb`` holds a violet-to-orange fill at 100% smoothness.

    Its preview shows the ramp through a 16x94 mask, laid out on that mask's
    box; drawn on the same box, the fill's own blend has to match Photoshop's
    pixels down the opaque column, and the linear blend must not.
    """
    psd = PSDImage.open(full_name("gradient-fill.psb"))
    fill = psd[2]
    mask = fill.mask
    assert fill.name == "Gradient Fill 1" and mask is not None
    assert mask.bbox == (44, 6, 60, 100)
    desc = fill.tagged_blocks.get_data(Tag.GRADIENT_FILL_SETTING)
    assert float(desc[Key.Gradient][Key.Interpolation]) == 4096.0
    column, rows = 50, slice(8, 98)
    coverage = fill.numpy("mask")
    assert coverage is not None
    assert (coverage[rows.start - 6 : rows.stop - 6, column - 44, 0] == 1.0).all()
    photoshop = psd.numpy()[rows, column, :3]

    smooth, _ = paint.draw_gradient_fill(mask.bbox, ColorMode.RGB, desc)
    assert smooth is not None
    rendered = smooth[rows.start - 6 : rows.stop - 6, column - 44]
    assert np.abs(rendered - photoshop).max() < 3 / 255

    desc[Key.Gradient][Key.Interpolation] = Double(0.0)
    linear, _ = paint.draw_gradient_fill(mask.bbox, ColorMode.RGB, desc)
    assert linear is not None
    assert (
        np.abs(linear[rows.start - 6 : rows.stop - 6, column - 44] - photoshop).max()
        > 8 / 255
    )


def test_a_transparency_midpoint_shapes_the_alpha_ramp_alone() -> None:
    desc = _gradient(
        transparency=[_opacity_stop(100.0, 0), _opacity_stop(0.0, 4096, 25)]
    )
    color, shape = paint.draw_gradient_fill((0, 0, 100, 1), ColorMode.RGB, desc)
    assert color is not None and shape is not None
    along = np.arange(100) / 99
    assert np.allclose(shape[0, :, 0], 1.0 - _bent_at(0.25, along), atol=1e-5)
    assert np.allclose(color[0, :, 0], along, atol=1e-5)


# What a black-to-white ramp is halfway along in each ``gs99`` method: the
# classic blend is the sRGB mean, perceptual is Oklab lightness 0.5 (linear
# light 0.125) and linear is linear-light 0.5, both re-encoded as sRGB.
HALFWAY_GREY = {
    None: 0.5,
    b"XXXX": 0.5,
    GradientInterpolation.CLASSIC.value: 0.5,
    GradientInterpolation.PERCEPTUAL.value: 1.055 * 0.125 ** (1 / 2.4) - 0.055,
    GradientInterpolation.LINEAR.value: 1.055 * 0.5 ** (1 / 2.4) - 0.055,
}


@pytest.mark.parametrize("method", list(HALFWAY_GREY), ids=str)
def test_each_interpolation_method_blends_in_its_own_space(
    method: bytes | None,
) -> None:
    grey = _grey(_gradient(method=method), 101, 1)[0]
    assert grey[50] == pytest.approx(HALFWAY_GREY[method], abs=0.002)
    assert grey[0] == pytest.approx(0.0, abs=1e-3) and grey[100] == pytest.approx(
        1.0, abs=1e-3
    )
    assert (np.diff(grey) >= 0).all()


# The width each mode's color arrays carry; multichannel's is per file.
MODE_CHANNELS = {
    ColorMode.BITMAP: 1,
    ColorMode.GRAYSCALE: 1,
    ColorMode.INDEXED: 3,
    ColorMode.RGB: 3,
    ColorMode.CMYK: 4,
    ColorMode.DUOTONE: 1,
    ColorMode.LAB: 3,
}


@pytest.mark.parametrize(
    "method",
    [GradientInterpolation.PERCEPTUAL.value, GradientInterpolation.LINEAR.value],
)
@pytest.mark.parametrize("color_mode", list(MODE_CHANNELS))
def test_a_ramp_blended_outside_the_canvas_space_lands_in_the_documents_arrays(
    method: bytes, color_mode: ColorMode
) -> None:
    color, _ = paint.draw_gradient_fill(
        (0, 0, 16, 4), color_mode, _gradient(method=method)
    )
    assert color is not None
    assert color.shape == (4, 16, MODE_CHANNELS[color_mode])
    assert color.min() >= 0.0 and color.max() <= 1.0


def test_an_overlay_not_aligned_with_its_layer_is_laid_out_on_the_document() -> None:
    psd = PSDImage.open(full_name("layer_effects.psd"))
    layer = next(x for x in psd if x.name == "Gradient Overlay")
    desc = next(layer.effects.find("gradientoverlay")).descriptor
    assert layer.bbox == (99, 422, 699, 482) and psd.viewbox == (0, 0, 800, 640)

    assert paint.gradient_anchor(layer, desc) is None, "aligned by default"
    aligned, _ = _draw_gradient_overlay(layer, desc, 3)
    desc[Key.Alignment] = Bool(False)
    assert paint.gradient_anchor(layer, desc) == psd.viewbox
    unaligned, _ = _draw_gradient_overlay(layer, desc, 3)
    on_document, _ = paint.draw_gradient_fill(
        layer.bbox, psd.color_mode, desc, psd.viewbox
    )
    assert aligned is not None and unaligned is not None and on_document is not None
    assert np.array_equal(unaligned, on_document)
    assert not np.array_equal(unaligned, aligned)


def test_a_wide_overlay_at_a_near_vertical_angle_matches_photoshop() -> None:
    """A 600x60 overlay at 87 degrees against Photoshop's own render.

    Its ramp is as long as the box is tall; measured as long as the box was
    wide, only a slice of it showed.
    """
    psd = PSDImage.open(full_name("layer_effects.psd"))
    layer = next(x for x in psd if x.name == "Gradient Overlay")
    assert layer.bbox == (99, 422, 699, 482)
    left, top, right, bottom = layer.bbox
    reference = psd.numpy()[top:bottom, left:right]
    color, _, _ = composite(psd, force=True)
    assert reference.shape[2] == color.shape[2]
    error = np.mean((reference - color[top:bottom, left:right]) ** 2)
    on_main = 0.00082
    assert error < on_main / 4
