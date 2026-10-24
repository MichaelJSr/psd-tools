"""The ramp ``paint.draw_gradient_fill()`` lays out and the colours it blends.

A gradient's descriptor says where its ramp runs -- ``Angl``, ``Scl ``,
``Ofst``, ``Algn`` -- and how its stops blend -- ``Mdpn`` and the method, under
``gs99`` on an overlay and ``gradientsInterpolationMethod`` on a fill layer or
stroke effect. The closed forms below are for a black-to-white gradient, whose
rendered grey is the ramp position itself, on boxes whose aspect tells the
width from the height.
"""

import itertools
import math
from collections.abc import Callable

import numpy as np
import pytest
from PIL import Image

from psd_tools import PSDImage
from psd_tools.api.layers import Layer, PixelLayer
from psd_tools.composite import effects, paint
from psd_tools.composite.composite import (
    _draw_gradient_overlay,
    _opaque_bounds,
    _path_bbox,
    composite,
)
from psd_tools.composite.effects import draw_stroke_effect
from psd_tools.constants import GRADIENT_METHOD_KEY, ColorMode, GradientMethod, Tag
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

pytest.importorskip("scipy")

BLACK = (0.0, 0.0, 0.0)
WHITE = (255.0, 255.0, 255.0)


def _stop_of(color: Descriptor, location: int, midpoint: int = 50) -> Descriptor:
    stop = Descriptor(classID=b"Clrt")
    stop[Key.Color] = color
    stop[Key.Type] = Enumerated(
        typeID=Type.ColorStopType.value, enum=Enum.UserStop.value
    )
    stop[Key.Location] = Integer(location)
    stop[Key.Midpoint] = Integer(midpoint)
    return stop


def _stop(
    color: tuple[float, float, float], location: int, midpoint: int = 50
) -> Descriptor:
    rgb = Descriptor(classID=Klass.RGBColor.value)
    rgb[Key.Red], rgb[Key.Green], rgb[Key.Blue] = (Double(c) for c in color)
    return _stop_of(rgb, location, midpoint)


def _cmyk_stop(ink: tuple[float, float, float, float], location: int) -> Descriptor:
    cmyk = Descriptor(classID=Klass.CMYKColor.value)
    for key, value in zip(
        (Key.Cyan, Key.Magenta, Key.Yellow, Key.Black), ink, strict=True
    ):
        cmyk[key] = Double(value)
    return _stop_of(cmyk, location)


def _lab_stop(lab: tuple[float, float, float], location: int) -> Descriptor:
    color = Descriptor(classID=Klass.LabColor.value)
    color[Key.Luminance], color[Key.A], color[Key.B] = (Double(v) for v in lab)
    return _stop_of(color, location)


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
        desc[GRADIENT_METHOD_KEY] = Enumerated(
            typeID=b"gradientInterpolationMethodType", enum=method
        )
    return desc


def _grey(desc: Descriptor, width: int, height: int) -> np.ndarray:
    """The red plane of ``desc`` drawn on a ``width`` x ``height`` box."""
    color, _ = paint.draw_gradient_fill((0, 0, width, height), ColorMode.RGB, desc)
    assert color is not None and color.shape == (height, width, 3)
    return color[..., 0]


def _row(
    color_mode: ColorMode, desc: Descriptor
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """``desc`` drawn on a 101 px row whose ramp is laid out on its first 100 px.

    A pixel is sampled at its index, so pixel ``x`` is ``x``% along the ramp
    and the last one lands on the end stop.
    """
    return paint.draw_gradient_fill(
        (0, 0, 101, 1), color_mode, desc, anchor=(0, 0, 100, 1)
    )


def _grey_row(desc: Descriptor) -> np.ndarray:
    """The red plane of :py:func:`_row`'s row."""
    color, _ = _row(ColorMode.RGB, desc)
    assert color is not None and color.shape == (1, 101, 3)
    return color[0, :, 0]


def test_a_vertical_ramp_spans_the_height_of_a_wide_box() -> None:
    """The last stop sits on the top row, the first one row below the bottom.

    Photoshop samples a pixel at its index, so the box's top edge is the top
    row and its bottom edge lies past the last row.
    """
    grey = _grey(_gradient(angle=90.0), 800, 80)
    assert np.allclose(grey, grey[:, :1]), "every column is the same ramp"
    assert np.allclose(grey[:, 0], 1.0 - np.arange(80) / 80, atol=1e-5)


@pytest.mark.parametrize(
    "angle", [0.0, 90.0, -90.0, 180.0, 270.0, 90.00000000000001, 89.99999, -88.92]
)
def test_an_axis_ramp_spans_the_box_whatever_the_aspect(angle: float) -> None:
    """Down the box's middle the near edge is on a stop, the far one a step short."""
    width, height = 100, 20
    grey = _grey(_gradient(angle=angle), width, height)
    vertical = abs(math.sin(math.radians(angle))) > 0.5
    line = grey[:, width // 2] if vertical else grey[height // 2, :]
    near, far = float(line[0]), float(line[-1])
    assert min(near, 1.0 - near) == pytest.approx(0.0, abs=1 / 255)
    assert abs(far - near) == pytest.approx(1.0 - 1.0 / line.size, abs=1 / 255)


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
        (90.0, (0.0, 25.0), 75),
        (90.0, (0.0, -25.0), 25),
        (0.0, (25.0, 0.0), 75),
        (0.0, (-25.0, 0.0), 25),
    ],
)
def test_offset_moves_the_centre_by_a_share_of_the_box(
    angle: float, offset: tuple[float, float], centre: int
) -> None:
    """The halfway grey lands on the pixel whose index is the moved centre."""
    grey = _grey(_gradient(angle=angle, offset=offset), 100, 100)
    line = grey[:, 0] if angle == 90.0 else grey[0, :]
    halfway = np.flatnonzero(np.isclose(line, 0.5, atol=1 / 198))
    assert halfway.tolist() == [centre]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_non_finite_offset_reads_as_none(value: float) -> None:
    desc = _gradient(angle=90.0, offset=(value, value))
    assert np.array_equal(_grey(desc, 20, 20), _grey(_gradient(angle=90.0), 20, 20))


def test_an_offset_past_the_ramp_renders_flat_rather_than_overflowing() -> None:
    grey = _grey(_gradient(angle=90.0, offset=(0.0, 1e300)), 20, 20)
    assert np.isfinite(grey).all() and grey.min() == grey.max()


def test_co_located_stops_make_a_hard_edge() -> None:
    """Photoshop's editor authors a hard edge as two stops at one location.

    The sample on the edge takes the later stop, including on the last one.
    """
    red, blue = (255.0, 0.0, 0.0), (0.0, 0.0, 255.0)
    desc = _gradient(
        stops=[_stop(red, 0), _stop(red, 2048), _stop(blue, 2048), _stop(blue, 4096)],
        transparency=[
            _opacity_stop(100.0, 0),
            _opacity_stop(100.0, 2048),
            _opacity_stop(0.0, 2048),
            _opacity_stop(0.0, 4096),
        ],
    )
    color, shape = paint.draw_gradient_fill((0, 0, 100, 1), ColorMode.RGB, desc)
    assert color is not None and shape is not None
    assert np.allclose(color[0, :50], [1.0, 0.0, 0.0], atol=1e-6)
    assert np.allclose(color[0, 50:], [0.0, 0.0, 1.0], atol=1e-6)
    assert np.allclose(shape[0, :50, 0], 1.0) and np.allclose(shape[0, 50:, 0], 0.0)

    desc = _gradient(stops=[_stop(BLACK, 0), _stop(WHITE, 4096), _stop(red, 4096)])
    color, _ = _row(ColorMode.RGB, desc)
    assert color is not None
    assert color[0, 99, 0] == pytest.approx(0.99, abs=1e-5)
    assert np.allclose(color[0, 100], [1.0, 0.0, 0.0], atol=1e-6)


def test_the_midpoint_is_read_from_the_stop_that_ends_the_segment() -> None:
    """``Mdpn`` 25 on the last stop puts the halfway grey a quarter of the way in.

    On the first stop it names a segment that does not exist.
    """
    plain = _grey_row(_gradient())
    bent = _grey_row(_gradient(stops=[_stop(BLACK, 0), _stop(WHITE, 4096, 25)]))
    unbent = _grey_row(_gradient(stops=[_stop(BLACK, 0, 25), _stop(WHITE, 4096)]))
    assert np.allclose(bent, _bent_at(0.25, np.arange(101) / 100), atol=1e-5)
    assert np.array_equal(unbent, plain)


def test_stops_blend_in_ramp_order_whatever_order_they_are_listed() -> None:
    """A red stop listed after the white end stop still sits between the two."""
    red = (255.0, 0.0, 0.0)
    listed = [_stop(BLACK, 0), _stop(WHITE, 4096), _stop(red, 2048)]
    sorted_stops = [_stop(BLACK, 0), _stop(red, 2048), _stop(WHITE, 4096)]
    color, _ = _row(ColorMode.RGB, _gradient(stops=listed))
    expected, _ = _row(ColorMode.RGB, _gradient(stops=sorted_stops))
    assert color is not None and expected is not None
    assert np.array_equal(color, expected)
    assert color[0, 50].tolist() == pytest.approx([1.0, 0.0, 0.0], abs=1e-5)


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
    grey = _grey_row(_gradient(smoothness=smoothness))
    assert np.allclose(grey, _smoothed(along), atol=1e-5)
    assert grey[25] == pytest.approx(0.203125) and grey[50] == 0.5
    half = _grey_row(_gradient(smoothness=0.5))
    assert np.allclose(half, (along + _smoothed(along)) / 2, atol=1e-5)


def _infinite_angle(desc: Descriptor) -> None:
    desc[Key.Angle] = UnitFloat(math.inf, Unit.Angle)


def _offset_not_a_point(desc: Descriptor) -> None:
    desc[Key.Offset] = Integer(3)


def _offset_not_a_number(desc: Descriptor) -> None:
    point = Descriptor(classID=Klass.Point.value)
    point[Key.Horizontal] = List()
    point[Key.Vertical] = UnitFloat(math.nan, Unit.Percent)
    desc[Key.Offset] = point


def _smoothness_not_a_number(desc: Descriptor) -> None:
    desc[Key.Gradient][Key.Interpolation] = List()


def _midpoint_not_a_number(desc: Descriptor) -> None:
    desc[Key.Gradient][Key.Colors][1][Key.Midpoint] = List()


@pytest.mark.parametrize(
    "forge",
    [
        _infinite_angle,
        _offset_not_a_point,
        _offset_not_a_number,
        _smoothness_not_a_number,
        _midpoint_not_a_number,
    ],
)
def test_a_malformed_number_reads_as_if_it_were_absent(
    forge: Callable[[Descriptor], None],
) -> None:
    """A fill layer has no guard around its draw, so the ramp must not raise."""
    expected = _grey(_gradient(smoothness=None), 16, 4)
    desc = _gradient(smoothness=None)
    forge(desc)
    assert np.allclose(_grey(desc, 16, 4), expected, atol=1e-6)


def test_a_smooth_ramp_matches_photoshops_own_pixels() -> None:
    """``gradient-fill.psb`` holds a violet-to-orange fill at 100% smoothness.

    Its preview shows the ramp through a 16x94 mask, laid out on that mask's
    box; drawn on the same box, the fill's own blend has to match Photoshop's
    pixels down the opaque column to within half a level, and the linear
    blend must not.
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
    assert np.abs(rendered - photoshop).max() < 0.5 / 255

    desc[Key.Gradient][Key.Interpolation] = Double(0.0)
    linear, _ = paint.draw_gradient_fill(mask.bbox, ColorMode.RGB, desc)
    assert linear is not None
    assert (
        np.abs(linear[rows.start - 6 : rows.stop - 6, column - 44] - photoshop).max()
        > 8 / 255
    )


def _ramp(
    stops: list[Descriptor], smoothness: float = 1.0, method: bytes | None = None
) -> np.ndarray:
    """``stops`` drawn on a 4097 px row in 0-255 levels: pixel ``x`` is ``Lctn`` x.

    The ramp is laid out on the row's first 4096 px, so the last pixel is one
    past that box and lands on ``Lctn`` 4096.
    """
    desc = _gradient(stops=stops, smoothness=smoothness, method=method)
    color, _ = paint.draw_gradient_fill(
        (0, 0, 4097, 1), ColorMode.RGB, desc, anchor=(0, 0, 4096, 1)
    )
    assert color is not None
    return color[0] * 255.0


def _eased_per_segment(
    stops: list[tuple[tuple[float, float, float], int]], location: int
) -> np.ndarray:
    """The ramp at ``location`` if each segment were eased on its own."""
    for (a, start), (b, end) in itertools.pairwise(stops):
        if start <= location <= end:
            along = float(_smoothed(np.array((location - start) / (end - start))))
            return np.array(a) + (np.array(b) - np.array(a)) * along
    raise ValueError(location)


Stops = list[tuple[tuple[float, float, float], int]]
Readings = list[tuple[int, tuple[int, int, int]]]


def _photoshop_misses(
    stops: Stops, readings: Readings
) -> tuple[np.ndarray, np.ndarray]:
    """(ours, per-segment ease) each as its largest miss of Photoshop's readings."""
    ramp = _ramp([_stop(color, location) for color, location in stops])
    locations = [location for location, _ in readings]
    photoshop = np.array([color for _, color in readings], dtype=float)
    visible = [
        (c, x) for i, (c, x) in enumerate(stops) if i == 0 or x > stops[i - 1][1]
    ]
    eased = np.array([_eased_per_segment(visible, x) for x in locations])
    return np.abs(ramp[locations] - photoshop).max(), np.abs(eased - photoshop).max()


# Each card's stat bar at 100% smoothness: its stops, and Photoshop's ramp at
# a location, read off the card's export in the document's profile.
_JEH_OUL: tuple[Stops, Readings] = (
    [
        ((15.0, 67.0, 240.0), 0),
        ((132.0, 124.0, 241.0), 819),
        ((251.0, 28.0, 60.0), 2048),
        ((0.0, 0.0, 0.0), 4096),
    ],
    [
        (813, (131, 125, 241)),
        (1073, (163, 114, 213)),
        (1333, (200, 92, 172)),
        (1918, (251, 37, 76)),
        (2178, (245, 24, 53)),
        (2438, (223, 18, 40)),
        (2698, (190, 13, 30)),
        (2958, (151, 9, 22)),
        (3218, (109, 6, 15)),
    ],
)
_FUZZY_HORDE: tuple[Stops, Readings] = (
    [((255.0, 255.0, 255.0), 0), ((203.0, 8.0, 20.0), 2523), ((43.0, 1.0, 1.0), 4096)],
    [
        (813, (246, 185, 189)),
        (1073, (242, 155, 160)),
        (1333, (238, 123, 130)),
        (1593, (232, 92, 101)),
        (1853, (226, 63, 73)),
        (2113, (218, 37, 49)),
        (2373, (209, 17, 29)),
    ],
)
# Springtrap's bar: a stop hidden under the first one still lends its colour.
_SPRINGTRAP: tuple[Stops, Readings] = (
    [
        ((159.0, 126.0, 26.0), 0),
        ((229.0, 114.0, 13.0), 0),
        ((245.0, 214.0, 69.0), 4096),
    ],
    [
        (1983, (241, 161, 39)),
        (2276, (242, 170, 44)),
        (2536, (242, 178, 49)),
        (2796, (243, 186, 53)),
        (3056, (243, 193, 57)),
        (3316, (244, 200, 61)),
    ],
)
# Spuder's bar: the red would fall below its stop and is held there instead.
_SPUDER: tuple[Stops, Readings] = (
    [((232.0, 216.0, 145.0), 0), ((82.0, 48.0, 35.0), 0), ((72.0, 65.0, 55.0), 4096)],
    [
        (813, (72, 48, 35)),
        (1333, (72, 48, 35)),
        (1853, (72, 48, 37)),
        (2373, (72, 49, 42)),
        (2893, (72, 55, 47)),
        (3153, (72, 58, 49)),
    ],
)


@pytest.mark.parametrize(
    ("card", "tolerance", "eased_miss"),
    [
        (_JEH_OUL, 1.5, 20.0),
        (_FUZZY_HORDE, 1.0, 11.0),
        (_SPRINGTRAP, 1.0, 4.0),
        (_SPUDER, 1.0, 8.0),
    ],
    ids=["Jeh'Oul", "Fuzzy_Horde", "Springtrap", "Spuder"],
)
def test_a_smooth_ramp_runs_through_its_stops_as_photoshops_does(
    card: tuple[Stops, Readings], tolerance: float, eased_miss: float
) -> None:
    """A stop's tangent is half the difference between its neighbours.

    Easing each segment on its own flattens the ramp at every stop instead,
    and misses Photoshop's readings by the levels ``eased_miss`` names.
    """
    ours, eased = _photoshop_misses(*card)
    assert ours < tolerance
    assert eased > eased_miss


def test_the_fixtures_smooth_gradient_stroke_runs_through_its_crest_as_photoshops_does() -> (
    None
):
    """``stroke-effects.psd``'s ``Gradient`` stroke against the composite it saved.

    The Blue-Yellow-Blue ramp runs up column 33 over 30 rows from row 157, so
    its stops land on rows 154, 142 and 130. Easing each segment on its own
    would land halfway to yellow at a segment's midpoint.
    """
    psd = PSDImage.open(full_name("effects/stroke-effects.psd"))
    layer = next(x for x in psd.descendants() if x.name == "Gradient")
    gradient = next(layer.effects.find("stroke")).descriptor[Key.Gradient]
    assert float(gradient[Key.Interpolation]) == 4096.0
    stops: Stops = []
    for stop in gradient[Key.Colors]:
        rgb = stop[Key.Color]
        color = (float(rgb[Key.Red]), float(rgb[Key.Green]), float(rgb[Key.Blue]))
        stops.append((color, int(stop[Key.Location])))
    assert [location for _, location in stops] == [410, 2048, 3686]
    bottom, rows = 157, 30
    pixels = psd.numpy()[:, 33, :3] * 255.0
    readings: Readings = []
    for row in range(130, 155):
        levels = [round(v) for v in pixels[row]]
        location = round((bottom - row) * 4096 / rows)
        readings.append((location, (levels[0], levels[1], levels[2])))
    ours, eased = _photoshop_misses(stops, readings)
    assert ours < 1.0
    assert eased > 15.0


def test_a_smooth_ramp_of_two_stops_has_no_inner_slope_to_carry() -> None:
    """With no inner stop, the ramp is the two-stop ease whatever the colours."""
    stops = [((15.0, 67.0, 240.0), 0), ((251.0, 28.0, 60.0), 4096)]
    ramp = _ramp([_stop(color, location) for color, location in stops])
    along = np.arange(4097) / 4096
    expected = np.array(stops[0][0]) + np.outer(
        _smoothed(along), np.subtract(stops[1][0], stops[0][0])
    )
    assert np.allclose(ramp, expected, atol=1e-3)


def test_a_segment_between_equal_stops_stays_flat() -> None:
    """The_Queen's bar: grey to the same grey, then white, keeps its grey level."""
    grey = (147.0,) * 3
    ramp = _ramp([_stop(grey, 0), _stop(grey, 1937), _stop(WHITE, 2351)])
    assert np.allclose(ramp[: 1937 + 1], 147.0)
    assert np.allclose(ramp[2351:], 255.0)
    assert ramp[2113, 0] == pytest.approx(191.0, abs=1.0)


def test_a_hard_edge_between_equal_stops_is_no_edge() -> None:
    """Two equal stops at one location ramp as one stop does."""
    a, b = (199.0, 117.0, 0.0), (255.0, 240.0, 0.0)
    doubled = _ramp([_stop(a, 0), _stop(a, 0), _stop(b, 4096)])
    single = _ramp([_stop(a, 0), _stop(b, 4096)])
    assert np.allclose(doubled, single, atol=1e-6)


def test_partial_smoothness_mixes_the_linear_and_the_smooth_ramp() -> None:
    stops = [_stop(color, location) for color, location in _JEH_OUL[0]]
    linear, smooth, half = _ramp(stops, 0.0), _ramp(stops, 1.0), _ramp(stops, 0.5)
    assert np.allclose(half, (linear + smooth) / 2, atol=1e-3)


@pytest.mark.parametrize(
    "method",
    [None, GradientMethod.PERCEPTUAL.value, GradientMethod.LINEAR.value],
    ids=["classic", "perceptual", "linear"],
)
def test_a_smooth_ramp_is_held_between_each_segments_stops(
    method: bytes | None,
) -> None:
    """A tangent carried across a stop cannot push a segment past its ends."""
    stops = [
        _stop(BLACK, 0),
        _stop(BLACK, 1638),
        _stop(WHITE, 2048),
        _stop(WHITE, 4096),
    ]
    ramp = _ramp(stops, method=method)
    assert np.allclose(ramp[: 1638 + 1], 0.0, atol=0.5)
    assert np.allclose(ramp[2048:], 255.0, atol=0.5)
    assert np.all(np.diff(ramp[1638:2049, 0]) >= -1e-6)


@pytest.mark.parametrize(
    "method",
    [None, GradientMethod.PERCEPTUAL.value],
    ids=["classic", "perceptual"],
)
def test_a_smooth_transparency_ramp_is_held_between_its_stops(
    method: bytes | None,
) -> None:
    """Opacity never leaves 0-100%, so the perceptual ease has no NaN to make."""
    desc = _gradient(
        transparency=[
            _opacity_stop(100.0, 0),
            _opacity_stop(0.0, 410),
            _opacity_stop(0.0, 4096),
        ],
        method=method,
        smoothness=1.0,
    )
    _, alpha = paint.draw_gradient_fill(
        (0, 0, 4097, 1), ColorMode.RGB, desc, anchor=(0, 0, 4096, 1)
    )
    assert alpha is not None
    assert np.isfinite(alpha).all()
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0
    assert np.allclose(alpha[0, 410:, 0], 0.0)


def test_a_transparency_midpoint_shapes_the_alpha_ramp_alone() -> None:
    desc = _gradient(
        transparency=[_opacity_stop(100.0, 0), _opacity_stop(0.0, 4096, 25)]
    )
    color, shape = _row(ColorMode.RGB, desc)
    assert color is not None and shape is not None
    along = np.arange(101) / 100
    assert np.allclose(shape[0, :, 0], 1.0 - _bent_at(0.25, along), atol=1e-5)
    assert np.allclose(color[0, :, 0], along, atol=1e-5)


# What a black-to-white ramp is halfway along in each method: the classic
# blend is the sRGB mean, perceptual is Oklab lightness 0.5 (linear light
# 0.125) and linear is linear-light 0.5, both re-encoded as sRGB. Smooth and
# an unknown method blend as classic.
HALFWAY_GREY = {
    None: 0.5,
    b"XXXX": 0.5,
    GradientMethod.CLASSIC.value: 0.5,
    GradientMethod.SMOOTH.value: 0.5,
    GradientMethod.PERCEPTUAL.value: 1.055 * 0.125 ** (1 / 2.4) - 0.055,
    GradientMethod.LINEAR.value: 1.055 * 0.5 ** (1 / 2.4) - 0.055,
}


@pytest.mark.parametrize("method", list(HALFWAY_GREY), ids=str)
def test_each_method_blends_in_its_own_space(method: bytes | None) -> None:
    grey = _grey_row(_gradient(method=method))
    assert grey[50] == pytest.approx(HALFWAY_GREY[method], abs=0.002)
    assert grey[0] == pytest.approx(0.0, abs=1e-3) and grey[100] == pytest.approx(
        1.0, abs=1e-3
    )
    assert (np.diff(grey) >= 0).all()


FILL_METHOD_KEY = b"gradientsInterpolationMethod"


def test_an_unknown_method_under_gs99_yields_to_the_fill_layers_key() -> None:
    """Both keys are read in order, skipping a value outside the enumeration."""
    desc = _gradient(method=b"XXXX")
    assert paint._gradient_method(desc) is None
    desc[FILL_METHOD_KEY] = Enumerated(
        typeID=b"gradientInterpolationMethodType",
        enum=GradientMethod.PERCEPTUAL.value,
    )
    assert paint._gradient_method(desc) == GradientMethod.PERCEPTUAL.value
    grey = _grey_row(desc)
    assert grey[50] == pytest.approx(
        HALFWAY_GREY[GradientMethod.PERCEPTUAL.value], abs=0.002
    )


def _fill_layer(method: bytes | None) -> tuple[Layer, Descriptor]:
    """The fixture's 32x32 red foreground-to-transparent fill at 90 degrees.

    A fill layer stores its interpolation method under its own key rather
    than an effect's ``gs99``; ``None`` leaves the key out, as the fixture does.
    """
    psd = PSDImage.open(full_name("layers/gradient-fill.psd"))
    layer = psd[0]
    desc = layer.tagged_blocks.get_data(Tag.GRADIENT_FILL_SETTING)
    if method is not None:
        desc[FILL_METHOD_KEY] = Enumerated(
            typeID=b"gradientInterpolationMethodType", enum=method
        )
    return layer, desc


@pytest.mark.parametrize(
    "method",
    [GradientMethod.CLASSIC.value, GradientMethod.PERCEPTUAL.value],
    ids=str,
)
def test_a_fill_layer_names_its_method_under_its_own_key(
    method: bytes,
) -> None:
    """The fixture's stops forged black to white blend to the method's halfway grey."""
    layer, desc = _fill_layer(method)
    stops = desc[Key.Gradient][Key.Colors]
    for stop, value in zip(stops, (BLACK, WHITE), strict=True):
        rgb = stop[Key.Color]
        rgb[Key.Red], rgb[Key.Green], rgb[Key.Blue] = (Double(c) for c in value)
    color, _ = paint.create_fill(layer, layer.bbox)
    assert color is not None and color.shape == (32, 32, 3)
    grey = color[:, 0, 0]
    assert np.allclose(color, grey[:, None, None])
    # The midpoint sits on the row whose index is half the height.
    assert grey[16] == pytest.approx(HALFWAY_GREY[method], abs=0.003)


def test_a_fill_layers_linear_method_renders_its_fixture_as_photoshop_did() -> None:
    """``adjustment_clipping.psd``'s two fills say ``Lnr `` under the fill key.

    Forced, so the fills are drawn rather than read back; the preview is 8-bit,
    so the render is held to a couple of its steps.
    """
    psd = PSDImage.open(full_name("adjustments/adjustment_clipping.psd"))
    fills = [layer for layer in psd if layer.kind == "gradientfill"]
    methods = [
        layer.tagged_blocks.get_data(Tag.GRADIENT_FILL_SETTING)[FILL_METHOD_KEY].enum
        for layer in fills
    ]
    assert methods == [GradientMethod.LINEAR.value] * 2
    color, _, _ = composite(psd, force=True)
    error = np.mean((psd.numpy()[..., :3] - color[..., :3]) ** 2)
    assert error < (2 / 255) ** 2


def test_a_gradient_stroke_names_its_method_under_the_fill_layers_key() -> None:
    """``stroke-effects.psd``'s ``Gradient`` stroke, its stops forged black to white.

    Under either key the stroke's paint is the same perceptual ramp, which the
    classic one it reads without a key is not.
    """
    psd = PSDImage.open(full_name("effects/stroke-effects.psd"))
    layer = next(x for x in psd.descendants() if x.name == "Gradient")
    desc = next(layer.effects.find("stroke")).descriptor
    assert desc[Key.PaintType].enum == Enum.GradientFill.value
    assert float(desc[Key.Angle]) == 90.0 and FILL_METHOD_KEY not in desc
    stops = List()
    stops.extend([_stop(BLACK, 0), _stop(WHITE, 4096)])
    desc[Key.Gradient][Key.Colors] = stops
    viewport = layer.bbox
    shape = np.ones((viewport[3] - viewport[1], viewport[2] - viewport[0], 1))

    def paint_of(key: bytes | None) -> np.ndarray:
        desc.pop(b"gs99", None)
        desc.pop(FILL_METHOD_KEY, None)
        if key is not None:
            desc[key] = Enumerated(
                typeID=b"gradientInterpolationMethodType",
                enum=GradientMethod.PERCEPTUAL.value,
            )
        color, _ = draw_stroke_effect(viewport, shape, desc, psd)
        return color

    classic, stroke_key, overlay_key = (
        paint_of(key) for key in (None, FILL_METHOD_KEY, b"gs99")
    )
    assert np.array_equal(stroke_key, overlay_key)
    assert not np.array_equal(stroke_key, classic)
    grey = stroke_key[:, 0, 0]
    assert np.allclose(stroke_key, grey[:, None, None])
    halfway = grey[len(grey) // 2]
    assert halfway == pytest.approx(
        HALFWAY_GREY[GradientMethod.PERCEPTUAL.value], abs=0.003
    )


@pytest.mark.parametrize(
    "method",
    [GradientMethod.PERCEPTUAL.value, GradientMethod.LINEAR.value],
)
def test_a_cmyk_documents_own_inks_land_on_the_ramp_whatever_the_method(
    method: bytes,
) -> None:
    """A fill of one CMYK colour is that colour, as the canvas stores it."""
    ink = (60.0, 40.0, 40.0, 100.0)
    stops = [_cmyk_stop(ink, 0), _cmyk_stop(ink, 4096)]
    color, _ = paint.draw_gradient_fill(
        (0, 0, 3, 1), ColorMode.CMYK, _gradient(stops=stops, method=method)
    )
    assert color is not None
    canvas = [1.0 - v / 100.0 for v in ink]
    assert np.allclose(color, np.array(canvas)[None, None, :], atol=1e-6)


@pytest.mark.parametrize(
    ("color_mode", "stops"),
    [
        (
            ColorMode.CMYK,
            [
                _cmyk_stop((100.0, 0.0, 0.0, 0.0), 0),
                _cmyk_stop((0.0, 0.0, 0.0, 100.0), 4096),
            ],
        ),
        (
            ColorMode.LAB,
            [
                _lab_stop((50.0, 120.0, -120.0), 0),
                _lab_stop((80.0, -100.0, 90.0), 4096),
            ],
        ),
    ],
    ids=["cmyk", "lab"],
)
def test_stops_native_to_a_non_rgb_canvas_land_on_the_ramps_ends(
    color_mode: ColorMode, stops: list[Descriptor]
) -> None:
    """The ends are the stops as the canvas stores them, whatever the method."""
    perceptual, _ = _row(
        color_mode,
        _gradient(stops=stops, method=GradientMethod.PERCEPTUAL.value),
    )
    assert perceptual is not None
    ends = [np.array(paint._get_color(color_mode, stop)) for stop in stops]
    assert np.allclose(perceptual[0, 0], ends[0], atol=1e-6)
    assert np.allclose(perceptual[0, 100], ends[1], atol=1e-6)
    # The transparency stops carry no colour, so they are eased all the same.
    fade = [_opacity_stop(100.0, 0), _opacity_stop(0.0, 4096)]
    _, classic_shape = _row(color_mode, _gradient(stops=stops, transparency=fade))
    _, eased_shape = _row(
        color_mode,
        _gradient(
            stops=stops,
            transparency=fade,
            method=GradientMethod.PERCEPTUAL.value,
        ),
    )
    assert classic_shape is not None and eased_shape is not None
    assert np.allclose(eased_shape, _eased(classic_shape), atol=1e-5)


def test_rgb_stops_in_a_cmyk_document_still_blend_perceptually() -> None:
    """Their trip through sRGB is the classic path's too, so only the middle moves."""
    classic, _ = _row(ColorMode.CMYK, _gradient())
    perceptual, _ = _row(
        ColorMode.CMYK,
        _gradient(method=GradientMethod.PERCEPTUAL.value),
    )
    assert classic is not None and perceptual is not None
    assert np.allclose(perceptual[0, 0], classic[0, 0], atol=1e-6)
    assert np.allclose(perceptual[0, 100], classic[0, 100], atol=1e-6)
    assert not np.allclose(perceptual[0, 50], classic[0, 50], atol=0.01)


def _grey_stop(black: float, location: int) -> Descriptor:
    """A ``Grsc`` stop; ``Gry `` is percent black."""
    color = Descriptor(classID=Klass.Grayscale.value)
    color[Key.Gray] = Double(black)
    stop = _stop(BLACK, location)
    stop[Key.Color] = color
    return stop


@pytest.mark.parametrize(
    "method",
    [GradientMethod.PERCEPTUAL.value, GradientMethod.LINEAR.value],
)
def test_grey_stops_on_a_grayscale_canvas_blend_as_their_rgb_spelling_does(
    method: bytes,
) -> None:
    """A grey survives the trip through sRGB, so how it is stored changes nothing."""
    as_grey = [_grey_stop(100.0, 0), _grey_stop(0.0, 4096)]
    from_grey, _ = _row(ColorMode.GRAYSCALE, _gradient(stops=as_grey, method=method))
    from_rgb, _ = _row(ColorMode.GRAYSCALE, _gradient(method=method))
    assert from_grey is not None and from_rgb is not None
    assert from_grey.shape == (1, 101, 1)
    assert np.allclose(from_grey, from_rgb, atol=1e-6)
    assert from_grey[0, 50, 0] == pytest.approx(HALFWAY_GREY[method], abs=0.002)


def _eased(opacity: np.ndarray) -> np.ndarray:
    """Photoshop's perceptual fade of a linearly blended opacity."""
    return opacity ** (1.0 + 1.42 * (1.0 - opacity))


def test_a_perceptual_fill_fades_along_the_fitted_ease() -> None:
    """Perceptual bends the classic opacity blend to Photoshop's convex fade.

    The fixture's own smoothness shapes the classic blend; the ease is a
    function of that blend, whatever shape it has.
    """
    layer, desc = _fill_layer(None)
    _, classic = paint.create_fill(layer, layer.bbox)
    desc[FILL_METHOD_KEY] = Enumerated(
        typeID=b"gradientInterpolationMethodType",
        enum=GradientMethod.PERCEPTUAL.value,
    )
    _, perceptual = paint.create_fill(layer, layer.bbox)
    assert classic is not None and perceptual is not None
    blend = classic[:, 0, 0]
    assert blend[0] == pytest.approx(0.0), "the top row is on the clear stop"
    assert 0.95 < blend[-1] < 1.0, "the bottom row is a row short of opaque"
    assert (np.diff(blend) > 0).all()
    assert np.allclose(perceptual[:, 0, 0], _eased(blend), atol=1e-5)
    assert np.allclose(perceptual, perceptual[:, :1, :])


@pytest.mark.parametrize(
    "ascending", [False, True], ids=["opaque_first", "transparent_first"]
)
def test_a_perceptual_transparency_ramp_is_the_eased_classic_one(
    ascending: bool,
) -> None:
    """The ease is a function of the blended opacity, so a mirrored ramp mirrors."""
    locations = (4096, 0) if ascending else (0, 4096)
    stops = [_opacity_stop(100.0, locations[0]), _opacity_stop(0.0, locations[1])]
    _, classic = _row(ColorMode.RGB, _gradient(transparency=stops))
    _, perceptual = _row(
        ColorMode.RGB,
        _gradient(transparency=stops, method=GradientMethod.PERCEPTUAL.value),
    )
    assert classic is not None and perceptual is not None
    along = np.arange(101) / 100
    assert np.allclose(classic[0, :, 0], along if ascending else 1.0 - along, atol=1e-5)
    assert np.allclose(perceptual[0, :, 0], _eased(classic[0, :, 0]), atol=1e-5)
    assert perceptual[0, 50, 0] == pytest.approx(0.5 ** (1 + 1.42 * 0.5), abs=1e-5)


def test_a_perceptual_ramp_between_partial_opacities_still_lands_on_its_stops() -> None:
    """What the ease's encode/decode pair guarantees between any two stops.

    The ends stay put and the ramp stays monotone and convex between them.
    Only the 100% to 0% fade is checked against Photoshop; this is the
    construction's own contract.
    """
    stops = [_opacity_stop(80.0, 0), _opacity_stop(20.0, 4096)]
    _, shape = _row(
        ColorMode.RGB,
        _gradient(transparency=stops, method=GradientMethod.PERCEPTUAL.value),
    )
    assert shape is not None
    ramp = shape[0, :, 0]
    assert ramp[0] == pytest.approx(0.8, abs=1e-4)
    assert ramp[100] == pytest.approx(0.2, abs=1e-4)
    assert (np.diff(ramp) <= 1e-6).all()
    # Convex: the fade sits below the straight line between its stops.
    assert ramp[50] < 0.5


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


BLENDED_METHODS = [GradientMethod.PERCEPTUAL.value, GradientMethod.LINEAR.value]


@pytest.mark.parametrize("method", BLENDED_METHODS)
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


# Rich black ink, the CMYK example in ``paint._get_color``'s docstring, and a
# Lab purple outside the sRGB gamut: neither survives a trip through sRGB.
RICH_BLACK = (83.04, 74.03, 80.99, 58.3)
LAB_PURPLE = (50.0, 80.0, -80.0)


@pytest.mark.parametrize("method", BLENDED_METHODS)
@pytest.mark.parametrize(
    ("color_mode", "stops"),
    [
        (ColorMode.CMYK, [_cmyk_stop(RICH_BLACK, 0), _cmyk_stop((0, 0, 0, 0), 4096)]),
        (ColorMode.LAB, [_lab_stop(LAB_PURPLE, 0), _lab_stop((100, 0, 0), 4096)]),
    ],
    ids=["cmyk", "lab"],
)
def test_a_ramp_blended_outside_the_canvas_space_keeps_the_stops_own_values(
    color_mode: ColorMode, stops: list[Descriptor], method: bytes
) -> None:
    """The stops sit where the classic blend puts them; only the way between moves."""
    classic, _ = _row(color_mode, _gradient(stops=stops))
    blended, _ = _row(color_mode, _gradient(stops=stops, method=method))
    assert classic is not None and blended is not None
    assert np.allclose(blended[0, 0], classic[0, 0], atol=1e-6)
    assert np.allclose(blended[0, 100], classic[0, 100], atol=1e-6)
    assert np.abs(blended[0, 50] - classic[0, 50]).max() > 0.01


def test_an_overlay_not_aligned_with_its_layer_is_laid_out_on_the_document() -> None:
    psd = PSDImage.open(full_name("layer_effects.psd"))
    layer = next(x for x in psd if x.name == "Gradient Overlay")
    desc = next(layer.effects.find("gradientoverlay")).descriptor
    assert layer.bbox == (99, 422, 699, 482) and psd.viewbox == (0, 0, 800, 640)

    assert paint.gradient_anchor(psd, desc) is None, "aligned by default"
    aligned, _ = _draw_gradient_overlay(layer, desc, 3, layer.bbox, layer.bbox)
    desc[Key.Alignment] = Bool(False)
    assert paint.gradient_anchor(psd, desc) == psd.viewbox
    unaligned, _ = _draw_gradient_overlay(layer, desc, 3, layer.bbox, layer.bbox)
    on_document, _ = paint.draw_gradient_fill(
        layer.bbox, psd.color_mode, desc, psd.viewbox
    )
    assert aligned is not None and unaligned is not None and on_document is not None
    assert np.array_equal(unaligned, on_document)
    assert not np.array_equal(unaligned, aligned)


def _soft_edged_layer() -> tuple[PSDImage, PixelLayer]:
    """A white 12x10 layer at (4, 3) whose rim is under half opaque.

    Its outer ring is a quarter opaque and the next one just over half, but
    for that ring's bottom row, just under at 127.
    """
    psd = PSDImage.new("RGBA", (20, 16), color=(0, 0, 0, 0))
    alpha = np.full((10, 12), 64, dtype=np.uint8)
    alpha[1:9, 1:11] = 128
    alpha[8, 1:11] = 127
    alpha[2:8, 2:10] = 255
    rgba = np.full((10, 12, 4), 255, dtype=np.uint8)
    rgba[..., 3] = alpha
    layer = PixelLayer.frompil(Image.fromarray(rgba, "RGBA"), psd, top=3, left=4)
    psd.append(layer)
    return psd, layer


def test_a_pixel_layers_overlay_is_laid_out_on_its_over_half_opaque_pixels() -> None:
    """The rim under half opaque stays out of the box the ramp spans.

    Black to white at 90 degrees: white on the top row over half opaque,
    a seventh of the way per row down the seven such rows, black below.
    """
    psd, layer = _soft_edged_layer()
    assert layer.bbox == (4, 3, 16, 13)
    opaque = (5, 4, 15, 11)
    assert _opaque_bounds(layer, layer.bbox) == opaque
    desc = _gradient(angle=90.0)
    color, _ = _draw_gradient_overlay(layer, desc, 3, layer.bbox, layer.bbox)
    expected, _ = paint.draw_gradient_fill(layer.bbox, psd.color_mode, desc, opaque)
    assert color is not None and expected is not None
    assert np.array_equal(color, expected)
    rows = np.clip(1.0 - (np.arange(10) - 1) / 7, 0.0, 1.0)
    assert np.allclose(color[..., 0], rows[:, None], atol=1e-6)


def test_an_overlay_keeps_its_layout_without_pixels_over_half_opaque() -> None:
    """A layer under half opaque throughout keeps the box given."""
    psd, _ = _soft_edged_layer()
    faint = PixelLayer.frompil(Image.new("RGBA", (6, 6), (255, 255, 255, 127)), psd)
    assert _opaque_bounds(faint, faint.bbox) == faint.bbox


def test_a_type_layers_overlay_is_laid_out_on_its_over_half_opaque_pixels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A glyph rim under half opaque stays out of the box, as on a pixel layer.

    The text's top row is a quarter opaque and its right column a third, so
    the box loses both; its bottom row, at 128, stays in.
    """
    psd = PSDImage.open(full_name("layer_effects.psd"))
    text = next(x for x in psd if x.name == "Gradient Overlay")
    assert text.kind == "type"
    left, top, right, bottom = text.bbox
    shape = np.ones((bottom - top, right - left, 1), dtype=np.float32)
    shape[0] = 64 / 255
    shape[:, -1] = 85 / 255
    shape[-1, :-1] = 128 / 255
    monkeypatch.setattr(text, "numpy", lambda *args, **kwargs: shape)
    opaque = (left, top + 1, right - 1, bottom)
    assert _opaque_bounds(text, text.bbox) == opaque
    desc = _gradient(angle=90.0)
    color, _ = _draw_gradient_overlay(text, desc, 3, text.bbox, text.bbox)
    expected, _ = paint.draw_gradient_fill(text.bbox, psd.color_mode, desc, opaque)
    assert color is not None and expected is not None
    assert np.array_equal(color, expected)


def test_an_overlay_keeps_its_layout_when_its_pixels_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, layer = _soft_edged_layer()

    def unreadable(*args: object, **kwargs: object) -> None:
        raise ValueError("corrupt channel")

    monkeypatch.setattr(layer, "numpy", unreadable)
    assert _opaque_bounds(layer, layer.bbox) == layer.bbox


def test_an_overlay_not_aligned_with_its_layer_leaves_its_pixels_unread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    psd, layer = _soft_edged_layer()
    desc = _gradient(angle=90.0)
    desc[Key.Alignment] = Bool(False)
    expected, _ = paint.draw_gradient_fill(
        layer.bbox, psd.color_mode, desc, psd.viewbox
    )

    def unread(*args: object, **kwargs: object) -> None:
        raise AssertionError("read the layer's pixels")

    monkeypatch.setattr(layer, "numpy", unread)
    color, _ = _draw_gradient_overlay(layer, desc, 3, layer.bbox, layer.bbox)
    assert color is not None and expected is not None
    assert np.array_equal(color, expected)


def test_a_gradient_stroke_effect_not_aligned_with_its_layer_is_laid_out_on_the_document() -> (
    None
):
    """``layer_params.psb``'s caption has a gradient stroke with the box unticked."""
    psd = PSDImage.open(full_name("layer_params.psb"))
    layer = next(x for x in psd.descendants() if x.name == "Test caption")
    desc = next(layer.effects.find("stroke")).descriptor
    assert desc[Key.PaintType].enum == Enum.GradientFill and not desc[Key.Alignment]
    bbox = effects.stroke_bbox(layer.bbox, desc, None)
    shape = np.ones((bbox[3] - bbox[1], bbox[2] - bbox[0], 1), dtype=np.float32)
    color, _ = effects.draw_stroke_effect(bbox, shape, desc, psd)
    on_document, _ = paint.draw_gradient_fill(bbox, psd.color_mode, desc, psd.viewbox)
    on_box, _ = paint.draw_gradient_fill(bbox, psd.color_mode, desc)
    assert on_document is not None and on_box is not None
    assert np.array_equal(color, on_document)
    assert not np.array_equal(color, on_box)


def test_a_vector_stroke_gradient_not_aligned_with_its_layer_is_laid_out_on_the_document() -> (
    None
):
    psd = PSDImage.open(full_name("effects/stroke-effects.psd"))
    layer = next(x for x in psd.descendants() if x.name == "Stroke OutsetFrame")
    desc = layer.tagged_blocks.get_data(Tag.VECTOR_STROKE_CONTENT_DATA)
    assert Key.Gradient in desc and Key.Alignment not in desc
    # The fixture's ramp fades red to clear, so its layout shows in the shape.
    _, on_layer = paint.create_fill(layer, layer.bbox)
    desc[Key.Alignment] = Bool(False)
    color, on_document = paint.create_fill(layer, layer.bbox)
    expected_color, expected = paint.draw_gradient_fill(
        layer.bbox, psd.color_mode, desc, psd.viewbox
    )
    assert on_layer is not None and on_document is not None and expected is not None
    assert color is not None and expected_color is not None
    assert np.array_equal(color, expected_color)
    assert np.array_equal(on_document, expected)
    assert not np.array_equal(on_document, on_layer)


def test_a_shape_stroke_gradient_not_aligned_with_its_layer_is_laid_out_on_the_document() -> (
    None
):
    psd = PSDImage.open(full_name("stroke.psd"))
    layer = next(x for x in psd if x.name == "Shape 1")
    assert layer.stroke is not None
    desc = layer.stroke._data[b"strokeStyleContent"]
    assert desc.classID == b"gradientLayer" and desc[Key.Alignment]
    anchor = _path_bbox(layer)
    on_path, _ = paint.create_fill_desc(layer, desc, psd.viewbox, anchor)
    desc[Key.Alignment] = Bool(False)
    on_document, _ = paint.create_fill_desc(layer, desc, psd.viewbox, anchor)
    expected, _ = paint.draw_gradient_fill(psd.viewbox, psd.color_mode, desc)
    assert on_path is not None and on_document is not None and expected is not None
    assert np.array_equal(on_document, expected)
    assert not np.array_equal(on_document, on_path)


def test_a_wide_overlay_at_a_near_vertical_angle_matches_photoshop() -> None:
    """A 600x60 overlay at 87 degrees against Photoshop's own render.

    Its ramp is as long as the box is tall, so the whole of it shows across
    the 60 rows.
    """
    psd = PSDImage.open(full_name("layer_effects.psd"))
    layer = next(x for x in psd if x.name == "Gradient Overlay")
    assert layer.bbox == (99, 422, 699, 482)
    left, top, right, bottom = layer.bbox
    reference = psd.numpy()[top:bottom, left:right]
    color, _, _ = composite(psd, force=True)
    assert reference.shape[2] == color.shape[2]
    error = np.mean((reference - color[top:bottom, left:right]) ** 2)
    ramp_across_the_width = 0.00082
    assert error < ramp_across_the_width / 4
