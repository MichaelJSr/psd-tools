"""Paint and fill operations for compositing."""

import logging
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from psd_tools.api import numpy_io
from psd_tools.api.utils import check_pixel_size
from psd_tools.color_convert import (
    SRGB_CUTOFF,
    SRGB_GAMMA,
    SRGB_OFFSET,
    SRGB_SLOPE,
    cmyk_to_rgb,
    gray_to_cmyk,
    gray_to_rgb,
    hsb_to_rgb,
    lab_to_rgb,
    rgb_to_cmyk,
    rgb_to_grayscale,
    rgb_to_lab,
)
from psd_tools.composite._compat import require_scipy
from psd_tools.composite.utils import is_fill_disabled
from psd_tools.constants import GRADIENT_METHOD_KEY, ColorMode, GradientMethod, Tag
from psd_tools.psd.descriptor import Descriptor
from psd_tools.terminology import Enum, Key, Klass, Type

if TYPE_CHECKING:
    from psd_tools.api.layers import Layer
    from psd_tools.api.protocols import PSDProtocol

logger = logging.getLogger(__name__)

# The key a gradient fill layer or stroke effect stores its method under;
# not in the terminology either.
_FILL_METHOD_KEY = b"gradientsInterpolationMethod"

# Photoshop's range for a pattern's scale, in percent.
_MIN_SCALE = 1.0
_MAX_SCALE = 1000.0

# Per pixel, what a gradient fill peaks at: its coordinate grids and ramp
# planes outweigh the float32 colour it returns.
GRADIENT_FILL_BYTES = 128


# The modes whose color array is a single channel, so a descriptor color has to
# be reduced to one component to be a legal source for them. Grayscale, bitmap
# and duotone genuinely are one grayscale channel -- duotone's inks live in the
# color mode data section and were never a channel count (#733). Multichannel
# is here for a different reason: its channels are spot inks with no colorimetric relation to RGB, so
# there is no conversion to N of them -- one channel is the honest answer, and
# what a single channel means once widened to N is #722's question -- #749's
# where the single channel is a source, as this one is -- not this function's.
_SINGLE_CHANNEL_MODES = (
    ColorMode.BITMAP,
    ColorMode.GRAYSCALE,
    ColorMode.DUOTONE,
    ColorMode.MULTICHANNEL,
)


def _clamp01(value: float) -> float:
    """Hold *value* inside the range a color array may carry (#757).

    A descriptor component is unconstrained, and ``utils.clip()`` runs too late
    once the color is blended. ``composite_pil()``'s uint8 clip is independent
    on purpose; do not drop this one because of it. Applied where the untrusted
    number enters, so ``color_convert`` keeps its ``[0, 1]`` input contracts.

    NaN maps to 0.0 because ``max`` keeps its first argument when ``nan > 0.0``
    is false; do not reverse the argument order.
    """
    return min(1.0, max(0.0, value))


def _lab_to_canvas(lightness: float, a: float, b: float) -> tuple[float, ...]:
    """Encode native CIE L*a*b* into the compositor's Lab color array.

    PIL mode "LAB" stores ``L * 255/100`` with the chroma axes offset by 128, so
    this is a relabelling, not a conversion (#743). Shared by a Lab descriptor
    on a Lab document and by other color classes via
    :py:func:`psd_tools.color_convert.rgb_to_lab` (#752).
    """
    return (
        _clamp01(lightness / 100.0),
        _clamp01((a + 128.0) / 255.0),
        _clamp01((b + 128.0) / 255.0),
    )


def _ink_to_canvas(ink: tuple[float, ...]) -> tuple[float, ...]:
    """Invert an ink-space CMYK tuple into the compositor's canvas convention.

    Ink space has white at all zeros; the canvas stores what is *left*, so 1.0
    is no ink (#747). Only conversions *into* CMYK need this; ``_get_cmyk()``
    already lands in canvas space.
    """
    return tuple(1.0 - v for v in ink)


def _from_rgb(color_mode: ColorMode, rgb: tuple[float, ...]) -> tuple[float, ...]:
    """Convert a canonical RGB triple to *color_mode*'s color array width.

    Every descriptor color class goes through here, so a fill matches the
    document's array width, which ``Compositor._assert_source_fits()`` checks.
    Indexed is three because its stored channel expands through the palette.
    Lab is a real conversion, not a width choice (#752).
    """
    if color_mode == ColorMode.CMYK:
        return _ink_to_canvas(rgb_to_cmyk(*rgb))
    if color_mode == ColorMode.LAB:
        return _lab_to_canvas(*rgb_to_lab(*rgb))
    if color_mode in _SINGLE_CHANNEL_MODES:
        return (rgb_to_grayscale(*rgb),)
    return rgb


def _get_color(color_mode: ColorMode, desc: Descriptor) -> tuple[float, ...]:
    r"""Return color tuple from descriptor.

    Example descriptor::

        Descriptor(b'solidColorLayer'){
            'Clr ': Descriptor(b'CMYC'){
                'Cyn ': 83.04,
                'Mgnt': 74.03,
                'Ylw ': 80.99,
                'Blck': 58.3
                }
            }

        Descriptor(b'solidColorLayer'){
            'Clr ': Descriptor(b'RGBC'){
                'Rd  ': 235.90926200151443,
                'Grn ': 232.29671984910965,
                'Bl  ': 25.424751117825508,
                'Bk  ': 'PANTONE+® Solid Coated\x00',
                'Nm  ': 'PANTONE 395 C\x00',
                'bookID': 3060,
                'bookKey': RawData(value=b'1123SC')
                }
            }
    """

    def _get_int_color(color_desc: Descriptor, keys: tuple) -> tuple[float, ...]:
        return tuple(_clamp01(float(color_desc[key]) / 255.0) for key in keys)

    def _get_rgb(color_mode: ColorMode, color_desc: Descriptor) -> tuple[float, ...]:
        if Key.Red in color_desc:
            rgb = _get_int_color(color_desc, (Key.Red, Key.Green, Key.Blue))
        else:
            # No divisor, unlike ``Rd  ``/``Grn ``/``Bl  ``: these components
            # are the format's own normalized spelling. Nothing under
            # tests/psd_files carries one, so that scale is taken on the
            # format's word rather than measured here -- which is exactly why
            # the clamp matters on this path. If the scale is what it claims,
            # the clamp never fires; if it is not, an unexpected value
            # saturates instead of wrapping to an unrelated colour (#757).
            rgb = tuple(
                _clamp01(float(color_desc[key]))
                for key in (Key.RedFloat, Key.GreenFloat, Key.BlueFloat)
            )
        return _from_rgb(color_mode, rgb)

    def _get_hsb(color_mode: ColorMode, color_desc: Descriptor) -> tuple[float, ...]:
        # ``H   `` is an angle in degrees, so the full turn is 360 and not 300
        # (#754).
        hue = float(color_desc[Key.Hue]) / 360.0
        # Not clamped, unlike the other classes. Hue is cyclic, so 400 deg
        # names a real angle and ``hsb_to_rgb`` wraps it; clamping would turn
        # it into 360. Saturation and brightness are clamped by ``hsb_to_rgb``
        # itself, which is total, so guarding them again here would defend the
        # same two numbers twice (#757).
        saturation = float(color_desc[Key.Saturation]) / 100.0
        brightness = float(color_desc[Key.Brightness]) / 100.0
        return _from_rgb(color_mode, hsb_to_rgb(hue, saturation, brightness))

    def _get_gray(color_mode: ColorMode, x: Descriptor) -> tuple[float, ...]:
        # ``Gry `` is percent *black*, so 0 is white. Inverting it yields a
        # luminance, which is what every helper below takes -- a real
        # conversion, not the canvas-convention flip that ``_ink_to_canvas()``
        # performs.
        gray = _clamp01((100.0 - float(x[Key.Gray])) / 100.0)
        if color_mode == ColorMode.RGB:
            return gray_to_rgb(gray)
        if color_mode == ColorMode.CMYK:
            return _ink_to_canvas(gray_to_cmyk(gray))
        if color_mode == ColorMode.LAB:
            # Not left to widen(): one channel is a legal width, so a grey fill
            # was reaching a Lab canvas as a bare lightness and being widened
            # with a neutral a/b. That put it on the right axis but at the wrong
            # height -- the grey itself rather than its L* -- and disagreed with
            # the same grey written as an RGB descriptor (#752).
            return _from_rgb(color_mode, gray_to_rgb(gray))
        return (gray,)

    def _get_cmyk(color_mode: ColorMode, x: Descriptor) -> tuple[float, ...]:
        # Read as ink -- 0.0 is no ink -- which is both the descriptor's own
        # spelling and ``color_convert``'s documented contract. The compositor's
        # arrays store what is *left*, so only the CMYK branch flips, and it
        # says so by name (#747, #763).
        ink = tuple(
            _clamp01(float(x[key]) / 100.0)
            for key in (Key.Cyan, Key.Magenta, Key.Yellow, Key.Black)
        )
        if color_mode == ColorMode.CMYK:
            return _ink_to_canvas(ink)
        return _from_rgb(color_mode, cmyk_to_rgb(*ink))

    def _get_lab(color_mode: ColorMode, x: Descriptor) -> tuple[float, ...]:
        lightness = float(x[Key.Luminance])
        a = float(x[Key.A])
        b = float(x[Key.B])
        if color_mode == ColorMode.LAB:
            # Straight into the array's own encoding, with no trip through RGB
            # to lose anything on (#743).
            return _lab_to_canvas(lightness, a, b)
        # Everything else is a real conversion, through sRGB: the chroma axes
        # are signed, so neither the raw triple nor ``L`` alone stands in for
        # the colour (the second half of #743).
        return _from_rgb(color_mode, lab_to_rgb(lightness, a, b))

    _COLOR_FUNC = {
        Klass.RGBColor: _get_rgb,
        Klass.Grayscale: _get_gray,
        Klass.CMYKColor: _get_cmyk,
        Klass.LabColor: _get_lab,
        Klass.HSBColor: _get_hsb,
    }
    color_desc = desc.get(Key.Color)
    if not color_desc:
        raise ValueError(f"Could not find a color descriptor {desc}")
    return _COLOR_FUNC[color_desc.classID](color_mode, color_desc)


def create_fill_desc(
    layer: "Layer",
    desc: Descriptor,
    viewport: tuple[int, int, int, int],
    anchor: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Create a fill image, laid out on ``anchor`` and drawn on ``viewport``."""
    if desc.classID == b"solidColorLayer":
        return draw_solid_color_fill(viewport, layer._psd.color_mode, desc)
    if desc.classID == b"patternLayer":
        return draw_pattern_fill(viewport, layer._psd, desc, anchor)
    if desc.classID == b"gradientLayer":
        return draw_gradient_fill(
            viewport,
            layer._psd.color_mode,
            desc,
            gradient_anchor(layer._psd, desc, anchor),
        )
    return None, None


def gradient_anchor(
    psd: "PSDProtocol",
    desc: Descriptor,
    box: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int] | None:
    """The box a gradient is laid out on: ``box``, or the document's with ``Algn`` off.

    ``None`` lays the gradient out on the viewport it is drawn on.
    """
    if bool(desc.get(Key.Alignment, True)):
        return box
    return psd.viewbox


def pattern_origin(
    desc: Descriptor,
    box: tuple[int, int, int, int],
    reference_point: Sequence[float] | None = None,
) -> tuple[int, int]:
    """Where a pattern's tiling starts, in document coordinates.

    Aligned with the layer, it starts at the layer's reference point, or at
    ``box``'s top-left without one, moved by the phase. With ``Algn`` off the
    phase is measured from the document's origin.
    """
    phase = desc.get(b"phase")
    shift = (0.0, 0.0)
    if phase is not None:
        shift = (
            float(phase.get(Key.Horizontal, 0.0)),
            float(phase.get(Key.Vertical, 0.0)),
        )
    start = (0.0, 0.0)
    if bool(desc.get(Key.Alignment, True)):
        start = (
            (float(box[0]), float(box[1]))
            if reference_point is None
            else (float(reference_point[0]), float(reference_point[1]))
        )
    # Rounded to the pixel grid the tiling is drawn on.
    return (
        math.floor(start[0] + shift[0] + 0.5),
        math.floor(start[1] + shift[1] + 0.5),
    )


def create_fill(
    layer: "Layer",
    viewport: tuple[int, int, int, int],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Create a fill image."""
    if is_fill_disabled(layer):
        return None, None
    blocks = layer.tagged_blocks
    if (desc := blocks.get_data(Tag.SOLID_COLOR_SHEET_SETTING)) is not None:
        return draw_solid_color_fill(viewport, layer._psd.color_mode, desc)
    if (desc := blocks.get_data(Tag.PATTERN_FILL_SETTING)) is not None:
        return draw_pattern_fill(viewport, layer._psd, desc)
    if (desc := blocks.get_data(Tag.GRADIENT_FILL_SETTING)) is not None:
        return draw_gradient_fill(
            viewport, layer._psd.color_mode, desc, gradient_anchor(layer._psd, desc)
        )
    if (desc := blocks.get_data(Tag.VECTOR_STROKE_CONTENT_DATA)) is not None:
        if Key.Color in desc:
            return draw_solid_color_fill(viewport, layer._psd.color_mode, desc)
        elif Key.Pattern in desc:
            return draw_pattern_fill(viewport, layer._psd, desc)
        elif Key.Gradient in desc:
            return draw_gradient_fill(
                viewport, layer._psd.color_mode, desc, gradient_anchor(layer._psd, desc)
            )
    return None, None


def _pattern_scale(desc: Descriptor) -> float:
    """The pattern's ``Scl `` as a factor, within Photoshop's 1-1000% range."""
    scale = float(desc.get(Key.Scale, 100.0))
    if not math.isfinite(scale):
        raise ValueError("Pattern scale is not finite.")
    return min(max(scale, _MIN_SCALE), _MAX_SCALE) / 100.0


def _turn_panel(panel: np.ndarray, desc: Descriptor) -> np.ndarray:
    """The panel turned counter-clockwise by ``Angl``, a multiple of 90 degrees.

    Turned about the tiling's origin, the tiles are the turned panel laid
    from that origin. Any other angle is left unturned.
    """
    angle = float(desc.get(Key.Angle, 0.0))
    if not math.isfinite(angle):
        raise ValueError("Pattern angle is not finite.")
    turns = angle / 90.0
    if turns != round(turns):
        logger.debug("Pattern angle %s is not a right angle; not applied", angle)
        return panel
    return np.rot90(panel, int(round(turns)) % 4)


def _reduction_box(scale: float) -> int:
    """The box a pattern reduced by ``scale`` is averaged over: 2**L samples.

    L is the deepest halving that does not reduce the panel past the scale.
    """
    width = 1
    # A percentage over 100 can fall a hair short of the halving it names.
    while width * 2 * scale <= 1.0 + 1e-9:
        width *= 2
    return width


def _box_average(panel: np.ndarray, width: int) -> np.ndarray:
    """Each sample averaged with the ``width - 1`` after it, wrapping, on both axes."""
    for axis in (0, 1):
        work = np.moveaxis(panel, axis, 0)
        size = work.shape[0]
        sums = np.zeros((size + width + 1,) + work.shape[1:], dtype=np.float64)
        np.cumsum(work[np.arange(size + width) % size], axis=0, out=sums[1:])
        total = np.subtract(sums[width : width + size], sums[:size])
        total /= width
        panel = np.moveaxis(total.astype(np.float32), 0, axis)
    return panel


def _box_peak_bytes(panel: np.ndarray, width: int) -> int:
    """What :py:func:`_box_average` holds at its peak, for the shorter axis."""
    size = min(panel.shape[0], panel.shape[1])
    # The wrapped copy, its float64 sums and cumsum's cast of it, at 5x the
    # panel per wrapped row; the float64 totals, the result and the input.
    return -(-panel.nbytes * (5 * (size + width + 1) + 5 * size) // size)


def _tile_axis(
    start: int, length: int, origin: int, size: int, scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Panel indices and weights along one axis for a tiling scaled by ``scale``.

    The sample at ``origin`` is the panel's first, and each output pixel
    blends the two panel samples either side of its scaled position.
    """
    position = (np.arange(start, start + length, dtype=np.float64) - origin) / scale
    first = np.floor(position)
    weight = (position - first).astype(np.float32)
    index = first.astype(np.int64) % size
    return index, (index + 1) % size, weight


def _sample_tiling(
    panel: np.ndarray,
    viewport: tuple[int, int, int, int],
    origin: tuple[int, int],
    scale: float,
) -> np.ndarray:
    """The panel tiled across ``viewport`` from ``origin``, bilinear and wrapping."""
    height, width = viewport[3] - viewport[1], viewport[2] - viewport[0]
    top, bottom, down = _tile_axis(
        viewport[1], height, origin[1], panel.shape[0], scale
    )
    left, right, across = _tile_axis(
        viewport[0], width, origin[0], panel.shape[1], scale
    )
    rows = panel[top] * (1 - down)[:, None, None] + panel[bottom] * down[:, None, None]
    return (
        rows[:, left] * (1 - across)[None, :, None]
        + rows[:, right] * across[None, :, None]
    )


def _sample_peak_bytes(panel: np.ndarray, viewport: tuple[int, int, int, int]) -> int:
    """What :py:func:`_sample_tiling` holds at its peak."""
    height, width = viewport[3] - viewport[1], viewport[2] - viewport[0]
    plane = panel.shape[2] * panel.dtype.itemsize
    # The panel, three row-blend arrays, the output with its two terms, and
    # per row and column two int64 indices, their float64 position and weights.
    return (
        panel.nbytes
        + 3 * height * panel.shape[1] * plane
        + 3 * height * width * plane
        + 48 * (height + width)
    )


def draw_solid_color_fill(
    viewport: tuple[int, int, int, int],
    color_mode: ColorMode,
    desc: Descriptor,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Create a solid color fill."""
    fill = _get_color(color_mode, desc)
    height, width = viewport[3] - viewport[1], viewport[2] - viewport[0]
    color = np.full((height, width, len(fill)), fill, dtype=np.float32)
    return color, None


def draw_pattern_fill(
    viewport: tuple[int, int, int, int],
    psd: Any,
    desc: Descriptor,
    anchor: tuple[int, int, int, int] | None = None,
    origin: tuple[int, int] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    r"""
    Create a pattern fill.

    The tiling starts at ``origin``, in document coordinates, or else at the
    top-left of ``anchor``, which defaults to ``viewport``, so a viewport
    larger than the anchor continues the same tiling. ``Angl`` turns the
    tiles about that point, in right angles only; ``Scl `` scales them about
    it, with a bilinear sample of the panel's box average on a reduction.

    Example descriptor::

        Descriptor(b'patternFill'){
            'enab': True,
            'present': True,
            'showInDialog': True,
            'Md  ': (b'BlnM', b'CBrn'),
            'Opct': 100.0 Percent,
            'Ptrn': Descriptor(b'Ptrn'){
                'Nm  ': 'foo\x00',
                'Idnt': '5e1713ab-e968-4c4c-8855-c8fa2cde8610\x00'
                },
            'Angl': 0.0 Angle,
            'Scl ': 87.0 Percent,
            'Algn': True,
            'phase': Descriptor(b'Pnt '){'Hrzn': 0.0, 'Vrtc': 0.0}
            }

    .. todo:: Test this.
    """
    pattern_id = desc[Enum.Pattern][Key.ID].value.rstrip("\x00")
    pattern = psd._get_pattern(pattern_id)
    if not pattern:
        logger.error("Pattern not found: %s", pattern_id)
        return None, None

    budget = getattr(psd, "_max_alloc_bytes", None)
    panel = numpy_io.get_pattern(pattern, budget)
    assert panel.shape[0] > 0

    scale = _pattern_scale(desc)
    panel = _turn_panel(panel, desc)
    box = _reduction_box(scale)
    if box > 1:
        check_pixel_size(
            panel.shape[1],
            panel.shape[0],
            panel.shape[2],
            budget,
            estimated_bytes=_box_peak_bytes(panel, box),
            warn=False,
        )
        panel = _box_average(panel, box)

    height, width = viewport[3] - viewport[1], viewport[2] - viewport[0]
    if origin is None:
        origin = (
            (viewport[0], viewport[1]) if anchor is None else (anchor[0], anchor[1])
        )
    if height > 0 and width > 0:
        check_pixel_size(
            width,
            height,
            panel.shape[2],
            budget,
            estimated_bytes=_sample_peak_bytes(panel, viewport),
            warn=False,
        )
    # Taken from the pattern's own slot layout rather than from its color mode.
    # A mode-keyed count is only ever right by coincidence -- when the mode's
    # constant happens to equal the width this pattern stored -- and
    # multichannel's is 64, the format's maximum, which no pattern can equal.
    # ``shape[2] > channels`` was therefore never true there, the alpha was
    # never split off, and the array reached the canvas one plane too wide and
    # was rejected as inconsistent with it (#741).
    channels = numpy_io.get_pattern_color_channels(pattern)
    pixels = _sample_tiling(panel, viewport, origin, scale)
    if pixels.shape[2] > channels:
        return pixels[:, :, :channels], pixels[:, :, -1:]
    return pixels, None


@require_scipy
def draw_gradient_fill(
    viewport: tuple[int, int, int, int],
    color_mode: ColorMode,
    desc: Descriptor,
    anchor: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Create a gradient fill image.

    The ramp is laid out across ``anchor``, which defaults to ``viewport``; a
    viewport larger than the anchor continues the ramp past its ends.

    Requires scipy for gradient color interpolation.
    """
    height, width = viewport[3] - viewport[1], viewport[2] - viewport[0]
    box = viewport if anchor is None else anchor
    # A path with no extent along an axis ramps as a single pixel there.
    box_height, box_width = max(box[3] - box[1], 1), max(box[2] - box[0], 1)

    angle = float(desc.get(Key.Angle, 0))
    scale = float(desc.get(Key.Scale, 100.0)) / 100.0
    scale *= _gradient_length(angle, box_width, box_height)
    # ``Ofst`` moves the ramp's centre by a share of the box.
    offset = desc.get(Key.Offset)
    shift_x = box_width * _offset_share(offset, Key.Horizontal)
    shift_y = box_height * _offset_share(offset, Key.Vertical)
    X, Y = np.meshgrid(
        _ramp_axis(viewport[0] - box[0] - shift_x, width, box_width, scale),
        _ramp_axis(viewport[1] - box[1] - shift_y, height, box_height, scale),
    )

    gradient_kind = desc.get(Key.Type).enum
    if gradient_kind == Enum.Linear:
        Z = _make_linear_gradient(X, Y, angle)
    elif gradient_kind == Enum.Radial:
        Z = _make_radial_gradient(X, Y)
    elif gradient_kind == Enum.Angle:
        Z = _make_angle_gradient(X, Y, angle)
    elif gradient_kind == Enum.Reflected:
        Z = _make_reflected_gradient(X, Y, angle)
    elif gradient_kind == Enum.Diamond:
        Z = _make_diamond_gradient(X, Y, angle)
    else:
        # Unsupported: b'shapeburst', only avail in stroke effect
        logger.warning("Unknown gradient style: %s.", gradient_kind)
        Z = np.full((height, width), 0.5, dtype=np.float32)

    Z = np.maximum(0.0, np.minimum(1.0, Z))
    if bool(desc.get(Key.Reverse, False)):
        Z = 1.0 - Z

    G, Ga = _make_gradient_color(
        color_mode, desc.get(Key.Gradient), _gradient_method(desc)
    )
    color = G(Z) if G is not None else None
    shape = np.expand_dims(Ga(Z), 2) if Ga is not None else None
    return color, shape


_METHODS = frozenset(method.value for method in GradientMethod)


def _gradient_method(desc: Descriptor) -> bytes | None:
    """The ``gs99`` method, or the one under a fill layer's or stroke's own key.

    A key holding a value outside :py:class:`GradientMethod` is skipped, as
    ``Stroke.method`` skips it.
    """
    for key in (GRADIENT_METHOD_KEY, _FILL_METHOD_KEY):
        method = getattr(desc.get(key), "enum", None)
        if method in _METHODS:
            return method
    return None


def _gradient_length(angle: float, width: float, height: float) -> float:
    """Length of the line through the box centre at ``angle``, clipped by the box.

    The ramp of a 100% gradient spans exactly this, so a vertical ramp spans
    the height and a horizontal one the width whatever the box's aspect.
    """
    theta = math.radians(angle)
    cosine, sine = abs(math.cos(theta)), abs(math.sin(theta))
    if cosine < 1e-9:
        return float(height)
    if sine < 1e-9:
        return float(width)
    return min(width / cosine, height / sine)


# ``Ofst`` percent beyond this leaves the box wholly past the ramp at any scale
# the dialog allows; the clamp keeps a forged offset finite in float32.
_OFFSET_LIMIT = 10_000.0


def _offset_share(point: Descriptor | None, key: bytes) -> float:
    """One axis of ``Ofst`` as a share of the box; NaN and infinity read as none."""
    if point is None:
        return 0.0
    value = float(point.get(key, 0.0))
    if not math.isfinite(value):
        return 0.0
    return min(_OFFSET_LIMIT, max(-_OFFSET_LIMIT, value)) / 100.0


def _ramp_axis(start: float, count: int, extent: int, scale: float) -> np.ndarray:
    """``count`` ramp coordinates from pixel ``start``.

    The box's first and last pixels sit at -extent/scale and extent/scale, so
    a ramp as long as the box puts a stop on each end pixel, as Photoshop does.
    """
    step = 2.0 * extent / scale / (extent - 1) if extent > 1 else 0.0
    return (-extent / scale + (start + np.arange(count)) * step).astype(np.float32)


def _make_linear_gradient(X: np.ndarray, Y: np.ndarray, angle: float) -> np.ndarray:
    """Generate index map for linear gradients."""
    theta = np.radians(angle % 360)
    Z = 0.5 * (np.cos(theta) * X - np.sin(theta) * Y + 1)
    return Z


def _make_radial_gradient(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Generate index map for radial gradients."""
    Z = np.sqrt(np.power(X, 2) + np.power(Y, 2))
    return Z


def _make_angle_gradient(X: np.ndarray, Y: np.ndarray, angle: float) -> np.ndarray:
    """Generate index map for angle gradients."""
    Z = (((180 * np.arctan2(Y, X) / np.pi) + angle) % 360) / 360
    return Z


def _make_reflected_gradient(X: np.ndarray, Y: np.ndarray, angle: float) -> np.ndarray:
    """Generate index map for reflected gradients."""
    theta = np.radians(angle % 360)
    Z = np.abs(np.cos(theta) * X - np.sin(theta) * Y)
    return Z


def _make_diamond_gradient(X: np.ndarray, Y: np.ndarray, angle: float) -> np.ndarray:
    """Generate index map for diamond gradients."""
    theta = np.radians(angle % 360)
    Z = np.abs(np.cos(theta) * X - np.sin(theta) * Y) + np.abs(
        np.sin(theta) * X + np.cos(theta) * Y
    )
    return Z


def _make_gradient_color(
    color_mode: ColorMode, grad: Descriptor, method: bytes | None = None
) -> tuple[Any | None, Any | None]:
    gradient_form = grad.get(Type.GradientForm).enum
    if gradient_form == Enum.ColorNoise:
        return _make_noise_gradient_color(color_mode, grad)
    elif gradient_form == Enum.CustomStops:
        return _make_linear_gradient_color(color_mode, grad, method)

    logger.error("Unknown gradient form: %s", gradient_form)
    return None, None


def _collect_stops(
    stops: Sequence[Any], value_fn: Callable[[Any], np.ndarray]
) -> tuple[list[float], list[np.ndarray], list[float]]:
    """Collect gradient stop (location, value, midpoint) triples in ramp order.

    A location is the file's, so it is held to the gradient's length as a
    colour component is (#757). Co-located stops make a step: at their
    location the ramp takes the later listed, and the sort is stable so two
    that are not neighbours in the file still come out in the file's order. A
    stop's ``Mdpn`` is where, as a fraction of the segment that ends at it,
    the blend from the previous stop is halfway.
    """
    triples = sorted(
        (
            (
                _clamp01(float(stop.get(Key.Location)) / 4096.0),
                value_fn(stop),
                float(stop.get(Key.Midpoint, 50.0)) / 100.0,
            )
            for stop in stops
        ),
        key=lambda triple: triple[0],
    )
    if not triples:
        raise ValueError("Gradient has no stops.")
    if len(triples) == 1:
        value = triples[0][1]
        return [0.0, 1.0], [value, value], [0.5, 0.5]
    X = [location for location, _, _ in triples]
    Y = [value for _, value, _ in triples]
    M = [midpoint for _, _, midpoint in triples]
    return X, Y, M


# One sample per ``Lctn`` unit, so every stop is a sample.
_RAMP_SAMPLES = 4097


def _blend_stops(
    X: list[float], Y: list[np.ndarray], M: list[float], smoothness: float
) -> np.ndarray:
    """Sample the ramp through the stops at every ``Lctn`` unit.

    Within a segment the blend runs linearly to halfway at the end stop's
    midpoint and linearly on to the end; a midpoint on either end of its
    segment is ignored, as Photoshop's dialog cannot author one. Smoothness
    then eases the blend towards a smoothstep, with 100% landing halfway
    between the two.
    """
    t = np.linspace(0.0, 1.0, _RAMP_SAMPLES)
    locations = np.asarray(X)
    k = np.clip(np.searchsorted(locations, t, side="right") - 1, 0, len(X) - 2)
    span = locations[k + 1] - locations[k]
    m = np.asarray(M)[k + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        # A sample falls in a segment of no width only on its end stop.
        u = np.where(span > 0.0, np.clip((t - locations[k]) / span, 0.0, 1.0), 1.0)
        bent = np.where(u <= m, 0.5 * u / m, 0.5 + 0.5 * (u - m) / (1.0 - m))
    f = np.where((m > 0.0) & (m < 1.0), bent, u)
    f += smoothness * ((3.0 - 2.0 * f) * f * f - f) / 2.0
    values = np.asarray(Y, dtype=np.float64)
    f = f.reshape((-1,) + (1,) * (values.ndim - 1))
    return values[k] * (1.0 - f) + values[k + 1] * f


# ``color_convert``'s sRGB transfer functions over an array. That module is
# scalar by design, so it is the pieces that are shared, not the functions.
def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(
        rgb <= SRGB_SLOPE * SRGB_CUTOFF,
        rgb / SRGB_SLOPE,
        ((rgb + SRGB_OFFSET) / (1.0 + SRGB_OFFSET)) ** SRGB_GAMMA,
    )


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(
        rgb <= SRGB_CUTOFF,
        rgb * SRGB_SLOPE,
        (1.0 + SRGB_OFFSET) * rgb ** (1.0 / SRGB_GAMMA) - SRGB_OFFSET,
    )


# Oklab's two matrices (Ottosson 2020): linear sRGB to LMS, and cube-rooted
# LMS to Lab.
_OKLAB_LMS = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_OKLAB_LAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
_OKLAB_LMS_INV = np.linalg.inv(_OKLAB_LMS)
_OKLAB_LAB_INV = np.linalg.inv(_OKLAB_LAB)


def _srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    return np.cbrt(_srgb_to_linear(rgb) @ _OKLAB_LMS.T) @ _OKLAB_LAB.T


def _oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    return _linear_to_srgb(((lab @ _OKLAB_LAB_INV.T) ** 3) @ _OKLAB_LMS_INV.T)


# The perceptual method fades opacity along a power of the blend whose
# exponent runs from 1 at the opaque end to 1 + this at the transparent end.
# Fitted to Photoshop's renders of foreground-to-transparent fills.
_PERCEPTUAL_ALPHA_EASE = 1.42


def _ease_alpha(alpha: np.ndarray) -> np.ndarray:
    """Perceptual opacity from a linearly blended one."""
    return alpha ** (1.0 + _PERCEPTUAL_ALPHA_EASE * (1.0 - alpha))


_ALPHA_SAMPLES = np.linspace(0.0, 1.0, _RAMP_SAMPLES)
_EASED_ALPHA_SAMPLES = _ease_alpha(_ALPHA_SAMPLES)


def _unease_alpha(alpha: np.ndarray) -> np.ndarray:
    """Inverse of :py:func:`_ease_alpha`, read off its samples."""
    return np.interp(alpha, _EASED_ALPHA_SAMPLES, _ALPHA_SAMPLES)


_BlendSpace = tuple[
    Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]
]

# Methods that blend somewhere other than the document's own arrays:
# perceptual in Oklab, linear in linear-light sRGB. Classic and smooth, and a
# missing or unknown key, blend the canvas values directly.
_BLEND_SPACES: dict[bytes, _BlendSpace] = {
    GradientMethod.PERCEPTUAL.value: (_srgb_to_oklab, _oklab_to_srgb),
    GradientMethod.LINEAR.value: (_srgb_to_linear, _linear_to_srgb),
}

# The same for the transparency stops; only the perceptual method eases them.
_ALPHA_BLEND_SPACES: dict[bytes, _BlendSpace] = {
    GradientMethod.PERCEPTUAL.value: (_unease_alpha, _ease_alpha),
}


def _canvas_rows(color_mode: ColorMode, rgb: np.ndarray) -> np.ndarray:
    """``rgb``'s rows brought into the document's arrays."""
    return np.array([_from_rgb(color_mode, tuple(row)) for row in rgb.tolist()])


def _make_linear_gradient_color(
    color_mode: ColorMode, grad: Descriptor, method: bytes | None = None
) -> tuple[Any | None, Any | None]:
    from scipy import interpolate  # type: ignore[import-untyped]  # noqa: PLC0415

    smoothness = _clamp01(float(grad.get(Key.Interpolation, 4096.0)) / 4096.0)
    colors = grad.get(Key.Colors, [])
    table = _blend_stops(
        *_collect_stops(colors, lambda stop: np.array(_get_color(color_mode, stop))),
        smoothness,
    )
    space = _BLEND_SPACES.get(method) if method is not None else None
    if space is not None:
        encode, decode = space
        X, Y, M = _collect_stops(
            colors, lambda stop: np.array(_get_color(ColorMode.RGB, stop))
        )
        classic = _canvas_rows(color_mode, _blend_stops(X, Y, M, smoothness))
        blended = _canvas_rows(
            color_mode, decode(_blend_stops(X, [encode(y) for y in Y], M, smoothness))
        )
        # The method's blend as a shift from the classic one, so each stop
        # keeps its stored value: sRGB does not round-trip CMYK ink or a Lab
        # colour beyond its gamut.
        table = np.clip(blended + (table - classic), 0.0, 1.0)

    samples = np.linspace(0.0, 1.0, _RAMP_SAMPLES)

    def lookup(table: np.ndarray) -> Any:
        table = table.astype(np.float32)
        return interpolate.interp1d(
            samples, table, axis=0, bounds_error=False, fill_value=(table[0], table[-1])
        )

    G = lookup(table)
    if Key.Transparency not in grad:
        return G, None

    def opacity(stop: Any) -> np.ndarray:
        return np.array(_clamp01(float(stop.get(Key.Opacity)) / 100.0))

    stops = grad.get(Key.Transparency)
    alpha_space = _ALPHA_BLEND_SPACES.get(method) if method is not None else None
    if alpha_space is None:
        alpha = _blend_stops(*_collect_stops(stops, opacity), smoothness)
    else:
        encode, decode = alpha_space
        alpha = decode(
            _blend_stops(
                *_collect_stops(stops, lambda stop: encode(opacity(stop))), smoothness
            )
        )
    return G, lookup(alpha)


def _noise_color_components(table: np.ndarray) -> np.ndarray:
    """The three color columns of a noise gradient's table.

    Narrower is not something any writer here has produced; replicating the
    first column keeps such a file rendering as a grey ramp rather than
    raising on the unpack in :py:func:`_noise_to_canvas`.
    """
    if table.shape[1] >= 3:
        return table[:, :3]
    logger.debug("Noise gradient has %d components, expected 4.", table.shape[1])
    return np.repeat(table[:, :1], 3, axis=1)


def _noise_to_canvas(
    color_mode: ColorMode, space: bytes, table: np.ndarray
) -> np.ndarray:
    """Map a noise gradient's lookup table from *space* into the document's arrays.

    The three colour components of a noise gradient are stored as percentages
    of their own space's normalized encoding rather than as native values.
    Measured against Photoshop 2026 by authoring flat gradients -- ``Mnm ``
    equal to ``Mxm ``, which pins the noise field to one colour and so gives a
    render comparable without reproducing Photoshop's noise synthesis -- and
    reading the stored channels back:

    - ``RGBC``: ``v / 100`` is the channel, so 50 is 128/255.
    - ``HSBl``: ``v / 100`` is the fraction of a full turn for the hue, and the
      fraction of the axis for saturation and brightness; 33 rendered as 118.8
      degrees.
    - ``LbCl``: ``v / 100`` is the *byte* of the eight-bit Lab encoding, which
      is the compositor's own Lab array convention. So ``L* = v``, and
      ``a = b = 0`` at 50 rather than at 0. In a Lab document the table needs
      no conversion at all; the value stored for a flat ``[60, 60, 60]``
      gradient is exactly ``(153, 153, 153)``.

    Every one of the three is a colour space independent of the document's
    mode, which is the width bug this fixes: the table came back three wide
    whatever the document was (#730).
    """
    if space not in (Enum.RGBColor, Enum.HSBColor, Enum.LabColor):
        # Photoshop offers exactly those three, so anything else is another
        # writer's. Reading it as RGB is what this did for all three before,
        # and a wrong color beats refusing to render.
        logger.debug("Unknown noise gradient color space: %s", space)

    rows: list[tuple[float, ...]] = []
    for row in table:
        c0, c1, c2 = (float(value) for value in row)
        if space == Enum.HSBColor:
            rgb = hsb_to_rgb(c0, c1, c2)
        elif space == Enum.LabColor:
            if color_mode == ColorMode.LAB:
                # Already the destination encoding, so a trip through RGB
                # would only lose the out-of-gamut values on the way.
                rows.append((_clamp01(c0), _clamp01(c1), _clamp01(c2)))
                continue
            rgb = lab_to_rgb(c0 * 100.0, c1 * 255.0 - 128.0, c2 * 255.0 - 128.0)
        else:
            # Clamped for the same reason the descriptor readers are: these
            # components come from ``Mnm ``/``Mxm ``, which are raw file
            # values, so a band at ``Mxm = 150`` leaves [0, 1]. The other two
            # spaces are already covered -- HSB by ``hsb_to_rgb`` and Lab by
            # ``lab_to_rgb`` and ``_clamp01`` above -- which left RGB as the
            # one unguarded noise path (#757).
            rgb = (_clamp01(c0), _clamp01(c1), _clamp01(c2))
        rows.append(_from_rgb(color_mode, rgb))
    return np.array(rows, dtype=np.float32)


def _make_noise_gradient_color(
    color_mode: ColorMode, grad: Descriptor
) -> tuple[Any | None, Any | None]:
    r"""
    Make a noise gradient color.

    TODO: Improve noise gradient quality.

    Example:
        Descriptor(b'Grdn'){
            'Nm  ': 'Custom\x00',
            'GrdF': (b'GrdF', b'ClNs'),
            'ShTr': False,
            'VctC': False,
            'ClrS': (b'ClrS', b'RGBC'),
            'RndS': 3650322,
            'Smth': 2048,
            'Mnm ': [0, 0, 0, 0],
            'Mxm ': [0, 100, 100, 100]
        }
    """
    from scipy import interpolate  # type: ignore[import-untyped]  # noqa: PLC0415
    from scipy.ndimage import (  # type: ignore[import-untyped]  # noqa: PLC0415
        maximum_filter1d,
        uniform_filter1d,
    )

    logger.debug("Noise gradient is not accurate.")
    roughness = grad.get(Key.Smoothness).value / 4096.0  # Larger is sharper.
    maximum = np.array([x.value for x in grad.get(Key.Maximum)], dtype=np.float32)
    minimum = np.array([x.value for x in grad.get(Key.Minimum)], dtype=np.float32)
    seed = grad.get(Key.RandomSeed).value
    rng = np.random.RandomState(seed)
    Y = rng.binomial(1, 0.5, (256, len(maximum))).astype(np.float32)
    size = max(1, int(roughness))
    Y = maximum_filter1d(Y, size, axis=0)
    Y = uniform_filter1d(Y, size * 64, axis=0)
    Y = Y / np.max(Y, axis=0)
    Y = ((maximum - minimum) * Y + minimum) / 100.0
    X = np.linspace(0, 1, 256, dtype=np.float32)
    # Photoshop writes four components for each of its three noise color
    # spaces -- three color and one transparency -- and writes the fourth
    # whether or not ``ShTr`` is set, so the color triple is the leading three
    # either way.
    color_space = grad.get(Key.ColorSpace)
    Yc = _noise_to_canvas(
        color_mode,
        color_space.enum if color_space is not None else Enum.RGBColor,
        _noise_color_components(Y),
    )
    G = interpolate.interp1d(
        X, Yc, axis=0, bounds_error=False, fill_value=(Yc[0], Yc[-1])
    )
    if not grad.get(Key.ShowTransparency):
        return G, None
    # Clamped like the color components, and from the same raw ``Mnm ``/``Mxm ``
    # values (#757). ``np.clip`` keeps a NaN, so it goes to 0.0 first, as
    # ``_clamp01`` sends it.
    Ya = np.clip(np.nan_to_num(Y[:, -1], nan=0.0), 0.0, 1.0)
    Ga = interpolate.interp1d(
        X, Ya, axis=0, bounds_error=False, fill_value=(Ya[0], Ya[-1])
    )
    return G, Ga
