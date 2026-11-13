"""Adjustment layer operations for the compositor."""

import functools
import logging
import math
import struct
from collections.abc import Callable
from importlib import resources
from typing import Literal, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from psd_tools.api.adjustments import (
    BlackAndWhite,
    BrightnessContrast,
    ChannelMixer,
    ColorBalance,
    Curves,
    Exposure,
    GradientMap,
    HueSaturation,
    Invert,
    Levels,
    Posterize,
    Threshold,
    Vibrance,
)
from psd_tools.api.layers import Layer
from psd_tools.color_convert import SRGB_CUTOFF, SRGB_GAMMA, SRGB_OFFSET, SRGB_SLOPE
from psd_tools.composite import paint
from psd_tools.composite._compat import require_scipy
from psd_tools.composite.blend import (
    _lum,
    hsl2hsv,
    hsl2rgb,
    hsv2hsl,
    rgb2hsl,
)
from psd_tools.constants import ColorMode, Resource
from psd_tools.psd.adjustments import LevelRecord
from psd_tools.psd.adjustments import Levels as LevelsData

try:
    from psd_tools.composite import _blend as _cblend  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    _cblend = None

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)

# Float32 constants used throughout adjustments.
_0 = np.float32(0.0)
_1 = np.float32(1.0)
_100 = np.float32(100.0)
_255 = np.float32(255.0)
_360 = np.float32(360.0)

# Small epsilon to prevent division by zero in adjustment calculations.
_FLOAT_EPSILON = np.float32(1e-9)

# Contrast cubic spline control points (Photoshop empirical values, 0–255 scale).
# Inner midpoints move ±_CONTRAST_INNER_DELTA per unit of contrast.
_CONTRAST_CONTROL_X = np.array([0.0, 63.0, 191.0, 255.0], dtype=np.float32) / _255
_CONTRAST_INNER_DELTA = np.float32(25.0 / 255.0)

# Level the legacy brightness and contrast line pivots on (0–255 scale).
# Only mean 127 with lab False is verified; neither property is read here.
_LEGACY_PIVOT = np.float32(127.0) / _255

# Modern brightness curve, fitted to Photoshop exports (16 and 32 bits; 8-bit
# documents read Photoshop's tables). Positive brightness
# scales the shadows by a gain of exp(brightness / _BRIGHTNESS_GAIN_SCALE) up
# to a knee, then rolls off into white; negative brightness is its inverse.
# With s = 1 - 1 / gain, the knee's output level is k0 + k1 * s and the
# shoulder's exponent is 1 + s * (p1 + p2 * s), so the curve flattens to the
# identity as the gain goes to 1.
_BRIGHTNESS_GAIN_SCALE = 160.0
_BRIGHTNESS_KNEE = (0.48, 0.21)  # k0, k1
_BRIGHTNESS_SHOULDER = (3.2, -4.65)  # p1, p2
_BRIGHTNESS_GRID = np.linspace(0.0, 1.0, 4097)

# ICC profile gamma approximations for the exposure adjustment.
# Dot Gray 20% (Photoshop default grayscale profile) ≈ 1.75.
# sRGB (simplified power-law approximation) ≈ 2.2.
_GRAY_COLOR_GAMMA = np.float32(1.75)
_SRGB_COLOR_GAMMA = np.float32(2.2)

# Photoshop's exposure brings RGB back from linear light along the power curve
# or this line through black, whichever is lower. Read off 8-bit ramps through
# exposures -7.88 to 1.68.
_EXPOSURE_TOE_SLOPE = np.float32(94.5)

# Gamma values lower and upper bounds.
_GAMMA_LOWER = np.float32(0.01)
_GAMMA_UPPER = np.float32(9.99)

# Photoshop's Levels gamma above 1 leaves black at a slope of 2 ** gamma and
# joins the power curve, slope and all, at 2 ** -(gamma + 1 / (gamma - 1)),
# twice where that line meets the curve, along a cubic. Read off ramps put
# through Levels at gammas 1.05 to 9.99 in 8 and 16 bits.
_LEVELS_TOE_BASE = 2.0

# LUT of the f(x,y) function that blends two color range saturation vectors x, y to a new saturation vector.
# Uses empirical values tested using the regular grid (-100, -90, ... , 90, 100)^2.
# See: https://colab.research.google.com/drive/1CGU4kxaVgv01vAMdNKbdTlLAhhQeOEOy?usp=sharing
# fmt: off
_SATURATION_RANGE_INTERPOLATION_GRID = np.array([
    [100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100],
    [100, 94, 92, 91, 91, 90, 90, 90, 90, 90, 90, 89, 89, 89, 89, 89, 89, 89, 88, 88, 88],
    [100, 92, 89, 86, 84, 83, 82, 81, 81, 80, 80, 79, 79, 78, 78, 77, 77, 76, 76, 75, 74],
    [100, 91, 86, 82, 79, 77, 74, 73, 71, 70, 70, 68, 67, 66, 65, 64, 62, 61, 59, 58, 56],
    [100, 91, 84, 79, 75, 71, 68, 65, 63, 61, 60, 58, 56, 54, 52, 49, 47, 44, 41, 36, 32],
    [100, 90, 83, 77, 71, 66, 62, 59, 55, 52, 50, 47, 44, 41, 37, 33, 28, 22, 17,  9, -1],
    [100, 90, 82, 74, 68, 62, 57, 52, 47, 43, 40, 36, 31, 26, 21, 13,  5, -5,-15,-25,-35],
    [100, 90, 81, 73, 65, 59, 52, 46, 39, 35, 30, 24, 19, 11,  2, -8,-18,-28,-38,-48,-58],
    [100, 90, 81, 71, 63, 55, 47, 39, 33, 26, 20, 13,  4, -6,-16,-26,-36,-46,-56,-66,-76],
    [100, 90, 80, 70, 61, 52, 43, 35, 26, 17, 10,  1, -9,-19,-29,-39,-49,-59,-69,-79,-89],
    [100, 90, 80, 70, 60, 50, 40, 30, 20, 10,  0,-10,-20,-30,-40,-49,-60,-70,-80,-90,-100],
    [100, 89, 79, 68, 58, 47, 36, 24, 13,  1,-10,-19,-30,-40,-50,-60,-70,-79,-89,-99,-100],
    [100, 89, 79, 67, 56, 44, 31, 19,  4, -9,-20,-30,-40,-50,-60,-70,-79,-90,-99,-100,-100],
    [100, 89, 78, 66, 54, 41, 26, 11, -6,-19,-30,-40,-50,-59,-70,-79,-89,-99,-100,-100,-100],
    [100, 89, 78, 65, 52, 37, 21,  2,-16,-29,-40,-50,-60,-70,-79,-90,-99,-100,-100,-100,-100],
    [100, 89, 77, 64, 49, 33, 13, -8,-26,-39,-49,-60,-70,-79,-90,-99,-100,-100,-100,-100,-100],
    [100, 89, 77, 62, 47, 28,  5,-18,-36,-49,-60,-70,-79,-89,-99,-100,-100,-100,-100,-100,-100],
    [100, 89, 76, 61, 44, 22, -5,-28,-46,-59,-70,-79,-90,-99,-100,-100,-100,-100,-100,-100,-100],
    [100, 88, 76, 59, 41, 17,-15,-38,-56,-69,-80,-89,-99,-100,-100,-100,-100,-100,-100,-100,-100],
    [100, 88, 75, 58, 36,  9,-25,-48,-66,-79,-90,-99,-100,-100,-100,-100,-100,-100,-100,-100,-100],
    [100, 88, 74, 56, 32, -1,-35,-58,-76,-89,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100,-100],
], dtype=np.float32)[::-1,::-1] / _100
# fmt: on

# Offset used by threshold adjustments to compensate values on 16 bits.
_THRESHOLD_OFFSET = 1 / _255


def _preserve_alpha(func: F) -> F:
    """
    Decorator to apply adjustments only on color channels, leaving the alpha channel intact.

    If `n_channels == color_channels + 1`, the last channel is assumed to be alpha.
    If `n_channels == color_channels`, it is assumed no alpha channel exists.
    In any other case, a value error is raised.

    Raises:
        ValueError: If the number of channels of the image doesn't align with the colormode attribute.
    """

    @functools.wraps(func)
    def wrapper(img: np.ndarray, colormode: ColorMode, *args, **kwargs):
        color_channels = ColorMode.channels(colormode)
        n_channels = img.shape[2]

        if not (n_channels == color_channels or n_channels == color_channels + 1):
            raise ValueError("Channel count does not match colormode.")

        if n_channels > color_channels:
            color = img[..., :color_channels]
            alpha = img[..., color_channels:]

            out = func(color, colormode, *args, **kwargs)
            if out is None:
                return None
            if out.shape[2] != color_channels:
                raise ValueError(
                    f"Adjustment function returned {out.shape[2]} channels, "
                    f"expected {color_channels}."
                )
            return np.concatenate([out, alpha], axis=2)

        return func(img, colormode, *args, **kwargs)

    return wrapper  # type: ignore[return-value]


def _get_lut_size(layer: Layer) -> Literal[256, 65536]:
    bits = layer._psd.depth
    lut_size = min(2**bits, 65536)
    logger.debug("Lut size: %s", lut_size)
    return lut_size


@functools.lru_cache(maxsize=2)
def _lut_domain(lut_size: int) -> NDArray[np.float32]:
    """
    Return the normalized domain [0, 1] used for LUT interpolation.

    Gives `lut_size` evenly spaced samples. Cached per size.
    """
    return np.linspace(_0, _1, lut_size, dtype=np.float32)


def _apply_luts(
    luts: dict[int, NDArray[np.float32]], img: np.ndarray, colormode: ColorMode
) -> NDArray[np.float32]:
    if img.ndim != 3:
        raise ValueError(f"Expected 3D array (H, W, C), got ndim={img.ndim}.")
    n_channels = ColorMode.channels(colormode)
    if (
        colormode != ColorMode.GRAYSCALE
        and list(luts) == [0]
        and img.dtype == np.float32
    ):
        # The master alone: read it off the image, no copy first.
        return _apply_lut(img, luts[0])
    out = img.astype(np.float32, copy=True)
    channels = range(1, n_channels + 1)

    # individual adjustments get applied independently of each other, then the master lut its applied if exists
    for channel_id in channels:
        if channel_id in luts:
            lut = luts[channel_id]
            out[:, :, channel_id - 1] = _apply_lut(img[:, :, channel_id - 1], lut)

    if colormode != ColorMode.GRAYSCALE and 0 in luts:
        out = _apply_lut(out, luts[0])

    return out


def _level_index(values: np.ndarray, lut_size: int) -> NDArray[np.intp]:
    """The index of the LUT's level nearest each of ``values`` (0 to 1)."""
    index = np.floor(values * np.float32(lut_size - 1) + np.float32(0.5))
    return np.clip(index, 0, lut_size - 1).astype(np.intp)


def _nearest_level(values: np.ndarray, lut_size: int) -> NDArray[np.float32]:
    """``values`` (0 to 1) moved to the nearest of the LUT's levels."""
    return _lut_domain(lut_size)[_level_index(values, lut_size)]


def _apply_lut(values: np.ndarray, lut: np.ndarray) -> NDArray[np.float32]:
    """``lut`` read at the level nearest each value.

    Photoshop's backdrop is in whole levels, rounded, and its curves chain
    on them, the master reading each channel's output at its nearest.
    """
    lut_size = int(lut.shape[0])
    if (
        _cblend is not None
        and 0 < lut_size <= 2**16
        and lut.ndim == 1
        and lut.dtype == np.float32
        and values.dtype == np.float32
        and values.ndim in (2, 3)
    ):
        out = _cblend.lut_levels(
            values if values.ndim == 3 else values[:, :, None],
            np.ascontiguousarray(lut),
        )
        return out if values.ndim == 3 else out[:, :, 0]
    if lut_size <= 2**16:
        nan = np.isnan(values)
        if nan.any():
            # NaN has no level: it reads as NaN.
            out = lut[_level_index(np.where(nan, 0.0, values), lut_size)]
            out = out.astype(np.float32)
            out[nan] = np.nan
            return out
        return lut[_level_index(values, lut_size)].astype(np.float32, copy=False)
    xp = _lut_domain(lut_size)
    return np.interp(values, xp, lut).astype(np.float32, copy=False)


def _get_luminance(
    img: np.ndarray,
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB],
) -> np.ndarray:
    if colormode == ColorMode.RGB:
        return _lum(img)
    elif colormode == ColorMode.GRAYSCALE:
        return img[..., 0:1]
    else:  # CMYK requires accurate luminance conversion
        return img


@require_scipy
@_preserve_alpha
def apply_brightnesscontrast(
    img: np.ndarray,
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB],
    layer: BrightnessContrast,
) -> np.ndarray:
    """
    Apply a brightness & contrast adjustment to an image.

    Brightness is applied first and contrast to its result, through one curve
    for every channel: Photoshop's own in 8-bit documents, else a fit.
    Requires scipy for cubic spline interpolation.
    """
    t = _lut_domain(_get_lut_size(layer))

    if layer.use_legacy:
        lut = _legacy_brightnesscontrast_lut(layer.brightness, layer.contrast, t)
        channel_id = 1 if colormode == ColorMode.GRAYSCALE else 0
        return _apply_luts({channel_id: lut}, img, colormode)

    if _get_lut_size(layer) == 256 and (layer.brightness, layer.contrast) != (0, 0):
        levels = _brightnesscontrast_levels(layer.brightness, layer.contrast)
        if levels is not None:
            return _apply_lut(img, levels.astype(np.float32) / _255)

    from scipy import interpolate  # type: ignore[import-untyped]  # noqa: PLC0415

    c = np.float32(layer.contrast) / _100

    x = _CONTRAST_CONTROL_X
    y = np.array(
        [_0, x[1] - c * _CONTRAST_INNER_DELTA, x[2] + c * _CONTRAST_INNER_DELTA, _1],
        dtype=np.float32,
    )
    contrast = interpolate.CubicSpline(x, y, bc_type="natural")
    lut = contrast(_brightness_curve(layer.brightness, t)).clip(0.0, 1.0)

    # Interpolated rather than read at the nearest level as _apply_luts
    # does: the curve is steep near white, and an identity adjustment must
    # leave the pixels as they are.
    return np.interp(img, t, lut).astype(np.float32, copy=False)


@functools.cache
def _brightnesscontrast_tables() -> tuple[
    NDArray[np.uint8], NDArray[np.uint8], dict[tuple[int, int], NDArray[np.uint8]]
]:
    """Photoshop's 8-bit tables: by brightness, by contrast, and by pair."""
    source = resources.files(__package__).joinpath("brightnesscontrast_tables.npz")
    with source.open("rb") as stream, np.load(stream) as data:
        pairs = {
            (int(b), int(c)): table
            for (b, c), table in zip(data["pairs"], data["pair_tables"], strict=True)
        }
        return data["brightness"], data["contrast"], pairs


def _brightnesscontrast_levels(
    brightness: int, contrast: int
) -> NDArray[np.uint8] | None:
    """
    Photoshop 2023's modern brightness and contrast on 8-bit levels.

    Read off ramps (tools/brightnesscontrast_tables.py): every brightness
    alone, every contrast alone and the pairs measured. Another pair takes
    the contrast of the brightness's level, at most a level off. None out
    of range.
    """
    if not (-150 <= brightness <= 150 and -50 <= contrast <= 100):
        return None
    bright, contrasts, pairs = _brightnesscontrast_tables()
    table = pairs.get((brightness, contrast))
    if table is None:
        table = contrasts[contrast + 50][bright[brightness + 150]]
    return table


def _brightness_curve(brightness: int, t: NDArray[np.float32]) -> NDArray[np.float64]:
    """Map ``t`` through the modern brightness curve, brightness in [-150, 150]."""
    if brightness == 0:
        return t.astype(np.float64)
    gain = np.exp(abs(brightness) / _BRIGHTNESS_GAIN_SCALE)
    s = 1.0 - 1.0 / gain
    knee = _BRIGHTNESS_KNEE[0] + _BRIGHTNESS_KNEE[1] * s
    power = 1.0 + s * (_BRIGHTNESS_SHOULDER[0] + _BRIGHTNESS_SHOULDER[1] * s)
    start = knee / gain
    # The shoulder leaves the knee at the gain's slope and reaches white at 1.
    rate = gain * (1.0 - start) / (1.0 - knee) - power
    grid = _BRIGHTNESS_GRID
    w = np.clip((grid - start) / (1.0 - start), 0.0, 1.0)
    shoulder = 1.0 - (1.0 - knee) * (1.0 - w) ** power * np.exp(-rate * w)
    curve = np.where(grid <= start, gain * grid, shoulder)
    if brightness > 0:
        return np.interp(t, grid, curve)
    return np.interp(t, curve, grid)


def _legacy_brightnesscontrast_lut(
    brightness: int, contrast: int, t: NDArray[np.float32]
) -> NDArray[np.float32]:
    """
    Return Photoshop's legacy brightness and contrast, a line through mid-gray.

    Brightness shifts by that many of 255 levels. A positive contrast then
    steepens the line by ``100 / (100 - contrast)``, so 100 is a threshold; a
    negative contrast flattens it by ``(100 + contrast) / 100`` before the shift.
    """
    shift = np.float32(brightness) / _255
    if contrast > 0:
        slope = _100 / max(_100 - np.float32(contrast), _FLOAT_EPSILON)
        lut = (t + shift - _LEGACY_PIVOT) * slope + _LEGACY_PIVOT
    else:
        slope = (_100 + np.float32(contrast)) / _100
        lut = (t - _LEGACY_PIVOT) * slope + _LEGACY_PIVOT + shift
    return lut.clip(_0, _1).astype(np.float32, copy=False)


@_preserve_alpha
def apply_levels(
    img: np.ndarray,
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB],
    layer: Levels,
) -> np.ndarray:
    """Apply a levels adjustment to an image."""
    levels_data = layer.data
    if not levels_data:
        return img

    lut_size = _get_lut_size(layer)
    t = _lut_domain(lut_size)

    luts: dict[int, NDArray[np.float32]] = {}
    # A master record that only sets a gamma multiplies into each channel's
    # gamma, one curve rather than two in a row.
    folded = _master_gamma(levels_data, colormode)

    for channel_id, channel_data in enumerate(levels_data):
        if channel_id == 0 and folded != 1.0:
            continue
        in_black = np.float32(channel_data.input_floor) / _255
        in_white = np.float32(channel_data.input_ceiling) / _255
        gamma = _record_gamma(channel_data) * (folded if channel_id else 1.0)
        out_black = np.float32(channel_data.output_floor) / _255
        out_white = np.float32(channel_data.output_ceiling) / _255

        # input adjustments, to the LUT's own levels
        scale = (
            (in_white - in_black) if abs(in_white - in_black) > _FLOAT_EPSILON else _1
        )
        out = (t - in_black) / scale
        out = _nearest_level(out, lut_size)

        # gamma midtone adjustment
        out = _levels_gamma(out, float(gamma), lut_size)
        np.clip(out, _0, _1, out=out)

        # output adjustments
        lut = out * (out_white - out_black) + out_black
        luts[channel_id] = lut.clip(_0, _1)

    return _apply_luts(luts, img, colormode)


def _record_gamma(record: LevelRecord) -> float:
    """A Levels record's gamma, held to [0.01, 9.99] as Photoshop holds it."""
    return float(min(max(_GAMMA_LOWER, np.float32(record.gamma / _100)), _GAMMA_UPPER))


def _master_gamma(records: LevelsData, colormode: ColorMode) -> float:
    """The gamma of a master record that sets nothing else, else 1.

    Only where every colour channel has its own record to carry it, with
    its output left at 0 to 255: the master's power then follows the
    channel's directly. The product is not held to 9.99; no fixture says
    Photoshop holds it.
    """
    channels = ColorMode.channels(colormode)
    if colormode == ColorMode.GRAYSCALE or len(records) <= channels:
        return 1.0
    if any(
        (record.output_floor, record.output_ceiling) != (0, 255)
        for record in records[: channels + 1]
    ):
        return 1.0
    master = records[0]
    if (master.input_floor, master.input_ceiling) != (0, 255):
        return 1.0
    return _record_gamma(master)


def _levels_gamma(
    values: np.ndarray,
    gamma: float,
    lut_size: int,
    power: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """``values`` raised to ``1 / gamma``, with Photoshop's toe above gamma 1.

    The toe is a cubic from black, at a slope of ``2 ** gamma``, to the point
    where it joins the power curve at the curve's own slope; that point falls
    on the LUT's levels. ``power``: that power, if worked out already.
    """
    if power is None:
        power = np.power(values, np.float32(1.0 / gamma), dtype=np.float32)
    if gamma <= 1.0:
        return power
    steps = lut_size - 1
    join = math.floor(_LEVELS_TOE_BASE ** -(gamma + 1.0 / (gamma - 1.0)) * steps + 0.5)
    if join < 1:
        return power
    end = join / steps
    rise = end ** (1.0 / gamma)
    t = values.astype(np.float64) / end
    toe = (
        _LEVELS_TOE_BASE**gamma * end * (t**3 - 2.0 * t**2 + t)
        + rise * (3.0 * t**2 - 2.0 * t**3)
        + rise / gamma * (t**3 - t**2)
    )
    return np.where(values < end, toe, power).astype(np.float32)


@_preserve_alpha
def apply_curves(
    img: np.ndarray,
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB],
    layer: Curves,
) -> np.ndarray:
    """Apply a curves adjustment to an image: a natural cubic spline per channel."""
    if layer.extra is None:
        return img
    info_dict = {data.channel_id: data.points for data in layer.extra}

    lut_size = _get_lut_size(layer)
    t = _lut_domain(lut_size)

    luts: dict[int, NDArray[np.float32]] = {}

    for channel_id, points in info_dict.items():
        if len(points) < 2:
            continue

        x = np.array([p[1] for p in points], dtype=np.float32) / _255
        y = np.array([p[0] for p in points], dtype=np.float32) / _255

        t_clamped = np.clip(t, x[0], x[-1])
        lut = _natural_spline(x, y, t_clamped).astype(np.float32, copy=False)
        luts[channel_id] = lut.clip(_0, _1)

    return _apply_luts(luts, img, colormode)


def _natural_spline(
    x: NDArray[np.float32], y: NDArray[np.float32], t: NDArray[np.float32]
) -> NDArray[np.float64]:
    """
    ``scipy.interpolate.CubicSpline(x, y, bc_type="natural")(t)``.

    scipy's float64 steps (1.18): its banded system for the slopes, solved as
    LAPACK's reference gtsv does, CubicHermiteSpline's coefficients, PPoly's
    intervals and Horner order. Plain float64, so the same on every CPU
    (scipy's LAPACK differs by platform in the last bits).
    """
    xs, ys = x.astype(np.float64), y.astype(np.float64)
    dx = np.diff(xs)
    if np.any(dx <= 0):
        raise ValueError("`x` must be strictly increasing sequence.")
    slope = np.diff(ys) / dx
    n = len(xs)

    # CubicSpline's banded system; its end rows hold the zero second derivatives.
    A = np.zeros((3, n))
    A[1, 1:-1] = 2 * (dx[:-1] + dx[1:])
    A[0, 2:] = dx[:-1]
    A[-1, :-2] = dx[1:]
    A[1, 0], A[0, 1] = 2 * dx[0], dx[0]
    A[1, -1], A[-1, -2] = 2 * dx[-1], dx[-1]
    b = np.empty(n)
    b[1:-1] = 3 * (dx[1:] * slope[:-1] + dx[:-1] * slope[1:])
    b[0], b[-1] = 3 * (ys[1] - ys[0]), 3 * (ys[-1] - ys[-2])
    s = np.array(
        _gtsv(A[2, :-1].tolist(), A[1].tolist(), A[0, 1:].tolist(), b.tolist())
    )

    w = (s[:-1] + s[1:] - 2 * slope) / dx
    c0, c1, c2, c3 = w / dx, (slope - s[:-1]) / dx - w, s[:-1], ys[:-1]

    tt = t.astype(np.float64)
    i = np.clip(np.searchsorted(xs, tt, side="right") - 1, 0, n - 2)
    h = tt - xs[i]
    return 0.0 + c3[i] + c2[i] * h + c1[i] * (h * h) + c0[i] * (h * h * h)


def _gtsv(
    dl: list[float], d: list[float], du: list[float], b: list[float]
) -> list[float]:
    """LAPACK's reference dgtsv, one right-hand side: partial pivoting."""
    n = len(d)
    for i in range(n - 1):
        if abs(d[i]) >= abs(dl[i]):
            fact = dl[i] / d[i]
            d[i + 1] = d[i + 1] - fact * du[i]
            b[i + 1] = b[i + 1] - fact * b[i]
            dl[i] = 0.0
        else:
            fact = d[i] / dl[i]
            d[i], temp = dl[i], d[i + 1]
            d[i + 1] = du[i] - fact * temp
            if i < n - 2:
                dl[i] = du[i + 1]
                du[i + 1] = -fact * dl[i]
            du[i] = temp
            b[i], b[i + 1] = b[i + 1], b[i] - fact * b[i + 1]
    b[-1] = b[-1] / d[-1]
    b[-2] = (b[-2] - du[-1] * b[-1]) / d[-2]
    for i in range(n - 3, -1, -1):
        b[i] = (b[i] - du[i] * b[i + 1] - dl[i] * b[i + 2]) / d[i]
    return b


@_preserve_alpha
def apply_exposure(
    img: np.ndarray,
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB],
    layer: Exposure,
) -> np.ndarray:
    """Apply an exposure adjustment to an image."""
    if colormode == ColorMode.CMYK:
        logger.debug("Exposure doesn't support CMYK in Photoshop.")
        return img

    if layer.exposure is None or layer.exposure_offset is None or layer.gamma is None:
        return img

    exposure = np.float32(layer.exposure)
    offset = np.float32(layer.exposure_offset)
    gamma = min(
        max(_GAMMA_LOWER, np.float32(layer.gamma)), _GAMMA_UPPER
    )  # photoshop clamps gamma values into [0.01, 9.99]

    lut_size = _get_lut_size(layer)
    values = _lut_domain(lut_size)

    # color_gamma approximates the TRC of the ICC profile used.
    # Defaults to 2.2 for sRGB (RGB) and 1.75 for Dot Gray 20% (grayscale).
    # TODO: generalize for all ICC profiles.
    color_gamma = (
        _GRAY_COLOR_GAMMA if colormode == ColorMode.GRAYSCALE else _SRGB_COLOR_GAMMA
    )

    lut = _powf(values, color_gamma)  # image is converted to linear color space first

    # exposure operates in linear space
    lut *= _powf(np.float32(2.0), exposure)
    lut = (lut + offset).clip(_0, _1)
    lut = _powf(lut, _1 / gamma)
    toe = lut * _EXPOSURE_TOE_SLOPE

    # convert back to original space using inverse TRC approximation
    lut = _powf(lut, (_1 / color_gamma).clip(_0, _1))
    if colormode != ColorMode.GRAYSCALE:
        lut = np.minimum(lut, toe)

    channel_id = 1 if colormode == ColorMode.GRAYSCALE else 0

    return _apply_luts({channel_id: lut}, img, colormode)


def _powf(
    x: np.ndarray | np.float32, y: np.ndarray | np.float32
) -> NDArray[np.float32]:
    """``x ** y`` in float32, correctly rounded: worked in float64, rounded once.

    float32 powers differ in the last bits between CPUs (SVML, libm).
    """
    return np.power(np.asarray(x, np.float64), np.float64(y)).astype(np.float32)


@require_scipy
@_preserve_alpha
def apply_huesaturation(
    img: np.ndarray,
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB],
    layer: HueSaturation,
) -> np.ndarray:
    """Apply a hue/saturation adjustment to an image."""
    if colormode == ColorMode.GRAYSCALE:
        logger.debug("Hue/Saturation doesn't support grayscale in Photoshop.")
        return img

    # CMYK requires accurate luminance conversion
    if colormode == ColorMode.CMYK:
        logger.info("Hue/Saturation isn't currently supported for CMYK.")
        return img

    items, colorization, master = layer.data, layer.colorization, layer.master
    if items is None or colorization is None or master is None:
        return img

    if layer.enable_colorization:
        if _get_lut_size(layer) == 256:
            return _colorize_8bit(img, *colorization)
        hsl_colorize_tuple = _normalize_hsl(colorization)
        return (
            _huesaturation_colorize(img, hsl_colorize_tuple)
            if hsl_colorize_tuple != (_0, _0, _0)
            else img
        )

    color_ranges = [
        (hue_range, _normalize_hsl(hsl_tuple))
        for hue_range, hsl_tuple in items
        if hsl_tuple != (_0, _0, _0)
    ]
    hue, saturation, lightness = (int(value) for value in master)
    if (
        not color_ranges
        and hue == 0
        and saturation in (0, -100)
        and (saturation or lightness)
        and colormode == ColorMode.RGB
        and layer._psd.depth == 8
    ):
        return _huesaturation_levels(img, saturation == -100, lightness)
    hsl_master_tuple = _normalize_hsl(master)
    return (
        _huesaturation(img, color_ranges, hsl_master_tuple)
        if color_ranges or hsl_master_tuple != (_0, _0, _0)
        else img
    )


def _huesaturation_levels(
    img: np.ndarray, desaturate: bool, lightness: int
) -> NDArray[np.float32]:
    """Photoshop 2023's 8-bit master lightness, then desaturation.

    On each channel's nearest level ``c``, ``K`` a whole level: lightness
    below 0 scales ``c`` by ``K / 255``, ``K = ceil(255 * (100 + l) / 100)``;
    above 0 it moves ``c`` that far toward white, ``K = floor(255 * l / 100)``;
    both rounded. Desaturation then sets every channel to the floored
    midrange. Exact in float64: ``c * K / 255`` is never within 1/510 of a
    half, so every CPU agrees.
    """
    level = np.floor(np.clip(img, _0, _1) * _255 + np.float32(0.5))
    level = level.astype(np.float64)
    if lightness < 0:
        k = -((-255 * (100 + lightness)) // 100)
        level = np.floor(level * k / 255.0 + 0.5)
    elif lightness > 0:
        k = 255 * lightness // 100
        level = np.floor(level + (255.0 - level) * k / 255.0 + 0.5)
    if desaturate:
        mid = np.floor((level.max(axis=2) + level.min(axis=2)) / 2.0)
        level = np.repeat(mid[..., None], img.shape[2], axis=2)
    return level.astype(np.float32) / _255


def _normalize_hsl(
    color: tuple[int, int, int],
) -> tuple[np.float32, np.float32, np.float32]:
    hue, saturation, lightness = color
    return hue / _360, saturation / _100, lightness / _100


def _huesaturation_colorize(
    img: np.ndarray, hsl_colorize_tuple: tuple[np.float32, np.float32, np.float32]
) -> np.ndarray:
    hue, saturation, lightness = hsl_colorize_tuple
    hue = hue % _1

    hsl = rgb2hsl(img)

    hsl[..., 0:1] = hue
    hsl[..., 1:2] = saturation
    hsl[..., 2:3] = _apply_lightness(hsl[..., 2:3], lightness)

    return hsl2rgb(hsl)


def _colorize_8bit(
    img: np.ndarray, hue: int, saturation: int, lightness: int
) -> NDArray[np.float32]:
    """Photoshop 2023's Colorize in 8-bit documents, exact.

    The pixel's whole levels give its lightness level, ``(max + min) // 2``,
    and that level reads the setting's table.
    """
    nan = np.isnan(img)
    levels = _level_index(np.where(nan, _0, img), 256)
    out = _colorize_levels(hue, saturation, lightness)[
        (levels.max(axis=2) + levels.min(axis=2)) // 2
    ]
    out[nan.any(axis=2)] = np.nan
    return out


def _colorize_ramp() -> tuple[int, ...]:
    # A sector's rising channel at steps of 60/42.5 degrees: the mix of its
    # two ends at that angle on a circle, not on a hexagon.
    angles = [math.radians(k * 60 / 42.5) for k in range(42)]
    return tuple(
        math.floor(255 * math.sin(a) / math.sin(a + math.pi / 3) + 0.5) for a in angles
    )


_COLORIZE_RAMP = _colorize_ramp()


@functools.lru_cache(maxsize=16)
def _colorize_levels(hue: int, saturation: int, lightness: int) -> NDArray[np.float32]:
    """Colorize's output (256, 3) by lightness level, in Photoshop's integers.

    Read off Photoshop 2023 (every hue, saturation and lightness): the hue
    in 256ths of a turn, its channels a third of a turn apart (85 and 170),
    the saturation in 255ths over 127, the lightness in 255ths. 16-bit
    documents mix plain HSL instead.
    """
    sat = max(0, min(100, saturation)) * 255 // 100
    light = min(100, abs(lightness)) * 255 // 100
    level = np.arange(256, dtype=np.int64)
    if lightness < 0:
        level = (level * (255 - light) + 127) // 255
    else:
        level = (level * (255 - light) + 255 * light + 127) // 255
    chroma = np.minimum(level, 255 - level) * sat // 127
    high = np.minimum(255, (2 * level + chroma + 1) // 2)

    index = hue % 360 * 256 // 360
    shares = []
    for offset in (85, 0, 170):
        t = (index + offset) % 256
        if t < 42:
            shares.append(_COLORIZE_RAMP[t])
        elif t <= 128:
            shares.append(255)
        elif t <= 170:
            shares.append(_COLORIZE_RAMP[170 - t])
        else:
            shares.append(0)
    out = (high - chroma)[:, None] + (chroma[:, None] * np.array(shares) + 127) // 255
    table = (out / _255).astype(np.float32)
    table.flags.writeable = False
    return table


def _huesaturation(
    img: np.ndarray,
    color_ranges: list[
        tuple[tuple[int, int, int, int], tuple[np.float32, np.float32, np.float32]]
    ],
    master_tuple: tuple[np.float32, np.float32, np.float32],
) -> np.ndarray:
    # master lightness is applied before converting to hsl
    master_hue, master_saturation, master_lightness = master_tuple
    img = _apply_lightness(img, master_lightness)

    hsl = rgb2hsl(img)
    colorrange_hue, colorrange_saturation, colorrange_lightness = (
        _get_colorrange_hsl_values(hsl, color_ranges)
    )

    # color range lightness is applied in HSV space
    hsl = hsl2hsv(hsl)
    saturation_channel, value_channel = hsl[..., 1:2], hsl[..., 2:3]

    hsl[..., 2:3] = np.where(
        colorrange_lightness >= 0,
        value_channel,
        value_channel * (_1 + colorrange_lightness * saturation_channel),
    ).clip(_0, _1)
    hsl[..., 1:2] = _correct_saturation(colorrange_lightness, saturation_channel)

    # hue and saturation are applied in HSL space
    # master and color range hues are applied simultaneously
    # master saturation is applied first, then color range saturation follows
    hsl = hsv2hsl(hsl)
    hsl[..., 0:1] = (hsl[..., 0:1] + colorrange_hue + master_hue) % _1
    hsl[..., 1:2] = _apply_saturation(
        _apply_saturation(hsl[..., 1:2], master_saturation), colorrange_saturation
    )

    return hsl2rgb(hsl).astype(np.float32, copy=False)


def _apply_lightness(img: np.ndarray, lightness: np.float32) -> np.ndarray:
    if lightness > 0:
        return (img * (_1 - lightness) + lightness).clip(_0, _1)
    if lightness < 0:
        return (img * (_1 + lightness)).clip(_0, _1)
    return img


# TODO: improve saturation algorithm accuracy when saturation nears 1.0
def _apply_saturation(
    img: np.ndarray, saturation: np.float32 | NDArray[np.float32]
) -> NDArray[np.float32]:
    out = np.where(
        saturation > 0,
        img / (_1 - saturation + np.float32(1e-2)),
        img * (_1 + saturation),
    )
    np.clip(out, _0, _1, out=out)
    return out


def _correct_saturation(
    colorrange_lightness: np.ndarray, saturation_channel: np.ndarray
) -> NDArray[np.float32]:
    """Correct saturation values when colorrange lightness is applied to the hsl image."""
    div = _1 / (saturation_channel + np.float32(1e-3))
    out = np.where(
        colorrange_lightness >= 0,
        saturation_channel * (_1 - colorrange_lightness),
        _1 + (div - _1) / (np.abs(colorrange_lightness) - div + np.float32(1e-3)),
    )
    np.clip(out, _0, _1, out=out)
    return out


# TODO: find exact mapping to avoid interpolation
@functools.lru_cache(maxsize=1)
def _get_huesaturation_interpolator():
    from scipy.interpolate import (  # type: ignore[import-untyped]  # noqa: PLC0415
        RegularGridInterpolator,
    )

    axis = np.linspace(-_1, _1, 21, dtype=np.float32)
    return RegularGridInterpolator(
        (axis, axis),
        _SATURATION_RANGE_INTERPOLATION_GRID,
        method="linear",
    )


def _get_colorrange_hsl_values(
    hsl_img: np.ndarray,
    color_ranges: list[
        tuple[tuple[int, int, int, int], tuple[np.float32, np.float32, np.float32]]
    ],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-pixel hue, saturation, and lightness values derived from color ranges."""
    base_hue = hsl_img[..., 0:1]

    colorrange_hue = np.zeros_like(base_hue)
    colorrange_saturation = np.zeros_like(base_hue)
    colorrange_lightness = np.zeros_like(base_hue)
    if not color_ranges:  # master only: no scipy
        return colorrange_hue, colorrange_saturation, colorrange_lightness

    interpolator = _get_huesaturation_interpolator()
    points = np.empty((*colorrange_saturation.shape, 2), dtype=np.float32)

    # construction of the hue, saturation and lightness color range vectors depends on loop order
    for color_range_tuple, (hue, saturation, lightness) in color_ranges:
        range_mask = _get_huesaturation_range_mask(base_hue, color_range_tuple)

        colorrange_hue += hue * range_mask
        colorrange_lightness += lightness * range_mask
        np.clip(colorrange_lightness, -_1, _1, out=colorrange_lightness)

        saturation_contribution = saturation * (
            (
                _1
                - np.power(
                    _1 - range_mask,
                    np.float32(1.5)
                    / (_1 - saturation ** np.float32(4.0) + np.float32(5e-2)),
                    dtype=np.float32,
                )
            )
            if saturation > 0
            else range_mask
        )
        # saturation values get mapped using a 3D surface
        points[..., 0] = colorrange_saturation
        points[..., 1] = saturation_contribution
        colorrange_saturation = interpolator(points.clip(-_1, _1)).astype(
            np.float32, copy=False
        )

    return colorrange_hue, colorrange_saturation, colorrange_lightness


def _get_huesaturation_range_mask(
    hue: np.ndarray, color_range: tuple[int, int, int, int]
) -> NDArray[np.float32]:
    """
    Compute a trapezoidal hue mask for a Hue/Saturation adjustment.

    The range is defined by four hue control points (in degrees):
    BL (bottom-left), TL (top-left), TR (top-right), BR (bottom-right).

    The resulting mask:
    - is 0.0 outside [BL, BR]
    - rises linearly from 0.0->1.0 in [BL, TL]
    - is flat (1.0) in [TL, TR]
    - falls linearly from 1.0->0.0 in [TR, BR]
    """
    BL, TL, TR, BR = cast(
        tuple[np.float32, np.float32, np.float32, np.float32],
        np.array(color_range, dtype=np.float32) / _360,
    )

    # hue is treated circularly (mod 360)
    centered_hue = (hue - BL) % _1
    left_supp = (TL - BL) % _1
    center_supp = (TR - BL) % _1
    right_supp = (BR - BL) % _1

    # values are 0.0 outside the trapezoid support
    mask = np.zeros_like(centered_hue)

    center = (centered_hue >= left_supp) & (centered_hue <= center_supp)
    mask[center] = _1

    left = (centered_hue >= _0) & (centered_hue < left_supp)
    mask[left] = np.divide(
        centered_hue[left],
        left_supp,
        out=np.zeros_like(centered_hue[left]),
        where=left_supp > _FLOAT_EPSILON,
    )

    right = (centered_hue > center_supp) & (centered_hue <= right_supp)
    mask[right] = np.divide(
        right_supp - centered_hue[right],
        right_supp - center_supp,
        out=np.zeros_like(centered_hue[right]),
        where=(right_supp - center_supp) > _FLOAT_EPSILON,
    )

    return mask.astype(np.float32, copy=False)


@_preserve_alpha
def apply_invert(
    img: np.ndarray,
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB],
    layer: Invert,
) -> np.ndarray:
    """Apply an invert adjustment to an image."""
    return _1 - img


@_preserve_alpha
def apply_posterize(
    img: np.ndarray,
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB],
    layer: Posterize,
) -> np.ndarray:
    """Apply a posterize adjustment to an image."""
    if layer.posterize is None:
        return img

    lut_size = _get_lut_size(layer)
    # layer.posterize is a 1–255 integer from the PSD spec. The [2, 255] clamp is
    # intentional: PS minimum is 2. This range is correct for both 8-bit and 16-bit
    # because correction_factor below compensates for bit depth — the level count
    # itself does not change with bit depth. Do not widen the upper bound.
    levels = max(2, min(layer.posterize, 255))
    correction_factor = (
        (lut_size - _1) / lut_size
    )  # including this factor makes the output more accurate, on 8 bits and 16 bits

    level = np.floor(correction_factor * img * levels)
    if lut_size == 256:
        # Photoshop truncates an 8-bit level to a whole byte, which the
        # compositor's rounding cast then keeps.
        return np.floor(level * _255 / (levels - _1)) / _255
    return level / (levels - _1)


@_preserve_alpha
def apply_threshold(
    img: np.ndarray,
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB],
    layer: Threshold,
) -> np.ndarray:
    """Apply a threshold adjustment to an image."""
    # CMYK requires accurate luminance conversion
    if colormode == ColorMode.CMYK:
        logger.info("Threshold isn't currently supported for CMYK.")
        return img

    if layer.threshold is None:
        return img

    lut_size = _get_lut_size(layer)

    correction_factor = (lut_size - _1) / lut_size
    bit_factor = (lut_size - 1) / _255  # constant used to compensate values on 16 bits
    threshold = (layer.threshold - _THRESHOLD_OFFSET) * correction_factor * bit_factor

    luminance = np.round(_get_luminance(img, colormode) * (lut_size - 1))
    filtered = (luminance > threshold).astype(np.float32, copy=False)

    if colormode == ColorMode.RGB:
        out = np.repeat(filtered, 3, axis=2)
        return out

    return filtered


@_preserve_alpha
def apply_channelmixer(
    img: np.ndarray,
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB],
    layer: ChannelMixer,
) -> np.ndarray | None:
    """Apply a channel mixer adjustment to an image, or None where it is not rendered."""
    if colormode != ColorMode.RGB:
        logger.info("Channel mixer isn't currently supported for %s.", colormode)
        return None

    records = _channelmixer_records(layer)
    if records is None:
        return None
    if layer.monochrome:
        records = np.repeat(records[:1], 3, axis=0)

    weights = records[:3, :3]
    constant = records[:3, _MIXER_FIELDS - 1]
    out = _dot3(img, weights.T) + constant
    return out.clip(_0, _1).astype(np.float32, copy=False)


# A channel mixer block is four output records of five int16 percentages: the
# weights of the red, green, blue and fourth (CMYK black) source channels, then
# a constant. The low-level reader parses the first record only; the others sit
# in its trailing bytes.
_MIXER_RECORDS = 4
_MIXER_FIELDS = 5


def _channelmixer_records(layer: ChannelMixer) -> NDArray[np.float32] | None:
    block = layer._data
    data = layer.data
    if block is None or data is None or len(data) < _MIXER_FIELDS:
        return None
    rest = (_MIXER_RECORDS - 1) * _MIXER_FIELDS
    trailing: bytes = block.unknown
    if len(trailing) < 2 * rest:
        logger.debug("Channel mixer block is short: %d bytes.", len(trailing))
        return None
    others = struct.unpack(f">{rest}h", trailing[: 2 * rest])
    records = np.array(list(data[:_MIXER_FIELDS]) + list(others), dtype=np.float32)
    return records.reshape(_MIXER_RECORDS, _MIXER_FIELDS) / _100


@_preserve_alpha
def apply_blackandwhite(
    img: np.ndarray,
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB],
    layer: BlackAndWhite,
) -> np.ndarray | None:
    """Apply a black & white adjustment to an image, or None where it is not rendered."""
    if colormode != ColorMode.RGB:
        logger.info("Black & White isn't currently supported for %s.", colormode)
        return None
    if layer._data is None:
        return None
    if layer.use_tint:
        logger.info("Black & White tint is not rendered.")

    # The gray of a pixel is its minimum, plus the span between its minimum and
    # middle channel weighted by the slider of the hue those two channels mix
    # to, plus the span to its maximum weighted by the slider of the maximum
    # channel's primary.
    primary = np.array([layer.red, layer.green, layer.blue], dtype=np.float32) / _100
    secondary = (
        np.array([layer.cyan, layer.magenta, layer.yellow], dtype=np.float32) / _100
    )
    high = img.max(axis=2, keepdims=True)
    low = img.min(axis=2, keepdims=True)
    mid = img.sum(axis=2, keepdims=True) - high - low
    weight_high = primary[img.argmax(axis=2)][..., None]
    weight_mid = secondary[img.argmin(axis=2)][..., None]
    gray = low + (mid - low) * weight_mid + (high - mid) * weight_high
    np.clip(gray, _0, _1, out=gray)
    return np.repeat(gray, 3, axis=2).astype(np.float32, copy=False)


@_preserve_alpha
def apply_gradientmap(
    img: np.ndarray,
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB],
    layer: GradientMap,
) -> np.ndarray | None:
    """Apply a gradient map adjustment to an image, or None where it is not rendered."""
    lut = _gradientmap_lut(layer, colormode)
    if lut is None:
        return None

    luminance = np.clip(_gradientmap_luminance(img, colormode), 0, 1)
    if layer.reversed:
        luminance = _1 - luminance
    size = lut.shape[0]
    if size == _GRADIENTMAP_STEPS:
        # Read at rint(luminance * 4096), as Photoshop's table is.
        index = np.minimum(np.rint(luminance[..., 0] * np.float32(size)), size - 1)
    else:
        index = np.rint(luminance[..., 0] * np.float32(size - 1))
    return lut[index.astype(np.intp)]


def _gradientmap_luminance(
    img: np.ndarray,
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB],
) -> np.ndarray:
    if colormode == ColorMode.CMYK:
        # The canvas holds what is left of each ink, so the naive RGB of a
        # pixel is its CMY leftovers scaled by the K leftover.
        return _lum(img[..., :3] * img[..., 3:4])
    return _get_luminance(img, colormode)


# A gradient map stop's colour space id and the scale of its components. A
# gray component is ink, so 0 is white.
_STOP_SPACE_RGB = 0
_STOP_SPACE_GRAYSCALE = 8
_STOP_COMPONENT_MAX = np.float32(65535.0)
_STOP_GRAY_MAX = np.float32(10000.0)
_STOP_LOCATION_MAX = 4096.0
# The block's smoothness at 100%, as a gradient descriptor's ``Intr`` is.
_SMOOTHNESS_MAX = 4096.0
# Photoshop's table for an 8-bit document: 4096 steps, k / 4096.
_GRADIENTMAP_STEPS = 4096


@functools.lru_cache(maxsize=1)
def _gradientmap_steps() -> NDArray[np.float32]:
    return np.arange(_GRADIENTMAP_STEPS, dtype=np.float32) / np.float32(
        _GRADIENTMAP_STEPS
    )


def _gradientmap_lut(
    layer: GradientMap,
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB],
) -> NDArray[np.float32] | None:
    block = layer._data
    stops = layer.color_stops
    if block is None or not stops:
        return None
    if block.mode != 0:
        logger.info("Noise gradient maps are not rendered.")
        return None
    locations: list[float] = []
    midpoints: list[float] = []
    colors: list[list[float]] = []
    for stop in stops:
        if stop.mode == _STOP_SPACE_RGB:
            rgb = [component / _STOP_COMPONENT_MAX for component in stop.color[:3]]
        elif stop.mode == _STOP_SPACE_GRAYSCALE:
            rgb = [1.0 - stop.color[0] / _STOP_GRAY_MAX] * 3
        else:
            logger.info(
                "Gradient map stop colour space %d is not supported.", stop.mode
            )
            return None
        locations.append(stop.location / _STOP_LOCATION_MAX)
        midpoints.append(float(stop.midpoint) / 100.0)
        colors.append(rgb)

    size = _get_lut_size(layer)
    smoothness = min(max(float(block.interpolation) / _SMOOTHNESS_MAX, 0.0), 1.0)
    ramp = _gradient_ramp(
        _gradientmap_steps() if size == 256 else _lut_domain(size),
        np.array(locations),
        np.array(midpoints),
        np.clip(np.array(colors, dtype=np.float32), _0, _1),
        block.method,
        smoothness,
    )
    return _rows_from_rgb(colormode, ramp)


def _rows_from_rgb(
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB],
    rgb: NDArray[np.float32],
) -> NDArray[np.float32]:
    """``paint._from_rgb`` over the rows of an (N, 3) array."""
    if colormode == ColorMode.RGB:
        return rgb
    if colormode == ColorMode.CMYK:
        ink = _1 - rgb
        key = ink.min(axis=1, keepdims=True)
        cmy = (ink - key) / np.maximum(_1 - key, _FLOAT_EPSILON)
        return (_1 - np.concatenate([cmy, key], axis=1)).astype(np.float32, copy=False)
    return (rgb @ _BT601_LUMINANCE)[:, None].astype(np.float32, copy=False)


_BT601_LUMINANCE = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _gradient_ramp(
    t: NDArray[np.float32],
    locations: NDArray[np.float64],
    midpoints: NDArray[np.float64],
    colors: NDArray[np.float32],
    method: bytes,
    smoothness: float,
) -> NDArray[np.float32]:
    """Sample a stop gradient at ``t``; ends are clamped to the end stops.

    Stops are taken in location order. Each segment blends as a gradient
    fill's does (``paint._smooth_segments``), in the space the method names:
    perceptual and smooth in Oklab, linear in linear light, classic and the
    rest on the sRGB values. Two stops at one location make a hard edge: the
    segment into it ends in the first stop's colour and the segment out of it
    starts in the second's.
    """
    if len(locations) == 1:
        return np.repeat(colors[:1], len(t), axis=0)
    order = np.argsort(locations, kind="stable")
    locations, midpoints = locations[order], midpoints[order]
    space = paint._BLEND_SPACES.get(method)
    values = colors[order] if space is None else space[0](colors[order])
    # The segment holding each sample starts at the last stop at or before it,
    # so a hard edge's zero-length segment is never picked.
    k = np.clip(np.searchsorted(locations, t, side="right") - 1, 0, len(locations) - 2)
    left, span = locations[k], locations[k + 1] - locations[k]
    u = np.divide(t - left, span, out=np.ones_like(left), where=span > 0)
    f = paint._segment_blend(u, midpoints[k + 1])
    out = paint._smooth_segments(values, k, f[:, None], smoothness)
    out[t < locations[0]] = values[0]
    out[t >= locations[-1]] = values[-1]
    if space is not None:
        out = space[1](out)
    return np.clip(out, _0, _1).astype(np.float32, copy=False)


@_preserve_alpha
def apply_colorbalance(
    img: np.ndarray,
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB],
    layer: ColorBalance,
) -> np.ndarray | None:
    """Apply a color balance adjustment to an image, or None where it is not rendered."""
    if colormode != ColorMode.RGB:
        logger.info("Color balance isn't currently supported for %s.", colormode)
        return None
    shadows, midtones, highlights = layer.shadows, layer.midtones, layer.highlights
    if shadows is None or midtones is None or highlights is None:
        return None
    if not any(shadows) and not any(highlights):
        # Midtones alone: Photoshop 2023's Levels gamma per channel.
        size = _get_lut_size(layer)
        curves = {
            channel + 1: _colorbalance_gamma_lut(gamma, size)
            for channel, gamma in enumerate(
                _colorbalance_gammas(midtones, bool(layer.luminosity))
            )
        }
        return _apply_luts(curves, img, colormode)

    # With shadows or highlights, a stand-in. A midtone amount is a per-channel
    # gamma. A shadow or highlight amount adds up to that many levels, weighted
    # toward its own end of the range and zero at black and white; that weight
    # is a shape choice, not a fit.
    # Preserve Luminosity keeps every channel a curve of its own: Photoshop
    # takes the midrange of the three midtone amounts off each, so that a shift
    # toward one primary darkens the other two. Shadows and highlights lose
    # their mean, a stand-in.
    amounts = np.array([shadows, midtones, highlights], dtype=np.float32)
    if layer.luminosity:
        centre = amounts.mean(axis=1)
        centre[1] = (amounts[1].max() + amounts[1].min()) / 2
        amounts -= centre[:, None]
    lows, gammas, highs = amounts[0] / _255, amounts[1] / _100, amounts[2] / _255
    level = _lut_domain(_get_lut_size(layer))
    luts: dict[int, NDArray[np.float32]] = {}
    for channel in range(3):
        value = level
        if lows[channel]:
            weight = _colorbalance_shadows(value)
            value = np.clip(value + lows[channel] * weight, _0, _1)
        if highs[channel]:
            weight = _colorbalance_shadows(_1 - value)
            value = np.clip(value + highs[channel] * weight, _0, _1)
        value = _colorbalance_midtones(value, float(gammas[channel]))
        luts[channel + 1] = value.astype(np.float32, copy=False)
    return _apply_luts(luts, img, colormode)


# ``t * (1 - t) ** 2`` peaks at t = 1/3 with 4/27.
_COLORBALANCE_PEAK = np.float32(27 / 4)
# A brightening midtone gamma leaves black on a parabola instead of the power
# curve's vertical tangent, and joins the curve where its slope falls to this.
_COLORBALANCE_TOE_SLOPE = 1.4


def _colorbalance_gammas(
    midtones: tuple[int, int, int], luminosity: bool
) -> list[float]:
    """Photoshop 2023's Levels gamma per channel for midtones alone.

    ``2 ** (amount / 100)`` to 0.01. Preserve Luminosity first takes the
    midrange of the three amounts, truncated toward zero, off each.
    """
    amounts = [int(amount) for amount in midtones]
    if luminosity:
        centre = int((max(amounts) + min(amounts)) / 2)
        amounts = [amount - centre for amount in amounts]
    return [math.floor(100 * 2.0 ** (amount / 100) + 0.5) / 100 for amount in amounts]


def _colorbalance_gamma_lut(gamma: float, lut_size: int) -> NDArray[np.float32]:
    """Levels' curve at ``gamma``, its power in float64 and cast once.

    So every CPU gets the same table (numpy's float32 power differs by CPU).
    """
    level = _lut_domain(lut_size)
    exponent = 1.0 / gamma
    lut = np.array(
        [math.pow(value, exponent) for value in level.tolist()], np.float64
    ).astype(np.float32)
    if gamma > 1.0:
        lut = _levels_gamma(level, gamma, lut_size, lut)
    return np.clip(lut, _0, _1)


def _colorbalance_shadows(level: np.ndarray) -> np.ndarray:
    """How much of a shadow shift reaches a level; a highlight shift mirrors it."""
    return _COLORBALANCE_PEAK * level * (_1 - level) ** 2


def _colorbalance_midtones(level: np.ndarray, amount: float) -> np.ndarray:
    """``level ** 2 ** -amount``; a brightening one starts from black on a parabola."""
    exponent = 2.0**-amount
    out = level**exponent
    if exponent >= 1:
        return out
    slope = _COLORBALANCE_TOE_SLOPE
    join = (slope / exponent) ** (1 / (exponent - 1))
    bend = (slope * join - join**exponent) / join**2
    toe = (slope - 2 * bend * join + bend * level) * level
    return np.where(level < join, toe, out)


@_preserve_alpha
def apply_vibrance(
    img: np.ndarray,
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB],
    layer: Vibrance,
) -> np.ndarray | None:
    """Apply a vibrance adjustment to an image, or None where it is not rendered."""
    if colormode != ColorMode.RGB:
        logger.info("Vibrance isn't currently supported for %s.", colormode)
        return None
    vibrance = layer.vibrance / 100.0
    saturation = layer.saturation / 100.0
    if vibrance == 0 and saturation == 0:
        return img
    linear = _srgb_to_linear(img)
    space = _vibrance_space(layer)
    if space is None:
        out = _vibrance(linear, vibrance, saturation)
    else:
        out = _vibrance(_dot3(linear, space.T), vibrance, saturation)
        out = np.clip(_dot3(out, np.linalg.inv(space).T), _0, _1)
    # In float64, cast once: float32 powers differ between CPUs.
    out = paint._linear_to_srgb(out.astype(np.float64, copy=False))
    return out.astype(np.float32, copy=False)


# The float32 exponent ``** SRGB_GAMMA`` applies to a float32 array.
_SRGB_GAMMA32 = np.float32(SRGB_GAMMA)


@functools.cache
def _srgb_level_powers() -> NDArray[np.float32]:
    """The sRGB curve's power at each whole 8-bit level, in float64, cast once."""
    base = (np.arange(256, dtype=np.float32) / _255 + SRGB_OFFSET) / (1.0 + SRGB_OFFSET)
    exponent = float(_SRGB_GAMMA32)
    return np.array([math.pow(b, exponent) for b in base.tolist()], np.float32)


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """``paint._srgb_to_linear`` with its float32 power the same on every CPU.

    Whole 8-bit levels read a table; other values are worked in float64.
    """
    if rgb.dtype != np.float32:
        return paint._srgb_to_linear(rgb)
    level = np.rint(rgb * _255)
    if (
        level.size
        and np.array_equal(level / _255, rgb)
        and level.min() >= 0
        and level.max() <= 255
    ):
        power = _srgb_level_powers()[level.astype(np.intp)]
    else:
        power = _powf((rgb + SRGB_OFFSET) / (1.0 + SRGB_OFFSET), _SRGB_GAMMA32)
    return np.where(rgb <= SRGB_SLOPE * SRGB_CUTOFF, rgb / SRGB_SLOPE, power)


# In linear light vibrance keeps each pixel's hue: its brightest channel H
# becomes H' and every channel c becomes H' - k (H - c). H' / H - 1 and k - 1
# depend on the pixel's encoded value (H) and linear saturation (1 - low / H)
# alone, and are read off ``vibrance_tables.npz`` (rebuilt by
# tools/vibrance_tables.py from Photoshop 2023 renders of every 8-bit colour):
# k - 1 at amounts 1/4, 1/2, 3/4 and 1 each way, read between them along a
# quadratic through the nearest three and 0 at 0; H' - H is linear in the
# amount. A positive amount spares the reds: one whose red is the brightest,
# with green and blue equal, gets the k of a smaller amount and part of the
# rise, both tabulated; toward magenta that share of the spared amount and rise
# falls off linearly with the hue, toward yellow it holds to halfway and is gone
# by three quarters of the way.
# The saturation slider then scales chroma about 0.288 R + 0.712 G,
# Photoshop's gray for it: blue has no say. Only that result is clipped.
_VIBRANCE_GRAY_WEIGHTS = np.array([0.288, 0.712, 0.0], dtype=np.float32)
# The tables' nodes: encoded values every 4 levels, and linear saturations
# crowded at both ends, where the tables bend most, then a few past 1, which a
# document profile's space can take a pixel to.
_VIBRANCE_VALUES = np.r_[np.arange(0.0, 256.0, 4.0), 255.0]
_VIBRANCE_SATURATIONS = np.r_[
    (1 - np.cos(np.pi * np.linspace(0.0, 1.0, 41))) / 2, 1.1, 1.25, 1.5, 2.0
]


def _vibrance_space(layer: Layer) -> NDArray[np.float64] | None:
    """The linear RGB Photoshop runs vibrance in, from the document's, or None.

    It is the document's own unless its profile is version 2 with a chromatic
    adaptation tag, a white other than D50 and primaries other than sRGB's.
    Photoshop then reads the primaries as adapted from D50 to that white by
    Bradford and back by scaling XYZ, which moves them a little.
    """
    resources = layer._psd.image_resources
    if Resource.ICC_PROFILE not in resources:
        return None
    profile = paint._icc_tags(resources.get_data(Resource.ICC_PROFILE))
    if profile is None:
        return None
    version, tags = profile
    if version != 2 or b"chad" not in tags:
        return None
    try:
        white = np.array(paint._icc_xyz(tags[b"wtpt"]))
        colorants = np.array(
            [paint._icc_xyz(tags[key]) for key in (b"rXYZ", b"gXYZ", b"bXYZ")]
        ).T
    except (KeyError, struct.error):
        return None
    if np.allclose(white, paint._ICC_D50, atol=2e-3) or np.allclose(
        colorants, paint._ICC_SRGB_COLORANTS, atol=2e-3
    ):
        return None
    cone = paint._BRADFORD @ white / (paint._BRADFORD @ paint._ICC_D50)
    adapt = np.linalg.solve(paint._BRADFORD, cone[:, None] * paint._BRADFORD)
    moved = (paint._ICC_D50 / white)[:, None] * adapt @ colorants
    return np.linalg.solve(colorants, moved)


def _vibrance(linear: np.ndarray, vibrance: float, saturation: float) -> np.ndarray:
    """Linear-light result of the vibrance and saturation sliders."""
    out = _vibrance_step(linear, vibrance) if vibrance else linear
    if saturation:
        gray = _dot3(out, _VIBRANCE_GRAY_WEIGHTS)
        out = gray[..., None] + (out - gray[..., None]) * (_1 + saturation)
    return np.clip(out, _0, _1)


def _dot3(rgb: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``rgb @ weights`` (3, or 3 by n), each pixel's sum in channel order.

    BLAS may fuse or block those sums by the array's shape; the compositor
    runs each distinct colour alone (``composite._adjust``).
    """
    out = rgb[..., 0:1] * weights[0]
    out += rgb[..., 1:2] * weights[1]
    out += rgb[..., 2:3] * weights[2]
    return out[..., 0] if weights.ndim == 1 else out


@functools.cache
def _vibrance_tables() -> dict[str, np.ndarray]:
    """The vibrance tables by name; see ``_vibrance_step``."""
    source = resources.files(__package__).joinpath("vibrance_tables.npz")
    with source.open("rb") as stream, np.load(stream) as data:
        return {key: data[key].astype(np.float32) for key in data.files}


def _vibrance_amount_weights(
    amount: np.ndarray | float,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Weights of the quadratic through three of the amounts 0, 1/4 ... 1.

    Returns the three weights and the index of the first amount.
    """
    position = np.asarray(amount, dtype=np.float32) * 4
    # A NaN index is 0, as on arm64 and in _blend (x86 casts it to INT64_MIN).
    first = np.nan_to_num(np.clip(np.ceil(position) - 1, 0, 2)).astype(np.intp)
    u = position - first
    return ((u - 1) * (u - 2) / 2, -u * (u - 2), u * (u - 1) / 2), first


def _vibrance_at(tables: np.ndarray, amount: float) -> np.ndarray:
    """``tables`` at amounts 1/4 ... 1, read linearly at ``amount``, held below 1/4."""
    position = (max(amount, 0.25) - 0.25) * 4
    first = min(int(position), len(tables) - 2)
    fraction = position - first
    return tables[first] * (1 - fraction) + tables[first + 1] * fraction


def _vibrance_step(linear: np.ndarray, amount: float) -> np.ndarray:
    """Linear light after the vibrance slider at ``amount``, in [-1, 1]."""
    tables = _vibrance_tables()
    size = abs(amount)
    name = "raise" if amount > 0 else "lower"
    chroma = tables[f"{name}_chroma"]
    anchors = np.concatenate([np.zeros_like(chroma[:1]), chroma])
    weights, first = _vibrance_amount_weights(size)
    at_amount = sum(w * anchors[first + j] for j, w in enumerate(weights))
    kernel = getattr(_cblend, "vibrance_step", None)
    if (
        kernel is not None
        and type(linear) is np.ndarray
        and linear.dtype in (np.float32, np.float64)
        and linear.ndim == 3
        and linear.shape[2] == 3
        and linear.flags.c_contiguous
    ):
        # The same steps per pixel, compiled; the tables above stay numpy's.
        skin_amount = skin_rise = tables[f"{name}_value"]
        if amount > 0:
            skin_amount = _vibrance_at(tables["skin_amount"], size)
            rises = np.concatenate([tables["skin_value"], np.ones_like(chroma[:1])])
            skin_rise = _vibrance_at(rises, size)
        return kernel(
            linear,
            float(amount),
            at_amount,
            tables[f"{name}_value"],
            skin_amount,
            chroma,
            skin_rise,
            _VIBRANCE_VALUES,
            _VIBRANCE_SATURATIONS,
            SRGB_SLOPE,
            SRGB_CUTOFF,
            1.0 + SRGB_OFFSET,
            1.0 / SRGB_GAMMA,
            SRGB_OFFSET,
        )
    high = linear.max(axis=-1)
    low = linear.min(axis=-1)
    span = high - low
    lit = high > 0
    sat = np.where(lit, span / np.where(lit, high, _1), _0)
    # Past white, as a document profile's space can take a pixel, the tables
    # extrapolate.
    value = SRGB_SLOPE * high
    bright = high > SRGB_CUTOFF
    value[bright] = (1.0 + SRGB_OFFSET) * high[bright] ** (
        1.0 / SRGB_GAMMA
    ) - SRGB_OFFSET
    node = _vibrance_node(value * _255, sat)
    scale = _1 + _bilinear(at_amount, node)
    rise = high * np.float32(size) * _bilinear(tables[f"{name}_value"], node)
    if amount > 0:
        red, green, blue = linear[..., 0], linear[..., 1], linear[..., 2]
        reds = np.nonzero((red >= green) & (red >= blue) & (span > 0))
        if reds[0].size:
            spared = _vibrance_spared(linear[reds], high[reds], low[reds])
            row, col, down, across = node
            at = (row[reds], col[reds], down[reds], across[reds])
            share = _bilinear(_vibrance_at(tables["skin_amount"], size), at)
            effective = size * (_1 - spared * (_1 - share))
            own = np.concatenate(
                [np.zeros((1, *effective.shape), np.float32), _bilinear(chroma, at)]
            )
            weights, first = _vibrance_amount_weights(effective)
            scale[reds] = _1 + sum(
                w * np.take_along_axis(own, (first + j)[None], 0)[0]
                for j, w in enumerate(weights)
            )
            rises = np.concatenate([tables["skin_value"], np.ones_like(chroma[:1])])
            share = _bilinear(_vibrance_at(rises, size), at)
            rise[reds] *= _1 - spared * (_1 - share)
    return (high + rise)[..., None] - scale[..., None] * (high[..., None] - linear)


def _vibrance_spared(
    linear: np.ndarray, high: np.ndarray, low: np.ndarray
) -> np.ndarray:
    """How much of a red-brightest pixel the skin tones' spare reaches, 0 to 1."""
    hue = (linear.sum(axis=-1) - high - low - low) / (high - low)
    toward_yellow = linear[..., 1] >= linear[..., 2]
    return np.where(toward_yellow, np.clip(3 - 4 * hue, _0, _1), _1 - hue)


def _vibrance_node(
    value: np.ndarray, sat: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Each pixel's cell in the tables and where in it, extrapolating past the last."""
    values, sats = _VIBRANCE_VALUES, _VIBRANCE_SATURATIONS
    row = np.clip(np.searchsorted(values, value, side="right") - 1, 0, len(values) - 2)
    col = np.clip(np.searchsorted(sats, sat, side="right") - 1, 0, len(sats) - 2)
    down = np.maximum((value - values[row]) / (values[row + 1] - values[row]), 0)
    across = np.maximum((sat - sats[col]) / (sats[col + 1] - sats[col]), 0)
    return row, col, down.astype(np.float32), across.astype(np.float32)


def _bilinear(
    table: np.ndarray, node: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    """``table`` (its last two axes the nodes) read bilinearly at ``node``."""
    row, col, down, across = node
    top = table[..., row, col] * (1 - across) + table[..., row, col + 1] * across
    bottom = (
        table[..., row + 1, col] * (1 - across) + table[..., row + 1, col + 1] * across
    )
    return top * (1 - down) + bottom * down


"""Adjustment function table."""
# A function returns None for an adjustment it does not render, so that the
# compositor leaves the canvas alone instead of blending it with itself.
AdjustmentFn = Callable[..., np.ndarray | None]
ADJUSTMENT_FUNC: dict[str, AdjustmentFn] = {
    "brightnesscontrast": apply_brightnesscontrast,
    "levels": apply_levels,
    "curves": apply_curves,
    "exposure": apply_exposure,
    "huesaturation": apply_huesaturation,
    "invert": apply_invert,
    "posterize": apply_posterize,
    "threshold": apply_threshold,
    "channelmixer": apply_channelmixer,
    "blackandwhite": apply_blackandwhite,
    "gradientmap": apply_gradientmap,
    "colorbalance": apply_colorbalance,
    "vibrance": apply_vibrance,
}
