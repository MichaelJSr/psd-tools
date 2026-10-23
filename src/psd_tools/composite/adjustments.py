"""Adjustment layer operations for the compositor."""

import functools
import logging
import struct
from collections.abc import Callable
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

# Modern brightness curve, fitted to Photoshop exports. Positive brightness
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

# Gamma values lower and upper bounds.
_GAMMA_LOWER = np.float32(0.01)
_GAMMA_UPPER = np.float32(9.99)

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
    out = img.astype(np.float32, copy=True)
    n_channels = ColorMode.channels(colormode)
    channels = range(1, n_channels + 1)

    # individual adjustments get applied independently of each other, then the master lut its applied if exists
    for channel_id in channels:
        if channel_id in luts:
            lut = luts[channel_id]
            out[:, :, channel_id - 1] = _apply_lut(img[:, :, channel_id - 1], lut)

    if colormode != ColorMode.GRAYSCALE and 0 in luts:
        out = _apply_lut(out, luts[0])

    return out


def _apply_lut(values: np.ndarray, lut: np.ndarray) -> NDArray[np.float32]:
    lut_size = int(lut.shape[0])

    if lut_size <= 2**16:
        depth = lut_size - 1
        values = np.floor(values * depth) / depth

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
    for every channel. Requires scipy for cubic spline interpolation.
    """
    from scipy import interpolate  # type: ignore[import-untyped]  # noqa: PLC0415

    if layer.use_legacy:  # these layers are skipped during composing as they are recognized as PixelLayers with no bounding box
        return img

    c = np.float32(layer.contrast) / _100
    t = _lut_domain(_get_lut_size(layer))

    x = _CONTRAST_CONTROL_X
    y = np.array(
        [_0, x[1] - c * _CONTRAST_INNER_DELTA, x[2] + c * _CONTRAST_INNER_DELTA, _1],
        dtype=np.float32,
    )
    contrast = interpolate.CubicSpline(x, y, bc_type="natural")
    lut = contrast(_brightness_curve(layer.brightness, t)).clip(0.0, 1.0)

    # Interpolated rather than truncated to the LUT's levels as _apply_luts
    # does: the curve is steep near white, and an identity adjustment must
    # leave the pixels as they are.
    return np.interp(img, t, lut).astype(np.float32, copy=False)


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

    for channel_id, channel_data in enumerate(levels_data):
        in_black = np.float32(channel_data.input_floor) / _255
        in_white = np.float32(channel_data.input_ceiling) / _255
        gamma = min(
            max(_GAMMA_LOWER, np.float32(channel_data.gamma / _100)), _GAMMA_UPPER
        )  # photoshop clamps gamma values into [0.01, 9.99]
        out_black = np.float32(channel_data.output_floor) / _255
        out_white = np.float32(channel_data.output_ceiling) / _255

        # input adjustments
        scale = (
            (in_white - in_black) if abs(in_white - in_black) > _FLOAT_EPSILON else _1
        )
        out = (t - in_black) / scale
        np.clip(out, _0, _1, out=out)

        # gamma midtone adjustment
        out = np.power(out, _1 / gamma, dtype=np.float32)
        np.clip(out, _0, _1, out=out)

        # output adjustments
        lut = out * (out_white - out_black) + out_black
        luts[channel_id] = lut.clip(_0, _1)

    return _apply_luts(luts, img, colormode)


@require_scipy
@_preserve_alpha
def apply_curves(
    img: np.ndarray,
    colormode: Literal[ColorMode.CMYK, ColorMode.GRAYSCALE, ColorMode.RGB],
    layer: Curves,
) -> np.ndarray:
    """
    Apply a curves adjustment to an image.

    Requires scipy for cubic spline interpolation.
    """
    from scipy import interpolate  # type: ignore[import-untyped]  # noqa: PLC0415

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

        cs = interpolate.CubicSpline(x, y, bc_type="natural")

        x_min, x_max = x[0], x[-1]
        t_clamped = np.clip(t, x_min, x_max)

        lut = cs(t_clamped).astype(np.float32, copy=False).clip(_0, _1)
        luts[channel_id] = lut

    return _apply_luts(luts, img, colormode)


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

    lut = cast(
        NDArray[np.float32], np.power(values, color_gamma, dtype=np.float32)
    )  # image is converted to linear color space first

    # exposure operates in linear space
    lut *= np.float32(np.exp2(exposure))
    lut = (lut + offset).clip(_0, _1)
    lut **= _1 / gamma

    # convert back to original space using inverse TRC approximation
    lut **= (_1 / color_gamma).clip(_0, _1)

    channel_id = 1 if colormode == ColorMode.GRAYSCALE else 0

    return _apply_luts({channel_id: lut}, img, colormode)


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
    hsl_master_tuple = _normalize_hsl(master)
    return (
        _huesaturation(img, color_ranges, hsl_master_tuple)
        if color_ranges or hsl_master_tuple != (_0, _0, _0)
        else img
    )


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
    out = img @ weights.T + constant
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
    index = np.rint(luminance[..., 0] * (lut.shape[0] - 1)).astype(np.intp)
    return lut[index]


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
        _lut_domain(size),
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
    perceptual in Oklab, linear in linear light, classic and the rest on the
    sRGB values. Two stops at one location make a hard edge: the segment into
    it ends in the first stop's colour and the segment out of it starts in
    the second's.
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

    # A midtone amount is a per-channel gamma. A shadow or highlight amount
    # adds up to that many levels, weighted toward its own end of the range and
    # zero at black and white; that weight is a shape choice, not a fit.
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
    linear = paint._srgb_to_linear(img)
    space = _vibrance_space(layer)
    if space is None:
        out = _vibrance(img, linear, vibrance, saturation)
    else:
        linear = linear @ space.T
        encoded = np.sign(linear) * paint._linear_to_srgb(np.abs(linear))
        out = _vibrance(encoded, linear, vibrance, saturation)
        out = np.clip(out @ np.linalg.inv(space).T, _0, _1)
    return paint._linear_to_srgb(out).astype(np.float32, copy=False)


# In linear light each pixel is scaled away from or toward a gray on the line
# through its darkest and brightest channel, which keeps its hue.
# A positive amount is a fit: it boosts low-saturation pixels most, and spares
# bright, very dark and reddish (skin) ones; its saturation is measured on the
# encoded values, and its gray sits at a position in units of the difference.
# A negative amount is fitted to Photoshop renders of every pair of darkest and
# brightest levels, which alone decide it: it lowers the linear HSV saturation
# of a pale pixel furthest and of a pure one by a quarter of the amount, more
# below an encoded value of about a quarter, and dims mid-saturated pixels away
# from white.
# The saturation slider then scales chroma about 0.288 R + 0.712 G,
# Photoshop's gray for it: blue has no say. Only that result is clipped.
_VIBRANCE_BOOST = 2.44
_VIBRANCE_BOOST_CURVE = 2.6
_VIBRANCE_BOOST_TAIL = 0.25
_VIBRANCE_BOOST_AMOUNT_CURVE = 1.33
_VIBRANCE_BOOST_BRIGHT = 0.73
_VIBRANCE_BOOST_DARK = 0.023
_VIBRANCE_BOOST_GRAY = 0.79
_VIBRANCE_BOOST_GRAY_SLOPE = -0.29
_VIBRANCE_SKIN_WIDTH = 0.058
_VIBRANCE_SKIN_CURVE = 1.8
_VIBRANCE_LOSS = 0.25
_VIBRANCE_LOSS_PALE = 1.5
_VIBRANCE_LOSS_PALE2 = -0.5
_VIBRANCE_REGAIN_PALE = 0.38
_VIBRANCE_REGAIN_PALE2 = -0.14
_VIBRANCE_DARK = 1.18
_VIBRANCE_DARK_LEVEL = 0.26
_VIBRANCE_DARK_CURVE = 1.14
_VIBRANCE_DIM = 1.79
_VIBRANCE_DIM_DARK_SCALE = 0.02
_VIBRANCE_GRAY_WEIGHTS = np.array([0.288, 0.712, 0.0], dtype=np.float32)
_ICC_D50 = np.array([0.9642, 1.0, 0.8249])
_ICC_SRGB_COLORANTS = np.array(
    [[0.4361, 0.3851, 0.1431], [0.2225, 0.7169, 0.0606], [0.0139, 0.0971, 0.7141]]
)
_BRADFORD = np.array(
    [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]]
)


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
    tags = _icc_tags(resources.get_data(Resource.ICC_PROFILE))
    if tags is None or tags.get(b"version") != 2 or b"chad" not in tags:
        return None
    try:
        white = np.array(_icc_xyz(tags[b"wtpt"]))
        colorants = np.array(
            [_icc_xyz(tags[key]) for key in (b"rXYZ", b"gXYZ", b"bXYZ")]
        ).T
    except (KeyError, struct.error):
        return None
    if np.allclose(white, _ICC_D50, atol=2e-3) or np.allclose(
        colorants, _ICC_SRGB_COLORANTS, atol=2e-3
    ):
        return None
    cone = _BRADFORD @ white / (_BRADFORD @ _ICC_D50)
    adapt = np.linalg.solve(_BRADFORD, cone[:, None] * _BRADFORD)
    moved = (_ICC_D50 / white)[:, None] * adapt @ colorants
    return np.linalg.solve(colorants, moved)


def _icc_tags(data: bytes) -> dict[bytes, bytes | int] | None:
    """An ICC profile's tags by signature, and its major version, or None."""
    try:
        (count,) = struct.unpack_from(">I", data, 128)
        tags: dict[bytes, bytes | int] = {b"version": data[8]}
        for index in range(count):
            name, start, size = struct.unpack_from(">4sII", data, 132 + 12 * index)
            tags[name] = data[start : start + size]
    except (struct.error, IndexError):
        return None
    return tags


def _icc_xyz(tag: bytes | int) -> tuple[float, float, float]:
    """The first value of an ICC ``XYZ `` tag."""
    if not isinstance(tag, bytes):
        raise KeyError(tag)
    x, y, z = struct.unpack_from(">3i", tag, 8)
    return (x / 65536, y / 65536, z / 65536)


def _vibrance(
    img: np.ndarray, linear: np.ndarray, vibrance: float, saturation: float
) -> np.ndarray:
    """Linear-light result of the vibrance and saturation sliders.

    ``img`` is ``linear`` encoded.
    """
    high = linear.max(axis=2, keepdims=True)
    low = linear.min(axis=2, keepdims=True)
    out = linear
    if vibrance > 0:
        out = _raise_vibrance(img, linear, high, low, vibrance)
    elif vibrance < 0:
        encoded_high = img.max(axis=2, keepdims=True)
        out = _lower_vibrance(linear, high, low, encoded_high, -vibrance)
    if saturation:
        gray = out @ _VIBRANCE_GRAY_WEIGHTS
        out = gray[..., None] + (out - gray[..., None]) * (_1 + saturation)
    return np.clip(out, _0, _1)


def _lower_vibrance(
    linear: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    encoded_high: np.ndarray,
    amount: float,
) -> np.ndarray:
    """Lower the linear HSV saturation and value of ``linear``, hue kept.

    ``high`` and ``low`` are its per-pixel extremes, ``encoded_high`` the
    brightest encoded channel; ``amount`` is in (0, 1].
    """
    sat = (high - low) / np.maximum(high, _FLOAT_EPSILON)
    pale = _1 - sat
    loss = _VIBRANCE_LOSS + (_VIBRANCE_LOSS_PALE + _VIBRANCE_LOSS_PALE2 * pale) * pale
    regain = (_VIBRANCE_REGAIN_PALE + _VIBRANCE_REGAIN_PALE2 * pale) * pale
    dark = np.clip(_1 - encoded_high / _VIBRANCE_DARK_LEVEL, _0, _1)
    dark = _1 - amount * _VIBRANCE_DARK * dark**_VIBRANCE_DARK_CURVE
    new_sat = np.clip(sat * (_1 - amount * loss + amount**2 * regain) * dark, _0, sat)
    dim = _1 - np.exp(-high / _VIBRANCE_DIM_DARK_SCALE)
    value = high * (_1 - amount * _VIBRANCE_DIM * (_1 - high) * dim * sat * pale)
    scale = value * new_sat / np.maximum(high * sat, _FLOAT_EPSILON)
    return value - scale * (high - linear)


def _raise_vibrance(
    img: np.ndarray,
    linear: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    vibrance: float,
) -> np.ndarray:
    """Scale chroma up, most for pale, mid-bright pixels away from red.

    ``high`` and ``low`` are the per-pixel extremes of ``linear``.
    """
    encoded_high = img.max(axis=2, keepdims=True)
    sat = (encoded_high - img.min(axis=2, keepdims=True)) / np.maximum(
        encoded_high, _FLOAT_EPSILON
    )
    sat = np.clip(sat, _0, _1)
    pale = _1 - sat
    hue = rgb2hsl(np.clip(img, _0, _1))[..., 0:1]
    from_red = np.minimum(hue, _1 - hue) / _VIBRANCE_SKIN_WIDTH
    skin = np.exp(-0.5 * from_red**2) * pale**_VIBRANCE_SKIN_CURVE
    boost = _VIBRANCE_BOOST * (
        pale**_VIBRANCE_BOOST_CURVE + _VIBRANCE_BOOST_TAIL * sat * pale
    )
    boost = boost * (_1 - _VIBRANCE_BOOST_BRIGHT * encoded_high)
    boost = boost * (_1 - np.exp(-high / _VIBRANCE_BOOST_DARK)) * (_1 - skin)
    scale = _1 + vibrance**_VIBRANCE_BOOST_AMOUNT_CURVE * boost
    position = _VIBRANCE_BOOST_GRAY + _VIBRANCE_BOOST_GRAY_SLOPE * sat
    gray = low + position * (high - low)
    return gray + scale * (linear - gray)


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
