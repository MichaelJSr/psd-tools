"""Photoshop 2023's Reduce Noise at Strength 10, Preserve Details 0%.

Photoshop takes the picture to Lab (D50, the document's values as sRGB) and
smooths L alone, a and b kept (Reduce Color Noise 0%). L becomes a base plus
:py:data:`KEEP` of the detail over it. The base is a Gaussian along each
pixel's isophote (the structure tensor's) where the picture has a direction,
an isotropic Gaussian where it has none, mixed by the tensor's coherence:
edges and lines stay, grain and small bumps fade.

Fitted on Photoshop 2023's own pixels (Burly Bomb's stored layer; noise,
bumps, edges, gratings and pictures under the filter): near its result by
eye, not level for level (its own runs differ by its dither).

Plain float64 numpy in a fixed order, sRGB through tables made in decimal and
cube roots by Newton's method, so every CPU gives the same result.
"""

from __future__ import annotations

import math
from decimal import Context, Decimal

import numpy as np

from psd_tools.color_convert import (
    _LAB_D50,
    _LAB_EPS,
    _LAB_KAPPA,
    _SRGB_TO_XYZ_D50,
    _XYZ_D50_TO_SRGB,
    SRGB_GAMMA,
    SRGB_OFFSET,
    SRGB_SLOPE,
)

#: Share of L's detail over the base kept.
KEEP = 0.2
_REACH = 12  # px each way along the isophote
_ALONG = 6.0  # its Gaussian's sigma
_ROUND = 2.5  # the isotropic Gaussian's sigma
_PRE = 0.5  # sigma L is smoothed by before its gradients
_TENSOR = 2.0  # sigma the tensor is smoothed by
_TINY = 1e-12


def _decoded(v: Decimal, context: Context) -> Decimal:
    """Linear light of the sRGB value *v* (0..1)."""
    if v <= Decimal(repr(SRGB_SLOPE)) * Decimal("0.0031308"):
        return v / Decimal(repr(SRGB_SLOPE))
    offset = Decimal(repr(SRGB_OFFSET))
    return context.power((v + offset) / (1 + offset), Decimal(repr(SRGB_GAMMA)))


def _tables() -> tuple[np.ndarray, np.ndarray]:
    """Each level's linear light, and the linear light each level ends at."""
    context = Context(prec=34)
    levels = [float(_decoded(Decimal(i) / 255, context)) for i in range(256)]
    ends = [
        float(_decoded((Decimal(i) + Decimal("0.5")) / 255, context))
        for i in range(255)
    ]
    return np.array(levels), np.array(ends)


_LINEAR, _ENDS = _tables()


def _cbrt(t: np.ndarray) -> np.ndarray:
    """Cube root of positive *t*: Newton's method from the exponent's third."""
    mantissa, exponent = np.frexp(t)
    third, rest = np.divmod(exponent, 3)
    m = np.ldexp(mantissa, rest)  # 0.5 .. 4
    y = 0.681 + 0.2266 * m
    for _ in range(6):
        y = (y + y + m / (y * y)) / 3.0
    return np.ldexp(y, third)


def _f(t: np.ndarray) -> np.ndarray:
    return np.where(
        t > _LAB_EPS,
        _cbrt(np.maximum(t, _LAB_EPS)),
        (_LAB_KAPPA * t + 16.0) / 116.0,
    )


def _f_inverse(f: np.ndarray) -> np.ndarray:
    cube = f * f * f
    return np.where(cube > _LAB_EPS, cube, (116.0 * f - 16.0) / _LAB_KAPPA)


def _lab(levels: np.ndarray) -> list[np.ndarray]:
    """L, a, b of whole sRGB *levels* ``(h, w, 3)``."""
    linear = [_LINEAR[levels[..., k]] for k in range(3)]
    f = []
    for row, white in zip(_SRGB_TO_XYZ_D50, _LAB_D50, strict=True):
        f.append(
            _f((row[0] * linear[0] + row[1] * linear[1] + row[2] * linear[2]) / white)
        )
    return [116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])]


def _levels(lightness: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Whole sRGB levels ``(h, w, 3)`` of Lab, rounded, out of gamut clipped."""
    fy = (lightness + 16.0) / 116.0
    xyz = [
        _f_inverse(f) * white
        for f, white in zip((fy + a / 500.0, fy, fy - b / 200.0), _LAB_D50, strict=True)
    ]
    out = np.empty((*lightness.shape, 3), np.int64)
    for k, row in enumerate(_XYZ_D50_TO_SRGB):
        linear = row[0] * xyz[0] + row[1] * xyz[1] + row[2] * xyz[2]
        out[..., k] = np.searchsorted(_ENDS, linear, side="right")
    return out


def _weights(sigma: float, reach: int) -> list[float]:
    """Gaussian weights at 0 .. *reach*, the centre's once, summing to 1."""
    context = Context(prec=34)
    raw = [
        float(context.exp(Decimal(-0.5 / (sigma * sigma)) * Decimal(k * k)))
        for k in range(reach + 1)
    ]
    total = math.fsum(raw + raw[1:])
    return [w / total for w in raw]


def _blur(plane: np.ndarray, sigma: float) -> np.ndarray:
    """A separable Gaussian, 3 sigma each way, edges repeated, float64."""
    taps = _weights(sigma, int(3.0 * sigma + 0.5))
    for axis in (0, 1):
        reach, n = len(taps) - 1, plane.shape[axis]
        widths = [(0, 0), (0, 0)]
        widths[axis] = (reach, reach)
        src = np.pad(plane, widths, mode="edge")
        at = (
            (lambda k: src[reach + k : reach + k + n])
            if axis == 0
            else (lambda k: src[:, reach + k : reach + k + n])
        )
        total = at(0) * taps[0]
        for k in range(reach, 0, -1):
            total = total + (at(-k) + at(k)) * taps[k]
        plane = total
    return plane


def _gradients(plane: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sobel x and y over 8, edges repeated."""
    p = np.pad(plane, 1, mode="edge")
    h, w = plane.shape

    def at(dy: int, dx: int) -> np.ndarray:
        return p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]

    gx = (at(-1, 1) + 2.0 * at(0, 1) + at(1, 1)) - (
        at(-1, -1) + 2.0 * at(0, -1) + at(1, -1)
    )
    gy = (at(1, -1) + 2.0 * at(1, 0) + at(1, 1)) - (
        at(-1, -1) + 2.0 * at(-1, 0) + at(-1, 1)
    )
    return gx / 8.0, gy / 8.0


def _sample(
    planes: tuple[np.ndarray, ...], y: np.ndarray, x: np.ndarray
) -> list[np.ndarray]:
    """Each of *planes* at points, bilinear, coordinates clamped to them."""
    h, w = planes[0].shape
    y = np.clip(y, 0.0, h - 1.0)
    x = np.clip(x, 0.0, w - 1.0)
    iy = np.minimum(np.floor(y), max(h - 2, 0)).astype(np.int64)
    ix = np.minimum(np.floor(x), max(w - 2, 0)).astype(np.int64)
    fy, fx = y - iy, x - ix
    down, right = np.minimum(iy + 1, h - 1) * w, np.minimum(ix + 1, w - 1)
    iy = iy * w
    out = []
    for plane in planes:
        flat = plane.ravel()
        top = flat[iy + ix] * (1.0 - fx) + flat[iy + right] * fx
        bottom = flat[down + ix] * (1.0 - fx) + flat[down + right] * fx
        out.append(top * (1.0 - fy) + bottom * fy)
    return out


def _mean(num: np.ndarray, den: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """*num* over *den* (an alpha-weighted mean); *fallback* where none shows."""
    shown = den > _TINY
    return np.where(shown, num / np.where(shown, den, 1.0), fallback)


def _base(lightness: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """L's base: along the isophote by coherence, else round.

    Means weighted by *alpha*: a clear pixel's colour counts for nothing, as
    in Photoshop (its tensor sees it filled from its neighbours).
    """
    weighted = alpha * lightness
    filled = np.where(
        alpha > 0,
        lightness,
        _mean(_blur(weighted, _ROUND), _blur(alpha, _ROUND), lightness),
    )
    gx, gy = _gradients(_blur(filled, _PRE))
    jxx, jxy, jyy = (
        _blur(gx * gx, _TENSOR),
        _blur(gx * gy, _TENSOR),
        _blur(gy * gy, _TENSOR),
    )
    trace = jxx + jyy
    spread = np.sqrt((jxx - jyy) * (jxx - jyy) + 4.0 * jxy * jxy)
    coherence = np.where(trace > _TINY, spread / np.maximum(trace, _TINY), 0.0)
    least = (trace - spread) / 2.0
    wide = jxx >= jyy  # the isophote, the least eigenvalue's vector
    vx = np.where(wide, jxy, jyy - least)
    vy = np.where(wide, least - jxx, -jxy)
    norm = np.sqrt(vx * vx + vy * vy)
    flat = norm <= _TINY
    vx = np.where(flat, 1.0, vx / np.where(flat, 1.0, norm))
    vy = np.where(flat, 0.0, vy / np.where(flat, 1.0, norm))
    h, w = lightness.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    taps = _weights(_ALONG, _REACH)
    planes = (np.ascontiguousarray(weighted), np.ascontiguousarray(alpha))
    num, den = weighted * taps[0], alpha * taps[0]
    for k in range(_REACH, 0, -1):
        (n0, d0), (n1, d1) = (
            _sample(planes, yy - k * vy, xx - k * vx),
            _sample(planes, yy + k * vy, xx + k * vx),
        )
        num = num + (n0 + n1) * taps[k]
        den = den + (d0 + d1) * taps[k]
    along = _mean(num, den, lightness)
    round_ = _mean(_blur(weighted, _ROUND), _blur(alpha, _ROUND), lightness)
    return coherence * along + (1.0 - coherence) * round_


def reduce_noise(rgba: np.ndarray) -> np.ndarray:
    """Straight float RGBA ``(h, w, 4)`` under Reduce Noise, alpha kept.

    Colours on whole levels, as Photoshop's 8-bit filter source.
    """
    out = np.asarray(rgba, np.float32).copy()
    if not out.size:
        return out
    levels = np.rint(np.clip(out[..., :3].astype(np.float64), 0.0, 1.0) * 255).astype(
        np.int64
    )
    lightness, a, b = _lab(levels)
    base = _base(lightness, np.clip(out[..., 3].astype(np.float64), 0.0, 1.0))
    smoothed = base + KEEP * (lightness - base)
    out[..., :3] = _levels(smoothed, a, b) / np.float32(255)
    return out
