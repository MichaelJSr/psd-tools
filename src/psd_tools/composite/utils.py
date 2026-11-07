"""Utility functions for composite operations."""

from collections.abc import Sequence
from typing import Any, overload

import numpy as np
from numpy.typing import NDArray

from psd_tools.api.layers import Layer
from psd_tools.constants import Tag

try:  # Compiled loops, bit-identical to the numpy ones they stand in for.
    from psd_tools.composite import _blend as _cblend  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    _cblend = None

# A divisor or a fill is usually a canvas, but ``draw_stroke_effect()``
# normalizes by a NumPy scalar and every caller may pass a plain float.
_Scalable = NDArray[np.floating] | np.floating[Any] | float


def divide(
    a: NDArray[np.floating],
    b: _Scalable,
    fill: _Scalable = 1.0,
) -> NDArray[np.floating]:
    """Divide ``a`` by ``b``, substituting ``fill`` where ``b`` is not positive.

    Every divisor in the compositor is an alpha or a coverage, so a zero one
    means "nothing here" rather than an arithmetic accident, and the quotient
    is undefined at exactly those pixels. What belongs there instead is the
    caller's to say: un-premultiplying a color wants a color to fall back to,
    while a ratio of two alphas wants an opacity. The default 1.0 reads as
    white in normalized color space and as fully opaque in alpha, which is what
    every caller that does not pass ``fill`` relies on.

    ``fill`` may be a full canvas as well as a scalar, so a caller can fall
    back per pixel to something it already has in hand.

    Skipping those pixels via ``where=`` also avoids computing an invalid
    quotient only to overwrite it -- the divisor is never negative here, so
    ``b > 0`` and "b is nonzero" are the same test.
    """
    # hasattr: a _blend built before this loop runs numpy's.
    if hasattr(_cblend, "divide") and in_rows((b,), (a,)):
        if type(fill) is float:
            return _cblend.divide(a, b, None, fill, True)
        if in_rows((b,), (a, fill)):
            return _cblend.divide(a, b, fill, 0.0, False)
    shape = np.broadcast_shapes(np.shape(a), np.shape(b))
    dtype = np.result_type(a, b)
    positive = np.asarray(b) > 0
    if shape and positive.all():
        # Every quotient is wanted: a plain divide, without the masked loop.
        if np.shape(a) == shape:
            return np.divide(a, b, dtype=dtype)
        return np.divide(np.broadcast_to(a, shape), b, dtype=dtype)
    out = np.full(shape, fill, dtype=dtype)
    return np.divide(a, b, out=out, where=positive)


def intersect(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """Calculate intersection of two bounding boxes."""
    inter = (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))
    if inter[0] >= inter[2] or inter[1] >= inter[3]:
        return (0, 0, 0, 0)
    return inter


def union_bbox(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """The smallest bounding box containing both, ignoring an empty one.

    The bounding-box twin of :py:func:`intersect`, and unrelated to
    :py:func:`union`, which is the generalized union of two *coverages*.

    An empty box means "nothing here" rather than a box at the origin, so it
    contributes nothing: growing a real box to reach ``(0, 0, 0, 0)`` would
    drag it to the top-left corner of the canvas.
    """
    if a[0] >= a[2] or a[1] >= a[3]:
        return b
    if b[0] >= b[2] or b[1] >= b[3]:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def is_fill_disabled(layer: Layer) -> bool:
    """Check if a shape layer's fill is switched off, whichever tag holds it."""
    stroke = layer.tagged_blocks.get_data(Tag.VECTOR_STROKE_DATA)
    return bool(stroke) and getattr(stroke.get("fillEnabled"), "value", True) is False


def has_fill(layer: Layer) -> bool:
    """Check if layer has fill settings."""
    FILL_TAGS = (
        Tag.SOLID_COLOR_SHEET_SETTING,
        Tag.PATTERN_FILL_SETTING,
        Tag.GRADIENT_FILL_SETTING,
        Tag.VECTOR_STROKE_CONTENT_DATA,
    )
    return any(tag in layer.tagged_blocks for tag in FILL_TAGS)


@overload
def union(backdrop: float, source: float) -> float: ...


@overload
def union(
    backdrop: NDArray[np.floating], source: NDArray[np.floating]
) -> NDArray[np.floating]: ...


@overload
def union(backdrop: float, source: NDArray[np.floating]) -> NDArray[np.floating]: ...


@overload
def union(backdrop: NDArray[np.floating], source: float) -> NDArray[np.floating]: ...


def union(
    backdrop: float | NDArray[np.floating],
    source: float | NDArray[np.floating],
) -> float | NDArray[np.floating]:
    """Generalized union of shape."""
    return backdrop + source - (backdrop * source)


def clip(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Clip between [0, 1]."""
    if hasattr(_cblend, "clip") and in_rows((), (x,)):
        return _cblend.clip(x)
    return np.clip(x, 0.0, 1.0)


def in_rows(
    ones: Sequence[object], wide: Sequence[object] = (), channels: int | None = None
) -> bool:
    """Whether :py:mod:`._blend` takes these operands as numpy would.

    Every one a float32 canvas of one height and width, ``ones`` a single
    channel and ``wide`` ``channels``, by default the first wide one's, laid
    out in rows of pixels as a box cut from a canvas is; anything else, a
    scalar, a float64 or a broadcast canvas, stays with numpy's own promotion
    and broadcasting.
    """
    first = ones[0] if ones else wide[0]
    if not isinstance(first, np.ndarray) or first.ndim != 3:
        return False
    size = first.shape[:2]
    if channels is None:
        channels = 1
        if wide:
            if not isinstance(wide[0], np.ndarray) or wide[0].ndim != 3:
                return False
            channels = wide[0].shape[2]
    for array, width in [(a, 1) for a in ones] + [(a, channels) for a in wide]:
        if not (
            isinstance(array, np.ndarray)
            and array.dtype == np.float32
            and array.shape == (*size, width)
            and (width == 1 or array.strides[2] == 4)
            and (size[1] < 2 or array.strides[1] == 4 * width)
        ):
            return False
    return True
