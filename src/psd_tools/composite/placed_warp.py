"""A placed layer's custom warp drawn on a new picture of its file.

Photoshop 2023 stores a ``warpCustom`` placed layer's pixels as its file bent
by a bicubic Bezier patch: 4 x 4 control points (``meshPoints``, in the
file's pixels over its content rect ``bounds``), the points' bounding box
placed on the layer's quad (``Trnf``). :py:func:`draw` cuts the patch into
:py:data:`GRID` x :py:data:`GRID` cells and gives each pixel the file's
pixel it comes from (nearest, in one step), as the card makers' Photoshop
stored it (proven on 4 layers of 2 cards). Plain float64 numpy in a fixed
order: every CPU gives the same result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from psd_tools.api._descriptor import get_scalar
from psd_tools.composite.smart_filters import Box, Quad, _affine, _map, _warp
from psd_tools.psd.descriptor import Descriptor, Double, ObjectArray

if TYPE_CHECKING:
    from psd_tools.api.smart_object import SmartObject

GRID = 64  # cells a side the patch is cut into
_ORDER = 4  # control points a side
_TOL = 0.01  # px: bounds this close to the content size are it
_BENDS = (b"warpValue", b"warpPerspective", b"warpPerspectiveOther")
_BOUNDS = (b"Left", b"Top ", b"Rght", b"Btom")


def _parts(so: SmartObject) -> tuple[tuple[int, int], np.ndarray, np.ndarray] | None:
    """``((width, height), xs, ys)`` of a warp :py:func:`draw` draws, else None.

    *xs*, *ys*: the control points, 4 x 4 rows top to bottom.
    """
    warp = so.warp
    if not isinstance(warp, Descriptor):
        return None
    if getattr(warp.get(b"warpStyle"), "enum", None) != b"warpCustom":
        return None
    if any(get_scalar(warp, key, float, 0.0) != 0 for key in _BENDS):
        return None
    if any(
        get_scalar(warp, key, int, _ORDER) != _ORDER for key in (b"uOrder", b"vOrder")
    ):
        return None
    envelope = warp.get(b"customEnvelopeWarp")
    bounds = warp.get(b"bounds")
    size = so._config.data.get(b"Sz  ") if so._config is not None else None
    if not all(isinstance(d, Descriptor) for d in (envelope, bounds, size)):
        return None
    assert isinstance(envelope, Descriptor) and isinstance(bounds, Descriptor)
    if list(envelope.keys()) != [b"meshPoints"]:  # a split grid has more
        return None
    mesh = envelope[b"meshPoints"]
    left, top, right, bottom = (get_scalar(bounds, k, float, np.nan) for k in _BOUNDS)
    width, height = right - left, bottom - top
    if not (np.isfinite(width) and np.isfinite(height)):
        return None
    w, h = round(width), round(height)
    if not (
        abs(left) < _TOL
        and abs(top) < _TOL
        and w > 0
        and h > 0
        and abs(width - w) < _TOL
        and abs(height - h) < _TOL
        and abs(get_scalar(size, b"Wdth", float, np.nan) - w) < _TOL
        and abs(get_scalar(size, b"Hght", float, np.nan) - h) < _TOL
    ):
        return None
    points = []
    for key in (b"Hrzn", b"Vrtc"):
        found = mesh.get(key) if isinstance(mesh, (Descriptor, ObjectArray)) else None
        values = getattr(found, "values", None)
        if values is None or len(values) != _ORDER * _ORDER:
            return None
        points.append(np.array([float(v) for v in values]).reshape(_ORDER, _ORDER))
    if not all(np.isfinite(p).all() for p in points):
        return None
    return (w, h), points[0], points[1]


def _mapping(
    xs: np.ndarray, ys: np.ndarray, quad: Quad | None
) -> tuple[float, ...] | None:
    """The map of the control points' bounding box onto *quad*."""
    if quad is None:
        return None
    x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
    return _affine([x0, y0, x1, y0, x1, y1, x0, y1], quad)


def drawable(so: SmartObject) -> bool:
    """Whether :py:func:`draw` draws the object's warp.

    A ``warpCustom`` of 4 x 4 points over its whole content (no split grid,
    no preset bend), on a quad that is a parallelogram.
    """
    parts = _parts(so)
    return (
        parts is not None and _mapping(parts[1], parts[2], so.transform_box) is not None
    )


def update(so: SmartObject, width: int, height: int, ppi: float) -> Quad:
    """Put the object's warp on a new picture of *width* x *height* pixels.

    As Photoshop 2023's Update Modified Content writes it (4 upright layers
    of 2 cards at 72 ppi, exact): ``Sz`` and every copy of the warp's
    ``bounds`` the new size, the mesh kept; the quad's corner and size each
    rounded to whole pixels, so the new picture fills the same box. The same
    size changes nothing.

    :param ppi: the new file's resolution
    :return: the layer's quad (TL TR BR BL), for the caller to write
    :raises ValueError: when :py:func:`drawable` is False or, at a new size,
        the quad is no upright rectangle or *ppi* is not the layer's
        ``Rslt`` (unproven)
    """
    parts = _parts(so)
    quad = so.transform_box
    if parts is None or _mapping(parts[1], parts[2], quad) is None:
        raise ValueError("this warp is not drawn")
    assert quad is not None and so._config is not None
    if (width, height) == parts[0]:
        return tuple(float(v) for v in quad)  # type: ignore[return-value]
    if width <= 0 or height <= 0:
        raise ValueError("a file has pixels")
    x0, y0, x1, y1, x2, y2, x3, y3 = (float(v) for v in quad)
    if not (
        abs(y0 - y1) < _TOL
        and abs(y2 - y3) < _TOL
        and abs(x0 - x3) < _TOL
        and abs(x1 - x2) < _TOL
        and x1 > x0
        and y3 > y0
    ):
        raise ValueError("the layer's quad is no upright rectangle")
    stored = get_scalar(so._config.data, b"Rslt", float, None)
    if stored is None or abs(stored - ppi) > _TOL:
        raise ValueError("the file's ppi is not the layer's")
    left, top = np.floor(x0 + 0.5), np.floor(y0 + 0.5)
    right = left + np.floor(x1 - x0 + 0.5)
    bottom = top + np.floor(y3 - y0 + 0.5)
    size = so._config.data[b"Sz  "]
    size[b"Wdth"], size[b"Hght"] = Double(float(width)), Double(float(height))
    for warp in (so.warp, getattr(so._placed_layer, "warp", None)):
        bounds = warp.get(b"bounds") if isinstance(warp, Descriptor) else None
        if isinstance(bounds, Descriptor):  # an old copy shows on the next update
            for key, value in zip(_BOUNDS, (0, 0, width, height), strict=True):
                bounds[key] = Double(float(value))
    out = (left, top, right, top, right, bottom, left, bottom)
    return tuple(float(v) for v in out)  # type: ignore[return-value]


def _bernstein(t: np.ndarray) -> list[np.ndarray]:
    s = 1.0 - t
    return [s * s * s, 3.0 * (t * s * s), 3.0 * (t * t * s), t * t * t]


def _surface(
    points: np.ndarray, bu: list[np.ndarray], bv: list[np.ndarray]
) -> np.ndarray:
    """The patch's coordinate at each (v, u), summed in a fixed order."""
    out = np.zeros(np.broadcast_shapes(bu[0].shape, bv[0].shape))
    for i in range(_ORDER):
        row = np.zeros_like(bu[0])
        for j in range(_ORDER):
            row = row + points[i, j] * bu[j]
        out = out + bv[i] * row
    return out


def draw(
    so: SmartObject, rgba: np.ndarray, corners: Quad | None = None
) -> tuple[np.ndarray, Box]:
    """Draw the object's warp on a new picture of its file.

    :param rgba: the file's pixels, straight RGBA ``(h, w, 4)``, uint8 or
        float in 0..1, at the warp's content size
    :param corners: the layer's quad (TL TR BR BL); by default its
        :py:attr:`~SmartObject.transform_box`
    :return: ``(straight float32 RGBA, canvas box)``, the box trimmed to the
        pixels that show; ``(0, 0, 0, 0)`` with an empty array when none do
    :raises ValueError: when :py:func:`drawable` is False or *rgba* is
        another size
    """
    parts = _parts(so)
    if parts is None:
        raise ValueError("this warp is not drawn")
    (w, h), xs, ys = parts
    mapped = _mapping(xs, ys, corners if corners is not None else so.transform_box)
    if mapped is None:
        raise ValueError("the layer's quad is no parallelogram")
    pic = np.asarray(rgba)
    if pic.ndim != 3 or pic.shape[2] != 4 or pic.shape[:2] != (h, w):
        raise ValueError(f"the picture is not {w} x {h} RGBA")
    if pic.dtype == np.uint8:
        pic = pic.astype(np.float32) / np.float32(255)
    t = np.arange(GRID + 1) / GRID
    u, v = t[None, :], t[:, None]
    bu, bv = _bernstein(u), _bernstein(v)
    deformed = np.stack([_surface(xs, bu, bv).ravel(), _surface(ys, bu, bv).ravel()], 1)
    grid = (GRID + 1, GRID + 1)
    orig = np.stack(
        [np.broadcast_to(u * w, grid).ravel(), np.broadcast_to(v * h, grid).ravel()], 1
    )
    idx = np.arange((GRID + 1) ** 2).reshape(grid)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
    tri = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    mesh = (orig, _map(deformed, mapped), tri)
    return _warp(pic, (0, 0, w, h), mesh, (0.0, 0.0, 1.0, 0.0, 0.0, 1.0), True)
