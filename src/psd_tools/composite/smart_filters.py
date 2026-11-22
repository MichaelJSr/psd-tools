"""Smart filters drawn on a new picture of a placed layer.

A placed layer's pixels are its smart filters' result. When its file changes,
the caller draws the new picture on the layer's quad and :py:func:`draw` draws
the filters over it as Photoshop 2023 does: a Puppet Warp from its stored mesh,
an Invert, a Gaussian Blur, a Motion Blur at angle 0 of a distance
:py:data:`MOTION` holds, a Mosaic, a Hue/Saturation (master only), a Black &
White (no tint) and a Relative Selective Color of the six hues.
Plain float64 numpy, element by element in a fixed order, and exactly rounded
sums (no BLAS), so every CPU gives the same result.

A filter mask does not change what is drawn: Photoshop stores the filters'
whole result as the layer's pixels, and the composite blends them with the
filter source through the mask.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from decimal import Context, Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from psd_tools.api._descriptor import get_scalar
from psd_tools.composite import adjustments
from psd_tools.constants import ColorMode
from psd_tools.psd.descriptor import Descriptor

if TYPE_CHECKING:
    from psd_tools.api.smart_object import SmartObject

Box = tuple[int, int, int, int]
Quad = tuple[float, float, float, float, float, float, float, float]
Mesh = tuple[np.ndarray, np.ndarray, np.ndarray]
Shape = tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[float]]

_INVERT = int.from_bytes(b"Invr", "big")  # Invert's filterID; it has no Fltr
_PUPPET = b"rigidTransform"
_MOSAIC = b"Msc "
_GSNB = b"GsnB"  # Gaussian Blur
_MTNB = b"MtnB"  # Motion Blur
_HSTR = b"HStr"  # Hue/Saturation
_BANW = b"BanW"  # Black & White
_SLCC = b"SlcC"  # Selective Color
_HUES = (b"Rds ", b"Ylws", b"Grns", b"Cyns", b"Bls ", b"Mgnt")  # Selective Color's
_INKS = (b"Cyn ", b"Mgnt", b"Ylw ", b"Blck")
_GREYS = (  # Black & White's sliders and defaults
    ("red", b"Rd  ", 40),
    ("yellow", b"Yllw", 60),
    ("green", b"Grn ", 40),
    ("cyan", b"Cyn ", 60),
    ("blue", b"Bl  ", 20),
    ("magenta", b"Mgnt", 80),
)
#: Motion Blur distances (px) at angle 0 proven on Photoshop 2023's pixels.
MOTION = range(1, 257)
_PU = (b"PuX0", b"PuY0", b"PuX1", b"PuY1", b"PuX2", b"PuY2", b"PuX3", b"PuY3")
_QUAD_TOL = 0.01  # px: a quad this close to a parallelogram is one
_EDGE = 1e-9  # barycentric slack: a pixel centre on an edge is inside
_BAND = 1 << 15  # float64s a blur band holds (256 KiB)


def _on(desc: Descriptor | None, key: bytes = b"enab") -> bool:
    return get_scalar(desc, key, bool, True)


def _kind(item: Descriptor) -> str | None:
    if get_scalar(item, b"filterID", int, None) == _INVERT:
        return "invert"
    fltr = item.get(b"Fltr")
    cls = getattr(fltr, "classID", None)
    cls = cls.encode() if isinstance(cls, str) else cls
    if cls == _PUPPET:
        return "puppet"
    if cls == _MOSAIC and _cell(fltr) is not None:
        return "mosaic"
    if cls == _GSNB and get_scalar(fltr, b"Rds ", float, 0.0) > 0:
        return "gaussian"
    if cls == _MTNB and _distance(fltr) is not None:
        return "motion"
    if cls == _HSTR and _master(fltr) is not None:
        return "hue"
    if cls == _BANW and not get_scalar(fltr, b"useTint", bool, False):
        return "grey"
    if cls == _SLCC and _selective(fltr) is not None:
        return "selective"
    return None


def _enum(value: object) -> bytes | None:
    enum = getattr(value, "enum", None)
    return enum.encode() if isinstance(enum, str) else enum


def _master(fltr: Descriptor) -> tuple[int, int, int] | None:
    """A Hue/Saturation's master hue, saturation and lightness.

    ``None`` with Colorize on or a hue range set.
    """
    adjs = list(fltr.get(b"Adjs") or [])
    if (
        get_scalar(fltr, b"Clrz", bool, False)
        or len(adjs) != 1
        or not isinstance(adjs[0], Descriptor)
        or b"LclR" in adjs[0]
    ):
        return None
    hue, saturation, lightness = (
        get_scalar(adjs[0], k, int, 0) for k in (b"H   ", b"Strt", b"Lght")
    )
    return hue, saturation, lightness


def _selective(fltr: Descriptor) -> list[tuple[int, list[float]]] | None:
    """A Relative Selective Color's hues (index in :py:data:`_HUES`) and inks.

    Cyan, magenta, yellow and black in -1..1. ``None`` when Absolute or with
    Whites, Neutrals or Blacks set.
    """
    if _enum(fltr.get(b"Mthd")) != b"Rltv":
        return None
    found = []
    for item in fltr.get(b"ClrC") or []:
        hue = _enum(item.get(b"Clrs")) if isinstance(item, Descriptor) else None
        if hue not in _HUES:
            return None
        inks = [get_scalar(item, k, float, 0.0) / 100 for k in _INKS]
        found.append((_HUES.index(hue), inks))
    return found


def _cell(fltr: Descriptor) -> int | None:
    """A Mosaic's cell size in px (2 or more)."""
    size = get_scalar(fltr, b"ClSz", float, 0.0)
    return int(size) if math.isfinite(size) and size == int(size) >= 2 else None


def _distance(fltr: Descriptor) -> int | None:
    """A Motion Blur's distance when at angle 0 and in :py:data:`MOTION`."""
    distance = get_scalar(fltr, b"Dstn", float, 0.0)
    if get_scalar(fltr, b"Angl", float, 0.0) != 0 or distance not in MOTION:
        return None
    return int(distance)


def _items(so: SmartObject, corners: Quad | None = None) -> list[Descriptor] | None:
    """The filters on; ``None`` when they are not :py:func:`draw`'s.

    That is more than one, another kind, a blend other than Normal at 100%, a
    mesh that does not decode, a quad no parallelogram or a Mosaic off its
    grid; the quad *corners*, by default the layer's.
    """
    fx = so.filters
    if fx is None or not _on(fx):
        return []
    found = [
        item
        for item in fx.get(b"filterFXList") or []
        if not isinstance(item, Descriptor) or _on(item)
    ]
    if not found:
        return []
    if not isinstance(found[0], Descriptor):
        return None
    if len(found) != 1 or so._layer._psd.color_mode != ColorMode.RGB:
        return None
    blend = found[0].get(b"blendOptions")
    mode = getattr(blend.get(b"Md  "), "enum", b"Nrml") if blend else b"Nrml"
    if (
        _kind(found[0]) is None
        or mode not in (b"Nrml", b"normal")
        or get_scalar(blend, b"Opct", float, 100.0) < 100
    ):
        return None
    if _kind(found[0]) == "puppet" and (
        _shapes(found[0][b"Fltr"]) is None
        or _mapping(so, found[0][b"Fltr"], corners) is None
    ):
        return None
    if _kind(found[0]) == "mosaic" and not _whole_cells(so, found[0][b"Fltr"], corners):
        return None
    return found


def drawable(so: SmartObject, corners: Quad | None = None) -> bool:
    """Whether :py:func:`draw` draws the object's smart filters on *corners*.

    True with none on, or one on, a Puppet Warp, an Invert, a Gaussian Blur,
    a Motion Blur at angle 0 of a distance :py:data:`MOTION` holds, a Mosaic
    on a canvas of whole cells and a quad not above or left of it, a
    Hue/Saturation of the master alone, a Black & White without tint or a
    Relative Selective Color of the six hues, Normal at 100%, in an RGB
    document. A filter mask does not refuse. *corners*: the quad the new
    picture is drawn on, by default the layer's.
    """
    return _items(so, corners) is not None


def draw(
    so: SmartObject,
    rgba: np.ndarray,
    box: Box,
    corners: Quad | None = None,
    sampling: str | None = None,
) -> tuple[np.ndarray, Box]:
    """Draw the object's smart filters over a new picture of it.

    Call it before the layer's pixels or filter source change: a Puppet Warp
    reads them to find how the card maker's Photoshop sampled.

    :param rgba: the picture drawn on the layer's quad, straight float RGBA
        ``(h, w, 4)`` in 0..1
    :param box: its canvas box ``(left, top, right, bottom)``
    :param corners: the quad it was drawn on (TL TR BR BL); by default the
        layer's :py:attr:`~SmartObject.transform_box`
    :param sampling: a Puppet Warp's ``"bilinear"`` or ``"nearest"``; by
        default the one that redraws the layer's stored pixels from its
        stored filter source closer, else bilinear
    :return: ``(straight float32 RGBA, canvas box)``, the box trimmed to the
        pixels that show; ``(0, 0, 0, 0)`` with an empty array when none do
    :raises ValueError: when :py:func:`drawable` is False on *corners*
    """
    items = _items(so, corners)
    if items is None:
        raise ValueError("these smart filters are not drawn")
    out = np.asarray(rgba, np.float32)
    if not items or out.size == 0:
        return out, box
    item = items[0]
    kind = _kind(item)
    if kind == "invert":
        out = out.copy()
        out[..., :3] = 1.0 - out[..., :3]
        return out, box
    if kind in ("hue", "grey", "selective"):
        return _adjusted(so, kind, item[b"Fltr"], out), box
    if kind == "mosaic":
        cell = _cell(item[b"Fltr"])
        assert cell is not None
        return _mosaic(out, box, cell)
    if kind == "gaussian":
        psd = so._layer._psd
        radius = get_scalar(item[b"Fltr"], b"Rds ", float, 0.0)
        return _trim(*_gaussian(out, box, radius, (psd.width, psd.height)))
    if kind == "motion":
        distance = _distance(item[b"Fltr"])
        assert distance is not None
        return _trim(*_motion(out, box, distance))
    if sampling not in (None, "bilinear", "nearest"):
        raise ValueError(f"unknown sampling {sampling!r}")
    fltr = item[b"Fltr"]
    mesh = _mesh(fltr)
    assert mesh is not None
    if sampling is None:
        sampling = _sampling(so, fltr, mesh)
    mapped = _mapping(so, fltr, corners)
    assert mapped is not None
    return _warp(out, box, mesh, mapped, sampling == "nearest")


def _mapping(
    so: SmartObject, fltr: Descriptor, corners: Quad | None
) -> tuple[float, ...] | None:
    quad = corners if corners is not None else so.transform_box
    if quad is None:
        return None
    return _affine([get_scalar(fltr, k, float, 0.0) for k in _PU], quad)


def _sampling(so: SmartObject, fltr: Descriptor, mesh: Mesh) -> str:
    """The sampling the card maker's Photoshop warped with.

    The one that warps the stored filter source nearer the layer's stored
    pixels (fewer px over 8 levels, premultiplied); bilinear on a tie or
    without them.
    """
    try:
        source = so.filter_source()
        stored = so._layer.numpy()
    except ValueError:
        return "bilinear"
    mapped = _mapping(so, fltr, None)
    if source is None or stored is None or mapped is None or stored.shape[2] < 4:
        return "bilinear"
    (x0, y0, _, _), color, alpha = source
    if alpha is None:
        alpha = np.full(color.shape[:2], 255, np.uint8)
    rows, cols = np.flatnonzero(alpha.any(1)), np.flatnonzero(alpha.any(0))
    if not len(rows) or color.shape[2] != 3:
        return "bilinear"
    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    pic = np.empty((r1 - r0, c1 - c0, 4), np.float32)
    pic[..., :3] = color[r0:r1, c0:c1] / np.float32(255)
    pic[..., 3] = alpha[r0:r1, c0:c1] / np.float32(255)
    box = (x0 + c0, y0 + r0, x0 + c1, y0 + r1)
    left, top, right, bottom = so._layer.bbox
    own = (np.asarray(stored, np.float32), (left, top, right, bottom))
    misses = [
        _misses(own, _warp(pic, box, mesh, mapped, nearest))
        for nearest in (False, True)
    ]
    return "nearest" if misses[1] < misses[0] else "bilinear"


def _misses(a: tuple[np.ndarray, Box], b: tuple[np.ndarray, Box]) -> int:
    """Pixels over 8 levels apart, premultiplied, on both boxes."""
    boxes = [box for _, box in (a, b) if box[2] > box[0] and box[3] > box[1]]
    if not boxes:
        return 0
    x0, y0 = min(v[0] for v in boxes), min(v[1] for v in boxes)
    x1, y1 = max(v[2] for v in boxes), max(v[3] for v in boxes)
    planes = []
    for pic, (left, top, right, bottom) in (a, b):
        plane = np.zeros((y1 - y0, x1 - x0, 4), np.float32)
        if right > left and bottom > top:
            part = plane[top - y0 : bottom - y0, left - x0 : right - x0]
            part[..., :3] = pic[..., :3] * pic[..., 3:]
            part[..., 3] = pic[..., 3]
        planes.append(np.rint(plane * 255))
    return int((np.abs(planes[0] - planes[1]).max(2) > 8).sum())


def _shapes(fltr: Descriptor) -> list[Shape] | None:
    """Each mesh shape of a Puppet Warp; ``None`` when one does not decode.

    A shape is its original and deformed vertices ``(n, 2)`` (float64, canvas
    px as the filter ran), its triangles ``(m, 3)`` into them and its pins'
    vertices and depths.
    """
    shapes = []
    for shape in fltr.get(b"puppetShapeList") or []:
        try:
            orig = _floats(shape[b"originalVertexArray"])
            deformed = _floats(shape[b"deformedVertexArray"])
            index = np.frombuffer(_raw(shape[b"indexArray"]), "<i4")
            pins = [int(v) for v in shape.get(b"pinVertexIndices") or []]
            depth = [float(v) for v in shape.get(b"PnDp") or []]
        except (KeyError, ValueError, TypeError):
            return None
        if (
            orig.shape != deformed.shape
            or index.size % 3
            or not index.size
            or not (np.isfinite(orig).all() and np.isfinite(deformed).all())
            or index.min() < 0
            or index.max() >= len(orig)
            or len(depth) != len(pins)
            or any(p < 0 or p >= len(orig) for p in pins)
        ):
            return None
        tri = index.reshape(-1, 3).astype(np.int64)
        shapes.append((orig, deformed, tri, pins, depth))
    return shapes or None


def _mesh(fltr: Descriptor) -> Mesh | None:
    """Every shape's vertices and triangles in one.

    Triangles back to front by depth (:py:func:`_depths`).
    """
    shapes = _shapes(fltr)
    if shapes is None:
        return None
    tris, depths, count = [], [], 0
    for orig, _, tri, pins, depth in shapes:
        tris.append(tri + count)
        depths.append(_depths(len(orig), tri, pins, depth))
        count += len(orig)
    order = np.argsort(np.concatenate(depths), kind="stable")
    return (
        np.concatenate([shape[0] for shape in shapes]),
        np.concatenate([shape[1] for shape in shapes]),
        np.concatenate(tris)[order],
    )


def _raw(value: object) -> bytes:
    data = getattr(value, "value", value)
    if not isinstance(data, bytes):
        raise TypeError("no raw data")
    return data


def _floats(value: object) -> np.ndarray:
    data = _raw(value)
    if len(data) % 8:
        raise ValueError("odd vertex array")
    return np.frombuffer(data, "<f4").reshape(-1, 2).astype(np.float64)


def _depths(n: int, tri: np.ndarray, pins: list[int], depth: list[float]) -> np.ndarray:
    """Each triangle's depth (Pin Depth), 0 without pins.

    The sum of its vertices', harmonic over the mesh: pins at their ``PnDp``,
    every other vertex the mean of its neighbours'. Conjugate gradients with
    exactly rounded dot products, so the order is the same on every CPU.
    """
    if not pins or not len(tri):
        return np.zeros(len(tri))
    z = np.zeros(n)
    fixed = np.zeros(n, bool)
    for pin, d in zip(pins, depth, strict=True):
        z[pin], fixed[pin] = d, True
    pairs = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edges = np.unique(np.sort(pairs, 1), axis=0)
    near: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges.tolist():
        near[u].append(v)
        near[v].append(u)
    free = np.flatnonzero(~fixed)
    if len(free):
        at = np.full(n, len(free))
        at[free] = np.arange(len(free))
        width = max(1, max(len(near[i]) for i in free))
        table = np.full((len(free), width), len(free))  # a free neighbour's slot
        degree = np.zeros(len(free))
        rhs = np.zeros(len(free))
        for row, i in enumerate(free.tolist()):
            degree[row] = len(near[i])
            rhs[row] = math.fsum(z[j] for j in near[i] if fixed[j])
            slots = [int(at[j]) for j in near[i] if not fixed[j]]
            table[row, : len(slots)] = slots

        def times(x: np.ndarray) -> np.ndarray:
            padded = np.append(x, 0.0)
            total = np.zeros(len(x))
            for k in range(width):
                total = total + padded[table[:, k]]
            return degree * x - total

        z[free] = _solve(times, rhs)
    return z[tri[:, 0]] + z[tri[:, 1]] + z[tri[:, 2]]


def _solve(times: Callable[[np.ndarray], np.ndarray], rhs: np.ndarray) -> np.ndarray:
    """``times(x) = rhs`` for a symmetric positive (semi)definite map."""
    x = np.zeros(len(rhs))
    r = rhs.copy()
    p = r.copy()
    rr = math.fsum(r * r)
    goal = 1e-24 * max(rr, 1e-300)
    for _ in range(4 * len(rhs) + 8):
        if rr <= goal:
            break
        ap = times(p)
        pap = math.fsum(p * ap)
        if pap <= 0:
            break
        step = rr / pap
        x = x + step * p
        r = r - step * ap
        new = math.fsum(r * r)
        p = r + (new / rr) * p
        rr = new
    return x


def _affine(src: list[float], dst: Quad | list[float]) -> tuple[float, ...] | None:
    """The map of parallelogram *src* onto *dst* (TL TR BR BL each).

    ``(ox, oy, ax, ay, bx, by)``: a point goes to ``ox + ax * x + bx * y``,
    ``oy + ay * x + by * y``. ``None`` when either is no parallelogram or is
    flat.
    """
    quads = []
    for q in (src, dst):
        if len(q) != 8:
            return None
        x0, y0, x1, y1, x2, y2, x3, y3 = (float(v) for v in q)
        if abs(x1 + x3 - x0 - x2) > _QUAD_TOL or abs(y1 + y3 - y0 - y2) > _QUAD_TOL:
            return None
        quads.append((x0, y0, x1 - x0, y1 - y0, x3 - x0, y3 - y0))
    (sx, sy, ux, uy, vx, vy), (tx, ty, px, py, qx, qy) = quads
    det = ux * vy - uy * vx
    if abs(det) < 1e-9 or abs(px * qy - py * qx) < 1e-9:
        return None
    # (x, y) -> (a, b) on src's edges -> tx + a * p + b * q
    ia, ib = vy / det, -vx / det
    ja, jb = -uy / det, ux / det
    ax, ay = px * ia + qx * ja, py * ia + qy * ja
    bx, by = px * ib + qx * jb, py * ib + qy * jb
    ox = tx - (ax * sx + bx * sy)
    oy = ty - (ay * sx + by * sy)
    return ox, oy, ax, ay, bx, by


def _map(v: np.ndarray, m: tuple[float, ...]) -> np.ndarray:
    ox, oy, ax, ay, bx, by = m
    x, y = v[:, 0], v[:, 1]
    return np.stack([ox + (ax * x + bx * y), oy + (ay * x + by * y)], 1)


def _warp(
    rgba: np.ndarray, box: Box, mesh: Mesh, m: tuple[float, ...], nearest: bool
) -> tuple[np.ndarray, Box]:
    """Warp *rgba* on canvas *box* by *mesh* mapped by *m*.

    Each pixel centre in a deformed triangle takes the picture at the same
    barycentric point of its original one, premultiplied, bilinear or
    nearest; a pixel in no triangle is clear. Later triangles cover earlier.
    """
    orig, deformed, tri = _map(mesh[0], m), _map(mesh[1], m), mesh[2]
    pic = rgba.astype(np.float64)
    pre = np.empty(pic.shape)
    pre[..., :3] = pic[..., :3] * pic[..., 3:]
    pre[..., 3] = pic[..., 3]
    x0 = int(np.floor(deformed[:, 0].min())) - 1
    y0 = int(np.floor(deformed[:, 1].min())) - 1
    x1 = int(np.ceil(deformed[:, 0].max())) + 1
    y1 = int(np.ceil(deformed[:, 1].max())) + 1
    out = np.zeros((y1 - y0, x1 - x0, 4))
    for a, b, c in tri:
        (p0x, p0y), (p1x, p1y), (p2x, p2y) = deformed[a], deformed[b], deformed[c]
        den = (p1y - p2y) * (p0x - p2x) + (p2x - p1x) * (p0y - p2y)
        if abs(den) < 1e-12:
            continue
        lx = int(np.floor(min(p0x, p1x, p2x) - 0.5))
        ly = int(np.floor(min(p0y, p1y, p2y) - 0.5))
        hx = int(np.ceil(max(p0x, p1x, p2x) - 0.5))
        hy = int(np.ceil(max(p0y, p1y, p2y) - 0.5))
        ys, xs = np.mgrid[ly : hy + 1, lx : hx + 1]
        px, py = xs.ravel() + 0.5, ys.ravel() + 0.5
        l0 = ((p1y - p2y) * (px - p2x) + (p2x - p1x) * (py - p2y)) / den
        l1 = ((p2y - p0y) * (px - p2x) + (p0x - p2x) * (py - p2y)) / den
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -_EDGE) & (l1 >= -_EDGE) & (l2 >= -_EDGE)
        if not inside.any():
            continue
        l0, l1, l2 = l0[inside], l1[inside], l2[inside]
        sx = l0 * orig[a, 0] + l1 * orig[b, 0] + l2 * orig[c, 0]
        sy = l0 * orig[a, 1] + l1 * orig[b, 1] + l2 * orig[c, 1]
        out[ys.ravel()[inside] - y0, xs.ravel()[inside] - x0] = (
            _nearest(pre, box, sx, sy) if nearest else _bilinear(pre, box, sx, sy)
        )
    return _straight(out, (x0, y0, x1, y1))


def _bilinear(pre: np.ndarray, box: Box, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """*pre* (premultiplied, on canvas *box*) at canvas points; clear outside."""
    h, w = pre.shape[:2]
    u, v = sx - box[0] - 0.5, sy - box[1] - 0.5
    iu, iv = np.floor(u), np.floor(v)
    fu, fv = (u - iu)[:, None], (v - iv)[:, None]
    iu, iv = iu.astype(np.int64), iv.astype(np.int64)
    out = np.zeros((len(u), 4))
    for du, dv, wt in (
        (0, 0, (1.0 - fu) * (1.0 - fv)),
        (1, 0, fu * (1.0 - fv)),
        (0, 1, (1.0 - fu) * fv),
        (1, 1, fu * fv),
    ):
        x, y = iu + du, iv + dv
        ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        tap = np.zeros((len(u), 4))
        tap[ok] = pre[y[ok], x[ok]]
        out = out + wt * tap
    return out


def _nearest(pre: np.ndarray, box: Box, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    """*pre* at canvas points, the pixel each lies in; clear outside."""
    h, w = pre.shape[:2]
    x = np.floor(sx - box[0]).astype(np.int64)
    y = np.floor(sy - box[1]).astype(np.int64)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    out = np.zeros((len(sx), 4))
    out[ok] = pre[y[ok], x[ok]]
    return out


def _whole_cells(so: SmartObject, fltr: Descriptor, corners: Quad | None) -> bool:
    """A Mosaic's grid is the canvas's (proven).

    Whole cells across it and the quad (*corners* or the layer's) not above or
    left of it.
    """
    quad = corners if corners is not None else so.transform_box
    cell, psd = _cell(fltr), so._layer._psd
    return (
        cell is not None
        and quad is not None
        and psd.width % cell == 0
        and psd.height % cell == 0
        and min(quad[0::2]) >= 0
        and min(quad[1::2]) >= 0
    )


def _mosaic(rgba: np.ndarray, box: Box, cell: int) -> tuple[np.ndarray, Box]:
    """*rgba* on canvas *box* under a Mosaic of *cell* px, as stored.

    Cells on the canvas's grid, each the mean of its pixels, premultiplied;
    a pixel outside the picture counts clear. Summed in a fixed order.
    """
    left, top, right, bottom = (int(v) for v in box)
    x0, y0 = left - left % cell, top - top % cell
    x1, y1 = right + (-right) % cell, bottom + (-bottom) % cell
    pic = rgba.astype(np.float64)
    pre = np.zeros((y1 - y0, x1 - x0, 4))
    part = pre[top - y0 : bottom - y0, left - x0 : right - x0]
    part[..., :3] = pic[..., :3] * pic[..., 3:]
    part[..., 3] = pic[..., 3]
    total = np.zeros(((y1 - y0) // cell, (x1 - x0) // cell, 4))
    for dy in range(cell):
        for dx in range(cell):
            total = total + pre[dy::cell, dx::cell]
    mean = np.repeat(np.repeat(total / (cell * cell), cell, 0), cell, 1)
    return _straight(mean, (x0, y0, x1, y1))


def _straight(pre: np.ndarray, box: Box) -> tuple[np.ndarray, Box]:
    """Premultiplied pixels on *box* as straight float32, trimmed to ink."""
    alpha = np.clip(pre[..., 3], 0.0, 1.0)
    shown = np.rint(alpha * 255) > 0
    rows, cols = np.flatnonzero(shown.any(1)), np.flatnonzero(shown.any(0))
    if not len(rows):
        return np.zeros((0, 0, 4), np.float32), (0, 0, 0, 0)
    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    pre, alpha = pre[r0:r1, c0:c1], alpha[r0:r1, c0:c1]
    safe = np.where(alpha > 0, alpha, 1.0)[..., None]
    rgb = np.where(alpha[..., None] > 0, np.clip(pre[..., :3] / safe, 0.0, 1.0), 0.0)
    out = np.empty((*alpha.shape, 4), np.float32)
    out[..., :3] = rgb
    out[..., 3] = alpha
    return out, (box[0] + c0, box[1] + r0, box[0] + c1, box[1] + r1)


def _gaussian(
    rgba: np.ndarray, box: Box, radius: float, size: tuple[int, int]
) -> tuple[np.ndarray, Box]:
    """*rgba* on canvas *box* under a Gaussian Blur of *radius*, as stored.

    Premultiplied, sigma = radius, 3 sigma each way. A side at or past the
    canvas's edge repeats its edge pixels (no halo); a side inside it fades
    out over the pad it grows by.
    """
    pre = np.dstack([rgba[..., :3] * rgba[..., 3:], rgba[..., 3]])
    pad = math.ceil(3 * radius) + 1
    x0, y0, x1, y1 = box
    edge = (x0 <= 0, y0 <= 0, x1 >= size[0], y1 >= size[1])
    for axis, before, after in ((1, edge[0], edge[2]), (0, edge[1], edge[3])):
        for side, at_edge in ((0, before), (1, after)):
            widths = [(0, 0)] * 3
            widths[axis] = (0, pad) if side else (pad, 0)
            pre = np.pad(pre, widths, mode="edge" if at_edge else "constant")
    taps = _taps(radius)
    pre = np.dstack(
        [_blur_axis(_blur_axis(pre[..., k], taps, 0), taps, 1) for k in range(4)]
    )
    out = (
        x0 - pad * (not edge[0]),
        y0 - pad * (not edge[1]),
        x1 + pad * (not edge[2]),
        y1 + pad * (not edge[3]),
    )
    left, top = pad * edge[0], pad * edge[1]
    pre = pre[top : top + out[3] - out[1], left : left + out[2] - out[0]]
    alpha = np.clip(pre[..., 3], 0.0, 1.0)
    safe = np.maximum(alpha, np.float32(1e-6))[..., None]
    rgb = np.where(alpha[..., None] > 0, pre[..., :3] / safe, np.float32(0))
    return np.dstack([np.clip(rgb, 0.0, 1.0), alpha]).astype(np.float32), out


def _taps(sigma: float) -> list[float]:
    """Gaussian weights, the centre's then each distance's, summing to 1.

    Out to ``int(3 * sigma + 0.5)``; exp in decimal and the sum exactly
    rounded, so they are the same on every CPU.
    """
    scale = -0.5 / (sigma * sigma)
    context = Context(prec=34)
    raw = [
        float(context.exp(Decimal(scale * (k * k))))
        for k in range(int(3.0 * sigma + 0.5) + 1)
    ]
    total = math.fsum(raw + raw[1:])
    return [w / total for w in raw]


def _blur_axis(plane: np.ndarray, taps: list[float], axis: int) -> np.ndarray:
    """One axis of a separable blur, edges repeated, float32 out.

    Float64, the centre tap then each pair from the farthest in, every step
    rounded (no fused multiply-add): the order scipy's ``correlate1d`` sums
    a symmetric filter in. In bands of rows that stay in cache.
    """
    reach, n = len(taps) - 1, plane.shape[axis]
    widths = [(0, 0), (0, 0)]
    widths[axis] = (reach, reach)
    src = np.pad(plane.astype(np.float64), widths, mode="edge")
    out = np.empty(plane.shape, np.float32)
    rows = max(1, _BAND // max(1, src.shape[1]))
    for r0 in range(0, plane.shape[0], rows):
        r1 = min(r0 + rows, plane.shape[0])
        if axis == 0:
            band = src[r0 : r1 + 2 * reach]
            at = lambda k: band[reach + k : reach + k + r1 - r0]  # noqa: E731
        else:
            band = src[r0:r1]
            at = lambda k: band[:, reach + k : reach + k + n]  # noqa: E731
        total = at(0) * taps[0]
        pair = np.empty_like(total)
        for k in range(reach, 0, -1):
            np.add(at(-k), at(k), out=pair)
            pair *= taps[k]
            total += pair
        out[r0:r1] = total
    return out


def _motion(rgba: np.ndarray, box: Box, distance: int) -> tuple[np.ndarray, Box]:
    """*rgba* on canvas *box* under a Motion Blur at angle 0, as stored.

    A box of distance + 1 taps over ``x - distance // 2 .. x + distance -
    distance // 2`` of 8-bit alpha and premultiplied colour, each ``(S + c)
    // taps`` with ``c = taps // 2``, less 1 unless taps is a power of 2.
    """
    taps, lo = distance + 1, distance // 2
    bias = taps // 2 - bool(taps & distance)
    hi = distance - lo
    alpha = np.round(np.clip(rgba[..., 3], 0, 1) * 255).astype(np.int64)
    color = np.round(np.clip(rgba[..., :3], 0, 1) * 255).astype(np.int64)
    color = (color * alpha[..., None] * 2 + 255) // 510  # rounded
    width = alpha.shape[1] + distance

    def sums(plane: np.ndarray) -> np.ndarray:
        widths = [(0, 0), (distance, distance)] + [(0, 0)] * (plane.ndim - 2)
        padded = np.pad(plane, widths)
        total = padded[:, :width].copy()
        for i in range(1, taps):
            total += padded[:, i : i + width]
        return total

    out_alpha = (sums(alpha) + bias) // taps
    inks = (sums(color) + bias) // taps
    shown = np.maximum(out_alpha, 1)[..., None]
    rgb = (inks * 510 + shown) // (shown * 2)  # unpremultiplied, rounded
    out = np.dstack([rgb / 255.0, out_alpha / 255.0]).astype(np.float32)
    return out, (box[0] - hi, box[1], box[2] + lo, box[3])


def _trim(rgba: np.ndarray, box: Box) -> tuple[np.ndarray, Box]:
    """Straight *rgba* on *box* trimmed to the pixels that show (8-bit alpha)."""
    shown = np.rint(np.clip(rgba[..., 3], 0.0, 1.0) * 255) > 0
    rows, cols = np.flatnonzero(shown.any(1)), np.flatnonzero(shown.any(0))
    if not len(rows):
        return np.zeros((0, 0, 4), np.float32), (0, 0, 0, 0)
    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    return rgba[r0:r1, c0:c1], (box[0] + c0, box[1] + r0, box[0] + c1, box[1] + r1)


def _adjusted(
    so: SmartObject, kind: str, fltr: Descriptor, rgba: np.ndarray
) -> np.ndarray:
    """*rgba*'s colours on whole levels under a colour filter; alpha kept.

    Hue/Saturation and Black & White by the adjustment layers' code.
    """
    levels = np.rint(np.clip(rgba[..., :3].astype(np.float64), 0.0, 1.0) * 255)
    out = np.empty(rgba.shape, np.float32)
    out[..., 3] = rgba[..., 3]
    if kind == "selective":
        ranges = _selective(fltr)
        assert ranges is not None
        out[..., :3] = _selective_color(levels, ranges)
        return out
    rgb = (levels / 255).astype(np.float32)
    psd = so._layer._psd
    if kind == "hue":
        layer = SimpleNamespace(
            data=[],
            colorization=(0, 0, 0),
            master=_master(fltr),
            enable_colorization=0,
            _psd=psd,
        )
        out[..., :3] = adjustments.apply_huesaturation(
            rgb, ColorMode.RGB, cast(Any, layer)
        )
    else:
        sliders = {name: get_scalar(fltr, key, int, v) for name, key, v in _GREYS}
        layer = SimpleNamespace(_data=fltr, use_tint=False, _psd=psd, **sliders)
        out[..., :3] = cast(
            np.ndarray,
            adjustments.apply_blackandwhite(rgb, ColorMode.RGB, cast(Any, layer)),
        )
    return out


def _selective_color(
    levels: np.ndarray, ranges: list[tuple[int, list[float]]]
) -> np.ndarray:
    """Relative Selective Color on whole *levels* (float64 0..255), 0..1 out.

    A hue's weight is its span: Reds where R is the largest channel, max -
    mid; Yellows where B is the smallest, mid - min; Greens G largest, Cyans
    R smallest, Blues B largest, Magentas G smallest. Each hue moves channel
    ``v`` (R by cyan, G magenta, B yellow; ink ``c``, black ``k``) by weight
    ``* clip(((-1 - c) * k - c) * (1 - v), -v, 1 - v)``, every hue from the
    same input, then rounded. Proven on Photoshop 2023's lattice of every
    colour (5 settings, at most 2 levels off).
    """
    high, low = levels.max(2), levels.min(2)
    mid = levels.sum(2) - high - low
    r, g, b = levels[..., 0], levels[..., 1], levels[..., 2]
    spans = (
        np.where(r == high, high - mid, 0.0),
        np.where(b == low, mid - low, 0.0),
        np.where(g == high, high - mid, 0.0),
        np.where(r == low, mid - low, 0.0),
        np.where(b == high, high - mid, 0.0),
        np.where(g == low, mid - low, 0.0),
    )
    v = levels / 255
    out = levels.copy()
    for hue, (*inks, black) in ranges:
        for channel, ink in enumerate(inks):
            amount = (-1.0 - ink) * black - ink
            if amount:
                x = v[..., channel]
                move = np.clip(amount * (1.0 - x), -x, 1.0 - x)
                out[..., channel] += spans[hue] * move
    return (np.clip(np.rint(out), 0, 255) / 255).astype(np.float32)
