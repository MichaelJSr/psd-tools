"""Smart filters drawn on a new picture of a placed layer.

A placed layer's pixels are its smart filters' result. When its file changes,
the caller draws the new picture on the layer's quad and :py:func:`draw` draws
the filters over it as Photoshop 2023 does: a Puppet Warp from its stored mesh,
an Invert. Plain float64 numpy, element by element in a fixed order, and
exactly rounded sums (no BLAS), so every CPU gives the same result.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from psd_tools.api._descriptor import get_scalar
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
_PU = (b"PuX0", b"PuY0", b"PuX1", b"PuY1", b"PuX2", b"PuY2", b"PuX3", b"PuY3")
_QUAD_TOL = 0.01  # px: a quad this close to a parallelogram is one
_EDGE = 1e-9  # barycentric slack: a pixel centre on an edge is inside


def _on(desc: Descriptor | None, key: bytes = b"enab") -> bool:
    return get_scalar(desc, key, bool, True)


def _kind(item: Descriptor) -> str | None:
    if get_scalar(item, b"filterID", int, None) == _INVERT:
        return "invert"
    if getattr(item.get(b"Fltr"), "classID", None) in (_PUPPET, _PUPPET.decode()):
        return "puppet"
    return None


def _items(so: SmartObject) -> list[Descriptor] | None:
    """The filters on; ``None`` when they are not :py:func:`draw`'s.

    That is more than one, another kind, a blend other than Normal at 100%, a
    painted mask, a mesh that does not decode or a quad no parallelogram.
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
    if so.filter_mask_enabled:
        try:
            painted = so.filter_mask()
        except ValueError:
            return None
        if painted is not None and (
            not so.filter_mask_extends_white or int(painted[1].min()) < 255
        ):
            return None
    if _kind(found[0]) == "puppet" and (
        _shapes(found[0][b"Fltr"]) is None
        or _mapping(so, found[0][b"Fltr"], None) is None
    ):
        return None
    return found


def drawable(so: SmartObject) -> bool:
    """Whether :py:func:`draw` draws the object's smart filters.

    True with none on, or one on, a Puppet Warp or an Invert, Normal at 100%,
    its mask off or white, in an RGB document.
    """
    return _items(so) is not None


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
    :raises ValueError: when :py:func:`drawable` is False
    """
    items = _items(so)
    if items is None:
        raise ValueError("these smart filters are not drawn")
    out = np.asarray(rgba, np.float32)
    if not items or out.size == 0:
        return out, box
    item = items[0]
    if _kind(item) == "invert":
        out = out.copy()
        out[..., :3] = 1.0 - out[..., :3]
        return out, box
    if sampling not in (None, "bilinear", "nearest"):
        raise ValueError(f"unknown sampling {sampling!r}")
    fltr = item[b"Fltr"]
    mesh = _mesh(fltr)
    assert mesh is not None
    if sampling is None:
        sampling = _sampling(so, fltr, mesh)
    mapped = _mapping(so, fltr, corners)
    if mapped is None:
        raise ValueError("the filter's quad or the layer's is no parallelogram")
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
