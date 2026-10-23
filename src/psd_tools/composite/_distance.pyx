# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""C loops behind effects._nearest_boundary and effects._chamfer_transform.

The numpy code's float32 (pruned minimum) and float64 (chamfer) operations in
the same order, so the results are bit-identical. Memoryviews only, so the
module builds against the Limited API without numpy headers.
"""
import numpy as np
from libc.math cimport INFINITY, sqrt


def prune_min(
    const Py_ssize_t[::1] rows,
    const Py_ssize_t[::1] cols,
    Py_ssize_t span,
    const Py_ssize_t[:, ::1] offsets,
    const float[::1] lengths,
    const unsigned char[:, ::1] stating_pad,
    const float[:, ::1] near_pad,
    bint outward,
):
    cdef Py_ssize_t n = rows.shape[0], m = lengths.shape[0], i, k, r, c
    cdef float best, cand, length
    result = np.empty(n, dtype=np.float32)
    cdef float[::1] out = result
    with nogil:
        for i in range(n):
            r = rows[i] + span
            c = cols[i] + span
            if outward:
                best = INFINITY
                for k in range(m):
                    length = lengths[k]
                    if stating_pad[r + offsets[k, 0], c + offsets[k, 1]]:
                        cand = near_pad[r + offsets[k, 0], c + offsets[k, 1]] + length
                        if cand < best:
                            best = cand
                    if best <= length - <float>0.5:
                        break
            else:
                best = -INFINITY
                for k in range(m):
                    length = lengths[k]
                    if stating_pad[r + offsets[k, 0], c + offsets[k, 1]]:
                        cand = near_pad[r + offsets[k, 0], c + offsets[k, 1]] - length
                        if cand > best:
                            best = cand
                    if best >= <float>0.5 - length:
                        break
            out[i] = best
    return result


cdef double DIAG = 1.4142135623730951
cdef double KNIGHT = 2.23606797749979


cdef inline void _relax(double[:, ::1] f, Py_ssize_t row, Py_ssize_t before, Py_ssize_t across, double length, Py_ssize_t width) noexcept nogil:
    cdef Py_ssize_t j
    cdef double s
    if across == 0:
        for j in range(width):
            s = f[before, j] + length
            if s < f[row, j]:
                f[row, j] = s
        return
    for j in range(across, width):
        s = f[before, j - across] + length
        if s < f[row, j]:
            f[row, j] = s
    for j in range(0, width - across):
        s = f[before, j + across] + length
        if s < f[row, j]:
            f[row, j] = s


def chamfer_transform(mask, double diag, double knight):
    mask = np.ascontiguousarray(mask, dtype=bool)
    cdef Py_ssize_t height = mask.shape[0], width = mask.shape[1]
    out = np.where(mask, np.inf, 0.0)
    cdef double[:, ::1] f = out
    cdef double[::1] tmp = np.empty(max(width, 1), dtype=np.float64)
    cdef Py_ssize_t passno, idx, row, step, before, j
    cdef double run, v
    cdef int k
    cdef Py_ssize_t downs[4]
    cdef Py_ssize_t acrosses[4]
    cdef double lens[4]
    downs[:] = [1, 1, 1, 2]
    acrosses[:] = [0, 1, 2, 1]
    lens[:] = [1.0, diag, knight, knight]
    with nogil:
        for passno in range(2):
            step = 1 if passno == 0 else -1
            for idx in range(height):
                row = idx if passno == 0 else height - 1 - idx
                for k in range(4):
                    before = row - step * downs[k]
                    if before < 0 or before >= height or acrosses[k] >= width:
                        continue
                    _relax(f, row, before, acrosses[k], lens[k], width)
                run = INFINITY
                for j in range(width):
                    v = f[row, j] - <double>j
                    if v < run or j == 0:
                        run = v
                    tmp[j] = run + <double>j
                for j in range(width):
                    if tmp[j] < f[row, j]:
                        f[row, j] = tmp[j]
                run = INFINITY
                for j in range(width - 1, -1, -1):
                    v = f[row, j] + <double>j
                    if v < run or j == width - 1:
                        run = v
                    tmp[j] = run - <double>j
                for j in range(width):
                    if tmp[j] < f[row, j]:
                        f[row, j] = tmp[j]
    return out.astype(np.float32)


def disk_reach(const unsigned char[:, ::1] stating, double radius):
    """Pixels whose Euclidean distance to a stating pixel is at most ``radius``.

    ``distance_transform_edt(~stating) <= radius`` without the transform: the
    distance is ``sqrt(dy * dy + dx * dx)`` in double either way, so marking
    each stating pixel's disk is exact.
    """
    cdef Py_ssize_t h = stating.shape[0], w = stating.shape[1], y, x, dy, x0, x1, yy, j
    cdef Py_ssize_t r = <Py_ssize_t>radius + 1
    cdef Py_ssize_t[::1] reach = np.full(2 * r + 1, -1, dtype=np.intp)
    cdef Py_ssize_t dx
    for dy in range(-r, r + 1):
        dx = -1
        while sqrt(<double>(dy * dy + (dx + 1) * (dx + 1))) <= radius:
            dx += 1
        reach[dy + r] = dx
    # Each row: +1 where a covered run starts, -1 past its end.
    cdef int[:, ::1] runs = np.zeros((h, w + 1), dtype=np.intc)
    with nogil:
        for y in range(h):
            for x in range(w):
                if not stating[y, x]:
                    continue
                for dy in range(-r, r + 1):
                    yy = y + dy
                    if yy < 0 or yy >= h or reach[dy + r] < 0:
                        continue
                    x0 = x - reach[dy + r]
                    x1 = x + reach[dy + r] + 1
                    if x0 < 0:
                        x0 = 0
                    if x1 > w:
                        x1 = w
                    runs[yy, x0] += 1
                    runs[yy, x1] -= 1
    out = np.empty((h, w), dtype=bool)
    cdef unsigned char[:, ::1] o = out.view(np.uint8)
    cdef int acc
    with nogil:
        for y in range(h):
            acc = 0
            for j in range(w):
                acc = acc + runs[y, j]
                o[y, j] = acc > 0
    return out
