# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""C loops for effects._nearest_boundary's pruned minimum and effects._chamfer_transform.

Same float32 (pruned minimum) and float64 (chamfer) operations as the numpy
code, in the same order, so the results are bit-identical.
"""
import numpy as np
cimport numpy as cnp
from libc.math cimport INFINITY

cnp.import_array()


def prune_min(
    cnp.intp_t[::1] rows,
    cnp.intp_t[::1] cols,
    Py_ssize_t span,
    cnp.intp_t[:, ::1] offsets,
    float[::1] lengths,
    cnp.uint8_t[:, ::1] stating_pad,
    float[:, ::1] near_pad,
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
    cdef cnp.ndarray[cnp.uint8_t, ndim=2, cast=True] m = np.ascontiguousarray(mask)
    cdef Py_ssize_t height = m.shape[0], width = m.shape[1]
    out = np.where(m, np.inf, 0.0)
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
