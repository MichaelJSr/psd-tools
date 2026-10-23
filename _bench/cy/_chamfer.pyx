# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""PROTO(speed): effects._chamfer_transform as a C loop, the same float64 operations in the same order."""
import numpy as np
cimport numpy as cnp
from libc.math cimport INFINITY

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
    # line[across:] vs source[:-across], then line[:-across] vs source[across:]
    # (numpy evaluates the first slice pair fully before the second)
    for j in range(across, width):
        s = f[before, j - across] + length
        if s < f[row, j]:
            f[row, j] = s
    for j in range(0, width - across):
        s = f[before, j + across] + length
        if s < f[row, j]:
            f[row, j] = s


def chamfer_transform(mask):
    cdef cnp.ndarray[cnp.uint8_t, ndim=2, cast=True] m = np.ascontiguousarray(mask)
    cdef Py_ssize_t height = m.shape[0], width = m.shape[1]
    out = np.where(m, np.inf, 0.0)
    cdef double[:, ::1] f = out
    cdef double[::1] tmp = np.empty(width, dtype=np.float64)
    cdef Py_ssize_t passno, idx, row, step, before, j
    cdef double run, v
    cdef int k
    cdef Py_ssize_t downs[4]
    cdef Py_ssize_t acrosses[4]
    cdef double lengths[4]
    downs[:] = [1, 1, 1, 2]
    acrosses[:] = [0, 1, 2, 1]
    lengths[:] = [1.0, DIAG, KNIGHT, KNIGHT]
    with nogil:
        for passno in range(2):
            step = 1 if passno == 0 else -1
            for idx in range(height):
                row = idx if passno == 0 else height - 1 - idx
                for k in range(4):
                    before = row - step * downs[k]
                    if before < 0 or before >= height or acrosses[k] >= width:
                        continue
                    _relax(f, row, before, acrosses[k], lengths[k], width)
                # np.minimum(line, np.minimum.accumulate(line - cols) + cols)
                run = INFINITY
                for j in range(width):
                    v = f[row, j] - <double>j
                    if v < run or j == 0:
                        run = v
                    tmp[j] = run + <double>j
                for j in range(width):
                    if tmp[j] < f[row, j]:
                        f[row, j] = tmp[j]
                # back = np.minimum.accumulate((line + cols)[::-1])[::-1] - cols
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
