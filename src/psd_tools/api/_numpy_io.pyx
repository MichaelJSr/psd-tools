# cython: boundscheck=False, wraparound=False, language_level=3
"""C loop behind ``numpy_io``'s 8-bit channel reads.

Memoryviews only, so the module builds against the Limited API without numpy
headers.
"""
import numpy as np


# numpy's own levels, so a read is ``_parse_array``'s bit for bit.
cdef float _LEVEL[256]
cdef const float[::1] _levels = np.arange(256, dtype=np.uint8).astype(np.float32) / 255.0
cdef int _i
for _i in range(256):
    _LEVEL[_i] = _levels[_i]


def planes8(list planes, Py_ssize_t h, Py_ssize_t w):
    """8-bit planes of ``h * w`` bytes as float32 levels, side by side: ``(h, w, n)``."""
    cdef Py_ssize_t n = len(planes), size = h * w, i, k
    cdef const unsigned char[::1] a
    cdef const unsigned char[::1] b
    cdef const unsigned char[::1] c
    for k in range(n):
        a = planes[k]
        if a.shape[0] != size:
            raise ValueError("plane is not h * w bytes")
    result = np.empty((h, w, n), dtype=np.float32)
    if size == 0 or n == 0:
        return result
    cdef float[:, :, ::1] out = result
    cdef float *o = &out[0, 0, 0]
    cdef const unsigned char *p
    cdef const unsigned char *q
    cdef const unsigned char *r
    if n == 1:
        a = planes[0]
        p = &a[0]
        with nogil:
            for i in range(size):
                o[i] = _LEVEL[p[i]]
    elif n == 3:
        a, b, c = planes[0], planes[1], planes[2]
        p, q, r = &a[0], &b[0], &c[0]
        with nogil:
            for i in range(size):
                o[3 * i] = _LEVEL[p[i]]
                o[3 * i + 1] = _LEVEL[q[i]]
                o[3 * i + 2] = _LEVEL[r[i]]
    else:
        for k in range(n):
            a = planes[k]
            p = &a[0]
            with nogil:
                for i in range(size):
                    o[n * i + k] = _LEVEL[p[i]]
    return result
