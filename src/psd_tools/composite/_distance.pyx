# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""C loops behind effects._nearest_boundary, _chamfer_transform, _chamfer_relax and _tent_blur.

The numpy code's float32 (pruned minimum) and float64 (chamfer) operations in
the same order, so the results are bit-identical. Memoryviews only, so the
module builds against the Limited API without numpy headers.
"""
import numpy as np
from libc.math cimport INFINITY, sqrt
from libc.stdlib cimport free, malloc


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


def chamfer_transform(mask, double diag, double knight):
    mask = np.ascontiguousarray(mask, dtype=bool)
    # Unpacking checks the length (ValueError, as the numpy code); shape[1]
    # of a 1-D mask would read past the tuple under boundscheck=False.
    height, width = mask.shape
    out = np.where(mask, np.inf, 0.0)
    chamfer_relax(out, diag, knight)
    return out.astype(np.float32)


def chamfer_relax(double[:, ::1] f, double diag, double knight):
    """Lower each value of ``f`` to the least of every value plus its chamfer length, in place."""
    cdef Py_ssize_t height = f.shape[0], width = f.shape[1]
    cdef Py_ssize_t passno, idx, row, step, b1, b2, j
    cdef double run, v, m, s
    cdef bint has1, has2
    cdef double* cur
    cdef double* p1
    cdef double* p2
    cdef Py_ssize_t edge[4]
    cdef int e
    with nogil:
        for passno in range(2):
            step = 1 if passno == 0 else -1
            for idx in range(height):
                row = idx if passno == 0 else height - 1 - idx
                b1 = row - step
                b2 = row - 2 * step
                has1 = 0 <= b1 < height
                has2 = 0 <= b2 < height and width > 1
                cur = &f[row, 0]
                # Every move from the one or two rows before at once: each is
                # a minimum against a finished row, so the order is free.
                if has1 and has2 and width > 4:
                    p1 = &f[b1, 0]
                    p2 = &f[b2, 0]
                    for j in range(2, width - 2):
                        m = cur[j]
                        s = p1[j] + 1.0
                        m = s if s < m else m
                        s = p1[j - 1] + diag
                        m = s if s < m else m
                        s = p1[j + 1] + diag
                        m = s if s < m else m
                        s = p1[j - 2] + knight
                        m = s if s < m else m
                        s = p1[j + 2] + knight
                        m = s if s < m else m
                        s = p2[j - 1] + knight
                        m = s if s < m else m
                        s = p2[j + 1] + knight
                        m = s if s < m else m
                        cur[j] = m
                    edge[0] = 0
                    edge[1] = 1
                    edge[2] = width - 2
                    edge[3] = width - 1
                    for e in range(4):
                        j = edge[e]
                        m = cur[j]
                        s = p1[j] + 1.0
                        if s < m:
                            m = s
                        if j >= 1:
                            s = p1[j - 1] + diag
                            if s < m:
                                m = s
                        if j + 1 < width:
                            s = p1[j + 1] + diag
                            if s < m:
                                m = s
                        if j >= 2:
                            s = p1[j - 2] + knight
                            if s < m:
                                m = s
                        if j + 2 < width:
                            s = p1[j + 2] + knight
                            if s < m:
                                m = s
                        if j >= 1:
                            s = p2[j - 1] + knight
                            if s < m:
                                m = s
                        if j + 1 < width:
                            s = p2[j + 1] + knight
                            if s < m:
                                m = s
                        cur[j] = m
                elif has1 or has2:
                    p1 = &f[b1, 0] if has1 else cur
                    p2 = &f[b2, 0] if has2 else cur
                    for j in range(width):
                        m = cur[j]
                        if has1:
                            s = p1[j] + 1.0
                            if s < m:
                                m = s
                            if j >= 1:
                                s = p1[j - 1] + diag
                                if s < m:
                                    m = s
                            if j + 1 < width:
                                s = p1[j + 1] + diag
                                if s < m:
                                    m = s
                            if j >= 2:
                                s = p1[j - 2] + knight
                                if s < m:
                                    m = s
                            if j + 2 < width:
                                s = p1[j + 2] + knight
                                if s < m:
                                    m = s
                        if has2:
                            if j >= 1:
                                s = p2[j - 1] + knight
                                if s < m:
                                    m = s
                            if j + 1 < width:
                                s = p2[j + 1] + knight
                                if s < m:
                                    m = s
                        cur[j] = m
                run = INFINITY
                for j in range(width):
                    v = cur[j] - <double>j
                    if v < run or j == 0:
                        run = v
                    v = run + <double>j
                    if v < cur[j]:
                        cur[j] = v
                run = INFINITY
                for j in range(width - 1, -1, -1):
                    v = cur[j] + <double>j
                    if v < run or j == width - 1:
                        run = v
                    v = run - <double>j
                    if v < cur[j]:
                        cur[j] = v


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


cdef void _tent_rows(const float* src, float* dst, Py_ssize_t h, Py_ssize_t n, const double* w,
                     Py_ssize_t s, const float* crow, double* acc, const float** rp) noexcept nogil:
    # Along axis 0: each output row sums whole input rows, so the loop over a
    # row's pixels is the vector one and each pixel sees scipy's order.
    cdef Py_ssize_t i, k, jj, t, j
    cdef double w0 = w[s], v, w1, w2, w3, w4
    cdef const float* a1
    cdef const float* b1
    cdef const float* a2
    cdef const float* b2
    cdef const float* a3
    cdef const float* b3
    cdef const float* a4
    cdef const float* b4
    for i in range(h):
        for t in range(2 * s + 1):
            j = i - s + t
            rp[t] = src + j * n if 0 <= j < h else crow
        a1 = rp[s]
        for k in range(n):
            acc[k] = <double>a1[k] * w0
        jj = -s
        while jj + 4 <= 0:
            a1 = rp[s + jj]
            b1 = rp[s - jj]
            a2 = rp[s + jj + 1]
            b2 = rp[s - jj - 1]
            a3 = rp[s + jj + 2]
            b3 = rp[s - jj - 2]
            a4 = rp[s + jj + 3]
            b4 = rp[s - jj - 3]
            w1 = w[s + jj]
            w2 = w[s + jj + 1]
            w3 = w[s + jj + 2]
            w4 = w[s + jj + 3]
            for k in range(n):
                v = acc[k]
                v = v + (<double>a1[k] + <double>b1[k]) * w1
                v = v + (<double>a2[k] + <double>b2[k]) * w2
                v = v + (<double>a3[k] + <double>b3[k]) * w3
                v = v + (<double>a4[k] + <double>b4[k]) * w4
                acc[k] = v
            jj = jj + 4
        while jj < 0:
            a1 = rp[s + jj]
            b1 = rp[s - jj]
            w1 = w[s + jj]
            for k in range(n):
                acc[k] = acc[k] + (<double>a1[k] + <double>b1[k]) * w1
            jj = jj + 1
        for k in range(n):
            dst[i * n + k] = <float>acc[k]


cdef void _tent_cols(const float* src, float* dst, Py_ssize_t h, Py_ssize_t n, const double* w,
                     Py_ssize_t s, double cval, double* acc, double* pad) noexcept nogil:
    # Along axis 1, over one padded row at a time.
    cdef Py_ssize_t i, k, jj
    cdef double w0 = w[s], v, w1, w2, w3, w4
    cdef double* p = pad + s
    cdef const float* row
    for k in range(s):
        pad[k] = cval
        pad[s + n + k] = cval
    for i in range(h):
        row = src + i * n
        for k in range(n):
            p[k] = <double>row[k]
        for k in range(n):
            acc[k] = p[k] * w0
        jj = -s
        while jj + 4 <= 0:
            w1 = w[s + jj]
            w2 = w[s + jj + 1]
            w3 = w[s + jj + 2]
            w4 = w[s + jj + 3]
            for k in range(n):
                v = acc[k]
                v = v + (p[k + jj] + p[k - jj]) * w1
                v = v + (p[k + jj + 1] + p[k - jj - 1]) * w2
                v = v + (p[k + jj + 2] + p[k - jj - 2]) * w3
                v = v + (p[k + jj + 3] + p[k - jj - 3]) * w4
                acc[k] = v
            jj = jj + 4
        while jj < 0:
            w1 = w[s + jj]
            for k in range(n):
                acc[k] = acc[k] + (p[k + jj] + p[k - jj]) * w1
            jj = jj + 1
        for k in range(n):
            dst[i * n + k] = <float>acc[k]


def tent_blur(const float[:, ::1] alpha, const double[::1] weights, float outside):
    """ndimage.correlate1d over axis 0, then axis 1, mode "constant".

    scipy's sum for an odd, symmetric kernel: in double, the centre tap,
    then each pair from the farthest in, rounded to float32 after each axis.
    Never fused, so every machine gets the same bits.
    """
    if weights.shape[0] % 2 == 0:
        raise ValueError("odd kernel only")
    cdef Py_ssize_t h = alpha.shape[0], n = alpha.shape[1], s = weights.shape[0] // 2, k
    mid = np.empty((h, n), dtype=np.float32)
    out = np.empty((h, n), dtype=np.float32)
    if h == 0 or n == 0:
        return out
    cdef float[:, ::1] m = mid
    cdef float[:, ::1] o = out
    cdef double* acc = <double*>malloc(n * sizeof(double))
    cdef double* pad = <double*>malloc((n + 2 * s) * sizeof(double))
    cdef float* crow = <float*>malloc(n * sizeof(float))
    cdef const float** rp = <const float**>malloc((2 * s + 1) * sizeof(float*))
    if acc == NULL or pad == NULL or crow == NULL or rp == NULL:
        free(acc)
        free(pad)
        free(crow)
        free(rp)
        raise MemoryError()
    with nogil:
        for k in range(n):
            crow[k] = outside
        _tent_rows(&alpha[0, 0], &m[0, 0], h, n, &weights[0], s, crow, acc, rp)
        _tent_cols(&m[0, 0], &o[0, 0], h, n, &weights[0], s, outside, acc, pad)
    free(acc)
    free(pad)
    free(crow)
    free(rp)
    return out
