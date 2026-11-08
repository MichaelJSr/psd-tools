# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""C loops behind effects._signed_distance, _nearest_boundary, _chamfer_transform, _chamfer_relax and _tent_blur.

The numpy code's float32 (pruned minimum) and float64 (chamfer) operations in
the same order, so the results are bit-identical. Memoryviews only, so the
module builds against the Limited API without numpy headers.
"""
import numpy as np
from libc.math cimport INFINITY, sqrt
from libc.stdlib cimport calloc, free, malloc


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
    if height == 0 or width == 0:
        return
    with nogil:
        _relax(&f[0, 0], height, width, diag, knight)


def disk_reach(const unsigned char[:, ::1] stating, double radius):
    """Pixels whose Euclidean distance to a stating pixel is at most ``radius``.

    ``distance_transform_edt(~stating) <= radius`` without the transform: the
    distance is ``sqrt(dy * dy + dx * dx)`` in double either way, so marking
    each stating pixel's disk is exact.
    """
    cdef Py_ssize_t h = stating.shape[0], w = stating.shape[1], y, j
    cdef Py_ssize_t r = <Py_ssize_t>radius + 1
    cdef Py_ssize_t[::1] reach = np.full(2 * r + 1, -1, dtype=np.intp)
    _disk_table(reach, r, radius)
    # Each row: +1 where a covered run starts, -1 past its end.
    cdef int[:, ::1] runs = np.zeros((h, w + 1), dtype=np.intc)
    with nogil:
        _disk_runs(stating, 0, 0, h, w, reach, r, runs)
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


cdef inline float _npmax(float a, float b) noexcept nogil:
    # np.maximum on values with no signed zero to tell apart: NaN wins.
    if a != a:
        return a
    if b != b:
        return b
    return a if a >= b else b


cdef inline float _npmin(float a, float b) noexcept nogil:
    if a != a:
        return a
    if b != b:
        return b
    return a if a <= b else b


def sd_prepare(const float[:, ::1] alpha, float[:, ::1] field, unsigned char[:, ::1] partial,
               unsigned char[:, ::1] boundary):
    """_signed_distance's masks in one pass: partial, boundary (partial, or opaque beside clear), and
    the field's start (near on partial, -inf opaque, inf else). Returns (any partial, any opaque)."""
    cdef Py_ssize_t h = alpha.shape[0], w = alpha.shape[1], y, x
    cdef float a
    cdef bint anyp = False, anyo = False, grown
    with nogil:
        for y in range(h):
            for x in range(w):
                a = alpha[y, x]
                if a > 0 and a < 1:
                    partial[y, x] = 1
                    anyp = True
                    field[y, x] = <float>0.5 - a
                else:
                    partial[y, x] = 0
                    field[y, x] = -INFINITY if a >= 1 else INFINITY
        for y in range(h):
            for x in range(w):
                if partial[y, x]:
                    boundary[y, x] = 1
                    continue
                boundary[y, x] = 0
                if not alpha[y, x] >= 1:
                    continue
                anyo = True
                grown = (y > 0 and alpha[y - 1, x] <= 0) or (y + 1 < h and alpha[y + 1, x] <= 0) or \
                    (x > 0 and alpha[y, x - 1] <= 0) or (x + 1 < w and alpha[y, x + 1] <= 0)
                if grown:
                    boundary[y, x] = 1
    return bool(anyp), bool(anyo)


cdef void _relax(double* f, Py_ssize_t height, Py_ssize_t width, double diag, double knight) noexcept nogil:
    # chamfer_relax's passes on a C-contiguous buffer.
    cdef Py_ssize_t passno, idx, row, step, b1, b2, j
    cdef double run, v, m, s
    cdef bint has1, has2
    cdef double* cur
    cdef double* p1
    cdef double* p2
    cdef Py_ssize_t edge[4]
    cdef int e
    for passno in range(2):
        step = 1 if passno == 0 else -1
        for idx in range(height):
            row = idx if passno == 0 else height - 1 - idx
            b1 = row - step
            b2 = row - 2 * step
            has1 = 0 <= b1 < height
            has2 = 0 <= b2 < height and width > 1
            cur = f + row * width
            # Every move from the one or two rows before at once: each is
            # a minimum against a finished row, so the order is free.
            if has1 and has2 and width > 4:
                p1 = f + b1 * width
                p2 = f + b2 * width
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
                p1 = f + b1 * width if has1 else cur
                p2 = f + b2 * width if has2 else cur
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


cdef void _relax2(double* f, double* g, Py_ssize_t height, Py_ssize_t width, double diag, double knight) noexcept nogil:
    # _relax on two fields in the same loops: each field sees the same moves in the same order.
    cdef Py_ssize_t passno, idx, row, step, b1, b2, j
    cdef double runf, rung, v, w2, m, s, n, t
    cdef bint has1, has2
    cdef double* cf
    cdef double* cg
    cdef double* f1
    cdef double* f2
    cdef double* g1
    cdef double* g2
    if not (height > 2 and width > 4):
        _relax(f, height, width, diag, knight)
        _relax(g, height, width, diag, knight)
        return
    for passno in range(2):
        step = 1 if passno == 0 else -1
        for idx in range(height):
            row = idx if passno == 0 else height - 1 - idx
            b1 = row - step
            b2 = row - 2 * step
            has1 = 0 <= b1 < height
            has2 = 0 <= b2 < height
            cf = f + row * width
            cg = g + row * width
            if has1 and has2:
                f1 = f + b1 * width
                f2 = f + b2 * width
                g1 = g + b1 * width
                g2 = g + b2 * width
                for j in range(2, width - 2):
                    m = cf[j]
                    s = f1[j] + 1.0
                    m = s if s < m else m
                    s = f1[j - 1] + diag
                    m = s if s < m else m
                    s = f1[j + 1] + diag
                    m = s if s < m else m
                    s = f1[j - 2] + knight
                    m = s if s < m else m
                    s = f1[j + 2] + knight
                    m = s if s < m else m
                    s = f2[j - 1] + knight
                    m = s if s < m else m
                    s = f2[j + 1] + knight
                    m = s if s < m else m
                    cf[j] = m
                    n = cg[j]
                    t = g1[j] + 1.0
                    n = t if t < n else n
                    t = g1[j - 1] + diag
                    n = t if t < n else n
                    t = g1[j + 1] + diag
                    n = t if t < n else n
                    t = g1[j - 2] + knight
                    n = t if t < n else n
                    t = g1[j + 2] + knight
                    n = t if t < n else n
                    t = g2[j - 1] + knight
                    n = t if t < n else n
                    t = g2[j + 1] + knight
                    n = t if t < n else n
                    cg[j] = n
                _relax_edges(cf, f1, f2, width, diag, knight)
                _relax_edges(cg, g1, g2, width, diag, knight)
            elif has1:
                _relax_row1(cf, f + b1 * width, width, diag, knight)
                _relax_row1(cg, g + b1 * width, width, diag, knight)
            runf = INFINITY
            rung = INFINITY
            for j in range(width):
                v = cf[j] - <double>j
                if v < runf or j == 0:
                    runf = v
                v = runf + <double>j
                if v < cf[j]:
                    cf[j] = v
                w2 = cg[j] - <double>j
                if w2 < rung or j == 0:
                    rung = w2
                w2 = rung + <double>j
                if w2 < cg[j]:
                    cg[j] = w2
            runf = INFINITY
            rung = INFINITY
            for j in range(width - 1, -1, -1):
                v = cf[j] + <double>j
                if v < runf or j == width - 1:
                    runf = v
                v = runf - <double>j
                if v < cf[j]:
                    cf[j] = v
                w2 = cg[j] + <double>j
                if w2 < rung or j == width - 1:
                    rung = w2
                w2 = rung - <double>j
                if w2 < cg[j]:
                    cg[j] = w2


cdef inline void _relax_edges(double* cur, const double* p1, const double* p2, Py_ssize_t width,
                              double diag, double knight) noexcept nogil:
    # _relax's four edge pixels of a row with both rows before it.
    cdef Py_ssize_t edge[4]
    cdef int e
    cdef Py_ssize_t j
    cdef double m, s
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


cdef inline void _relax_row1(double* cur, const double* p1, Py_ssize_t width, double diag,
                             double knight) noexcept nogil:
    # _relax's moves from the one row before (the second row of a pass).
    cdef Py_ssize_t j
    cdef double m, s
    for j in range(width):
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
        cur[j] = m


def distance_box(const float[:, ::1] alpha, float[:, ::1] field, const unsigned char[:, ::1] partial,
                 Py_ssize_t y0, Py_ssize_t y1, Py_ssize_t x0, Py_ssize_t x1, double radius, bint any_opaque,
                 Py_ssize_t span, const Py_ssize_t[:, ::1] offsets, const float[::1] lengths,
                 double diag, double knight):
    """effects._distance_on_box on alpha[y0:y1, x0:x1], written into field there, for a radius the disk
    marking takes and no ball search: the same float32 and float64 steps as the numpy code."""
    cdef Py_ssize_t h = y1 - y0, w = x1 - x0, y, x, rmin, rmax, cmin, cmax, r0, r1, c0, c1, hc, wc
    cdef Py_ssize_t m = lengths.shape[0], k, i, n
    cdef Py_ssize_t R = <Py_ssize_t>radius + 1
    cdef float a, best, cand, length, o, c
    cdef float freach = <float>radius
    cdef bint anyin
    cdef double* to_o
    cdef double* to_c
    cdef int acc
    cdef Py_ssize_t[::1] tab
    cdef int[:, ::1] runs_v
    cdef Py_ssize_t wp, hp
    cdef unsigned char* spad
    cdef float* npad
    cdef Py_ssize_t* lin
    if h <= 0 or w <= 0:
        return
    # The stating pixels' box, out by span.
    rmin = h
    rmax = -1
    cmin = w
    cmax = -1
    for y in range(h):
        for x in range(w):
            if partial[y0 + y, x0 + x]:
                if y < rmin:
                    rmin = y
                if y > rmax:
                    rmax = y
                if x < cmin:
                    cmin = x
                if x > cmax:
                    cmax = x
    # The field's start on the box is sd_prepare's: near, -inf or inf.
    if rmax >= 0:
        r0 = rmin - span if rmin - span > 0 else 0
        r1 = rmax + span + 1 if rmax + span + 1 < h else h
        c0 = cmin - span if cmin - span > 0 else 0
        c1 = cmax + span + 1 if cmax + span + 1 < w else w
        hc = r1 - r0
        wc = c1 - c0
        # disk_reach's marking of the crop, then each active pixel's pruned minimum.
        reach_tab = np.full(2 * R + 1, -1, dtype=np.intp)
        tab = reach_tab
        _disk_table(tab, R, radius)
        runs_arr = np.zeros((hc, wc + 1), dtype=np.intc)
        runs_v = runs_arr
        with nogil:
            _disk_runs(partial, y0 + r0, x0 + c0, hc, wc, tab, R, runs_v)
        wp = wc + 2 * span
        hp = hc + 2 * span
        spad = <unsigned char*>calloc(hp * wp, sizeof(unsigned char))
        npad = <float*>calloc(hp * wp, sizeof(float))
        lin = <Py_ssize_t*>malloc((m if m > 0 else 1) * sizeof(Py_ssize_t))
        if spad == NULL or npad == NULL or lin == NULL:
            free(spad)
            free(npad)
            free(lin)
            raise MemoryError()
        with nogil:
            for k in range(m):
                lin[k] = offsets[k, 0] * wp + offsets[k, 1]
            for y in range(hc):
                for x in range(wc):
                    if partial[y0 + r0 + y, x0 + c0 + x]:
                        spad[(y + span) * wp + x + span] = 1
                        npad[(y + span) * wp + x + span] = <float>0.5 - alpha[y0 + r0 + y, x0 + c0 + x]
            for y in range(hc):
                acc = 0
                for x in range(wc):
                    acc = acc + runs_v[y, x]
                    i = (y + span) * wp + x + span
                    if acc <= 0 or spad[i]:
                        continue
                    if alpha[y0 + r0 + y, x0 + c0 + x] >= 1:
                        best = -INFINITY
                        for k in range(m):
                            length = lengths[k]
                            if spad[i + lin[k]]:
                                cand = npad[i + lin[k]] - length
                                if cand > best:
                                    best = cand
                            if best >= <float>0.5 - length:
                                break
                    else:
                        best = INFINITY
                        for k in range(m):
                            length = lengths[k]
                            if spad[i + lin[k]]:
                                cand = npad[i + lin[k]] + length
                                if cand < best:
                                    best = cand
                            if best <= length - <float>0.5:
                                break
                    field[y0 + r0 + y, x0 + c0 + x] = best
        free(spad)
        free(npad)
        free(lin)
    if not any_opaque:
        return
    anyin = False
    for y in range(h):
        for x in range(w):
            if alpha[y0 + y, x0 + x] >= 1:
                anyin = True
                break
        if anyin:
            break
    if not anyin:
        return
    n = h * w
    to_o = <double*>malloc(n * sizeof(double))
    to_c = <double*>malloc(n * sizeof(double))
    if to_o == NULL or to_c == NULL:
        free(to_o)
        free(to_c)
        raise MemoryError()
    with nogil:
        for y in range(h):
            for x in range(w):
                a = alpha[y0 + y, x0 + x]
                to_o[y * w + x] = 0.0 if a >= 1 else INFINITY
                to_c[y * w + x] = 0.0 if a <= 0 else INFINITY
        _relax2(to_o, to_c, h, w, diag, knight)
        for y in range(h):
            for x in range(w):
                if partial[y0 + y, x0 + x]:
                    continue
                i = y * w + x
                if alpha[y0 + y, x0 + x] >= 1:
                    c = <float>to_c[i]
                    cand = <float>0.5 - c if c <= freach else -INFINITY
                    field[y0 + y, x0 + x] = _npmax(field[y0 + y, x0 + x], cand)
                else:
                    o = <float>to_o[i]
                    cand = o - <float>0.5 if o <= freach else INFINITY
                    field[y0 + y, x0 + x] = _npmin(field[y0 + y, x0 + x], cand)
    free(to_o)
    free(to_c)


cdef void _disk_table(Py_ssize_t[::1] reach, Py_ssize_t r, double radius) noexcept:
    cdef Py_ssize_t dy, dx
    for dy in range(-r, r + 1):
        dx = -1
        while sqrt(<double>(dy * dy + (dx + 1) * (dx + 1))) <= radius:
            dx += 1
        reach[dy + r] = dx


cdef void _disk_runs(const unsigned char[:, ::1] stating, Py_ssize_t oy, Py_ssize_t ox, Py_ssize_t h, Py_ssize_t w,
                     Py_ssize_t[::1] reach, Py_ssize_t r, int[:, ::1] runs) noexcept nogil:
    # disk_reach's run marks for the crop at (oy, ox) of size h x w.
    cdef Py_ssize_t y, x, dy, yy, xa, xb
    for y in range(h):
        for x in range(w):
            if not stating[oy + y, ox + x]:
                continue
            for dy in range(-r, r + 1):
                yy = y + dy
                if yy < 0 or yy >= h or reach[dy + r] < 0:
                    continue
                xa = x - reach[dy + r]
                xb = x + reach[dy + r] + 1
                if xa < 0:
                    xa = 0
                if xb > w:
                    xb = w
                runs[yy, xa] += 1
                runs[yy, xb] -= 1
