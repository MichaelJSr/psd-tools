# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""C loops behind effects._signed_distance, _nearest_boundary, _chamfer_transform, _chamfer_relax and _tent_blur, and scanline.fill_coverage's row sums.

The numpy code's float32 (pruned minimum) and float64 (chamfer) operations in
the same order, so the results are bit-identical. Memoryviews only, so the
module builds against the Limited API without numpy headers.
"""
import numpy as np
from libc.math cimport INFINITY, fabs, fmod, signbit, sqrt
from libc.stdlib cimport calloc, free, malloc
from libc.string cimport memcpy


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


cdef void _rows_span(const float** rp, float* out, Py_ssize_t k0, Py_ssize_t k1, const double* w,
                     Py_ssize_t s, double* acc, const Py_ssize_t* taps, const Py_ssize_t* tlo,
                     const Py_ssize_t* thi, Py_ssize_t ntaps) noexcept nogil:
    # _tent_rows' sum on columns k0 to k1 over the pairs in taps, each read only in [tlo, thi): both +0 elsewhere.
    cdef Py_ssize_t k, t, jj, lo, hi
    cdef double w0 = w[s], v, w1, w2, w3, w4
    cdef const float* a1 = rp[s]
    cdef const float* b1
    cdef const float* a2
    cdef const float* b2
    cdef const float* a3
    cdef const float* b3
    cdef const float* a4
    cdef const float* b4
    for k in range(k0, k1):
        acc[k] = <double>a1[k] * w0
    t = 0
    while t + 4 <= ntaps:
        lo = tlo[t]
        hi = thi[t]
        for k in range(t + 1, t + 4):
            if tlo[k] < lo:
                lo = tlo[k]
            if thi[k] > hi:
                hi = thi[k]
        if lo < k0:
            lo = k0
        if hi > k1:
            hi = k1
        if lo < hi:
            jj = taps[t]
            a1 = rp[s + jj]
            b1 = rp[s - jj]
            w1 = w[s + jj]
            jj = taps[t + 1]
            a2 = rp[s + jj]
            b2 = rp[s - jj]
            w2 = w[s + jj]
            jj = taps[t + 2]
            a3 = rp[s + jj]
            b3 = rp[s - jj]
            w3 = w[s + jj]
            jj = taps[t + 3]
            a4 = rp[s + jj]
            b4 = rp[s - jj]
            w4 = w[s + jj]
            for k in range(lo, hi):
                v = acc[k]
                v = v + (<double>a1[k] + <double>b1[k]) * w1
                v = v + (<double>a2[k] + <double>b2[k]) * w2
                v = v + (<double>a3[k] + <double>b3[k]) * w3
                v = v + (<double>a4[k] + <double>b4[k]) * w4
                acc[k] = v
        t = t + 4
    while t < ntaps:
        lo = tlo[t] if tlo[t] > k0 else k0
        hi = thi[t] if thi[t] < k1 else k1
        jj = taps[t]
        a1 = rp[s + jj]
        b1 = rp[s - jj]
        w1 = w[s + jj]
        for k in range(lo, hi):
            acc[k] = acc[k] + (<double>a1[k] + <double>b1[k]) * w1
        t = t + 1
    for k in range(k0, k1):
        out[k] = <float>acc[k]


cdef inline void _cols_taps(const double* p, double* acc, Py_ssize_t lo, Py_ssize_t hi, Py_ssize_t jj,
                            double w1, double w2, double w3, double w4) noexcept nogil:
    cdef Py_ssize_t k
    cdef double v
    for k in range(lo, hi):
        v = acc[k]
        v = v + (p[k + jj] + p[k - jj]) * w1
        v = v + (p[k + jj + 1] + p[k - jj - 1]) * w2
        v = v + (p[k + jj + 2] + p[k - jj - 2]) * w3
        v = v + (p[k + jj + 3] + p[k - jj - 3]) * w4
        acc[k] = v


cdef void _cols_span(const double* p, float* out, Py_ssize_t k0, Py_ssize_t k1, const double* w,
                     Py_ssize_t s, double* acc, Py_ssize_t lo, Py_ssize_t hi) noexcept nogil:
    # _tent_cols' sum on columns k0 to k1. p is +0 outside [lo, hi), pads too, so a pair m = -jj
    # apart is read only for k in [lo - m, hi - m) or [lo + m, hi + m).
    cdef Py_ssize_t k, jj, mx, mn, la, lb, ra, rb
    cdef double w0 = w[s], w1, w2, w3, w4
    for k in range(k0, k1):
        acc[k] = p[k] * w0
    jj = -s
    while jj + 4 <= 0:
        w1 = w[s + jj]
        w2 = w[s + jj + 1]
        w3 = w[s + jj + 2]
        w4 = w[s + jj + 3]
        mx = -jj
        mn = -(jj + 3)
        la = lo - mx
        lb = hi - mn
        ra = lo + mn
        rb = hi + mx
        if lb >= ra:
            lb = rb
            ra = rb
        if la < k0:
            la = k0
        if lb > k1:
            lb = k1
        if ra < k0:
            ra = k0
        if rb > k1:
            rb = k1
        if la < lb:
            _cols_taps(p, acc, la, lb, jj, w1, w2, w3, w4)
        if ra < rb:
            _cols_taps(p, acc, ra, rb, jj, w1, w2, w3, w4)
        jj = jj + 4
    while jj < 0:
        w1 = w[s + jj]
        mx = -jj
        la = lo - mx
        lb = hi - mx
        ra = lo + mx
        rb = hi + mx
        if lb >= ra:
            lb = rb
            ra = rb
        if la < k0:
            la = k0
        if lb > k1:
            lb = k1
        if ra < k0:
            ra = k0
        if rb > k1:
            rb = k1
        for k in range(la, lb):
            acc[k] = acc[k] + (p[k + jj] + p[k - jj]) * w1
        for k in range(ra, rb):
            acc[k] = acc[k] + (p[k + jj] + p[k - jj]) * w1
        jj = jj + 1
    for k in range(k0, k1):
        out[k] = <float>acc[k]


cdef double _flat_sum(double v, const double* w, Py_ssize_t s) noexcept nogil:
    # The sum over a window of v alone, in the spans' order (unrolled by 4, then the rest).
    cdef double acc = v * w[s]
    cdef Py_ssize_t jj = -s
    while jj + 4 <= 0:
        acc = acc + (v + v) * w[s + jj]
        acc = acc + (v + v) * w[s + jj + 1]
        acc = acc + (v + v) * w[s + jj + 2]
        acc = acc + (v + v) * w[s + jj + 3]
        jj = jj + 4
    while jj < 0:
        acc = acc + (v + v) * w[s + jj]
        jj = jj + 1
    return acc


cdef inline unsigned int _bits(float v) noexcept nogil:
    cdef unsigned int b
    memcpy(&b, &v, 4)
    return b


cdef void _flat_rows(const float* src, const unsigned int* bits, float* dst, Py_ssize_t h, Py_ssize_t n,
                     const double* w, Py_ssize_t s, const float* crow, unsigned int outside_bits,
                     unsigned int one_bits, double* acc, const float** rp, int* cz, int* co,
                     unsigned char* zr, Py_ssize_t* taps, Py_ssize_t* rlo, Py_ssize_t* rhi, Py_ssize_t* tlo,
                     Py_ssize_t* thi, bint skip) noexcept nogil:
    # _tent_rows; a window of one value (+0 or 1, by bits) takes its sum, worked once. The rest go
    # through _rows_span, with skip leaving out pairs of +0 rows (no sum is -0, so +0 adds nothing).
    cdef Py_ssize_t i, k, t, j, k0, nout, ntaps, lo, hi, width = 2 * s + 1
    cdef int zo = outside_bits == 0, oo = outside_bits == one_bits, ez, eo
    cdef float zero_v = <float>_flat_sum(0.0, w, s), one_v = <float>_flat_sum(1.0, w, s)
    cdef const unsigned int* row
    cdef unsigned int b
    cdef float* out
    cdef bint za, zb
    for k in range(n):
        cz[k] = 0
        co[k] = 0
    for j in range(h):
        row = bits + j * n
        rlo[j] = n
        rhi[j] = 0
        for k in range(n):
            if row[k] != 0:
                rlo[j] = k
                break
        k = n - 1
        while k >= rlo[j]:
            if row[k] != 0:
                rhi[j] = k + 1
                break
            k = k - 1
        zr[j] = rlo[j] >= rhi[j]
    for j in range(0, s + 1 if s + 1 < h else h):
        row = bits + j * n
        for k in range(n):
            b = row[k]
            cz[k] += b == 0
            co[k] += b == one_bits
    for i in range(h):
        for t in range(2 * s + 1):
            j = i - s + t
            rp[t] = src + j * n if 0 <= j < h else crow
        nout = (s - i if s > i else 0) + (i + s - (h - 1) if i + s > h - 1 else 0)
        ez = <int>(width - (nout if zo else 0))
        eo = <int>(width - (nout if oo else 0))
        out = dst + i * n
        ntaps = -1
        k0 = -1
        for k in range(n + 1):
            if k < n and cz[k] != ez and co[k] != eo:
                if k0 < 0:
                    k0 = k
                continue
            if k0 >= 0:
                if ntaps < 0:
                    ntaps = 0
                    for t in range(-s, 0):
                        if skip:
                            j = i + t
                            za = zr[j] if 0 <= j < h else zo
                            lo = rlo[j] if 0 <= j < h else (n if zo else 0)
                            hi = rhi[j] if 0 <= j < h else (0 if zo else n)
                            j = i - t
                            zb = zr[j] if 0 <= j < h else zo
                            if za and zb:
                                continue
                            if 0 <= j < h:
                                lo = rlo[j] if rlo[j] < lo else lo
                                hi = rhi[j] if rhi[j] > hi else hi
                            elif not zo:
                                lo = 0
                                hi = n
                        else:
                            lo = 0
                            hi = n
                        taps[ntaps] = t
                        tlo[ntaps] = lo
                        thi[ntaps] = hi
                        ntaps = ntaps + 1
                _rows_span(rp, out, k0, k, w, s, acc, taps, tlo, thi, ntaps)
                k0 = -1
            if k < n:
                out[k] = zero_v if cz[k] == ez else one_v
        # slide: row i - s leaves, row i + s + 1 comes in
        j = i - s
        if 0 <= j < h:
            row = bits + j * n
            for k in range(n):
                b = row[k]
                cz[k] -= b == 0
                co[k] -= b == one_bits
        j = i + s + 1
        if 0 <= j < h:
            row = bits + j * n
            for k in range(n):
                b = row[k]
                cz[k] += b == 0
                co[k] += b == one_bits


cdef void _flat_cols(const float* src, float* dst, Py_ssize_t h, Py_ssize_t n, const double* w,
                     Py_ssize_t s, double cval, double* acc, double* pad, bint skip) noexcept nogil:
    # _tent_cols; a window of one value (by bits, pads included) takes its sum, once per run.
    cdef Py_ssize_t i, k, k0, t, changes, lo, hi
    cdef double* p = pad + s
    cdef const float* row
    cdef unsigned int pad_bits = _bits(<float>cval), last_bits = 0, b
    cdef float last_v = <float>_flat_sum(0.0, w, s)
    cdef float* out
    for k in range(s):
        pad[k] = cval
        pad[s + n + k] = cval
    for i in range(h):
        row = src + i * n
        for k in range(n):
            p[k] = <double>row[k]
        out = dst + i * n
        # the row's span of nonzero bits; with pads not +0 (or skip off), all of it
        lo = -s
        hi = n + s
        if skip and pad_bits == 0:
            lo = n
            hi = 0
            for k in range(n):
                if _bits(row[k]) != 0:
                    lo = k
                    break
            k = n - 1
            while k >= lo:
                if _bits(row[k]) != 0:
                    hi = k + 1
                    break
                k = k - 1
        # neighbours that differ inside the window [k - s, k + s]
        changes = 0
        for t in range(-s + 1, s + 1):
            changes += _qbits(row, n, t, pad_bits) != _qbits(row, n, t - 1, pad_bits)
        k0 = -1
        for k in range(n + 1):
            if k < n and changes != 0:
                if k0 < 0:
                    k0 = k
            else:
                if k0 >= 0:
                    _cols_span(p, out, k0, k, w, s, acc, lo, hi)
                    k0 = -1
                if k < n:
                    b = _bits(row[k])
                    if b != last_bits:
                        last_bits = b
                        last_v = <float>_flat_sum(p[k], w, s)
                    out[k] = last_v
            if k < n:
                t = k + s + 1
                changes += _qbits(row, n, t, pad_bits) != _qbits(row, n, t - 1, pad_bits)
                t = k - s + 1
                changes -= _qbits(row, n, t, pad_bits) != _qbits(row, n, t - 1, pad_bits)


cdef inline unsigned int _qbits(const float* row, Py_ssize_t n, Py_ssize_t t, unsigned int pad_bits) noexcept nogil:
    return _bits(row[t]) if 0 <= t < n else pad_bits


cdef unsigned int NEG_ZERO = 0x80000000

# Below this reach the plain loops cost less than the window counts.
FLAT_MIN_REACH = 8


def tent_blur(const float[:, ::1] alpha, const double[::1] weights, float outside):
    """ndimage.correlate1d over axis 0, then axis 1, mode "constant".

    scipy's sum for an odd, symmetric kernel: in double, the centre tap,
    then each pair from the farthest in, rounded to float32 after each axis.
    Never fused, so every machine gets the same bits. From a reach of
    FLAT_MIN_REACH, a window holding one value (by its bits; +0 or 1 on
    axis 0) takes that value's sum, worked once in the same order, and with
    no sign bit anywhere terms that are +0 are left out. Same bits.
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
    cdef const unsigned int[:, ::1] bits
    cdef bint flat = s >= FLAT_MIN_REACH
    cdef unsigned int one_bits = 0x3F800000
    if flat:
        bits = np.asarray(alpha).view(np.uint32)
    cdef double* acc = <double*>malloc(n * sizeof(double))
    cdef double* pad = <double*>malloc((n + 2 * s) * sizeof(double))
    cdef float* crow = <float*>malloc(n * sizeof(float))
    cdef const float** rp = <const float**>malloc((2 * s + 1) * sizeof(float*))
    cdef int* cz = <int*>malloc(n * sizeof(int))
    cdef int* co = <int*>malloc(n * sizeof(int))
    cdef unsigned char* zr = <unsigned char*>malloc(h * sizeof(unsigned char))
    cdef Py_ssize_t* taps = <Py_ssize_t*>malloc((s + 1) * sizeof(Py_ssize_t))
    cdef Py_ssize_t* tlo = <Py_ssize_t*>malloc((s + 1) * sizeof(Py_ssize_t))
    cdef Py_ssize_t* thi = <Py_ssize_t*>malloc((s + 1) * sizeof(Py_ssize_t))
    cdef Py_ssize_t* rlo = <Py_ssize_t*>malloc((h + 1) * sizeof(Py_ssize_t))
    cdef Py_ssize_t* rhi = <Py_ssize_t*>malloc((h + 1) * sizeof(Py_ssize_t))
    cdef bint skip = False
    if acc == NULL or pad == NULL or crow == NULL or rp == NULL or cz == NULL or co == NULL or zr == NULL or taps == NULL \
            or tlo == NULL or thi == NULL or rlo == NULL or rhi == NULL:
        free(acc)
        free(pad)
        free(crow)
        free(rp)
        free(cz)
        free(co)
        free(zr)
        free(taps)
        free(tlo)
        free(thi)
        free(rlo)
        free(rhi)
        raise MemoryError()
    with nogil:
        for k in range(n):
            crow[k] = outside
        if flat:
            # skip only with no sign bit in alpha, outside or weights, and finite weights: no sum is -0
            skip = (_bits(outside) & NEG_ZERO) == 0
            for k in range(2 * s + 1):
                if not weights[k] < INFINITY or signbit(weights[k]):
                    skip = False
            for k in range(h * n if skip else 0):
                if (&bits[0, 0])[k] & NEG_ZERO:
                    skip = False
                    break
            _flat_rows(&alpha[0, 0], &bits[0, 0], &m[0, 0], h, n, &weights[0], s, crow, _bits(outside),
                       one_bits, acc, rp, cz, co, zr, taps, rlo, rhi, tlo, thi, skip)
            _flat_cols(&m[0, 0], &o[0, 0], h, n, &weights[0], s, outside, acc, pad, skip)
        else:
            _tent_rows(&alpha[0, 0], &m[0, 0], h, n, &weights[0], s, crow, acc, rp)
            _tent_cols(&m[0, 0], &o[0, 0], h, n, &weights[0], s, outside, acc, pad)
    free(acc)
    free(pad)
    free(crow)
    free(rp)
    free(cz)
    free(co)
    free(zr)
    free(taps)
    free(tlo)
    free(thi)
    free(rlo)
    free(rhi)
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


cdef int _gather_box(const float[:, ::1] alpha, float[:, ::1] field, const unsigned char[:, ::1] partial,
                     Py_ssize_t oy, Py_ssize_t ox, Py_ssize_t hc, Py_ssize_t wc, const int[:, ::1] runs_v,
                     Py_ssize_t span, const Py_ssize_t[:, ::1] offsets, const float[::1] lengths) except -1:
    # Each active pixel's pruned minimum over the stating pixels in the table.
    cdef Py_ssize_t m = lengths.shape[0], k, i, y, x, wp, hp
    cdef int acc
    cdef float best, cand, length
    cdef unsigned char* spad
    cdef float* npad
    cdef Py_ssize_t* lin
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
                if partial[oy + y, ox + x]:
                    spad[(y + span) * wp + x + span] = 1
                    npad[(y + span) * wp + x + span] = <float>0.5 - alpha[oy + y, ox + x]
        for y in range(hc):
            acc = 0
            for x in range(wc):
                acc = acc + runs_v[y, x]
                i = (y + span) * wp + x + span
                if acc <= 0 or spad[i]:
                    continue
                if alpha[oy + y, ox + x] >= 1:
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
                field[oy + y, ox + x] = best
    free(spad)
    free(npad)
    free(lin)
    return 0


# The scatter runs while the stating pixels' offers number at most this many per active pixel.
cdef Py_ssize_t _SCATTER_PER_ACTIVE = 64


def _set_scatter_limit(Py_ssize_t limit):
    """Set _SCATTER_PER_ACTIVE (0: always gather), returning the old one. For tests."""
    global _SCATTER_PER_ACTIVE
    old = _SCATTER_PER_ACTIVE
    _SCATTER_PER_ACTIVE = limit
    return old


cdef int _scatter_box(const float[:, ::1] alpha, float[:, ::1] field, const unsigned char[:, ::1] partial,
                      Py_ssize_t oy, Py_ssize_t ox, Py_ssize_t hc, Py_ssize_t wc, const int[:, ::1] runs_v,
                      const Py_ssize_t[:, ::1] offsets, const float[::1] lengths) except -1:
    # _gather_box turned round: each stating pixel offers near + length and near - length to the
    # pixels at the table's offsets. A symmetric table gives the same candidates, and min and max
    # are exact, so the same bits. 0 (nothing written) where the gather is cheaper or the table
    # is not symmetric.
    cdef Py_ssize_t m = lengths.shape[0], k, i, y, x, dy, dx, n, side = 0, row, py, px, xa, xb
    cdef Py_ssize_t nstat = 0, nact = 0
    cdef int acc
    cdef bint symmetric = True
    cdef float nq, c
    cdef Py_ssize_t* xr
    cdef float* lrow
    cdef float* lo
    cdef float* hi
    cdef const float* lr
    cdef float* plo
    cdef float* phi
    with nogil:
        for y in range(hc):
            acc = 0
            for x in range(wc):
                acc = acc + runs_v[y, x]
                if partial[oy + y, ox + x]:
                    nstat = nstat + 1
                elif acc > 0:
                    nact = nact + 1
    if nact == 0:
        return 1
    if m == 0 or nstat * m > nact * _SCATTER_PER_ACTIVE:
        return 0
    for k in range(m):
        side = max(side, offsets[k, 0], -offsets[k, 0], offsets[k, 1], -offsets[k, 1])
    n = 2 * side + 1
    xr = <Py_ssize_t*>malloc(n * sizeof(Py_ssize_t))
    lrow = <float*>malloc(n * n * sizeof(float))
    lo = <float*>malloc(hc * wc * sizeof(float))
    hi = <float*>malloc(hc * wc * sizeof(float))
    if xr == NULL or lrow == NULL or lo == NULL or hi == NULL:
        free(xr)
        free(lrow)
        free(lo)
        free(hi)
        raise MemoryError()
    with nogil:
        # Each row's lengths, -xr..xr; inf where the table has no offset (no candidate).
        for row in range(n):
            xr[row] = -1
        for i in range(n * n):
            lrow[i] = INFINITY
        for k in range(m):
            dy = offsets[k, 0] + side
            dx = offsets[k, 1]
            xr[dy] = max(xr[dy], dx, -dx)
            i = dy * n + dx + side
            if lengths[k] < lrow[i]:
                lrow[i] = lengths[k]
        for i in range(n * n):
            if lrow[i] != lrow[n * n - 1 - i]:
                symmetric = False
        if symmetric:
            for i in range(hc * wc):
                lo[i] = INFINITY
                hi[i] = -INFINITY
            for y in range(hc):
                for x in range(wc):
                    if not partial[oy + y, ox + x]:
                        continue
                    nq = <float>0.5 - alpha[oy + y, ox + x]
                    for row in range(n):
                        py = y + row - side
                        if xr[row] < 0 or py < 0 or py >= hc:
                            continue
                        xa = max(x - xr[row], 0)
                        xb = min(x + xr[row] + 1, wc)
                        lr = lrow + row * n + side - x
                        plo = lo + py * wc
                        phi = hi + py * wc
                        for px in range(xa, xb):
                            c = nq + lr[px]
                            plo[px] = c if c < plo[px] else plo[px]
                            c = nq - lr[px]
                            phi[px] = c if c > phi[px] else phi[px]
            for y in range(hc):
                acc = 0
                for x in range(wc):
                    acc = acc + runs_v[y, x]
                    if acc > 0 and not partial[oy + y, ox + x]:
                        i = y * wc + x
                        field[oy + y, ox + x] = hi[i] if alpha[oy + y, ox + x] >= 1 else lo[i]
    free(xr)
    free(lrow)
    free(lo)
    free(hi)
    return 1 if symmetric else 0


def distance_box(const float[:, ::1] alpha, float[:, ::1] field, const unsigned char[:, ::1] partial,
                 Py_ssize_t y0, Py_ssize_t y1, Py_ssize_t x0, Py_ssize_t x1, double radius, bint any_opaque,
                 Py_ssize_t span, const Py_ssize_t[:, ::1] offsets, const float[::1] lengths,
                 double diag, double knight):
    """effects._distance_on_box on alpha[y0:y1, x0:x1], written into field there, for a radius the disk
    marking takes and no ball search: the same float32 and float64 steps as the numpy code."""
    cdef Py_ssize_t h = y1 - y0, w = x1 - x0, y, x, rmin, rmax, cmin, cmax, r0, r1, c0, c1, hc, wc, i, n
    cdef Py_ssize_t R = <Py_ssize_t>radius + 1
    cdef float a, cand, o, c
    cdef float freach = <float>radius
    cdef bint anyin
    cdef double* to_o
    cdef double* to_c
    cdef Py_ssize_t[::1] tab
    cdef int[:, ::1] runs_v
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
        # disk_reach's marking of the crop, then each active pixel's least candidate.
        reach_tab = np.full(2 * R + 1, -1, dtype=np.intp)
        tab = reach_tab
        _disk_table(tab, R, radius)
        runs_arr = np.zeros((hc, wc + 1), dtype=np.intc)
        runs_v = runs_arr
        with nogil:
            _disk_runs(partial, y0 + r0, x0 + c0, hc, wc, tab, R, runs_v)
        if not _scatter_box(alpha, field, partial, y0 + r0, x0 + c0, hc, wc, runs_v, offsets, lengths):
            _gather_box(alpha, field, partial, y0 + r0, x0 + c0, hc, wc, runs_v, span, offsets, lengths)
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


def coverage_rows(double[:, ::1] rows, float[:, ::1] coverage, Py_ssize_t top, Py_ssize_t width,
                  bint nonzero, double residue):
    """``scanline.fill_coverage``'s row pass, numpy's float64 steps in its order.

    Each row's prefix sum (numpy's cumsum: the first cell, then one add a cell),
    its absolute value folded into [0, 1] (``np.mod`` by 2, then ``1 - |r - 1|``)
    or held to 1, under ``residue`` 0, then cast; columns 1 to ``width`` go to
    ``coverage`` from row ``top``. ``rows`` is left as it was.
    """
    cdef Py_ssize_t h = rows.shape[0], y, x
    cdef double run, r
    if rows.shape[1] < width + 1 or top < 0 or top + h > coverage.shape[0] or coverage.shape[1] < width:
        raise ValueError("rows do not fit the coverage")
    with nogil:
        for y in range(h):
            run = rows[y, 0]
            for x in range(1, width + 1):
                run = run + rows[y, x]
                r = fabs(run)
                if nonzero:
                    # np.minimum(r, 1.0): NaN passes.
                    if r > 1.0:
                        r = 1.0
                else:
                    # np.mod(r, 2.0) for r >= +0 is fmod's exact answer (+0 for 0).
                    if r >= 2.0:
                        r = fmod(r, 2.0)
                    r = 1.0 - fabs(r - 1.0)
                if r < residue:
                    r = 0.0
                coverage[top + y, x - 1] = <float>r
