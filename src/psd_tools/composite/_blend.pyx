# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""C loops behind the compositor's source arithmetic.

Each is the numpy expression it replaces in ``composite.py`` (``lut_levels``:
``adjustments._apply_lut``; ``to_bytes``: ``composite_pil``; ``divide`` and
``clip``: ``utils``; ``vibrance_step``: ``adjustments._vibrance_step``;
``text_mix``: ``Compositor._mix_in_text_space``; ``unfilter``:
``composite._blend_unfiltered``), its
operations in the same dtypes and order, so the results are bit-identical.
``distinct_pixels`` numbers a canvas' distinct colours for ``composite._adjust``.
``whole_levels``, and ``mix_toward`` and ``remix`` with ``levels`` set, round to
whole 8-bit levels (``composite._whole_levels``).
Memoryviews only, so the module builds against the Limited API without numpy
headers.

Every canvas but ``lut_levels``', ``to_bytes``', ``vibrance_step``'s and
``distinct_pixels``' is read a row at a time, so a pixel's channels must be
adjacent and its neighbour the next pixel along (``_rows``), as they are in a
box cut from a canvas; ``utils.in_rows`` checks that first.
"""
from libc.math cimport NAN, ceilf, fabs, floorf, fmaxf, fminf, pow, powf, rintf, signbit

import numpy as np


cdef inline int _rows(
    const float[:, :, :] canvas, Py_ssize_t h, Py_ssize_t w, Py_ssize_t channels
) except -1:
    """Raise unless ``canvas`` is ``h`` by ``w`` by ``channels``, in rows of pixels."""
    if (
        canvas.shape[0] != h
        or canvas.shape[1] != w
        or canvas.shape[2] != channels
        or channels > 1 and canvas.strides[2] != sizeof(float)
        or canvas.shape[1] > 1 and canvas.strides[1] != channels * sizeof(float)
    ):
        raise ValueError("canvas is not laid out in rows of pixels")
    return 0


cdef inline float _clip(float q) noexcept nogil:
    # np.clip's answer: NaN and -0.0 pass through.
    return <float>0.0 if q < 0 else (<float>1.0 if q > 1 else q)


cdef inline float _level(float q) noexcept nogil:
    # np.rint(q * 255) / 255 in float32.
    return rintf(q * <float>255.0) / <float>255.0


cdef inline void _levels_row(
    float* o, const float* k, Py_ssize_t w, Py_ssize_t n
) noexcept nogil:
    # o's pixels in whole levels where k > 0, every one with k NULL.
    cdef Py_ssize_t x, c, i
    if k == NULL:
        for i in range(w * n):
            o[i] = _level(o[i])
        return
    for x in range(w):
        if k[x] > 0:
            for c in range(n):
                i = x * n + c
                o[i] = _level(o[i])


def unions(
    const float[:, :, :] shape_g,
    const float[:, :, :] shape,
    const float[:, :, :] alpha_g,
    const float[:, :, :] alpha,
    const float[:, :, :] alpha_0,
):
    """``union(shape_g, shape)``, ``union(alpha_g, alpha)`` and the alpha over ``alpha_0``."""
    cdef Py_ssize_t h = shape.shape[0], w = shape.shape[1], y, x
    cdef float a, b, g
    _rows(shape_g, h, w, 1)
    _rows(shape, h, w, 1)
    _rows(alpha_g, h, w, 1)
    _rows(alpha, h, w, 1)
    _rows(alpha_0, h, w, 1)
    out_shape = np.empty((h, w, 1), dtype=np.float32)
    out_alpha_g = np.empty((h, w, 1), dtype=np.float32)
    out_alpha = np.empty((h, w, 1), dtype=np.float32)
    if h == 0 or w == 0:
        return out_shape, out_alpha_g, out_alpha
    cdef float[:, :, ::1] s = out_shape
    cdef float[:, :, ::1] ag = out_alpha_g
    cdef float[:, :, ::1] al = out_alpha
    cdef const float *p_sg
    cdef const float *p_s
    cdef const float *p_ag
    cdef const float *p_a
    cdef const float *p_a0
    cdef float *o_s
    cdef float *o_ag
    cdef float *o_al
    with nogil:
        for y in range(h):
            p_sg, p_s = &shape_g[y, 0, 0], &shape[y, 0, 0]
            p_ag, p_a, p_a0 = &alpha_g[y, 0, 0], &alpha[y, 0, 0], &alpha_0[y, 0, 0]
            o_s, o_ag, o_al = &s[y, 0, 0], &ag[y, 0, 0], &al[y, 0, 0]
            for x in range(w):
                a = p_sg[x]
                b = p_s[x]
                o_s[x] = (a + b) - (a * b)
                a = p_ag[x]
                b = p_a[x]
                g = (a + b) - (a * b)
                o_ag[x] = g
                a = p_a0[x]
                o_al[x] = (a + g) - (a * g)
    return out_shape, out_alpha_g, out_alpha


def over(
    const float[:, :, :] shape,
    const float[:, :, :] alpha,
    const float[:, :, :] alpha_b,
    const float[:, :, :] color_b,
    const float[:, :, :] color,
    const float[:, :, :] blended,
    const float[:, :, :] alpha_previous,
    const float[:, :, :] canvas,
    const float[:, :, :] canvas_alpha,
):
    """``composite._over``: the colour canvas once a source goes on."""
    cdef Py_ssize_t h = canvas.shape[0], w = canvas.shape[1], n = canvas.shape[2]
    cdef Py_ssize_t y, x, c, i
    cdef float s, a, ab, t, m, k, d, q, q1, q2
    _rows(shape, h, w, 1)
    _rows(alpha, h, w, 1)
    _rows(alpha_b, h, w, 1)
    _rows(alpha_previous, h, w, 1)
    _rows(canvas_alpha, h, w, 1)
    _rows(color_b, h, w, n)
    _rows(color, h, w, n)
    _rows(blended, h, w, n)
    _rows(canvas, h, w, n)
    result = np.empty((h, w, n), dtype=np.float32)
    if h == 0 or w == 0 or n == 0:
        return result
    cdef float[:, :, ::1] out = result
    cdef const float *p_s
    cdef const float *p_a
    cdef const float *p_ab
    cdef const float *p_ap
    cdef const float *p_d
    cdef const float *p_cb
    cdef const float *p_c
    cdef const float *p_bl
    cdef const float *p_cv
    cdef float *o
    with nogil:
        for y in range(h):
            p_s, p_a, p_ab = &shape[y, 0, 0], &alpha[y, 0, 0], &alpha_b[y, 0, 0]
            p_ap, p_d = &alpha_previous[y, 0, 0], &canvas_alpha[y, 0, 0]
            p_cb, p_c, p_bl = &color_b[y, 0, 0], &color[y, 0, 0], &blended[y, 0, 0]
            p_cv, o = &canvas[y, 0, 0], &out[y, 0, 0]
            for x in range(w):
                s = p_s[x]
                a = p_a[x]
                ab = p_ab[x]
                d = p_d[x]
                t = (s - a) * ab
                m = <float>1.0 - ab
                k = (<float>1.0 - s) * p_ap[x]
                if n == 3:
                    # Spelled out, the common case runs without a channel loop.
                    i = 3 * x
                    q = k * p_cv[i] + (t * p_cb[i] + a * (m * p_c[i] + ab * p_bl[i]))
                    i = i + 1
                    q1 = k * p_cv[i] + (t * p_cb[i] + a * (m * p_c[i] + ab * p_bl[i]))
                    i = i + 1
                    q2 = k * p_cv[i] + (t * p_cb[i] + a * (m * p_c[i] + ab * p_bl[i]))
                    if d == 1:
                        o[i - 2] = _clip(q)
                        o[i - 1] = _clip(q1)
                        o[i] = _clip(q2)
                        continue
                    o[i - 2] = _clip(q / d if d > 0 else <float>1.0)
                    o[i - 1] = _clip(q1 / d if d > 0 else <float>1.0)
                    o[i] = _clip(q2 / d if d > 0 else <float>1.0)
                    continue
                for c in range(n):
                    i = x * n + c
                    q = k * p_cv[i] + (t * p_cb[i] + a * (m * p_c[i] + ab * p_bl[i]))
                    o[i] = _clip(q / d if d > 0 else <float>1.0)
    return result


def passthrough(
    const float[:, :, :] mask,
    double mask_scalar,
    bint scalar,
    const float[:, :, :] alpha,
    const float[:, :, :] alpha_b,
    const float[:, :, :] color_b,
    const float[:, :, :] color,
    const float[:, :, :] canvas_alpha,
):
    """``Compositor._apply_passthrough_source``'s colour canvas.

    ``mask`` is read only when ``scalar`` is false; a scalar mask is a Python
    float, which numpy rounds to float32 after ``1.0 - mask``, not before.
    """
    cdef Py_ssize_t h = color.shape[0], w = color.shape[1], n = color.shape[2]
    cdef Py_ssize_t y, x, c, i
    cdef float mk, mk1, ab, a, d, p, r, q
    cdef float fixed = <float>mask_scalar
    cdef float fixed1 = <float>(1.0 - mask_scalar)
    _rows(mask, h, w, 1)
    _rows(alpha, h, w, 1)
    _rows(alpha_b, h, w, 1)
    _rows(canvas_alpha, h, w, 1)
    _rows(color_b, h, w, n)
    _rows(color, h, w, n)
    result = np.empty((h, w, n), dtype=np.float32)
    if h == 0 or w == 0 or n == 0:
        return result
    cdef float[:, :, ::1] out = result
    cdef const float *p_mk
    cdef const float *p_a
    cdef const float *p_ab
    cdef const float *p_d
    cdef const float *p_cb
    cdef const float *p_c
    cdef float *o
    with nogil:
        for y in range(h):
            p_mk, p_a, p_ab = &mask[y, 0, 0], &alpha[y, 0, 0], &alpha_b[y, 0, 0]
            p_d, p_cb, p_c = &canvas_alpha[y, 0, 0], &color_b[y, 0, 0], &color[y, 0, 0]
            o = &out[y, 0, 0]
            for x in range(w):
                if scalar:
                    mk = fixed
                    mk1 = fixed1
                else:
                    mk = p_mk[x]
                    mk1 = <float>1.0 - mk
                ab = p_ab[x]
                a = p_a[x]
                d = p_d[x]
                p = mk1 * ab
                r = mk * ab + a * (<float>1.0 - ab)
                if d == 1:
                    for c in range(n):
                        i = x * n + c
                        o[i] = _clip(p * p_cb[i] + r * p_c[i])
                    continue
                for c in range(n):
                    i = x * n + c
                    q = p * p_cb[i] + r * p_c[i]
                    o[i] = _clip(q / d if d > 0 else p_c[i])
    return result


cdef inline void _unmix3(
    const float* p_c, const float* p_bc, const float* p_ba, const float* p_a, float* o, Py_ssize_t w,
) noexcept nogil:
    # Three channels spelled out, so the pixel loop vectorizes.
    cdef Py_ssize_t x, i
    cdef float ba, a, r, q
    for x in range(w):
        ba = p_ba[x]
        a = p_a[x]
        r = (ba / a if a > 0 else <float>1.0) - ba
        i = 3 * x
        q = p_c[i]
        o[i] = _clip(q + (q - p_bc[i]) * r)
        q = p_c[i + 1]
        o[i + 1] = _clip(q + (q - p_bc[i + 1]) * r)
        q = p_c[i + 2]
        o[i + 2] = _clip(q + (q - p_bc[i + 2]) * r)


def unmix(
    const float[:, :, :] color,
    const float[:, :, :] backdrop_color,
    const float[:, :, :] backdrop_alpha,
    const float[:, :, :] alpha,
):
    """``composite._unmix``."""
    cdef Py_ssize_t h = color.shape[0], w = color.shape[1], n = color.shape[2]
    cdef Py_ssize_t y, x, c, i
    cdef float ba, a, r, q
    _rows(backdrop_alpha, h, w, 1)
    _rows(alpha, h, w, 1)
    _rows(color, h, w, n)
    _rows(backdrop_color, h, w, n)
    result = np.empty((h, w, n), dtype=np.float32)
    if h == 0 or w == 0 or n == 0:
        return result
    cdef float[:, :, ::1] out = result
    cdef const float *p_c
    cdef const float *p_bc
    cdef const float *p_ba
    cdef const float *p_a
    cdef float *o
    with nogil:
        for y in range(h):
            p_c, p_bc = &color[y, 0, 0], &backdrop_color[y, 0, 0]
            p_ba, p_a, o = &backdrop_alpha[y, 0, 0], &alpha[y, 0, 0], &out[y, 0, 0]
            if n == 3:
                _unmix3(p_c, p_bc, p_ba, p_a, o, w)
                continue
            for x in range(w):
                ba = p_ba[x]
                a = p_a[x]
                r = (ba / a if a > 0 else <float>1.0) - ba
                for c in range(n):
                    i = x * n + c
                    q = p_c[i]
                    o[i] = _clip(q + (q - p_bc[i]) * r)
    return result


def remix(
    const float[:, :, :] content,
    const float[:, :, :] backdrop_color,
    const float[:, :, :] backdrop_alpha,
    const float[:, :, :] alpha,
    const float[:, :, :] canvas_alpha,
):
    """``composite._remix``."""
    cdef Py_ssize_t h = content.shape[0], w = content.shape[1], n = content.shape[2]
    cdef Py_ssize_t y, x, c, i
    cdef float a, k, d, q
    _rows(backdrop_alpha, h, w, 1)
    _rows(alpha, h, w, 1)
    _rows(canvas_alpha, h, w, 1)
    _rows(content, h, w, n)
    _rows(backdrop_color, h, w, n)
    result = np.empty((h, w, n), dtype=np.float32)
    if h == 0 or w == 0 or n == 0:
        return result
    cdef float[:, :, ::1] out = result
    cdef const float *p_ct
    cdef const float *p_bc
    cdef const float *p_ba
    cdef const float *p_a
    cdef const float *p_d
    cdef float *o
    with nogil:
        for y in range(h):
            p_ct, p_bc, p_ba = &content[y, 0, 0], &backdrop_color[y, 0, 0], &backdrop_alpha[y, 0, 0]
            p_a, p_d, o = &alpha[y, 0, 0], &canvas_alpha[y, 0, 0], &out[y, 0, 0]
            for x in range(w):
                a = p_a[x]
                k = (<float>1.0 - a) * p_ba[x]
                d = p_d[x]
                if d == 1:
                    for c in range(n):
                        i = x * n + c
                        o[i] = _clip(a * p_ct[i] + k * p_bc[i])
                    continue
                for c in range(n):
                    i = x * n + c
                    q = a * p_ct[i] + k * p_bc[i]
                    o[i] = _clip(q / d if d > 0 else p_ct[i])
    return result


def lut_levels(const float[:, :, :] values, const float[::1] lut):
    """``adjustments._apply_lut`` for a float32 ``lut`` of at most 65536 levels."""
    cdef Py_ssize_t h = values.shape[0], w = values.shape[1], n = values.shape[2]
    cdef Py_ssize_t y, x, c
    cdef float top = <float>(lut.shape[0] - 1), t
    if lut.shape[0] == 0:
        raise ValueError("empty lut")
    result = np.empty((h, w, n), dtype=np.float32)
    if h == 0 or w == 0 or n == 0:
        return result
    cdef float[:, :, ::1] out = result
    with nogil:
        for y in range(h):
            for x in range(w):
                for c in range(n):
                    t = values[y, x, c]
                    if t != t:
                        # NaN has no level: np.nan, as numpy writes it.
                        out[y, x, c] = NAN
                        continue
                    t = floorf(t * top + <float>0.5)
                    out[y, x, c] = lut[0 if t < 0 else (<Py_ssize_t>top if t > top else <Py_ssize_t>t)]
    return result


def mix_toward(
    const float[:, :, :] content, const float[:, :, :] blended, strength, float scalar, bint levels=False
):
    """``utils.clip(content + strength * (blended - content))``, ``strength`` a canvas of one
    channel, or None for ``scalar``; with ``levels``, in whole 8-bit levels where the strength
    is above 0."""
    cdef Py_ssize_t h = content.shape[0], w = content.shape[1], n = content.shape[2]
    cdef Py_ssize_t y, x, c, i
    cdef float k
    cdef const float[:, :, :] s
    cdef const float* p_s
    cdef const float* p_c
    cdef const float* p_b
    cdef float* o
    _rows(content, h, w, n)
    _rows(blended, h, w, n)
    if strength is not None:
        s = strength
        _rows(s, h, w, 1)
    result = np.empty((h, w, n), dtype=np.float32)
    if h == 0 or w == 0 or n == 0:
        return result
    cdef float[:, :, ::1] out = result
    with nogil:
        for y in range(h):
            p_c, p_b, o = &content[y, 0, 0], &blended[y, 0, 0], &out[y, 0, 0]
            if strength is None:
                for i in range(w * n):
                    o[i] = _clip(p_c[i] + scalar * (p_b[i] - p_c[i]))
                if levels and scalar > 0:
                    _levels_row(o, NULL, w, n)
                continue
            p_s = &s[y, 0, 0]
            if n == 3:
                for x in range(w):
                    k = p_s[x]
                    i = 3 * x
                    o[i] = _clip(p_c[i] + k * (p_b[i] - p_c[i]))
                    o[i + 1] = _clip(p_c[i + 1] + k * (p_b[i + 1] - p_c[i + 1]))
                    o[i + 2] = _clip(p_c[i + 2] + k * (p_b[i + 2] - p_c[i + 2]))
            else:
                for x in range(w):
                    k = p_s[x]
                    for c in range(n):
                        i = x * n + c
                        o[i] = _clip(p_c[i] + k * (p_b[i] - p_c[i]))
            if levels:
                _levels_row(o, p_s, w, n)
    return result


def whole_levels(const float[:, :, :] values, strength=None):
    """``composite._whole_levels``: ``np.rint(np.clip(values, 0.0, 1.0) * 255) / 255``, only
    where ``strength``, a canvas of one channel, is above 0 if given."""
    cdef Py_ssize_t h = values.shape[0], w = values.shape[1], n = values.shape[2]
    cdef Py_ssize_t y, x, c, i
    cdef const float[:, :, :] s
    cdef const float *p
    cdef const float *p_s
    cdef float *o
    _rows(values, h, w, n)
    if strength is not None:
        s = strength
        _rows(s, h, w, 1)
    result = np.empty((h, w, n), dtype=np.float32)
    if h == 0 or w == 0 or n == 0:
        return result
    cdef float[:, :, ::1] out = result
    with nogil:
        for y in range(h):
            p, o = &values[y, 0, 0], &out[y, 0, 0]
            if strength is None:
                for i in range(w * n):
                    o[i] = _level(_clip(p[i]))
                continue
            p_s = &s[y, 0, 0]
            for x in range(w):
                for c in range(n):
                    i = x * n + c
                    o[i] = _level(_clip(p[i])) if p_s[x] > 0 else p[i]
    return result


def gather_rows(const float[:, ::1] table, const int[::1] index):
    """``table[index]``; None if an index is out of range."""
    cdef Py_ssize_t n = index.shape[0], m = table.shape[0], c = table.shape[1], p, k
    cdef int d
    cdef bint bad = False
    result = np.empty((n, c), dtype=np.float32)
    if n == 0 or c == 0:
        return result
    cdef float[:, ::1] out = result
    with nogil:
        for p in range(n):
            d = index[p]
            if d < 0 or d >= m:
                bad = True
                break
            for k in range(c):
                out[p, k] = table[d, k]
    return None if bad else result


cdef inline unsigned char _byte(float v) noexcept nogil:
    # 255 * v, np.clip to [0, 255], np.rint, astype(uint8). fmaxf takes NaN
    # to 0, which is what the cast makes of it.
    return <unsigned char>rintf(fminf(fmaxf(<float>255.0 * v, <float>0.0), <float>255.0))


def to_bytes(const float[:, :, :] color, const float[:, :, :] alpha=None):
    """``composite_pil``'s float32 canvas as bytes, ``alpha`` (one plane) appended last."""
    cdef Py_ssize_t h = color.shape[0], w = color.shape[1], n = color.shape[2]
    cdef Py_ssize_t m = n + (1 if alpha is not None else 0), y, x, c
    cdef bint has_alpha = alpha is not None
    cdef const float* row
    cdef const float* arow
    cdef unsigned char* orow
    if has_alpha and (alpha.shape[0] != h or alpha.shape[1] != w or alpha.shape[2] != 1):
        raise ValueError("alpha is not one plane of the canvas")
    result = np.empty((h, w, m), dtype=np.uint8)
    if h == 0 or w == 0 or m == 0:
        return result
    cdef unsigned char[:, :, ::1] out = result
    cdef bint packed = color.strides[2] == sizeof(float) and color.strides[1] == n * sizeof(float) and (
        not has_alpha or alpha.strides[1] == sizeof(float))
    with nogil:
        for y in range(h):
            orow = &out[y, 0, 0]
            if packed and n == 3 and has_alpha:
                row = &color[y, 0, 0]
                arow = &alpha[y, 0, 0]
                for x in range(w):
                    orow[4 * x] = _byte(row[3 * x])
                    orow[4 * x + 1] = _byte(row[3 * x + 1])
                    orow[4 * x + 2] = _byte(row[3 * x + 2])
                    orow[4 * x + 3] = _byte(arow[x])
                continue
            for x in range(w):
                for c in range(n):
                    orow[x * m + c] = _byte(color[y, x, c])
                if has_alpha:
                    orow[x * m + n] = _byte(alpha[y, x, 0])
    return result


def divide(
    const float[:, :, :] a,
    const float[:, :, :] b,
    const float[:, :, :] fill,
    float fill_scalar,
    bint scalar,
):
    """``utils.divide``: ``a / b`` where ``b > 0``, else ``fill``; ``b`` one channel.

    ``fill`` is ``a``'s size and read only when ``scalar`` is false.
    """
    cdef Py_ssize_t h = a.shape[0], w = a.shape[1], n = a.shape[2]
    cdef Py_ssize_t y, x, c, i
    cdef float d, f
    _rows(a, h, w, n)
    _rows(b, h, w, 1)
    if not scalar:
        _rows(fill, h, w, n)
    result = np.empty((h, w, n), dtype=np.float32)
    if h == 0 or w == 0 or n == 0:
        return result
    cdef float[:, :, ::1] out = result
    cdef const float *p_a
    cdef const float *p_b
    cdef const float *p_f = NULL
    cdef float *o
    with nogil:
        for y in range(h):
            p_a, p_b, o = &a[y, 0, 0], &b[y, 0, 0], &out[y, 0, 0]
            if not scalar:
                p_f = &fill[y, 0, 0]
            # Both sides computed and one picked, so the loop has no branch.
            if n == 1 and scalar:
                for x in range(w):
                    d = p_b[x]
                    f = p_a[x] / d
                    o[x] = f if d > 0 else fill_scalar
                continue
            if n == 1:
                for x in range(w):
                    d = p_b[x]
                    f = p_a[x] / d
                    o[x] = f if d > 0 else p_f[x]
                continue
            if scalar:
                for x in range(w):
                    d = p_b[x]
                    for c in range(n):
                        i = x * n + c
                        f = p_a[i] / d
                        o[i] = f if d > 0 else fill_scalar
                continue
            for x in range(w):
                d = p_b[x]
                for c in range(n):
                    i = x * n + c
                    f = p_a[i] / d
                    o[i] = f if d > 0 else p_f[i]
    return result


def clip(const float[:, :, :] values):
    """``utils.clip``: ``np.clip(values, 0.0, 1.0)``."""
    cdef Py_ssize_t h = values.shape[0], w = values.shape[1], n = values.shape[2]
    cdef Py_ssize_t y, i
    _rows(values, h, w, n)
    result = np.empty((h, w, n), dtype=np.float32)
    if h == 0 or w == 0 or n == 0:
        return result
    cdef float[:, :, ::1] out = result
    cdef const float *p
    cdef float *o
    with nogil:
        for y in range(h):
            p, o = &values[y, 0, 0], &out[y, 0, 0]
            for i in range(w * n):
                o[i] = _clip(p[i])
    return result


def effect_start(
    const float[:, :, :] source_alpha,
    const float[:, :, :] region,
    const float[:, :, :] color,
    float factor,
):
    """``_EffectCanvas._start``: its alpha and premultiplied colour."""
    cdef Py_ssize_t h = color.shape[0], w = color.shape[1], n = color.shape[2]
    cdef Py_ssize_t y, x, c, i
    cdef float r, q
    _rows(source_alpha, h, w, 1)
    _rows(region, h, w, 1)
    _rows(color, h, w, n)
    out_alpha = np.empty((h, w, 1), dtype=np.float32)
    out_premultiplied = np.empty((h, w, n), dtype=np.float32)
    if h == 0 or w == 0 or n == 0:
        return out_alpha, out_premultiplied
    cdef float[:, :, ::1] al = out_alpha
    cdef float[:, :, ::1] pm = out_premultiplied
    cdef const float *p_sa
    cdef const float *p_r
    cdef const float *p_c
    cdef float *o_al
    cdef float *o_pm
    with nogil:
        for y in range(h):
            p_sa, p_r, p_c = &source_alpha[y, 0, 0], &region[y, 0, 0], &color[y, 0, 0]
            o_al, o_pm = &al[y, 0, 0], &pm[y, 0, 0]
            for x in range(w):
                r = p_r[x]
                q = p_sa[x] / r
                q = (q if r > 0 else <float>0.0) * factor
                o_al[x] = q
                for c in range(n):
                    i = x * n + c
                    o_pm[i] = q * p_c[i]
    return out_alpha, out_premultiplied


def effect_over(
    const float[:, :, :] alpha,
    const float[:, :, :] premultiplied,
    const float[:, :, :] shape,
    float shape_scalar,
    bint scalar,
    const float[:, :, :] coverage,
    const float[:, :, :] color,
    float opacity,
):
    """``_EffectCanvas.over`` in Normal: the new alpha, premultiplied and shape.

    ``shape`` is read only when ``scalar`` is false. The blend's ``color`` is
    kept as numpy spells it, ``(1 - A) * c + A * c``.
    """
    cdef Py_ssize_t h = color.shape[0], w = color.shape[1], n = color.shape[2]
    cdef Py_ssize_t y, x, c, i
    cdef float k, a, s, e, e1, a1, q
    _rows(alpha, h, w, 1)
    _rows(coverage, h, w, 1)
    if not scalar:
        _rows(shape, h, w, 1)
    _rows(premultiplied, h, w, n)
    _rows(color, h, w, n)
    out_alpha = np.empty((h, w, 1), dtype=np.float32)
    out_premultiplied = np.empty((h, w, n), dtype=np.float32)
    out_shape = np.empty((h, w, 1), dtype=np.float32)
    if h == 0 or w == 0 or n == 0:
        return out_alpha, out_premultiplied, out_shape
    cdef float[:, :, ::1] al = out_alpha
    cdef float[:, :, ::1] pm = out_premultiplied
    cdef float[:, :, ::1] sh = out_shape
    cdef const float *p_a
    cdef const float *p_pm
    cdef const float *p_s = NULL
    cdef const float *p_k
    cdef const float *p_c
    cdef float *o_al
    cdef float *o_pm
    cdef float *o_sh
    with nogil:
        for y in range(h):
            p_a, p_pm, p_k = &alpha[y, 0, 0], &premultiplied[y, 0, 0], &coverage[y, 0, 0]
            p_c, o_al, o_pm, o_sh = &color[y, 0, 0], &al[y, 0, 0], &pm[y, 0, 0], &sh[y, 0, 0]
            if not scalar:
                p_s = &shape[y, 0, 0]
            for x in range(w):
                k = p_k[x]
                e = k * opacity
                e1 = <float>1.0 - e
                a = p_a[x]
                a1 = <float>1.0 - a
                for c in range(n):
                    i = x * n + c
                    q = p_c[i]
                    o_pm[i] = e1 * p_pm[i] + e * (a1 * q + a * q)
                o_al[x] = e + e1 * a
                s = shape_scalar if scalar else p_s[x]
                o_sh[x] = (s + k) - (s * k)
    return out_alpha, out_premultiplied, out_shape


def effect_source(
    const float[:, :, :] region,
    const float[:, :, :] alpha,
    const float[:, :, :] premultiplied,
    const float[:, :, :] shape,
    float shape_scalar,
    bint scalar,
    const float[:, :, :] color,
    float opacity,
):
    """``_EffectCanvas.source``: the colour, shape and alpha it hands on.

    ``shape`` is read only when ``scalar`` is false.
    """
    cdef Py_ssize_t h = color.shape[0], w = color.shape[1], n = color.shape[2]
    cdef Py_ssize_t y, x, c, i
    cdef float r, a, q, f
    _rows(region, h, w, 1)
    _rows(alpha, h, w, 1)
    if not scalar:
        _rows(shape, h, w, 1)
    _rows(premultiplied, h, w, n)
    _rows(color, h, w, n)
    out_color = np.empty((h, w, n), dtype=np.float32)
    out_shape = np.empty((h, w, 1), dtype=np.float32)
    out_alpha = np.empty((h, w, 1), dtype=np.float32)
    if h == 0 or w == 0 or n == 0:
        return out_color, out_shape, out_alpha
    cdef float[:, :, ::1] co = out_color
    cdef float[:, :, ::1] sh = out_shape
    cdef float[:, :, ::1] al = out_alpha
    cdef const float *p_r
    cdef const float *p_a
    cdef const float *p_pm
    cdef const float *p_s = NULL
    cdef const float *p_c
    cdef float *o_co
    cdef float *o_sh
    cdef float *o_al
    with nogil:
        for y in range(h):
            p_r, p_a, p_pm = &region[y, 0, 0], &alpha[y, 0, 0], &premultiplied[y, 0, 0]
            p_c, o_co, o_sh, o_al = &color[y, 0, 0], &co[y, 0, 0], &sh[y, 0, 0], &al[y, 0, 0]
            if not scalar:
                p_s = &shape[y, 0, 0]
            for x in range(w):
                r = p_r[x]
                a = (r * p_a[x]) * opacity
                for c in range(n):
                    i = x * n + c
                    q = (r * p_pm[i]) * opacity
                    f = q / a
                    o_co[i] = _clip(f if a > 0 else p_c[i])
                o_sh[x] = _clip(r * (shape_scalar if scalar else p_s[x]))
                o_al[x] = _clip(a)
    return out_color, out_shape, out_alpha


def unfilter(
    const float[:, :, :] color,
    const float[:, :, :] shape,
    const float[:, :, :] weight,
    const unsigned char[:, :, ::1] src_color,
    const unsigned char[:, ::1] src_alpha,
    Py_ssize_t oy,
    Py_ssize_t ox,
):
    """``composite._blend_unfiltered``: ``(out, alpha)``.

    The source's top left is at row ``oy``, column ``ox`` of the box; ``shape``
    and ``src_alpha`` may be None. ``weight`` is only read.
    """
    cdef Py_ssize_t h = color.shape[0], w = color.shape[1], n = color.shape[2]
    cdef Py_ssize_t sh = src_color.shape[0], sw = src_color.shape[1]
    cdef Py_ssize_t y, x, c, i, s, x0, x1
    cdef bint has_shape = shape is not None, has_alpha = src_alpha is not None
    cdef bint row_in, inside
    cdef float u, wt, a, q
    _rows(weight, h, w, 1)
    if has_shape:
        _rows(shape, h, w, 1)
    _rows(color, h, w, n)
    if src_color.shape[2] != n or has_alpha and (
        src_alpha.shape[0] != sh or src_alpha.shape[1] != sw
    ):
        raise ValueError("source planes do not match")
    out_color = np.empty((h, w, n), dtype=np.float32)
    out_alpha = np.empty((h, w, 1), dtype=np.float32)
    if h == 0 or w == 0 or n == 0:
        return out_color, out_alpha
    cdef float[:, :, ::1] co = out_color
    cdef float[:, :, ::1] al = out_alpha
    cdef const float *p_c
    cdef const float *p_w
    cdef const float *p_s = NULL
    cdef const unsigned char *p_sc = NULL
    cdef const unsigned char *p_sa = NULL
    cdef float *o_co
    cdef float *o_al
    # The source's columns in the box.
    x0 = min(max(ox, 0), w)
    x1 = max(min(ox + sw, w), x0)
    with nogil:
        for y in range(h):
            p_c, p_w, o_co, o_al = &color[y, 0, 0], &weight[y, 0, 0], &co[y, 0, 0], &al[y, 0, 0]
            if has_shape:
                p_s = &shape[y, 0, 0]
            row_in = 0 <= y - oy < sh and x0 < x1
            if row_in:
                p_sc = &src_color[y - oy, 0, 0]
                if has_alpha:
                    p_sa = &src_alpha[y - oy, 0]
            # numpy's steps per pixel, in its order and in float32.
            for x in range(w):
                inside = row_in and x0 <= x < x1
                s = x - ox
                u = (<float>1.0 if inside else <float>0.0) * (<float>1.0 - p_w[x])
                wt = <float>1.0 - u
                if has_alpha:
                    u = u * ((<float>p_sa[s] if inside else <float>0.0) / <float>255.0)
                if has_shape:
                    wt = wt * p_s[x]
                a = wt + u
                o_al[x] = a
                u = u / <float>255.0
                for c in range(n):
                    i = x * n + c
                    q = (<float>p_sc[s * n + c] if inside else <float>0.0) * u
                    q = q + wt * p_c[i]
                    if a > 0:
                        q = q / a
                    elif a <= 0:
                        q = p_c[i]
                    o_co[i] = q
    return out_color, out_alpha


ctypedef fused real:
    float
    double


cdef inline Py_ssize_t _right(const double* nodes, Py_ssize_t n, double v) noexcept nogil:
    # np.searchsorted(nodes, v, side="right"); NaN goes past the end.
    cdef Py_ssize_t lo = 0, hi = n, mid
    while lo < hi:
        mid = (lo + hi) >> 1
        if v < nodes[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


cdef inline double _node(double d) noexcept nogil:
    # np.maximum(d, 0): NaN stays, -0.0 becomes 0.0.
    return d if d > 0 or d != d else 0.0


cdef inline float _bil32(
    const float* t, Py_ssize_t ns, Py_ssize_t r, Py_ssize_t k, float dn, float ac
) noexcept nogil:
    cdef float oac = <float>1.0 - ac
    cdef float top = t[r * ns + k] * oac + t[r * ns + k + 1] * ac
    cdef float bottom = t[(r + 1) * ns + k] * oac + t[(r + 1) * ns + k + 1] * ac
    return top * (<float>1.0 - dn) + bottom * dn


cdef inline double _bil64(
    const double* t, Py_ssize_t ns, Py_ssize_t r, Py_ssize_t k, float dn, float ac
) noexcept nogil:
    cdef double oac = <double>(<float>1.0 - ac)
    cdef double top = t[r * ns + k] * oac + t[r * ns + k + 1] * <double>ac
    cdef double bottom = t[(r + 1) * ns + k] * oac + t[(r + 1) * ns + k + 1] * <double>ac
    return top * <double>(<float>1.0 - dn) + bottom * <double>dn


cdef inline real _max(real a, real b) noexcept nogil:
    # numpy's max reduction: NaN wins, max(-0.0, 0.0) is 0.0.
    if a != a:
        return a
    if b != b:
        return b
    if a == b:
        return b if signbit(a) else a
    return a if a > b else b


cdef inline real _min(real a, real b) noexcept nogil:
    # numpy's min reduction: NaN wins, min(-0.0, 0.0) is -0.0.
    if a != a:
        return a
    if b != b:
        return b
    if a == b:
        return a if signbit(a) else b
    return a if a < b else b


def vibrance_step(
    const real[:, :, ::1] linear,
    double amount,
    const double[:, ::1] at_amount,
    const float[:, ::1] value_table,
    const float[:, ::1] skin_amount,
    const float[:, :, ::1] chroma,
    const float[:, ::1] skin_rise,
    const double[::1] values,
    const double[::1] sats,
    double slope,
    double cutoff,
    double gain,
    double inv_gamma,
    double offset,
):
    """``adjustments._vibrance_step`` past its amount's tables, float64 out.

    ``at_amount`` is the chroma table at ``amount``, ``skin_amount`` and
    ``skin_rise`` the spared reds' tables there (unread when lowering), and
    ``chroma`` the four chroma tables; each pixel's own amount is read off
    those as numpy does.
    """
    cdef Py_ssize_t h = linear.shape[0], w = linear.shape[1], n = h * w
    cdef Py_ssize_t nv = values.shape[0], ns = sats.shape[0], tsz = nv * ns
    cdef Py_ssize_t i, r, k, first
    if linear.shape[2] != 3:
        raise ValueError("not three channels")
    if (
        nv < 2
        or ns < 2
        or chroma.shape[0] != 4
        or chroma.shape[1] != nv
        or chroma.shape[2] != ns
        or at_amount.shape[0] != nv
        or at_amount.shape[1] != ns
        or value_table.shape[0] != nv
        or value_table.shape[1] != ns
        or skin_amount.shape[0] != nv
        or skin_amount.shape[1] != ns
        or skin_rise.shape[0] != nv
        or skin_rise.shape[1] != ns
    ):
        raise ValueError("vibrance tables do not match their nodes")
    out = np.empty((h, w, 3), dtype=np.float64)
    if n == 0:
        return out
    cdef double[:, :, ::1] o = out
    cdef double* op = &o[0, 0, 0]
    cdef const real* lp = &linear[0, 0, 0]
    cdef const double* vp = &values[0]
    cdef const double* sp = &sats[0]
    cdef const double* atp = &at_amount[0, 0]
    cdef const float* vtp = &value_table[0, 0]
    cdef const float* skp = &skin_amount[0, 0]
    cdef const float* chp = &chroma[0, 0, 0]
    cdef const float* srp = &skin_rise[0, 0]
    cdef bint raising = amount > 0
    cdef double size = fabs(amount), scale, u, w0, w1, w2, total
    cdef float sizef = <float>size, dn, ac, share, pos, firstf, own0, own1, own2
    cdef real red, green, blue, high, low, span, sat, value, v255, rise, hue, spared, effective
    cdef real rslope = <real>slope, rcut = <real>cutoff, rgain = <real>gain
    cdef real rinv = <real>inv_gamma, roff = <real>offset, r255 = <real>(<float>255.0)
    with nogil:
        for i in range(n):
            red = lp[3 * i]
            green = lp[3 * i + 1]
            blue = lp[3 * i + 2]
            high = _max(_max(red, green), blue)
            low = _min(_min(red, green), blue)
            span = high - low
            sat = span / high if high > 0 else <real>0.0
            if high > rcut:
                if real is float:
                    value = rgain * powf(high, rinv) - roff
                else:
                    value = rgain * pow(high, rinv) - roff
            else:
                value = rslope * high
            v255 = value * r255
            # _vibrance_node
            r = _right(vp, nv, <double>v255) - 1
            r = 0 if r < 0 else (nv - 2 if r > nv - 2 else r)
            k = _right(sp, ns, <double>sat) - 1
            k = 0 if k < 0 else (ns - 2 if k > ns - 2 else k)
            dn = <float>_node((<double>v255 - vp[r]) / (vp[r + 1] - vp[r]))
            ac = <float>_node((<double>sat - sp[k]) / (sp[k + 1] - sp[k]))
            scale = <double>(<float>1.0) + _bil64(atp, ns, r, k, dn, ac)
            if real is float:
                rise = (high * sizef) * _bil32(vtp, ns, r, k, dn, ac)
            else:
                rise = (high * <double>sizef) * <double>_bil32(vtp, ns, r, k, dn, ac)
            if raising and red >= green and red >= blue and span > 0:
                # _vibrance_spared
                hue = ((((red + green) + blue) - high) - low - low) / (high - low)
                if green >= blue:
                    spared = <real>3.0 - <real>4.0 * hue
                    spared = <real>0.0 if spared < 0 else (<real>1.0 if spared > 1 else spared)
                else:
                    spared = <real>1.0 - hue
                share = _bil32(skp, ns, r, k, dn, ac)
                effective = <real>size * (<real>1.0 - spared * <real>(<float>1.0 - share))
                # _vibrance_amount_weights; a NaN index casts to 0 on arm64
                pos = (<float>effective) * <float>4.0
                firstf = ceilf(pos) - <float>1.0
                if firstf != firstf:
                    first = 0
                else:
                    first = 0 if firstf < 0 else (2 if firstf > 2 else <Py_ssize_t>firstf)
                u = <double>pos - <double>first
                w0 = (u - 1.0) * (u - 2.0) / 2.0
                w1 = -u * (u - 2.0)
                w2 = u * (u - 1.0) / 2.0
                own0 = <float>0.0 if first == 0 else _bil32(chp + (first - 1) * tsz, ns, r, k, dn, ac)
                own1 = _bil32(chp + first * tsz, ns, r, k, dn, ac)
                own2 = _bil32(chp + (first + 1) * tsz, ns, r, k, dn, ac)
                # Python's sum(): 0 + t0 + t1 + t2
                total = 0.0 + w0 * <double>own0
                total = total + w1 * <double>own1
                total = total + w2 * <double>own2
                scale = <double>(<float>1.0) + total
                share = _bil32(srp, ns, r, k, dn, ac)
                rise = rise * (<real>1.0 - spared * <real>(<float>1.0 - share))
            value = high + rise
            op[3 * i] = <double>value - scale * <double>(high - red)
            op[3 * i + 1] = <double>value - scale * <double>(high - green)
            op[3 * i + 2] = <double>value - scale * <double>(high - blue)
    return out


ctypedef struct _TextSpace:
    double knee, slope, cutoff, off, off1, sg, inv_sg, g, inv_g


cdef inline double _max0(double x) noexcept nogil:
    # np.maximum(x, 0.0): NaN passes, -0.0 becomes +0.0.
    return x if (x > 0 or x != x) else 0.0


cdef inline double _to_text(double x, const _TextSpace* t) noexcept nogil:
    """``text_blend.to_text_space``: the branch its np.where keeps."""
    x = _max0(x)
    if x <= t.knee:
        x = x / t.slope
    else:
        x = pow((x + t.off) / t.off1, t.sg)
    return pow(x, t.inv_g)


cdef inline double _from_text(double v, const _TextSpace* t) noexcept nogil:
    """``text_blend.from_text_space``."""
    v = pow(_max0(v), t.g)
    v = 0.0 if v < 0 else (1.0 if v > 1 else v)
    if v <= t.cutoff:
        return v * t.slope
    return t.off1 * pow(v, t.inv_sg) - t.off


def text_mix(
    float[:, :, ::1] result,
    const float[:, :, :] canvas,
    const float[:, :, :] canvas_alpha,
    const float[:, :, :] alpha_g,
    const float[:, :, :] color_0,
    const float[:, :, :] alpha_0,
    const float[:, :, :] color,
    const float[:, :, :] shape,
    const float[:, :, :] alpha,
    const float[:, :, :] alpha_previous,
    const float[:, :, :] alpha_g_previous,
    const float[:, :, :] alpha_b,
    bint isolated,
    Py_ssize_t top,
    Py_ssize_t bottom,
    Py_ssize_t left,
    Py_ssize_t right,
    double gamma,
    double slope,
    double cutoff,
    double offset,
    double srgb_gamma,
):
    """``Compositor._mix_in_text_space`` in Normal over the canvas, no ``inside``.

    Writes ``result``, a copy of ``canvas``, within the box. Three channels.
    """
    cdef Py_ssize_t h = canvas.shape[0], w = canvas.shape[1], y, x, c
    cdef float s, a, ap, now, agp, ag, share1, ag1, s1, sap, ssh
    cdef double ab, agpd, a0, col, prev, blended, source, prior, mixed, q, num
    cdef _TextSpace t
    # Each transfer's last input and answer per channel: a glyph's colour and a
    # flat backdrop repeat. Equal inputs (±0 too) give equal answers.
    cdef double kb[3]
    cdef double vb[3]
    cdef double kp[3]
    cdef double vp[3]
    cdef double kq[3]
    cdef double vq[3]
    _rows(canvas, h, w, 3)
    _rows(result, h, w, 3)
    _rows(color, h, w, 3)
    _rows(color_0, h, w, 3)
    _rows(canvas_alpha, h, w, 1)
    _rows(alpha_g, h, w, 1)
    _rows(alpha_0, h, w, 1)
    _rows(shape, h, w, 1)
    _rows(alpha, h, w, 1)
    _rows(alpha_previous, h, w, 1)
    _rows(alpha_g_previous, h, w, 1)
    _rows(alpha_b, h, w, 1)
    if not (0 <= top <= bottom <= h and 0 <= left <= right <= w):
        raise ValueError("box is off the canvas")
    t.knee = slope * cutoff
    t.slope = slope
    t.cutoff = cutoff
    t.off = offset
    t.off1 = 1.0 + offset
    t.sg = srgb_gamma
    t.inv_sg = 1.0 / srgb_gamma
    t.g = gamma
    t.inv_g = 1.0 / gamma
    for c in range(3):
        kb[c] = NAN
        kp[c] = NAN
        kq[c] = NAN
        vb[c] = vp[c] = vq[c] = 0.0
    with nogil:
        for y in range(top, bottom):
            for x in range(left, right):
                s = shape[y, x, 0]
                a = alpha[y, x, 0]
                ap = alpha_previous[y, x, 0]
                now = canvas_alpha[y, x, 0]
                agp = alpha_g_previous[y, x, 0]
                if not (
                    s != 0 or a != 0 or ap != now
                    or isolated and agp != 0 and agp != 1
                ):
                    # Untouched: the canvas stays, but for these two.
                    if isolated and agp == 0:
                        for c in range(3):
                            result[y, x, c] = color_0[y, x, c]
                    if (agp == 0 or not isolated) and now <= 0:
                        for c in range(3):
                            result[y, x, c] = <float>1.0
                    continue
                ab = <double>alpha_b[y, x, 0]
                if isolated:
                    agpd = <double>agp
                    a0 = <double>alpha_0[y, x, 0]
                    ag = alpha_g[y, x, 0]
                    share1 = <float>1.0 - a
                    ag1 = <float>1.0 - ag
                else:
                    s1 = <float>1.0 - s
                    sap = s1 * ap
                    ssh = s - a
                for c in range(3):
                    col = <double>color[y, x, c]
                    prev = <double>canvas[y, x, c]
                    blended = (1.0 - ab) * col + ab * col
                    if blended == kb[c]:
                        source = vb[c]
                    else:
                        source = _to_text(blended, &t)
                        kb[c] = blended
                        vb[c] = source
                    if isolated:
                        num = prev * <double>ap - (1.0 - agpd) * a0 * <double>color_0[y, x, c]
                        prev = num / agpd if agpd > 0 else 0.0
                    if prev == kp[c]:
                        prior = vp[c]
                    else:
                        prior = _to_text(prev, &t)
                        kp[c] = prev
                        vp[c] = prior
                    if isolated:
                        mixed = <double>share1 * agpd * prior
                        mixed = mixed + <double>a * source
                        q = mixed / <double>ag if ag > 0 else 1.0
                    else:
                        mixed = <double>sap * prior + <double>ssh * ab * prior
                        mixed = mixed + <double>a * source
                        q = mixed / <double>now if now > 0 else 1.0
                    if q == kq[c]:
                        mixed = vq[c]
                    else:
                        mixed = _from_text(q, &t)
                        kq[c] = q
                        vq[c] = mixed
                    if isolated:
                        num = <double>ag * mixed + <double>ag1 * a0 * <double>color_0[y, x, c]
                        mixed = num / <double>now if now > 0 else mixed
                    result[y, x, c] = <float>(0.0 if mixed < 0 else (1.0 if mixed > 1 else mixed))


def distinct_pixels(const float[:, ::1] pixels, Py_ssize_t limit):
    """``(index, first)``: each pixel's number among the distinct ones, equal
    bits being one, and each distinct pixel's first row; None past ``limit``.
    """
    cdef Py_ssize_t n = pixels.shape[0], c = pixels.shape[1]
    cdef Py_ssize_t cap = 1, mask, p, k, slot, d, count = 0
    cdef unsigned int hv
    cdef const unsigned int* bits
    cdef const unsigned int* own
    cdef const unsigned int* seen
    cdef bint same
    if limit < 0 or limit > 0x3FFFFFFF:
        raise ValueError("limit out of range")
    while cap < 2 * limit:
        cap <<= 1
    mask = cap - 1
    index_ = np.empty(n, dtype=np.int32)
    first_ = np.empty(min(limit, n), dtype=np.intp)
    if n == 0 or c == 0:
        return index_, first_[:0]
    if limit == 0:
        return None
    table_ = np.full(cap, -1, dtype=np.int32)
    cdef int[::1] table = table_
    cdef int[::1] index = index_
    cdef Py_ssize_t[::1] first = first_
    bits = <const unsigned int*>&pixels[0, 0]
    with nogil:
        for p in range(n):
            own = bits + p * c
            hv = 2166136261u
            for k in range(c):
                hv = (hv ^ own[k]) * 16777619u
                hv = hv ^ (hv >> 15)
            slot = hv & mask
            while True:
                d = table[slot]
                if d < 0:
                    if count == limit:
                        count = -1
                        break
                    table[slot] = <int>count
                    first[count] = p
                    index[p] = <int>count
                    count += 1
                    break
                seen = bits + first[d] * c
                same = True
                for k in range(c):
                    if seen[k] != own[k]:
                        same = False
                        break
                if same:
                    index[p] = <int>d
                    break
                slot = (slot + 1) & mask
            if count < 0:
                break
    if count < 0:
        return None
    return index_, first_[:count]
