# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""C loops behind the compositor's source arithmetic.

Each is the numpy expression it replaces in ``composite.py`` (``lut_levels``:
``adjustments._apply_lut``; ``to_bytes``: ``composite_pil``), its float32
operations in the same order, so the results are bit-identical. Memoryviews
only, so the module builds against the Limited API without numpy headers.

Every canvas but ``lut_levels``' and ``to_bytes``' is read a row at a time, so
a pixel's channels must be adjacent and its neighbour the next pixel along
(``_rows``), as they are in a box cut from a canvas; ``composite._compiled_fits``
checks that first.
"""
from libc.math cimport NAN, floorf, fmaxf, fminf, rintf

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
