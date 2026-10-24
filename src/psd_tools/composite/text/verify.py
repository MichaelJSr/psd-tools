"""Fidelity of a render against a stored raster.

The formulas are the scoring harness's: both rasters are placed premultiplied
on the union of their bboxes and compared pixel by pixel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from psd_tools.api.utils import AllocBudget, check_pixel_size
from psd_tools.composite.text.constants import ANY_ALPHA, BAD_DIFF, IOU_COVER
from psd_tools.composite.text.raster import Bbox
from psd_tools.composite.text.report import Fidelity
from psd_tools.constants import ColorMode

if TYPE_CHECKING:
    from psd_tools.api.layers import TypeLayer

#: A pixel with less alpha than this holds no ink for the ink box.
_INK = 0.5 / 255
#: Colour modes whose stored channels are the RGB or gray values themselves.
_DIRECT_MODES = (ColorMode.RGB, ColorMode.GRAYSCALE, ColorMode.DUOTONE)
#: Bytes per union pixel :func:`fidelity` holds at its peak: the two placed
#: premultiplied float32 rasters, their float32 difference and its maximum.
_FIDELITY_BYTES = 16 + 16 + 16 + 4


def stored_raster(layer: TypeLayer) -> tuple[np.ndarray | None, Bbox | None]:
    """The layer's stored pixels as straight RGBA ``float32`` and its bbox.

    ``(None, None)`` when the layer has no pixels. A grayscale document
    repeats its single channel; CMYK, Lab and the other modes take their
    colour through PIL's RGB conversion, at 8 bits. The alpha is the stored
    transparency channel in every mode, opaque when there is none.
    """
    if not layer.has_pixels():
        return None, None
    if layer._psd.color_mode in _DIRECT_MODES:
        color = layer.numpy("color")
        if color is None:
            return None, None
        if color.shape[2] == 1:
            color = np.repeat(color, 3, axis=2)
    else:
        image = layer.topil(apply_icc=False)
        if image is None:
            return None, None
        color = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    shape = layer.numpy("shape")
    if shape is None:
        shape = np.ones((*color.shape[:2], 1), np.float32)
    rgba = np.concatenate([color, shape], axis=2).astype(np.float32)
    x0, y0, x1, y1 = layer.bbox
    return rgba, (x0, y0, x1, y1)


def _union(a: Bbox | None, b: Bbox | None) -> Bbox:
    boxes = [x for x in (a, b) if x is not None]
    return (
        min(x[0] for x in boxes),
        min(x[1] for x in boxes),
        max(x[2] for x in boxes),
        max(x[3] for x in boxes),
    )


def _place(rgba: np.ndarray | None, bbox: Bbox | None, union: Bbox) -> np.ndarray:
    """Straight RGBA at ``bbox`` -> premultiplied HxWx4 on ``union``."""
    out = np.zeros((union[3] - union[1], union[2] - union[0], 4), np.float32)
    if rgba is None or bbox is None or not rgba.size:
        return out
    alpha = rgba[..., 3:4]
    out[
        bbox[1] - union[1] : bbox[3] - union[1], bbox[0] - union[0] : bbox[2] - union[0]
    ] = np.concatenate([rgba[..., :3] * alpha, alpha], axis=-1)
    return out


def _centroid(alpha: np.ndarray, union: Bbox) -> tuple[float, float] | None:
    mass = float(alpha.sum())
    if mass <= 0:
        return None
    ys, xs = np.indices(alpha.shape)
    return (
        float((xs * alpha).sum() / mass + union[0] + 0.5),
        float((ys * alpha).sum() / mass + union[1] + 0.5),
    )


def _ink_box(alpha: np.ndarray, union: Bbox) -> Bbox | None:
    ys, xs = np.nonzero(alpha > _INK)
    if not len(xs):
        return None
    return (
        int(xs.min()) + union[0],
        int(ys.min()) + union[1],
        int(xs.max()) + 1 + union[0],
        int(ys.max()) + 1 + union[1],
    )


def fidelity(
    stored_rgba: np.ndarray | None,
    stored_bbox: Bbox | None,
    rgba: np.ndarray | None,
    bbox: Bbox | None,
    status: Literal["ok", "empty", "missing"] | None = None,
    *,
    max_alloc_bytes: AllocBudget | None = None,
) -> Fidelity:
    """Compare two straight-alpha RGBA rasters on the union of their bboxes.

    A ``None`` raster is empty. ``status`` overrides the one derived from the
    render (``"empty"`` when it has no ink). The union is checked against
    the PSD per-axis limit and the ``max_alloc_bytes`` budget (the global
    default when ``None``) before it is allocated.
    """
    if status is None:
        status = (
            "ok" if rgba is not None and rgba.size and rgba[..., 3].any() else "empty"
        )
    if stored_bbox is None and bbox is None:
        return Fidelity(1.0, 0.0, 0.0, None, None, None, None, None, None, status)
    union = _union(stored_bbox, bbox)
    w, h = union[2] - union[0], union[3] - union[1]
    if w > 0 and h > 0:
        check_pixel_size(
            w,
            h,
            max_alloc_bytes=max_alloc_bytes,
            estimated_bytes=w * h * _FIDELITY_BYTES,
            warn=False,
        )
    stored = _place(stored_rgba, stored_bbox, union)
    rendered = _place(rgba, bbox, union)
    a_s, a_r = stored[..., 3], rendered[..., 3]
    diff = np.abs(stored - rendered).max(axis=-1)
    either = (a_s > ANY_ALPHA) | (a_r > ANY_ALPHA)
    cover_s, cover_r = a_s > IOU_COVER, a_r > IOU_COVER
    inter, uni = int((cover_s & cover_r).sum()), int((cover_s | cover_r).sum())
    if either.any():
        bad_pct = float(100 * (diff[either] > BAD_DIFF).mean())
        mae = float(np.abs(a_s - a_r)[either].mean())
    else:
        bad_pct, mae = 0.0, 0.0
    c_s, c_r = _centroid(a_s, union), _centroid(a_r, union)
    cdx = cdy = None
    if c_s is not None and c_r is not None:
        cdx, cdy = c_r[0] - c_s[0], c_r[1] - c_s[1]
    box_s, box_r = _ink_box(a_s, union), _ink_box(a_r, union)
    deltas: list[int | None] = [None] * 4
    if box_s is not None and box_r is not None:
        deltas = [r - s for r, s in zip(box_r, box_s, strict=True)]
    return Fidelity(
        iou=float(inter / uni) if uni else 1.0,
        bad_pct=bad_pct,
        mae=mae,
        cdx=cdx,
        cdy=cdy,
        dx0=deltas[0],
        dy0=deltas[1],
        dx1=deltas[2],
        dy1=deltas[3],
        status=status,
    )
