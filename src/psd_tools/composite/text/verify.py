"""Fidelity of a render against a stored raster; pure arrays, no I/O."""

from __future__ import annotations

import numpy as np

from psd_tools.composite.text.raster import Bbox
from psd_tools.composite.text.report import Fidelity


def fidelity(
    stored_rgba: np.ndarray,
    stored_bbox: Bbox,
    rgba: np.ndarray | None,
    bbox: Bbox | None,
) -> Fidelity:
    """Compare two straight-alpha RGBA rasters on the union of their bboxes."""
    raise NotImplementedError
