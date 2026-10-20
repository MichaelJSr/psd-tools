"""What a render reports about itself."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from psd_tools.composite.text.fonts import How
from psd_tools.composite.text.layout import LineInfo
from psd_tools.composite.text.raster import Bbox


@dataclass(frozen=True)
class RenderReport:
    """Fonts, lines, warnings and timing of one render."""

    fonts: Mapping[str, tuple[How, str | None]]  #: name -> (how, path)
    lines: tuple[LineInfo, ...]
    warnings: tuple[str, ...]
    ms: float


@dataclass(frozen=True, eq=False)
class RenderResult:
    """A render: straight RGBA ``float32`` HxWx4, its bbox and the report."""

    rgba: np.ndarray
    bbox: Bbox
    report: RenderReport


@dataclass(frozen=True)
class Fidelity:
    """Agreement between a render and a stored raster; the harness metrics.

    ``status`` is ``"ok"`` when both rasters have ink, ``"empty"`` when the
    render has none and ``"missing"`` when a font kept the layer from
    rendering; the shifts are ``None`` when either side has no ink.
    """

    iou: float  #: of the pixels either raster covers above ``IOU_COVER``
    bad_pct: float  #: % of pixels whose premultiplied channel diff exceeds ``BAD_DIFF``
    mae: float  #: mean |alpha difference| where either alpha exceeds ``ANY_ALPHA``
    cdx: float | None  #: alpha-centroid shift, rendered minus stored, px
    cdy: float | None
    dx0: int | None  #: ink-box deltas, rendered minus stored, px
    dy0: int | None
    dx1: int | None
    dy1: int | None
    status: Literal["ok", "empty", "missing"]
