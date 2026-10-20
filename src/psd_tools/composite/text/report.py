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
    """Agreement between a render and a stored raster; the harness metrics."""

    iou: float
    bad_pct: float
    mae: float
    cdx: float  #: alpha-centroid shift, rendered minus stored, px
    cdy: float
    dx0: int  #: ink-box deltas, rendered minus stored, px
    dy0: int
    dx1: int
    dy1: int
    status: Literal["ok", "empty", "missing"]
