"""The text layer as data: text, styled runs, paragraphs, shape and transform.

Built from a layer's EngineData; no font or raster dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from psd_tools.composite.text.constants import AUTO_LEADING, SUB_POSITION, SUB_SIZE

if TYPE_CHECKING:
    from psd_tools.api.layers import TypeLayer

#: ``(xx, xy, yx, yy, tx, ty)`` as :attr:`TypeLayer.transform` orders it.
Transform = tuple[float, float, float, float, float, float]
RGB = tuple[float, float, float]


@dataclass(frozen=True)
class Style:
    """A run's character style after the Normal sheet's defaults apply."""

    font: str  #: PostScript name from ``FontSet``
    size: float  #: ``FontSize`` in pt
    baseline: int  #: ``FontBaseline``: 0 normal, 1 superscript, 2 subscript
    tracking: float  #: ``Tracking`` in thousandths of an em
    auto_kerning: bool
    ligatures: bool
    horizontal_scale: float
    vertical_scale: float
    caps: int  #: ``FontCaps``: 0 normal, 1 small caps, 2 all caps
    faux_bold: bool
    faux_italic: bool
    underline: bool
    strikethrough: bool
    baseline_shift: float  #: pt, positive is up
    fill: RGB  #: straight RGB 0..1
    leading: float  #: pt, used when ``auto_leading`` is false
    auto_leading: bool


@dataclass(frozen=True)
class Run:
    """``text[start:end]`` shares one :class:`Style`; ``str`` indices."""

    start: int
    end: int
    style: Style


@dataclass(frozen=True)
class Paragraph:
    """``text[start:end]`` including its paragraph terminator; ``str`` indices."""

    start: int
    end: int
    justification: int  #: 0 left, 1 right, 2 center, 3-5 justify last left/right/center, 6 justify all


@dataclass(frozen=True)
class TextModel:
    """Everything layout needs from a type layer."""

    text: str
    runs: tuple[Run, ...]
    paragraphs: tuple[Paragraph, ...]
    shape: int  #: ``ShapeType``: 0 point, 1 box
    box: (
        tuple[float, float, float, float] | None
    )  #: ``BoxBounds`` (L, T, R, B) as stored, box text only
    antialias: int  #: ``AntA``: 0 none, 1 crisp, 2 strong, 3 smooth, 4 sharp
    transform: Transform
    warnings: tuple[str, ...]
    resolution: float = 72.0  #: document ppi
    sub_size: float = SUB_SIZE  #: ``ResourceDict`` SubscriptSize/SuperscriptSize
    sub_position: float = SUB_POSITION
    auto_leading: float = AUTO_LEADING  #: ``ResourceDict`` auto-leading ratio

    @classmethod
    def from_layer(cls, layer: TypeLayer) -> TextModel:
        """Read a :class:`TypeLayer`'s EngineData."""
        raise NotImplementedError
