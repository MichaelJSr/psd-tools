"""Placement: a :class:`TextModel` to glyph positions in text space (y down)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from psd_tools.composite.text.compose import Decision
from psd_tools.composite.text.fonts import Face, FontMap, Resolved
from psd_tools.composite.text.model import RGB, TextModel


@dataclass(frozen=True, eq=False)
class PlacedGlyph:
    """A glyph at its baseline origin in text-space pt."""

    face: Face
    gid: int
    x: float
    y: float
    em_w: float  #: rendered em times HorizontalScale
    em_h: float  #: rendered em times VerticalScale
    fill: RGB
    faux_bold: bool
    faux_italic: bool
    run: int


@dataclass(frozen=True)
class Rule:
    """An underline or strikethrough segment in text-space pt."""

    x0: float
    x1: float
    y: float
    thickness: float
    fill: RGB


@dataclass(frozen=True)
class LineInfo:
    """One laid-out line, for the report."""

    text: str
    x0: float
    width: float
    y: float  #: baseline
    n_spaces: int
    stretch: float
    decision: Decision


@dataclass(frozen=True, eq=False)
class Layout:
    """Placed glyphs and rules; the seam between layout and rasterization."""

    glyphs: tuple[PlacedGlyph, ...]
    rules: tuple[Rule, ...]
    lines: tuple[LineInfo, ...]
    fonts: Mapping[str, Resolved]  #: PostScript name -> how it resolved
    warnings: tuple[str, ...]


def layout(
    model: TextModel, fonts: FontMap, breaks: Sequence[int] | None = None
) -> Layout:
    """Shape, compose and place ``model``."""
    raise NotImplementedError
