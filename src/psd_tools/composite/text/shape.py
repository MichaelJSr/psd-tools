"""Shaping: text and runs to positioned glyphs with HarfBuzz."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from psd_tools.composite.text.fonts import Face, FontMap
from psd_tools.composite.text.model import Run


@dataclass(frozen=True, eq=False)
class Glyph:
    """One shaped glyph; distances in text-space pt."""

    face: Face
    gid: int
    cluster: int  #: ``str`` index of the character that produced it
    adv: float  #: horizontal advance including scale and faux-bold widening
    xoff: float
    yoff: float
    tracking: float  #: the tracking that follows this glyph
    run: int  #: index into :attr:`TextModel.runs`


@dataclass(frozen=True, eq=False)
class Shaped:
    """A shaped piece of text."""

    glyphs: tuple[Glyph, ...]
    #: ``measure(a, b)``: width in pt of ``text[a:b]`` from the pen positions,
    #: trailing tracking excluded, without re-shaping.
    measure: Callable[[int, int], float]
    #: advance of each space-class character by ``str`` index
    space_adv: Mapping[int, float]


def shape_piece(
    text: str, a: int, b: int, runs: Sequence[Run], fonts: FontMap
) -> Shaped:
    """Shape ``text[a:b]``, one segment per run of identical shaping style."""
    raise NotImplementedError
