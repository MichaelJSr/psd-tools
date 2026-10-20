"""Line breaking over a measure function; no font or glyph access."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

Decision = Literal["fit", "compress", "break", "forced", "paragraph"]


@dataclass(frozen=True)
class Line:
    """``text[start:end]`` on one line; ``end`` excludes the break character."""

    start: int
    end: int
    stretch: float  #: extra pt added to each space, 0 when not justified
    decision: Decision  #: why the line ended


def break_lines(
    text: str,
    a: int,
    b: int,
    measure: Callable[[int, int], float],
    space_positions: Sequence[int],
    width: float | None,
    justification: int,
    breaks: Sequence[int] | None = None,
) -> list[Line]:
    """Break the piece ``text[a:b]`` into lines.

    ``width`` is ``None`` for point text (one line). ``breaks`` lists ``str``
    indices that start a line and bypasses the greedy pass.
    """
    raise NotImplementedError
