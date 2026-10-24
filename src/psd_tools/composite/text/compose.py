"""Line breaking over a measure function; no font or glyph access."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from psd_tools.composite.text.constants import WORD_SPACING_MIN

Decision = Literal["fit", "compress", "break", "forced", "paragraph"]

#: Float noise in a sum of advances, not slack: a line fits exactly or not.
_FIT_EPS = 1e-6


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

    Decisions: ``fit`` the next word did not fit, ``compress`` a word was
    pulled up by compressing the spaces, ``break`` a break inside a word
    wider than the line (or one given in ``breaks``), ``forced`` the piece
    ends at a forced line break, ``paragraph`` at a paragraph end.
    """
    forced = b < len(text) and text[b] == "\x03"
    final: Decision = "forced" if forced else "paragraph"
    spaces = sorted(i for i in space_positions if a <= i < b)
    space_set = frozenset(spaces)
    justified = justification in (3, 4, 5, 6)

    def strip(p: int, q: int) -> int:
        while q > p and (q - 1) in space_set:
            q -= 1
        return q

    def n_spaces(p: int, q: int) -> int:
        return bisect_left(spaces, q) - bisect_left(spaces, p)

    raw: list[tuple[int, int, Decision]]
    if width is None:
        # Point text: trailing spaces count, so lines keep them.
        starts = [a, *sorted(i for i in (breaks or ()) if a < i < b)]
        ends = [*starts[1:], b]
        raw = [
            (s, e, final if e == b else "break")
            for s, e in zip(starts, ends, strict=True)
        ]
        return [Line(s, e, 0.0, d) for s, e, d in raw]
    if breaks is not None:
        starts = [a, *sorted(i for i in breaks if a < i < b)]
        ends = [*starts[1:], b]
        raw = [
            (s, strip(s, e), final if e == b else "break")
            for s, e in zip(starts, ends, strict=True)
        ]
    else:
        raw = _greedy(text, a, b, measure, spaces, space_set, width, justified, final)

    out: list[Line] = []
    for k, (s, e, decision) in enumerate(raw):
        stretch = 0.0
        count = n_spaces(s, e)
        if justified and count:
            last = k == len(raw) - 1 and not forced
            if justification == 6 or not last or decision == "compress":
                stretch = (width - measure(s, e)) / count
        out.append(Line(s, e, stretch, decision))
    return out


def _break_ops(
    text: str, a: int, b: int, space_set: frozenset[int]
) -> list[tuple[int, int]]:
    """``(end, start)`` per run of spaces.

    The line ends before the run and the next starts after it; a run at the
    piece start is not an opportunity.
    """
    ops: list[tuple[int, int]] = []
    i = a
    while i < b:
        if i in space_set:
            j = i
            while j < b and j in space_set:
                j += 1
            if i > a:
                ops.append((i, j))
            i = j
        else:
            i += 1
    return ops


def _greedy(
    text: str,
    a: int,
    b: int,
    measure: Callable[[int, int], float],
    spaces: list[int],
    space_set: frozenset[int],
    width: float,
    justified: bool,
    final: Decision,
) -> list[tuple[int, int, Decision]]:
    ops = _break_ops(text, a, b, space_set)

    def fits(p: int, q: int) -> bool:
        return measure(p, q) <= width + _FIT_EPS

    def factor(p: int, q: int) -> float | None:
        """Word-spacing factor that sets ``text[p:q]`` to ``width``; None without spaces."""
        k0, k1 = bisect_left(spaces, p), bisect_left(spaces, q)
        total = sum(measure(i, i + 1) for i in spaces[k0:k1])
        return None if total <= 0 else 1 + (width - measure(p, q)) / total

    def compressible(p: int, q: int, natural: int | None) -> bool:
        """Whether to pull the word ending at ``q`` onto the line.

        Only when compressing deviates less from normal spacing than
        stretching the line that ends at ``natural``.
        """
        f = factor(p, q)
        if f is None or f < WORD_SPACING_MIN - _FIT_EPS:
            return False
        if natural is None or natural <= p:
            return True
        f0 = factor(p, natural)
        return f0 is None or (1 - f) < (f0 - 1)

    def strip(p: int, q: int) -> int:
        while q > p and (q - 1) in space_set:
            q -= 1
        return q

    lines: list[tuple[int, int, Decision]] = []
    pos = a
    while True:
        rest = [(e, s) for e, s in ops if e > pos]
        end_all = strip(pos, b)
        natural = None
        for e, _s in rest:
            if not fits(pos, e):
                break
            natural = e
        if fits(pos, end_all):
            lines.append((pos, end_all, final))
            break
        if justified and compressible(pos, end_all, natural):
            lines.append((pos, end_all, "compress"))
            break
        best: tuple[int, int] | None = None
        natural = None
        compressed = False
        for e, s in rest:
            if fits(pos, e):
                best, natural = (e, s), e
            elif justified and compressible(pos, e, natural):
                best, natural, compressed = (e, s), e, True
            else:
                break
        if best is None:
            # A word wider than the line breaks at the overflowing character.
            e = pos + 1
            while e < b and fits(pos, e + 1):
                e += 1
            if rest and e >= rest[0][0]:
                lines.append((pos, rest[0][0], "fit"))
                pos = rest[0][1]
            else:
                lines.append((pos, e, "break"))
                pos = e
        else:
            lines.append((pos, best[0], "compress" if compressed else "fit"))
            pos = best[1]
        if pos >= b:
            break
    return lines
