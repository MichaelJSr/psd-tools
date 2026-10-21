"""Shaping: text and runs to positioned glyphs with HarfBuzz."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import uharfbuzz as hb

from psd_tools.composite.text.constants import FAUX_BOLD_EM, SUB_SIZE
from psd_tools.composite.text.fonts import Face, FontMap
from psd_tools.composite.text.model import Run, Style

#: Characters that take the space glyph's advance and offer a line break.
SPACE_CLASS = frozenset(" \t")
#: Terminators: never shaped, no glyph, no advance.
CONTROL = frozenset("\r\x03\n")


@dataclass(frozen=True, eq=False)
class Glyph:
    """One shaped glyph; distances in text-space pt."""

    face: Face
    gid: int
    cluster: int  #: ``str`` index of the character that produced it
    adv: float  #: horizontal advance including scale and faux-bold widening
    xoff: float
    yoff: float  #: positive raises the glyph (HarfBuzz's y_offset, scaled)
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


def rendered_em(style: Style, sub_size: float = SUB_SIZE) -> float:
    """The em a run is drawn at: ``FontSize`` scaled for super/subscript."""
    return style.size * (sub_size if style.baseline in (1, 2) else 1.0)


def shaping_key(style: Style) -> tuple[object, ...]:
    """What a segment shares; colour, tracking and underline do not cut.

    A ``BaselineShift`` change does: a shifted glyph is not kerned with its
    neighbours.
    """
    return (
        style.font,
        style.size,
        style.baseline,
        style.horizontal_scale,
        style.vertical_scale,
        style.caps,
        style.ligatures,
        style.faux_bold,
        style.faux_italic,
        style.baseline_shift,
    )


def run_locator(runs: Sequence[Run]) -> Callable[[int], int]:
    """``at(i)``: the run containing ``str`` index ``i`` (the last run past the end)."""
    starts = [r.start for r in runs]
    last = len(runs) - 1

    def at(i: int) -> int:
        return max(0, min(bisect_right(starts, i) - 1, last))

    return at


def run_index(runs: Sequence[Run], i: int) -> int:
    """The run containing ``str`` index ``i`` (the last run past the end)."""
    return run_locator(runs)(i)


def _segments(
    text: str, a: int, b: int, runs: Sequence[Run]
) -> list[tuple[int, int, int]]:
    """``(start, end, run index)`` of maximal same-key stretches without terminators."""
    out: list[tuple[int, int, int]] = []
    at = run_locator(runs)
    key = None
    start = seg_run = a
    for i in range(a, b):
        if text[i] in CONTROL:
            if key is not None:
                out.append((start, i, seg_run))
            key = None
            continue
        ri = at(i)
        k = shaping_key(runs[ri].style)
        if k != key:
            if key is not None:
                out.append((start, i, seg_run))
            key, start, seg_run = k, i, ri
    if key is not None:
        out.append((start, b, seg_run))
    return out


def _caps(text: str) -> str:
    # Only one-to-one upper-case mappings keep cluster indices aligned.
    return "".join(c.upper() if len(c.upper()) == 1 else c for c in text)


def shape_piece(
    text: str,
    a: int,
    b: int,
    runs: Sequence[Run],
    fonts: FontMap,
    *,
    sub_size: float = SUB_SIZE,
) -> Shaped:
    """Shape ``text[a:b]``, one segment per run of identical shaping style.

    A run whose font resolved to no face (``skip`` policy) produces no glyphs
    and no advance. Writing direction is left to right.
    """
    glyphs: list[Glyph] = []
    at = run_locator(runs)
    for s, e, ri in _segments(text, a, b, runs):
        style = runs[ri].style
        face = fonts.resolve(style.font).face
        if face is None:
            continue
        n = e - s
        chars = text[s:e].replace("\t", " ")
        if style.caps == 2:
            chars = _caps(chars)
        features: dict[str, object] = {
            "kern": True,
            "liga": style.ligatures,
            "clig": style.ligatures,
            "calt": False,
        }
        # AutoKerning off on a character removes the pair ending at it; the
        # segment's first character has no pair to remove.
        off = [j for j in range(1, n) if not runs[at(s + j)].style.auto_kerning]
        if off:
            features["kern"] = [(0, n, 1)] + [(j - 1, j + 1, 0) for j in off]
        buf = hb.Buffer()
        buf.add_str(chars)
        buf.guess_segment_properties()
        buf.direction = "ltr"
        buf.cluster_level = hb.BufferClusterLevel.MONOTONE_CHARACTERS
        hb.shape(face.hb, buf, features)
        infos, positions = buf.glyph_infos, buf.glyph_positions

        em = rendered_em(style, sub_size)
        upem = face.metrics.upem
        kx = em * style.horizontal_scale / upem
        ky = em * style.vertical_scale / upem
        bold = em * FAUX_BOLD_EM if style.faux_bold else 0.0
        for k, (info, pos) in enumerate(zip(infos, positions, strict=True)):
            j = info.cluster
            following = infos[k + 1].cluster if k + 1 < len(infos) else n
            # The tracking after a glyph comes from the last character it covers.
            tracked = runs[at(s + max(j, following - 1))].style
            glyphs.append(
                Glyph(
                    face=face,
                    gid=info.codepoint,
                    cluster=s + j,
                    adv=pos.x_advance * kx + bold,
                    xoff=pos.x_offset * kx,
                    yoff=pos.y_offset * ky,
                    tracking=tracked.tracking / 1000 * em,
                    run=at(s + j),
                )
            )

    clusters = [g.cluster for g in glyphs]
    pen = [0.0]
    for g in glyphs:
        pen.append(pen[-1] + g.adv + g.tracking)

    def measure(p: int, q: int) -> float:
        k0, k1 = bisect_left(clusters, p), bisect_left(clusters, q)
        if k1 <= k0:
            return 0.0
        return pen[k1] - pen[k0] - glyphs[k1 - 1].tracking

    space_adv: dict[int, float] = {}
    for g in glyphs:
        if text[g.cluster] in SPACE_CLASS:
            space_adv[g.cluster] = space_adv.get(g.cluster, 0.0) + g.adv
    return Shaped(glyphs=tuple(glyphs), measure=measure, space_adv=space_adv)
