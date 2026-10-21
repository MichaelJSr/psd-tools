"""Placement: a :class:`TextModel` to glyph positions in text space (y down)."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from psd_tools.composite.text.compose import Decision, Line, break_lines
from psd_tools.composite.text.constants import BOX_ABSOLUTE_FRAME_L, FAUX_BOLD_EM
from psd_tools.composite.text.fonts import Face, FontMap, Resolved
from psd_tools.composite.text.model import RGB, Style, TextModel
from psd_tools.composite.text.shape import (
    Glyph,
    Shaped,
    rendered_em,
    run_index,
    shape_piece,
)


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
    drawn: bool = True  #: ``False`` for a line below the box (``overflow``)


@dataclass(frozen=True, eq=False)
class Layout:
    """Placed glyphs and rules; the seam between layout and rasterization.

    ``warnings`` carries the model's warnings followed by the layout's own.
    """

    glyphs: tuple[PlacedGlyph, ...]
    rules: tuple[Rule, ...]
    lines: tuple[LineInfo, ...]
    fonts: Mapping[str, Resolved]  #: PostScript name -> how it resolved
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class _Piece:
    """``text[start:end]`` between terminators; ``end`` indexes the terminator."""

    paragraph: int
    start: int
    end: int


def pieces(model: TextModel) -> list[_Piece]:
    """Paragraphs split at forced breaks.

    An empty paragraph is a piece (an empty line); the empty remainder after a
    paragraph's last forced break is not.
    """
    text = model.text
    out: list[_Piece] = []
    for pi, p in enumerate(model.paragraphs):
        end = p.end - 1 if p.end > p.start and text[p.end - 1] == "\r" else p.end
        start = p.start
        while True:
            j = text.find("\x03", start, end)
            if j < 0:
                if start < end or start == p.start:
                    out.append(_Piece(pi, start, end))
                break
            out.append(_Piece(pi, start, j))
            start = j + 1
    return out


def frame(model: TextModel) -> tuple[float, float, float, float] | None:
    """The box as ``(left, top, width, height)``; ``None`` for point text.

    A box whose left edge is far from zero is stored in an absolute frame
    whose origin is the text-space origin.
    """
    if model.shape != 1 or model.box is None:
        return None
    left, top, right, bottom = model.box
    if abs(left) >= BOX_ABSOLUTE_FRAME_L:
        left, top = 0.0, 0.0
    return left, top, model.box[2] - model.box[0], model.box[3] - model.box[1]


def leading(style: Style, model: TextModel) -> float:
    """A run's leading in pt: ``Leading``, or the auto ratio times the nominal size."""
    return model.auto_leading * style.size if style.auto_leading else style.leading


def baseline_shift(style: Style, model: TextModel) -> float:
    """Downward shift of a run's glyph baseline from the line's nominal baseline.

    Super/subscript move by the position ratio of the nominal size scaled
    vertically; ``BaselineShift`` is unscaled and positive upwards.
    """
    sign = {1: -1.0, 2: 1.0}.get(style.baseline, 0.0)
    shift = sign * model.sub_position * style.size * style.vertical_scale
    return shift - style.baseline_shift


def _align(justification: int, left: float, free: float) -> float:
    """Line start for ``free`` = available width minus the line's set width."""
    if justification in (1, 4):
        return left + free
    if justification in (2, 5):
        return left + free / 2
    return left


def _line_glyphs(shaped: Shaped, line: Line) -> list[Glyph]:
    clusters = [g.cluster for g in shaped.glyphs]
    return list(
        shaped.glyphs[
            bisect_left(clusters, line.start) : bisect_left(clusters, line.end)
        ]
    )


def _runs_on(model: TextModel, start: int, end: int) -> list[int]:
    """Runs with a character in ``text[start:end]``.

    An empty line uses the run holding its terminator at ``end``.
    """
    if end <= start:
        return (
            [run_index(model.runs, min(end, len(model.text) - 1))] if model.text else []
        )
    first = run_index(model.runs, start)
    last = run_index(model.runs, end - 1)
    return list(range(first, last + 1))


def layout(
    model: TextModel, fonts: FontMap, breaks: Sequence[int] | None = None
) -> Layout:
    """Shape, compose and place ``model``."""
    text = model.text
    names = list(dict.fromkeys(r.style.font for r in model.runs if r.end > r.start))
    resolved = fonts.resolve_all(names)
    warnings = list(model.warnings)

    def warn(w: str) -> None:
        if w not in warnings:
            warnings.append(w)

    for r in resolved.values():
        if r.how == "loose":
            warn("font_loose")
        elif r.how == "substitute":
            warn("font_substitute")

    def face_of(style: Style) -> Face | None:
        r = resolved.get(style.font)
        return None if r is None else r.face

    def ascent(style: Style) -> float | None:
        face = face_of(style)
        if face is None:
            return None
        if face.metrics.ascent_glyph != "d":
            warn(f"ascent_fallback_{face.metrics.ascent_glyph}")
        # The faux-bold stroke raises the ascent by its half-width, at the
        # nominal size like the ascent itself.
        stroke = FAUX_BOLD_EM / 2 if style.faux_bold else 0.0
        return (face.metrics.ascent_units / face.metrics.upem + stroke) * style.size

    box = frame(model)
    width = box[2] if box else None
    left, top = (box[0], box[1]) if box else (0.0, 0.0)
    bottom = top + box[3] if box else None

    glyphs: list[PlacedGlyph] = []
    rules: list[Rule] = []
    infos: list[LineInfo] = []
    y = top
    started = False  # a line with content has been placed
    for piece in pieces(model):
        shaped = shape_piece(
            text, piece.start, piece.end, model.runs, fonts, sub_size=model.sub_size
        )
        justification = model.paragraphs[piece.paragraph].justification
        lines = break_lines(
            text,
            piece.start,
            piece.end,
            shaped.measure,
            sorted(shaped.space_adv),
            width,
            justification,
            breaks,
        )
        for k, line in enumerate(lines):
            # Trailing spaces hang on the line, so their runs set its metrics.
            hang_end = lines[k + 1].start if k + 1 < len(lines) else piece.end
            styles = [
                model.runs[i].style for i in _runs_on(model, line.start, hang_end)
            ]
            pitch = max((leading(s, model) for s in styles), default=0.0)
            y_prev = y
            if box is None:
                y = 0.0 if not started else y + pitch
                started = True
            elif not started:
                # The first line, empty or not, sits one ascent of its own
                # runs below the top.
                y += max((a for a in map(ascent, styles) if a is not None), default=0.0)
                started = True
            else:
                y += pitch
            drawn = bottom is None or y_prev <= bottom
            if not drawn:
                warn("overflow")

            line_glyphs = _line_glyphs(shaped, line)
            natural = shaped.measure(line.start, line.end)
            count = sum(1 for g in line_glyphs if g.cluster in shaped.space_adv)
            if width is None:
                x0 = _align(justification, 0.0, -natural)
            else:
                x0 = _align(justification, left, width - natural - count * line.stretch)
            infos.append(
                LineInfo(
                    text[line.start : line.end],
                    x0,
                    natural,
                    y,
                    count,
                    line.stretch,
                    line.decision,
                    drawn,
                )
            )
            if not drawn:
                continue

            pen = x0
            span_start, span_run = pen, -1
            for k, g in enumerate(line_glyphs):
                style = model.runs[g.run].style
                em = rendered_em(style, model.sub_size)
                base = y + baseline_shift(style, model)
                glyphs.append(
                    PlacedGlyph(
                        face=g.face,
                        gid=g.gid,
                        x=pen + g.xoff,
                        y=base - g.yoff,
                        em_w=em * style.horizontal_scale,
                        em_h=em * style.vertical_scale,
                        fill=style.fill,
                        faux_bold=style.faux_bold,
                        faux_italic=style.faux_italic,
                        run=g.run,
                    )
                )
                if g.run != span_run:
                    span_start, span_run = pen, g.run
                advance = g.adv + (
                    line.stretch if g.cluster in shaped.space_adv else 0.0
                )
                pen += advance
                last_of_run = (
                    k + 1 == len(line_glyphs) or line_glyphs[k + 1].run != g.run
                )
                if last_of_run and (style.underline or style.strikethrough):
                    rules.extend(_rules(style, g.face, span_start, pen, base, em))
                pen += g.tracking
    return Layout(
        glyphs=tuple(glyphs),
        rules=tuple(rules),
        lines=tuple(infos),
        fonts=resolved,
        warnings=tuple(warnings),
    )


def _rules(
    style: Style, face: Face, x0: float, x1: float, base: float, em: float
) -> list[Rule]:
    m = face.metrics
    thickness = m.underline_thickness / m.upem * em
    out: list[Rule] = []
    if style.underline:
        out.append(
            Rule(
                x0, x1, base - m.underline_position / m.upem * em, thickness, style.fill
            )
        )
    if style.strikethrough:
        # No strikeout metric is kept; the midpoint of the ascent stands in.
        out.append(
            Rule(x0, x1, base - m.ascent_units / m.upem * em / 2, thickness, style.fill)
        )
    return out
