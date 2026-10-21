from collections.abc import Callable, Mapping

import pytest

pytest.importorskip("freetype")
pytest.importorskip("uharfbuzz")

from psd_tools.composite.text.compose import Line, break_lines

W = 100.0


def widths(
    text: str, letter: float = 10.0, space: float = 10.0, **special: float
) -> tuple[Callable[[int, int], float], list[int]]:
    """A measure over fixed per-character widths and the space positions."""
    table: Mapping[str, float] = {" ": space, **special}

    def measure(a: int, b: int) -> float:
        return sum(table.get(c, letter) for c in text[a:b])

    return measure, [i for i, c in enumerate(text) if c == " "]


def spans(lines: list[Line]) -> list[tuple[int, int]]:
    return [(x.start, x.end) for x in lines]


def texts(text: str, lines: list[Line]) -> list[str]:
    return [text[x.start : x.end] for x in lines]


def test_point_text_is_one_line() -> None:
    text = "aa bb  \rcc"
    measure, sp = widths(text)
    lines = break_lines(text, 0, 7, measure, sp, None, 2)
    assert lines == [Line(0, 7, 0.0, "paragraph")]
    text = "aa\x03bb"
    measure, sp = widths(text)
    assert break_lines(text, 0, 2, measure, sp, None, 0)[0].decision == "forced"


def test_exact_fit_boundary() -> None:
    text = "aaaa bbbb"
    measure, sp = widths(text, space=5.0)
    assert break_lines(text, 0, 9, measure, sp, 85.0, 0) == [
        Line(0, 9, 0.0, "paragraph")
    ]
    lines = break_lines(text, 0, 9, measure, sp, 84.99, 0)
    assert lines == [Line(0, 4, 0.0, "fit"), Line(5, 9, 0.0, "paragraph")]


def test_trailing_spaces_hang() -> None:
    # The whole run of spaces hangs on the first line; the next starts after it.
    text = "aaaa   bbbb  "
    measure, sp = widths(text)
    lines = break_lines(text, 0, len(text), measure, sp, 40.0, 0)
    assert spans(lines) == [(0, 4), (7, 11)]
    # Hanging spaces are not part of the fit test.
    assert break_lines(text, 0, 7, measure, sp, 40.0, 0) == [
        Line(0, 4, 0.0, "paragraph")
    ]


def test_leading_and_interior_spaces_are_glyphs() -> None:
    text = "  aaaa bbbb"
    measure, sp = widths(text, space=5.0)
    lines = break_lines(text, 0, len(text), measure, sp, 50.0, 0)
    assert spans(lines) == [(0, 6), (7, 11)]
    text = "a   b"
    measure, sp = widths(text)
    lines = break_lines(text, 0, 5, measure, sp, W, 0)
    assert spans(lines) == [(0, 5)]


@pytest.mark.parametrize("sep", ["-", "/"])
def test_no_break_after_hyphen_or_slash(sep: str) -> None:
    text = f"aaaa{sep}bbbb cc"
    measure, sp = widths(text)
    lines = break_lines(text, 0, len(text), measure, sp, 60.0, 0)
    # The 9-letter word is wider than the line: it breaks at the overflowing
    # character, never after the separator.
    assert lines[0] == Line(0, 6, 0.0, "break")
    assert texts(text, lines) == [f"aaaa{sep}b", "bbb cc"]


def test_compression_floor_refuses() -> None:
    # One word, no spaces: stretching is infinitely bad, yet 0.5 is below the floor.
    text = "a b"
    measure, sp = widths(text, a=85.0, b=10.0)
    lines = break_lines(text, 0, 3, measure, sp, W, 3)
    assert lines == [Line(0, 1, 0.0, "fit"), Line(2, 3, 0.0, "paragraph")]


def test_compression_pulls_last_word() -> None:
    # 101 pt over 100 with one space: factor 0.9, closer than any stretch.
    text = "a b"
    measure, sp = widths(text, a=85.0, b=6.0)
    lines = break_lines(text, 0, 3, measure, sp, W, 3)
    assert lines == [Line(0, 3, -1.0, "compress")]


def test_compression_pull_by_deviation() -> None:
    # Five words (natural 93, factor 1.175) plus 'b' (104, factor 0.92).
    text = "a a a a a b ccc"
    measure, sp = widths(text, a=10.6, b=1.0, c=10.0)
    lines = break_lines(text, 0, len(text), measure, sp, W, 3)
    assert texts(text, lines) == ["a a a a a b", "ccc"]
    assert lines[0].decision == "compress"
    assert lines[0].stretch == pytest.approx(-4 / 5)


def test_compression_refused_by_deviation() -> None:
    # Natural 97 (factor 1.075) against 108 (factor 0.84): stretching is closer.
    text = "a a a a a b ccc"
    measure, sp = widths(text, a=11.4, b=1.0, c=10.0)
    lines = break_lines(text, 0, len(text), measure, sp, W, 3)
    assert texts(text, lines) == ["a a a a a", "b ccc"]
    assert lines[0].decision == "fit"
    assert lines[0].stretch == pytest.approx(3 / 4)


def test_compression_stops_after_one_word() -> None:
    # 50 one-unit words (99) pull a 51st (0.98) but not a 52nd (0.941):
    # a second compression is compared with a line already below normal spacing.
    text = " ".join(["a"] * 52)
    measure, sp = widths(text, letter=1.0, space=1.0)
    lines = break_lines(text, 0, len(text), measure, sp, W, 3)
    assert texts(text, lines)[0] == " ".join(["a"] * 51)
    assert lines[0].decision == "compress"
    assert lines[0].stretch == pytest.approx(-1 / 50)


def test_non_justified_never_compresses() -> None:
    text = "a a a a a b ccc"
    measure, sp = widths(text, a=10.6, b=1.0, c=10.0)
    for justification in (0, 1, 2):
        lines = break_lines(text, 0, len(text), measure, sp, W, justification)
        assert texts(text, lines) == ["a a a a a", "b ccc"]
        assert all(x.stretch == 0.0 for x in lines)


def test_stretch_into_spaces() -> None:
    text = "aa bb cc dd\r"
    measure, sp = widths(text)
    lines = break_lines(text, 0, 11, measure, sp, 60.0, 3)
    assert texts(text, lines) == ["aa bb", "cc dd"]
    # Unclamped: 60 - 50 over one space, factor 2.
    assert lines[0].stretch == pytest.approx(10.0)
    # The paragraph's last line is not stretched for 3/4/5.
    assert lines[1].stretch == 0.0
    assert [x.decision for x in lines] == ["fit", "paragraph"]


def test_justify_all_stretches_last_line() -> None:
    text = "aa bb cc dd"
    measure, sp = widths(text)
    lines = break_lines(text, 0, 11, measure, sp, 60.0, 6)
    assert [x.stretch for x in lines] == pytest.approx([10.0, 10.0])


def test_forced_break_line_is_stretched() -> None:
    text = "aa bb\x03cc dd\r"
    measure, sp = widths(text)
    lines = break_lines(text, 0, 5, measure, sp, W, 3)
    assert lines == [Line(0, 5, 50.0, "forced")]
    lines = break_lines(text, 6, 11, measure, sp, W, 3)
    assert lines == [Line(6, 11, 0.0, "paragraph")]


def test_breaks_override() -> None:
    text = "aaaa bbbb cccc"
    measure, sp = widths(text)
    lines = break_lines(text, 0, 14, measure, sp, W, 0, breaks=[5, 10, 99])
    assert lines == [
        Line(0, 4, 0.0, "break"),
        Line(5, 9, 0.0, "break"),
        Line(10, 14, 0.0, "paragraph"),
    ]
    # Point text keeps its trailing spaces.
    lines = break_lines(text, 0, 14, measure, sp, None, 0, breaks=[5, 10])
    assert spans(lines) == [(0, 5), (5, 10), (10, 14)]
    # Justified lines given explicitly are still stretched.
    lines = break_lines(text, 0, 14, measure, sp, W, 3, breaks=[10])
    assert lines[0] == Line(0, 9, (W - 90.0) / 1, "break")


def test_empty_piece() -> None:
    text = "\r"
    measure, sp = widths(text)
    assert break_lines(text, 0, 0, measure, sp, W, 2) == [Line(0, 0, 0.0, "paragraph")]
    assert break_lines(text, 0, 0, measure, sp, None, 2) == [
        Line(0, 0, 0.0, "paragraph")
    ]


def test_word_wider_than_line_then_fit() -> None:
    text = "aaaaaaaaaaaa bb"
    measure, sp = widths(text)
    lines = break_lines(text, 0, len(text), measure, sp, 50.0, 2)
    assert texts(text, lines) == ["aaaaa", "aaaaa", "aa bb"]
    assert [x.decision for x in lines] == ["break", "break", "paragraph"]
