import os
from collections.abc import Callable, Sequence
from typing import Any

import pytest

pytest.importorskip("freetype")
pytest.importorskip("uharfbuzz")

from psd_tools.composite.text import fonts as fonts_module
from psd_tools.composite.text.fonts import FontMap, MissingFontError
from psd_tools.composite.text.layout import Layout, frame, layout, pieces
from psd_tools.composite.text.model import Paragraph, Run, Style, TextModel

from .conftest import GLYPHS, PS_TT, UNDERLINE_POSITION, UNDERLINE_THICKNESS, UPEM

SIZE = 10.0
LEADING = 12.0
IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
ASCENT_D = GLYPHS["d"][1][3] / UPEM  # type: ignore[index]


def style(**kwargs: Any) -> Style:
    base: dict[str, Any] = dict(
        font=PS_TT,
        size=SIZE,
        baseline=0,
        tracking=0.0,
        auto_kerning=True,
        ligatures=True,
        horizontal_scale=1.0,
        vertical_scale=1.0,
        caps=0,
        faux_bold=False,
        faux_italic=False,
        underline=False,
        strikethrough=False,
        baseline_shift=0.0,
        fill=(0.0, 0.0, 0.0),
        leading=LEADING,
        auto_leading=False,
    )
    base.update(kwargs)
    return Style(**base)


def model(
    text: str,
    *styles: Style,
    cuts: Sequence[int] = (),
    box: tuple[float, float, float, float] | None = None,
    justification: int | Sequence[int] = 0,
    **kwargs: Any,
) -> TextModel:
    """A model over ``text`` with runs split at ``cuts`` and CR paragraphs."""
    styles = styles or (style(),)
    bounds = (0, *cuts, len(text))
    runs = tuple(
        Run(s, e, st) for s, e, st in zip(bounds[:-1], bounds[1:], styles, strict=True)
    )
    starts = [0] + [i + 1 for i, c in enumerate(text) if c == "\r"]
    if len(starts) > 1 and starts[-1] >= len(text):
        starts.pop()
    ends = starts[1:] + [len(text)]
    codes = (
        [justification] * len(starts)
        if isinstance(justification, int)
        else list(justification)
    )
    paragraphs = tuple(
        Paragraph(s, e, j) for s, e, j in zip(starts, ends, codes, strict=True)
    )
    return TextModel(
        text=text,
        runs=runs,
        paragraphs=paragraphs,
        shape=1 if box else 0,
        box=box,
        antialias=2,
        transform=IDENTITY,
        warnings=kwargs.pop("warnings", ()),
        **kwargs,
    )


def adv(glyph: str, size: float = SIZE) -> float:
    return GLYPHS[glyph][0] / UPEM * size


def xs(lay: Layout) -> list[float]:
    return [g.x for g in lay.glyphs]


def ys(lay: Layout) -> list[float]:
    return [g.y for g in lay.glyphs]


@pytest.fixture
def fonts(fixture_fonts_dir: str) -> FontMap:
    return FontMap(dirs=[fixture_fonts_dir])


@pytest.mark.parametrize(
    ("justification", "x0"),
    [(0, 0.0), (2, -4.5), (1, -9.0), (3, 0.0), (4, -9.0), (5, -4.5), (6, 0.0)],
)
def test_point_anchor(fonts: FontMap, justification: int, x0: float) -> None:
    # 'IA' is 9 pt wide; ty is the baseline.
    lay = layout(model("IA\r", justification=justification), fonts)
    assert xs(lay) == pytest.approx([x0, x0 + adv("I")])
    assert ys(lay) == [0.0, 0.0]
    line = lay.lines[0]
    assert (line.text, line.x0, line.width, line.y) == (
        "IA",
        pytest.approx(x0),
        9.0,
        0.0,
    )


def test_point_trailing_spaces_count(fonts: FontMap) -> None:
    lay = layout(model("IA \r", justification=2), fonts)
    assert xs(lay)[0] == pytest.approx(-(9.0 + adv("space")) / 2)
    assert len(lay.glyphs) == 3


def test_point_paragraphs_align_independently(fonts: FontMap) -> None:
    lay = layout(model("I\rIA\r", justification=2), fonts)
    assert [line.x0 for line in lay.lines] == pytest.approx([-1.0, -4.5])
    assert [line.y for line in lay.lines] == [0.0, LEADING]
    assert ys(lay) == [0.0, LEADING, LEADING]


def test_point_forced_break(fonts: FontMap) -> None:
    lay = layout(model("I\x03A\r"), fonts)
    assert [(line.text, line.y, line.decision) for line in lay.lines] == [
        ("I", 0.0, "forced"),
        ("A", LEADING, "paragraph"),
    ]


def test_box_first_baseline_is_d_ascent(fonts: FontMap) -> None:
    lay = layout(model("I\r", box=(0.0, 0.0, 100.0, 50.0)), fonts)
    assert ys(lay) == pytest.approx([ASCENT_D * SIZE])
    assert lay.lines[0].y == pytest.approx(7.5)
    # BoxBounds.T moves the top.
    lay = layout(model("I\r", box=(0.0, 2.0, 100.0, 50.0)), fonts)
    assert ys(lay) == pytest.approx([9.5])


def test_absolute_frame(fonts: FontMap) -> None:
    m = model("I\r", box=(14.0, 1364.0, 1040.0, 1873.0))
    assert frame(m) == pytest.approx((0.0, 0.0, 1026.0, 509.0))
    lay = layout(m, fonts)
    assert (lay.glyphs[0].x, lay.glyphs[0].y) == pytest.approx((0.0, 7.5))


@pytest.mark.parametrize(
    ("baseline", "vertical_scale", "shift"),
    [(2, 1.0, 3.33), (1, 1.0, -3.33), (2, 0.5, 1.665), (0, 0.5, 0.0)],
)
def test_font_baseline_shift(
    fonts: FontMap, baseline: int, vertical_scale: float, shift: float
) -> None:
    # The ascent uses the nominal size; the shift scales with VerticalScale.
    m = model(
        "I\r",
        style(baseline=baseline, vertical_scale=vertical_scale),
        box=(0, 0, 100, 50),
    )
    lay = layout(m, fonts)
    assert ys(lay) == pytest.approx([7.5 + shift])
    assert lay.lines[0].y == pytest.approx(7.5)
    em = SIZE * (0.583 if baseline else 1.0)
    assert (lay.glyphs[0].em_w, lay.glyphs[0].em_h) == pytest.approx(
        (em, em * vertical_scale)
    )


def test_baseline_shift_is_up_and_unscaled(fonts: FontMap) -> None:
    m = model("I\r", style(baseline=2, baseline_shift=2.0, vertical_scale=0.5))
    lay = layout(m, fonts)
    assert ys(lay) == pytest.approx([1.665 - 2.0])


def test_sub_position_from_model(fonts: FontMap) -> None:
    lay = layout(model("I\r", style(baseline=2), sub_position=0.5, sub_size=0.5), fonts)
    assert ys(lay) == pytest.approx([5.0])
    assert lay.glyphs[0].em_w == pytest.approx(5.0)


def test_max_ascent_on_line_one(fonts: FontMap) -> None:
    # A larger run on line 1 sets the first baseline, a space included.
    m = model(
        "I I\r", style(), style(size=20.0), style(), cuts=(1, 2), box=(0, 0, 100, 50)
    )
    lay = layout(m, fonts)
    assert lay.lines[0].y == pytest.approx(ASCENT_D * 20.0)
    # A run only on line 2 does not.
    m = model("I\rI\r", style(), style(size=20.0), cuts=(2,), box=(0, 0, 100, 50))
    lay = layout(m, fonts)
    assert lay.lines[0].y == pytest.approx(7.5)


def test_max_leading_on_line(fonts: FontMap) -> None:
    # Line 2's runs: 20 and 12 -> pitch 20, whatever line 1 carried.
    m = model("I\rII\r", style(), style(leading=20.0), style(), cuts=(2, 3))
    lay = layout(m, fonts)
    assert [line.y for line in lay.lines] == pytest.approx([0.0, 20.0])
    m = model("I\rII\r", style(leading=30.0), style(), cuts=(2,))
    lay = layout(m, fonts)
    assert [line.y for line in lay.lines] == pytest.approx([0.0, 12.0])


def test_auto_leading(fonts: FontMap) -> None:
    m = model("I\rI\r", style(auto_leading=True, leading=99.0))
    assert [line.y for line in layout(m, fonts).lines] == pytest.approx([0.0, 12.0])
    # Nominal size, also for subscript runs.
    m = model("I\rI\r", style(auto_leading=True, baseline=2))
    assert [line.y for line in layout(m, fonts).lines] == pytest.approx([0.0, 12.0])
    m = model("I\rI\r", style(auto_leading=True), auto_leading=1.5)
    assert [line.y for line in layout(m, fonts).lines] == pytest.approx([0.0, 15.0])


def test_empty_leading_lines_take_their_pitch(fonts: FontMap) -> None:
    m = model("\r\rI\r", style(leading=30.0), style(), cuts=(1,), box=(0, 0, 100, 200))
    lay = layout(m, fonts)
    # The first line, empty or not, sits one ascent of its own run below the
    # top; each later line adds the pitch of its terminator's run.
    assert [line.text for line in lay.lines] == ["", "", "I"]
    assert [line.y for line in lay.lines] == pytest.approx([7.5, 19.5, 31.5])
    assert ys(lay) == pytest.approx([31.5])


def test_empty_first_line_of_another_size(fonts: FontMap) -> None:
    # An empty first paragraph in a smaller size: its own ascent, then the
    # content line one leading below, not the leading plus the content's ascent.
    m = model(
        "\rI\r",
        style(size=20.0, leading=32.0),
        style(size=30.0, leading=32.0),
        cuts=(1,),
        box=(0, 0, 200, 200),
    )
    lay = layout(m, fonts)
    assert [line.y for line in lay.lines] == pytest.approx([15.0, 47.0])


def test_interior_empty_paragraph(fonts: FontMap) -> None:
    lay = layout(model("I\r\rI\r", box=(0, 0, 100, 200)), fonts)
    assert [line.y for line in lay.lines] == pytest.approx([7.5, 19.5, 31.5])
    assert ys(lay) == pytest.approx([7.5, 31.5])


def test_forced_break_pieces(fonts: FontMap) -> None:
    # The empty remainder after a final forced break is not a line; an
    # empty piece between two forced breaks is.
    assert [(p.start, p.end) for p in pieces(model("I\x03\r"))] == [(0, 1)]
    assert [(p.start, p.end) for p in pieces(model("I\x03\x03I\r"))] == [
        (0, 1),
        (2, 2),
        (3, 4),
    ]
    assert [(p.start, p.end) for p in pieces(model("\r"))] == [(0, 0)]
    assert [(p.start, p.end) for p in pieces(model("A\rB"))] == [(0, 1), (2, 3)]
    lay = layout(model("I\x03\x03I\r", box=(0, 0, 100, 200)), fonts)
    assert [line.y for line in lay.lines] == pytest.approx([7.5, 19.5, 31.5])


def test_mega_titan_arithmetic(fonts: FontMap) -> None:
    # 'Mega' 55.921 pt, forced break, 'Titan' 41.259 pt, Leading 30.518,
    # both subscript: the glyph baselines are Leading + sh(2) - sh(1) apart.
    big = style(size=55.921, baseline=2, leading=30.518)
    small = style(size=41.259, baseline=2, leading=30.518)
    lay = layout(model("I\x03I\r", big, small, cuts=(2,)), fonts)
    assert [line.y for line in lay.lines] == pytest.approx([0.0, 30.518])
    assert ys(lay)[1] - ys(lay)[0] == pytest.approx(25.636, abs=1e-3)
    assert ys(lay)[0] == pytest.approx(0.333 * 55.921)


def test_box_alignment_excludes_hanging_spaces(fonts: FontMap) -> None:
    box = (0.0, 0.0, 100.0, 50.0)
    for justification, x0 in ((0, 0.0), (2, 45.5), (1, 91.0)):
        lay = layout(model("IA  \r", box=box, justification=justification), fonts)
        assert len(lay.glyphs) == 2
        assert xs(lay)[0] == pytest.approx(x0)
        assert lay.lines[0].width == pytest.approx(9.0)
        assert lay.lines[0].text == "IA"


def test_box_wrap_and_stretch(fonts: FontMap) -> None:
    text = "I I I I I I I I I I\r"
    lay = layout(model(text, box=(0.0, 0.0, 21.0, 100.0), justification=3), fonts)
    assert [line.text for line in lay.lines] == ["I I I I I", "I I I I I"]
    assert [line.decision for line in lay.lines] == ["fit", "paragraph"]
    assert [line.stretch for line in lay.lines] == pytest.approx([0.25, 0.0])
    assert [line.n_spaces for line in lay.lines] == [4, 4]
    # Glyphs alternate I, space, I ...; each space grows by the stretch.
    pitch = adv("I") + adv("space")
    assert xs(lay)[0:9:2] == pytest.approx([k * (pitch + 0.25) for k in range(5)])
    assert xs(lay)[9::2] == pytest.approx([k * pitch for k in range(5)])
    assert [line.y for line in lay.lines] == pytest.approx([7.5, 19.5])


def test_justify_last_centre(fonts: FontMap) -> None:
    lay = layout(model("I I\r", box=(0.0, 0.0, 100.0, 50.0), justification=5), fonts)
    assert lay.lines[0].stretch == 0.0
    assert lay.lines[0].x0 == pytest.approx((100.0 - 6.5) / 2)


def test_overflow_cut(fonts: FontMap) -> None:
    # Lines are drawn while the previous baseline lies inside the box.
    lay = layout(model("I\rI\rI\rI\r", box=(0.0, 0.0, 100.0, 20.0)), fonts)
    assert [line.y for line in lay.lines] == pytest.approx([7.5, 19.5, 31.5, 43.5])
    assert ys(lay) == pytest.approx([7.5, 19.5, 31.5])
    assert "overflow" in lay.warnings
    lay = layout(model("I\rI\r", box=(0.0, 0.0, 100.0, 20.0)), fonts)
    assert "overflow" not in lay.warnings


def test_tracking_and_pen(fonts: FontMap) -> None:
    lay = layout(model("IA\r", style(tracking=100.0), justification=2), fonts)
    width = adv("I") + 1.0 + adv("A")
    assert lay.lines[0].width == pytest.approx(width)
    assert xs(lay) == pytest.approx([-width / 2, -width / 2 + adv("I") + 1.0])


def test_horizontal_scale_em(fonts: FontMap) -> None:
    lay = layout(model("A\r", style(horizontal_scale=0.5)), fonts)
    assert (lay.glyphs[0].em_w, lay.glyphs[0].em_h) == pytest.approx((5.0, 10.0))


def test_placed_glyph_fields(fonts: FontMap) -> None:
    red = (1.0, 0.0, 0.0)
    m = model(
        "IA\r", style(), style(fill=red, faux_bold=True, faux_italic=True), cuts=(1,)
    )
    lay = layout(m, fonts)
    g = lay.glyphs[1]
    assert (g.fill, g.faux_bold, g.faux_italic, g.run) == (red, True, True, 1)
    assert lay.glyphs[0].run == 0
    assert g.face is lay.glyphs[0].face
    assert g.gid == g.face.ft.get_char_index(ord("A"))


def test_underline_rule(fonts: FontMap) -> None:
    lay = layout(model("IA I\r", style(underline=True), style(), cuts=(2,)), fonts)
    assert len(lay.rules) == 1
    rule = lay.rules[0]
    position = (UNDERLINE_POSITION - UNDERLINE_THICKNESS // 2) / UPEM * SIZE
    assert (rule.x0, rule.x1) == pytest.approx((0.0, 9.0))
    assert rule.y == pytest.approx(-position)
    assert rule.thickness == pytest.approx(UNDERLINE_THICKNESS / UPEM * SIZE)
    assert rule.fill == (0.0, 0.0, 0.0)


def test_strikethrough_rule(fonts: FontMap) -> None:
    lay = layout(model("I\r", style(strikethrough=True)), fonts)
    assert len(lay.rules) == 1
    assert lay.rules[0].y == pytest.approx(-ASCENT_D * SIZE / 2)


def test_fonts_and_warnings(fonts: FontMap) -> None:
    m = model("I\r", style(font="psdtoolsfixture-tt"), warnings=("warp",))
    lay = layout(m, fonts)
    assert set(lay.fonts) == {"psdtoolsfixture-tt"}
    assert lay.fonts["psdtoolsfixture-tt"].how == "loose"
    assert lay.warnings == ("warp", "font_loose")


def test_ascent_fallback_warning(make_font: Callable[..., str]) -> None:
    path = make_font("no-d.ttf", omit=("d",))
    fonts = FontMap(dirs=[os.path.dirname(path)])
    lay = layout(model("I\r", box=(0, 0, 100, 50)), fonts)
    assert lay.lines[0].y == pytest.approx(GLYPHS["h"][1][3] / UPEM * SIZE)  # type: ignore[index]
    assert "ascent_fallback_h" in lay.warnings
    # Point text never reads the ascent.
    assert "ascent_fallback_h" not in layout(model("I\r"), fonts).warnings


def test_missing_font_policies(
    monkeypatch: pytest.MonkeyPatch, fixture_fonts_dir: str
) -> None:
    monkeypatch.setattr(fonts_module, "platform_font_dirs", lambda: [])
    m = model("IA\r", style(), style(font="NoSuchFont"), cuts=(1,), box=(0, 0, 100, 50))
    with pytest.raises(MissingFontError) as info:
        layout(m, FontMap(dirs=[fixture_fonts_dir]))
    assert info.value.names == ("NoSuchFont",)
    lay = layout(m, FontMap(dirs=[fixture_fonts_dir], policy="skip"))
    assert len(lay.glyphs) == 1
    assert lay.fonts["NoSuchFont"].face is None
    assert lay.lines[0].width == pytest.approx(adv("I"))
    lay = layout(
        m, FontMap(dirs=[fixture_fonts_dir], policy="substitute", fallback=PS_TT)
    )
    assert len(lay.glyphs) == 2
    assert "font_substitute" in lay.warnings


def test_breaks_override(fonts: FontMap) -> None:
    lay = layout(model("IA IA\r", box=(0.0, 0.0, 100.0, 50.0)), fonts, breaks=[3])
    assert [line.text for line in lay.lines] == ["IA", "IA"]
    assert [line.y for line in lay.lines] == pytest.approx([7.5, 19.5])
    assert len(lay.glyphs) == 4


def test_empty_text(fonts: FontMap) -> None:
    lay = layout(model(""), fonts)
    assert lay.glyphs == ()
    assert [line.text for line in lay.lines] == [""]
