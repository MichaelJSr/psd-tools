import os
from collections.abc import Callable
from typing import Any

import pytest

pytest.importorskip("freetype")
pytest.importorskip("uharfbuzz")

from psd_tools.composite.text import fonts as fonts_module
from psd_tools.composite.text.fonts import FontMap
from psd_tools.composite.text.model import Run, Style
from psd_tools.composite.text.shape import (
    run_index,
    shape_piece,
    shaping_key,
)

from .conftest import GLYPHS, KERN_A_V, PS_CFF, PS_TT, UPEM

SIZE = 10.0


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
        leading=12.0,
        auto_leading=False,
    )
    base.update(kwargs)
    return Style(**base)


def runs_for(text: str, *styles: Style, cuts: tuple[int, ...] = ()) -> list[Run]:
    """Runs over ``text`` split at ``cuts``, one style per span."""
    bounds = (0, *cuts, len(text))
    return [
        Run(s, e, st) for s, e, st in zip(bounds[:-1], bounds[1:], styles, strict=True)
    ]


def adv(glyph: str, size: float = SIZE) -> float:
    return GLYPHS[glyph][0] / UPEM * size


@pytest.fixture
def fonts(fixture_fonts_dir: str) -> FontMap:
    return FontMap(dirs=[fixture_fonts_dir])


def test_advances_at_em(fonts: FontMap) -> None:
    text = "IA"
    shaped = shape_piece(text, 0, 2, runs_for(text, style()), fonts)
    assert [g.adv for g in shaped.glyphs] == [adv("I"), adv("A")]
    assert [g.cluster for g in shaped.glyphs] == [0, 1]
    assert shaped.measure(0, 2) == pytest.approx(adv("I") + adv("A"))
    assert shaped.measure(1, 2) == pytest.approx(adv("A"))
    assert shaped.measure(2, 2) == 0.0
    assert shaped.glyphs[0].face.key[0].endswith(".ttf")


def test_subscript_em(fonts: FontMap) -> None:
    text = "I"
    shaped = shape_piece(text, 0, 1, runs_for(text, style(baseline=2)), fonts)
    assert shaped.glyphs[0].adv == pytest.approx(adv("I") * 0.583)
    shaped = shape_piece(
        text, 0, 1, runs_for(text, style(baseline=1)), fonts, sub_size=0.5
    )
    assert shaped.glyphs[0].adv == pytest.approx(adv("I") * 0.5)


def test_horizontal_scale(fonts: FontMap) -> None:
    text = "A"
    shaped = shape_piece(text, 0, 1, runs_for(text, style(horizontal_scale=0.5)), fonts)
    assert shaped.glyphs[0].adv == pytest.approx(adv("A") / 2)


def test_kerning(fonts: FontMap) -> None:
    text = "AV"
    shaped = shape_piece(text, 0, 2, runs_for(text, style()), fonts)
    kerned = adv("A") + KERN_A_V / UPEM * SIZE
    assert shaped.glyphs[0].adv == pytest.approx(kerned)
    assert shaped.measure(0, 2) == pytest.approx(kerned + adv("V"))


def test_auto_kerning_off_on_pair_end(fonts: FontMap) -> None:
    # The flag on 'V' removes the pair ending at it.
    text = "AV"
    runs = runs_for(text, style(), style(auto_kerning=False), cuts=(1,))
    shaped = shape_piece(text, 0, 2, runs, fonts)
    assert shaped.glyphs[0].adv == pytest.approx(adv("A"))
    # On the line's first character there is no pair: no effect.
    runs = runs_for(text, style(auto_kerning=False), style(), cuts=(1,))
    shaped = shape_piece(text, 0, 2, runs, fonts)
    assert shaped.glyphs[0].adv == pytest.approx(adv("A") + KERN_A_V / UPEM * SIZE)
    # Off for a whole run: no pair inside it is kerned.
    text = "AVAV"
    shaped = shape_piece(text, 0, 4, runs_for(text, style(auto_kerning=False)), fonts)
    assert [g.adv for g in shaped.glyphs] == [adv("A"), adv("V"), adv("A"), adv("V")]


def test_tracking_unit_and_width(fonts: FontMap) -> None:
    text = "IA"
    shaped = shape_piece(text, 0, 2, runs_for(text, style(tracking=100.0)), fonts)
    track = 100 / 1000 * SIZE
    assert [g.tracking for g in shaped.glyphs] == [track, track]
    # The pen advances by adv + tracking; the trailing tracking is not width.
    assert shaped.measure(0, 2) == pytest.approx(adv("I") + track + adv("A"))
    # Tracking is a fraction of the rendered em.
    shaped = shape_piece(
        text, 0, 2, runs_for(text, style(tracking=100.0, baseline=2)), fonts
    )
    assert shaped.glyphs[0].tracking == pytest.approx(track * 0.583)


def test_tracking_from_each_glyphs_run(fonts: FontMap) -> None:
    text = "IA"
    runs = runs_for(text, style(tracking=100.0), style(tracking=-50.0), cuts=(1,))
    shaped = shape_piece(text, 0, 2, runs, fonts)
    assert [g.tracking for g in shaped.glyphs] == pytest.approx([1.0, -0.5])
    assert [g.run for g in shaped.glyphs] == [0, 1]


def test_all_caps(fonts: FontMap) -> None:
    text = "i"
    shaped = shape_piece(text, 0, 1, runs_for(text, style(caps=2)), fonts)
    face = shaped.glyphs[0].face
    assert shaped.glyphs[0].gid == face.ft.get_char_index(ord("I"))
    plain = shape_piece(text, 0, 1, runs_for(text, style()), fonts)
    assert plain.glyphs[0].gid == face.ft.get_char_index(ord("i"))


def test_ligatures(fonts: FontMap) -> None:
    text = "fi"
    shaped = shape_piece(text, 0, 2, runs_for(text, style()), fonts)
    assert len(shaped.glyphs) == 1
    assert shaped.glyphs[0].adv == pytest.approx(adv("f_i"))
    assert shaped.glyphs[0].cluster == 0
    assert shaped.measure(0, 2) == pytest.approx(adv("f_i"))
    assert shaped.measure(0, 1) == pytest.approx(adv("f_i"))
    assert shaped.measure(1, 2) == 0.0
    off = shape_piece(text, 0, 2, runs_for(text, style(ligatures=False)), fonts)
    assert [g.adv for g in off.glyphs] == [adv("f"), adv("i")]


def test_faux_bold_widens(fonts: FontMap) -> None:
    text = "I"
    shaped = shape_piece(
        text, 0, 1, runs_for(text, style(size=30.0, faux_bold=True)), fonts
    )
    assert shaped.glyphs[0].adv == pytest.approx(adv("I", 30.0) + 30.0 / 30)


def test_segment_spans_colour_runs(fonts: FontMap) -> None:
    # Colour, tracking and underline do not cut a segment: the pair is kerned.
    text = "AV"
    runs = runs_for(
        text,
        style(fill=(1.0, 0.0, 0.0)),
        style(fill=(0.0, 0.0, 1.0), underline=True),
        cuts=(1,),
    )
    shaped = shape_piece(text, 0, 2, runs, fonts)
    assert shaped.glyphs[0].adv == pytest.approx(adv("A") + KERN_A_V / UPEM * SIZE)
    assert [g.run for g in shaped.glyphs] == [0, 1]
    assert shaping_key(runs[0].style) == shaping_key(runs[1].style)


def test_baseline_shift_cuts_kerning(fonts: FontMap) -> None:
    text = "AV"
    runs = runs_for(text, style(), style(baseline_shift=2.0), cuts=(1,))
    shaped = shape_piece(text, 0, 2, runs, fonts)
    assert shaped.glyphs[0].adv == pytest.approx(adv("A"))
    assert shaping_key(runs[0].style) != shaping_key(runs[1].style)


def test_size_change_cuts_segment(fonts: FontMap) -> None:
    text = "AV"
    runs = runs_for(text, style(), style(size=20.0), cuts=(1,))
    shaped = shape_piece(text, 0, 2, runs, fonts)
    assert [g.adv for g in shaped.glyphs] == [adv("A"), adv("V", 20.0)]


def test_two_fonts(fonts: FontMap) -> None:
    text = "II"
    runs = runs_for(text, style(), style(font=PS_CFF), cuts=(1,))
    shaped = shape_piece(text, 0, 2, runs, fonts)
    assert shaped.glyphs[0].face is not shaped.glyphs[1].face
    assert shaped.glyphs[1].face.key[0].endswith(".otf")
    assert shaped.measure(0, 2) == pytest.approx(2 * adv("I"))


def test_spaces_and_terminators(fonts: FontMap) -> None:
    text = "I I\tI\rI"
    shaped = shape_piece(text, 0, len(text), runs_for(text, style()), fonts)
    assert dict(shaped.space_adv) == {1: adv("space"), 3: adv("space")}
    assert [g.cluster for g in shaped.glyphs] == [0, 1, 2, 3, 4, 6]
    assert shaped.measure(0, len(text)) == pytest.approx(
        4 * adv("I") + 2 * adv("space")
    )
    assert shaped.measure(4, 6) == pytest.approx(adv("I"))


def test_piece_range(fonts: FontMap) -> None:
    text = "AV\rAV"
    shaped = shape_piece(text, 3, 5, runs_for(text, style()), fonts)
    assert [g.cluster for g in shaped.glyphs] == [3, 4]
    assert shaped.measure(0, 3) == 0.0
    assert shaped.measure(3, 5) == pytest.approx(
        adv("A") + KERN_A_V / UPEM * SIZE + adv("V")
    )


def test_faceless_run_is_skipped(
    monkeypatch: pytest.MonkeyPatch, fixture_fonts_dir: str
) -> None:
    monkeypatch.setattr(fonts_module, "platform_font_dirs", lambda: [])
    fonts = FontMap(dirs=[fixture_fonts_dir], policy="skip")
    text = "IA"
    runs = runs_for(text, style(), style(font="NoSuchFont"), cuts=(1,))
    shaped = shape_piece(text, 0, 2, runs, fonts)
    assert [g.cluster for g in shaped.glyphs] == [0]
    assert shaped.measure(0, 2) == pytest.approx(adv("I"))


def test_run_index() -> None:
    runs = runs_for("abcdef", style(), style(), style(), cuts=(2, 4))
    assert [run_index(runs, i) for i in range(7)] == [0, 0, 1, 1, 2, 2, 2]


def test_real_font(luckiest_guy_path: str) -> None:
    fonts = FontMap(dirs=[os.path.dirname(luckiest_guy_path)])
    text = "AV"
    shaped = shape_piece(
        text, 0, 2, runs_for(text, style(font="LuckiestGuy-Regular")), fonts
    )
    face = shaped.glyphs[0].face
    unkerned = (
        face.hb.get_glyph_h_advance(shaped.glyphs[0].gid) / face.metrics.upem * SIZE
    )
    assert shaped.glyphs[0].adv < unkerned


def test_make_font_adapter(make_font: Callable[..., str]) -> None:
    # A face without 'd' still shapes; its ascent falls back (layout warns).
    path = make_font("no-d.ttf", omit=("d",))
    fonts = FontMap(dirs=[os.path.dirname(path)])
    text = "I"
    shaped = shape_piece(text, 0, 1, runs_for(text, style()), fonts)
    assert shaped.glyphs[0].face.metrics.ascent_glyph == "h"
