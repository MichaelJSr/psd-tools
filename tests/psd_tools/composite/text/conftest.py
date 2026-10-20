"""Fixture fonts for the text engine tests.

Two synthetic fonts (TrueType and CFF) with rectangle glyphs, so coverage is
hand-computable, plus ``kern`` and ``liga`` features. Every test under this
directory carries the ``text`` marker; test modules ``importorskip`` the
optional libraries they use.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from typing import Any

import pytest

UPEM = 1000
PS_TT = "PsdToolsFixture-TT"
PS_CFF = "PsdToolsFixture-CFF"
FAMILY = "PSD Tools Fixture"
ASCENT = 800
DESCENT = -200
UNDERLINE_POSITION = -100
UNDERLINE_THICKNESS = 50
KERN_A_V = -80

#: glyph -> (advance, rectangle (x0, y0, x1, y1) or None)
GLYPHS: dict[str, tuple[int, tuple[int, int, int, int] | None]] = {
    ".notdef": (500, None),
    "space": (250, None),
    "I": (200, (0, 0, 100, 700)),
    "A": (700, (50, 0, 650, 700)),
    "V": (700, (50, 0, 650, 700)),
    "f": (300, (0, 0, 100, 700)),
    "i": (200, (0, 0, 100, 500)),
    "d": (600, (0, 0, 500, 750)),
    "h": (600, (0, 0, 500, 720)),
    "f_i": (500, (0, 0, 400, 700)),
}
CMAP = {ord(c): c for c in "IAVfidh"} | {ord(" "): "space"}
FEATURES = f"""
feature kern {{ pos A V {KERN_A_V}; }} kern;
feature liga {{ sub f i by f_i; }} liga;
"""

FONTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "fonts",
)
LUCKIEST_GUY = os.path.join(FONTS_DIR, "LuckiestGuy-Regular.ttf")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    here = os.path.dirname(__file__)
    for item in items:
        if str(item.fspath).startswith(here):
            item.add_marker(pytest.mark.text)


def _draw(pen: Any, rect: tuple[int, int, int, int] | None) -> None:
    if rect is None:
        return
    x0, y0, x1, y1 = rect
    pen.moveTo((x0, y0))
    pen.lineTo((x0, y1))
    pen.lineTo((x1, y1))
    pen.lineTo((x1, y0))
    pen.closePath()


def build_font(path: str, *, cff: bool, omit: Iterable[str] = ()) -> str:
    """Write a fixture font to ``path``; ``omit`` drops glyphs by name."""
    fontBuilder = pytest.importorskip("fontTools.fontBuilder")
    feaLib = pytest.importorskip("fontTools.feaLib.builder")
    names = [g for g in GLYPHS if g not in set(omit)]
    ps_name, style = (PS_CFF, "CFF") if cff else (PS_TT, "TT")
    fb = fontBuilder.FontBuilder(UPEM, isTTF=not cff)
    fb.setupGlyphOrder(names)
    fb.setupCharacterMap({cp: g for cp, g in CMAP.items() if g in names})
    metrics = {}
    if cff:
        from fontTools.pens.t2CharStringPen import T2CharStringPen  # noqa: PLC0415

        charstrings = {}
        for g in names:
            adv, rect = GLYPHS[g]
            pen = T2CharStringPen(adv, None)
            _draw(pen, rect)
            charstrings[g] = pen.getCharString()
            metrics[g] = (adv, rect[0] if rect else 0)
        fb.setupCFF(ps_name, {"FullName": ps_name}, charstrings, {})
    else:
        from fontTools.pens.ttGlyphPen import TTGlyphPen  # noqa: PLC0415

        glyphs = {}
        for g in names:
            adv, rect = GLYPHS[g]
            pen = TTGlyphPen(None)
            _draw(pen, rect)
            glyphs[g] = pen.glyph()
            metrics[g] = (adv, rect[0] if rect else 0)
        fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=ASCENT, descent=DESCENT)
    fb.setupNameTable({"familyName": FAMILY, "styleName": style, "psName": ps_name})
    fb.setupOS2(
        sTypoAscender=ASCENT,
        sTypoDescender=DESCENT,
        usWinAscent=ASCENT,
        usWinDescent=-DESCENT,
    )
    fb.setupPost(
        underlinePosition=UNDERLINE_POSITION, underlineThickness=UNDERLINE_THICKNESS
    )
    feaLib.addOpenTypeFeaturesFromString(fb.font, FEATURES)
    fb.save(path)
    return path


@pytest.fixture(autouse=True)
def _no_font_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PSD_TOOLS_FONT_DIRS", raising=False)


@pytest.fixture(scope="session")
def fixture_fonts_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    d = tmp_path_factory.mktemp("fixture-fonts")
    build_font(str(d / "PsdToolsFixture-TT.ttf"), cff=False)
    build_font(str(d / "PsdToolsFixture-CFF.otf"), cff=True)
    return str(d)


@pytest.fixture(scope="session")
def ttf_font_path(fixture_fonts_dir: str) -> str:
    return os.path.join(fixture_fonts_dir, "PsdToolsFixture-TT.ttf")


@pytest.fixture(scope="session")
def cff_font_path(fixture_fonts_dir: str) -> str:
    return os.path.join(fixture_fonts_dir, "PsdToolsFixture-CFF.otf")


@pytest.fixture
def make_font(tmp_path: Any) -> Callable[..., str]:
    """``make_font(name, cff=False, omit=())`` -> path of a fresh fixture font."""

    def make(name: str, *, cff: bool = False, omit: Iterable[str] = ()) -> str:
        return build_font(str(tmp_path / name), cff=cff, omit=omit)

    return make


@pytest.fixture(scope="session")
def luckiest_guy_path() -> str:
    return LUCKIEST_GUY
