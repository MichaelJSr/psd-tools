import os
import shutil
from collections.abc import Callable

import pytest

freetype = pytest.importorskip("freetype")
hb = pytest.importorskip("uharfbuzz")

from psd_tools.composite import _compat  # noqa: E402
from psd_tools.composite.text import fonts as fonts_module  # noqa: E402
from psd_tools.composite.text.fonts import (  # noqa: E402
    FontMap,
    MissingFontError,
    Resolved,
    load_face,
    norm,
    platform_font_dirs,
)

from .conftest import (  # noqa: E402
    ASCENT,
    GLYPHS,
    PS_CFF,
    PS_TT,
    UNDERLINE_POSITION,
    UNDERLINE_THICKNESS,
    UPEM,
)


def _platform_face() -> tuple[str, int, str] | None:
    """``(path, index, postscript_name)`` of some installed face, or ``None``."""
    for d in platform_font_dirs():
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            path = os.path.join(d, fn)
            if fn.lower().endswith((".ttf", ".otf", ".ttc")):
                try:
                    face = freetype.Face(path)
                except freetype.FT_Exception:
                    continue
                if face.postscript_name:
                    return path, 0, face.postscript_name.decode("latin-1")
    return None


def test_compat_flag() -> None:
    assert _compat.HAS_TEXT

    @_compat.require_text
    def f() -> int:
        return 1

    assert f() == 1


def test_norm() -> None:
    assert norm("Kung-Fu Panda_Regular") == "kungfupandaregular"


def test_exact_truetype(fixture_fonts_dir: str, ttf_font_path: str) -> None:
    r = FontMap(dirs=[fixture_fonts_dir]).resolve(PS_TT)
    assert r.how == "exact"
    assert r.path == ttf_font_path
    assert r.face is not None
    assert r.face.key == (ttf_font_path, 0)
    m = r.face.metrics
    assert m.upem == UPEM
    assert (m.ascent_units, m.ascent_glyph) == (GLYPHS["d"][1][3], "d")  # type: ignore[index]
    assert m.underline_position == UNDERLINE_POSITION - UNDERLINE_THICKNESS // 2
    assert m.underline_thickness == UNDERLINE_THICKNESS
    assert not m.has_svg
    assert m.strikes == frozenset()
    assert r.face.hb.scale == (UPEM, UPEM)
    assert r.face.ft.get_char_index(ord("A"))


def test_exact_cff(fixture_fonts_dir: str, cff_font_path: str) -> None:
    r = FontMap(dirs=[fixture_fonts_dir]).resolve(PS_CFF)
    assert r.how == "exact"
    assert r.path == cff_font_path
    assert r.face is not None
    assert r.face.metrics.ascent_glyph == "d"


def test_ascent_fallbacks(make_font: Callable[..., str]) -> None:
    face = load_face(make_font("no-d.ttf", omit=("d",)))
    assert (face.metrics.ascent_units, face.metrics.ascent_glyph) == (
        GLYPHS["h"][1][3],  # type: ignore[index]
        "h",
    )
    face = load_face(make_font("no-dh.ttf", omit=("d", "h")))
    assert (face.metrics.ascent_units, face.metrics.ascent_glyph) == (
        ASCENT,
        "ascender",
    )


@pytest.mark.parametrize(
    "name",
    ["psdtoolsfixture-tt", "PSD Tools Fixture TT", "PSDToolsFixtureTT-Regular"],
)
def test_loose(fixture_fonts_dir: str, ttf_font_path: str, name: str) -> None:
    r = FontMap(dirs=[fixture_fonts_dir]).resolve(name)
    assert (r.how, r.path) == ("loose", ttf_font_path)


def test_first_directory_wins(
    fixture_fonts_dir: str, ttf_font_path: str, tmp_path: object
) -> None:
    other = os.path.join(str(tmp_path), "other")
    os.mkdir(other)
    shadow = os.path.join(other, "shadow.ttf")
    shutil.copy(ttf_font_path, shadow)
    assert FontMap(dirs=[other, fixture_fonts_dir]).resolve(PS_TT).path == shadow
    assert FontMap(dirs=[fixture_fonts_dir, other]).resolve(PS_TT).path == ttf_font_path


def test_env_dirs(monkeypatch: pytest.MonkeyPatch, fixture_fonts_dir: str) -> None:
    monkeypatch.setenv("PSD_TOOLS_FONT_DIRS", fixture_fonts_dir)
    fm = FontMap()
    assert fm.dirs == (fixture_fonts_dir,)
    assert fm.resolve(PS_TT).how == "exact"


def test_system() -> None:
    hit = _platform_face()
    if hit is None:
        pytest.skip("no platform font found")
    path, index, name = hit
    fm = FontMap()
    assert fm.find(name) == (path, index, "system")
    assert fm.resolve(name).how == "system"


def test_ttc_indexing() -> None:
    ttcs = [
        os.path.join(d, fn)
        for d in platform_font_dirs()
        if os.path.isdir(d)
        for fn in os.listdir(d)
        if fn.lower().endswith(".ttc")
        and freetype.Face(os.path.join(d, fn)).num_faces > 1
    ]
    if not ttcs:
        pytest.skip("no font collection installed")
    path = ttcs[0]
    name = freetype.Face(path, 1).postscript_name.decode("latin-1")
    found = FontMap().find(name)
    assert found is not None
    assert found[0].lower().endswith(".ttc")
    face = load_face(found[0], found[1])
    assert face.ft.postscript_name.decode("latin-1") == name


def test_aliases(
    monkeypatch: pytest.MonkeyPatch, fixture_fonts_dir: str, ttf_font_path: str
) -> None:
    monkeypatch.setattr(fonts_module, "platform_font_dirs", lambda: [])
    fm = FontMap(aliases={"ArialMT": ttf_font_path})
    r = fm.resolve("ArialMT")
    assert (r.how, r.path) == ("exact", ttf_font_path)
    fm = FontMap(dirs=[fixture_fonts_dir], aliases={"ArialMT": PS_CFF})
    assert fm.resolve("ArialMT").how == "exact"
    assert fm.resolve("ArialMT").path is not None
    assert fm.resolve("ArialMT").path.endswith(".otf")  # type: ignore[union-attr]


def test_policy_raise(monkeypatch: pytest.MonkeyPatch, fixture_fonts_dir: str) -> None:
    monkeypatch.setattr(fonts_module, "platform_font_dirs", lambda: [])
    fm = FontMap(dirs=[fixture_fonts_dir])
    with pytest.raises(MissingFontError) as info:
        fm.resolve("NoSuchFont")
    assert info.value.names == ("NoSuchFont",)
    with pytest.raises(MissingFontError) as info:
        fm.resolve_all(["NoSuchFont", PS_TT, "Other"])
    assert info.value.names == ("NoSuchFont", "Other")


def test_policy_skip(monkeypatch: pytest.MonkeyPatch, fixture_fonts_dir: str) -> None:
    monkeypatch.setattr(fonts_module, "platform_font_dirs", lambda: [])
    fm = FontMap(dirs=[fixture_fonts_dir], policy="skip")
    r = fm.resolve("NoSuchFont")
    assert isinstance(r, Resolved)
    assert (r.face, r.how, r.path) == (None, "missing", None)
    assert set(fm.resolve_all(["NoSuchFont", PS_TT])) == {"NoSuchFont", PS_TT}


def test_policy_substitute(
    monkeypatch: pytest.MonkeyPatch, fixture_fonts_dir: str, ttf_font_path: str
) -> None:
    monkeypatch.setattr(fonts_module, "platform_font_dirs", lambda: [])
    fm = FontMap(dirs=[fixture_fonts_dir], policy="substitute", fallback=PS_TT)
    r = fm.resolve("NoSuchFont")
    assert (r.how, r.path) == ("substitute", ttf_font_path)
    fm = FontMap(dirs=[fixture_fonts_dir], policy="substitute", fallback="AlsoMissing")
    with pytest.raises(MissingFontError) as info:
        fm.resolve("NoSuchFont")
    assert info.value.names == ("NoSuchFont", "AlsoMissing")


def test_face_cache(fixture_fonts_dir: str) -> None:
    fm = FontMap(dirs=[fixture_fonts_dir])
    assert fm.resolve(PS_TT).face is fm.resolve(PS_TT).face


def test_real_font(luckiest_guy_path: str) -> None:
    fm = FontMap(dirs=[os.path.dirname(luckiest_guy_path)])
    r = fm.resolve("LuckiestGuy-Regular")
    assert (r.how, r.path) == ("exact", luckiest_guy_path)
    assert r.face is not None
    assert r.face.metrics.upem == 2048
    assert r.face.metrics.ascent_glyph == "d"
