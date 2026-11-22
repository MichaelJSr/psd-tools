import os
import shutil
import time
from collections import OrderedDict
from collections.abc import Callable

import pytest

freetype = pytest.importorskip("freetype")
hb = pytest.importorskip("uharfbuzz")

from psd_tools.composite.text import fonts as fonts_module  # noqa: E402
from psd_tools.composite.text.constants import ASCENT_PPEM  # noqa: E402
from psd_tools.composite.text.fonts import (  # noqa: E402
    FontMap,
    MissingFontError,
    font_files,
    load_face,
    norm,
)

from .conftest import (  # noqa: E402
    ASCENT,
    CVT_TOP,
    D_ROUND,
    D_TO_CVT,
    GLYPHS,
    PS_CFF,
    PS_TT,
    UNDERLINE_POSITION,
    UNDERLINE_THICKNESS,
    UPEM,
)


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
    # Midway from post's top edge to the box bottom, at least half a thickness
    # below it.
    box = min(r[1] for _, r in GLYPHS.values() if r)
    bottom = min(box, UNDERLINE_POSITION - UNDERLINE_THICKNESS / 2)
    assert m.underline_position == (UNDERLINE_POSITION + bottom) / 2
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


def test_cff_underline(cff_font_path: str, tmp_path: str) -> None:
    # CFF: the Top DICT's values, not post's, scaled by FontMatrix; the box
    # bottom is FontBBox's.
    ttLib = pytest.importorskip("fontTools.ttLib")
    font = ttLib.TTFont(cff_font_path, recalcBBoxes=False)
    top = font["CFF "].cff.topDictIndex[0]
    top.FontMatrix = [0.0005, 0, 0, 0.0005, 0, 0]
    top.UnderlinePosition, top.UnderlineThickness = -150, 60.5
    top.FontBBox = [0, -400, 1300, 1500]
    font["post"].underlinePosition = -75
    path = os.path.join(tmp_path, "top.otf")
    font.save(path)
    m = load_face(path).metrics
    # Top -75, thickness 30.25, box bottom -200.
    assert m.underline_position == pytest.approx((-75 - 200) / 2)
    assert m.underline_thickness == pytest.approx(30.25)


def test_underline_fallbacks(ttf_font_path: str, tmp_path: str) -> None:
    # No thickness: 0.05 em; no top edge either: -0.15 em.
    ttLib = pytest.importorskip("fontTools.ttLib")
    font = ttLib.TTFont(ttf_font_path)
    font["post"].underlinePosition = font["post"].underlineThickness = 0
    path = os.path.join(tmp_path, "zero.ttf")
    font.save(path)
    m = load_face(path).metrics
    assert m.underline_position == pytest.approx((-150 - 175) / 2)
    assert m.underline_thickness == pytest.approx(50)


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


def gid(face: fonts_module.Face, char: str) -> int:
    return face.ft.get_char_index(ord(char))


def test_ascent_runs_truetype_instructions(make_font: Callable[..., str]) -> None:
    face = load_face(make_font("cvt.ttf", programs={"d": D_TO_CVT}, cvt=[CVT_TOP]))
    assert face.metrics.truetype
    assert (face.metrics.ascent_units, face.metrics.ascent_glyph) == (CVT_TOP, "d")


def test_ascent_rounds_at_ascent_ppem(make_font: Callable[..., str]) -> None:
    upem = 2048
    face = load_face(make_font("round.ttf", upem=upem, programs={"d": D_ROUND}))
    top = GLYPHS["d"][1][3]  # type: ignore[index]
    rounded = round(top * ASCENT_PPEM / upem) * upem / ASCENT_PPEM
    assert face.metrics.ascent_units == pytest.approx(rounded)
    # At one pixel per unit the grid is the unit: the outline stays put.
    assert face.instructed(gid(face, "d")) is None


def test_instructed_points(make_font: Callable[..., str]) -> None:
    face = load_face(make_font("cvt.ttf", programs={"d": D_TO_CVT}, cvt=[CVT_TOP]))
    points = face.instructed(gid(face, "d"))
    assert points is not None
    x0, y0, x1, _ = GLYPHS["d"][1]  # type: ignore[misc]
    assert points.tolist() == [[x0, y0], [x0, CVT_TOP], [x1, CVT_TOP], [x1, y0]]
    # A glyph its instructions leave alone draws its own outline.
    assert face.instructed(gid(face, "h")) is None


def test_instructions_start_afresh(make_font: Callable[..., str]) -> None:
    # A prep that acts on what an earlier run of it stored: each run starts
    # from the same state, so one glyph at one ppem has one outline.
    prep = ["PUSHB[ ]", "0", "RS[ ]", "IF[ ]", "PUSHB[ ]", "0", "0", "WCVTP[ ]"]
    prep += ["EIF[ ]", "PUSHB[ ]", "0", "1", "WS[ ]"]
    path = make_font("prep.ttf", programs={"d": D_TO_CVT}, cvt=[CVT_TOP], prep=prep)
    face = load_face(path)
    assert UPEM == ASCENT_PPEM
    first = fonts_module.instructed_outline(face.ft, gid(face, "d"), UPEM)
    # Hinted loads on the face's own size, as Sharp text makes.
    for ppem in (UPEM // 2, UPEM):
        face.ft.set_char_size(ppem * 64, ppem * 64, 72, 72)
        face.ft.load_glyph(gid(face, "d"), freetype.FT_LOAD_TARGET_LIGHT)
    again = fonts_module.instructed_outline(face.ft, gid(face, "d"), UPEM)
    assert first is not None and again is not None
    assert (first[0].tolist(), first[1]) == (again[0].tolist(), again[1])
    assert face.metrics.ascent_units == first[1]


def test_cff_draws_its_outline(cff_font_path: str, ttf_font_path: str) -> None:
    cff = load_face(cff_font_path)
    assert not cff.metrics.truetype
    # Whole units: FreeType's own outline, as drawn before exact points.
    assert cff.instructed(gid(cff, "d")) is None
    # A TrueType font without instructions keeps its outline too.
    tt = load_face(ttf_font_path)
    assert tt.metrics.truetype
    assert tt.instructed(gid(tt, "d")) is None


def test_cff_exact_points(make_font: Callable[..., str]) -> None:
    # FreeType floors a CFF point to a whole unit; its exact point is drawn.
    face = load_face(make_font("frac.otf", cff=True, shift={"d": 0.625}))
    x0, y0, x1, y1 = GLYPHS["d"][1]  # type: ignore[misc]
    a, b = x0 + 0.625, x1 + 0.625
    points = face.instructed(gid(face, "d"))
    assert points is not None
    assert points.tolist() == [[a, y0], [a, y1], [b, y1], [b, y0]]
    assert face.instructed(gid(face, "h")) is None


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


def test_index_shared_until_a_folder_changes(
    monkeypatch: pytest.MonkeyPatch,
    ttf_font_path: str,
    cff_font_path: str,
    tmp_path: object,
) -> None:
    monkeypatch.setattr(fonts_module, "_INDEXES", OrderedDict())
    root = os.path.join(str(tmp_path), "fonts")
    sub = os.path.join(root, "sub")
    os.makedirs(sub)
    font = shutil.copy(ttf_font_path, sub)

    def index() -> object:
        return FontMap(dirs=[root])._indexes()[0]

    # Written just now: maybe within one timestamp tick, so built again.
    assert index() is not index()
    monkeypatch.setattr(fonts_module, "_STAMP_SLACK_NS", 0)
    old = time.time() - 60
    os.utime(sub, (old, old))
    # A coarse clock (Windows before Python 3.13) can stamp these in the
    # build's own tick, which is racy even without slack: build after it.
    newest = max(
        max(os.stat(p).st_mtime_ns, os.stat(p).st_ctime_ns) for p in (root, sub, font)
    )
    for _ in range(1000):
        if time.time_ns() > newest:
            break
        time.sleep(0.001)
    first = index()
    assert index() is first
    shutil.copy(cff_font_path, sub)
    fm = FontMap(dirs=[root])
    assert fm.find(PS_CFF) == (
        os.path.join(sub, os.path.basename(cff_font_path)),
        0,
        "exact",
    )
    assert fm._indexes()[0] is not first


def test_env_dirs(monkeypatch: pytest.MonkeyPatch, fixture_fonts_dir: str) -> None:
    monkeypatch.setenv("PSD_TOOLS_FONT_DIRS", fixture_fonts_dir)
    fm = FontMap()
    assert fm.dirs == (fixture_fonts_dir,)
    assert fm.resolve(PS_TT).how == "exact"


def test_nested_directories(ttf_font_path: str, tmp_path: object) -> None:
    root = os.path.join(str(tmp_path), "fonts")
    nested = os.path.join(root, "truetype", "fixture")
    os.makedirs(nested)
    top = shutil.copy(ttf_font_path, os.path.join(root, "top.ttf"))
    deep = shutil.copy(ttf_font_path, os.path.join(nested, "deep.ttf"))
    os.mkdir(os.path.join(root, "not-a-font.ttf"))
    assert font_files(root) == [top, deep]
    assert FontMap(dirs=[root]).resolve(PS_TT).path == top
    shutil.rmtree(os.path.dirname(nested))
    os.remove(top)
    os.makedirs(nested)
    shutil.copy(ttf_font_path, deep)
    assert FontMap(dirs=[root]).resolve(PS_TT).path == deep
    assert font_files(os.path.join(root, "missing")) == []


def test_alias_to_a_missing_file(
    monkeypatch: pytest.MonkeyPatch, fixture_fonts_dir: str, tmp_path: object
) -> None:
    gone = os.path.join(str(tmp_path), "gone.ttf")
    fm = FontMap(dirs=[fixture_fonts_dir], aliases={"ArialMT": gone})
    assert fm.find("ArialMT") is None
    with pytest.raises(MissingFontError) as info:
        fm.resolve("ArialMT")
    assert info.value.names == ("ArialMT",)
    sub = FontMap(
        dirs=[fixture_fonts_dir],
        aliases={"ArialMT": gone},
        policy="substitute",
        fallback=PS_TT,
    )
    assert sub.resolve("ArialMT").how == "substitute"


def test_indexed_file_removed(
    monkeypatch: pytest.MonkeyPatch, ttf_font_path: str, tmp_path: object
) -> None:
    copy = shutil.copy(ttf_font_path, os.path.join(str(tmp_path), "copy.ttf"))
    fm = FontMap(dirs=[str(tmp_path)])
    assert fm.find(PS_TT) == (copy, 0, "exact")
    os.remove(copy)
    with pytest.raises(MissingFontError):
        fm.resolve(PS_TT)
    with pytest.raises(MissingFontError):
        FontMap(dirs=[str(tmp_path)], aliases={PS_TT: copy}).resolve(PS_TT)


def test_aliases(
    monkeypatch: pytest.MonkeyPatch, fixture_fonts_dir: str, ttf_font_path: str
) -> None:
    fm = FontMap(aliases={"ArialMT": ttf_font_path})
    r = fm.resolve("ArialMT")
    assert (r.how, r.path) == ("exact", ttf_font_path)
    fm = FontMap(dirs=[fixture_fonts_dir], aliases={"ArialMT": PS_CFF})
    assert fm.resolve("ArialMT").how == "exact"
    assert fm.resolve("ArialMT").path is not None
    assert fm.resolve("ArialMT").path.endswith(".otf")  # type: ignore[union-attr]


def test_policy_raise(monkeypatch: pytest.MonkeyPatch, fixture_fonts_dir: str) -> None:
    fm = FontMap(dirs=[fixture_fonts_dir])
    with pytest.raises(MissingFontError) as info:
        fm.resolve("NoSuchFont")
    assert info.value.names == ("NoSuchFont",)
    with pytest.raises(MissingFontError) as info:
        fm.resolve_all(["NoSuchFont", PS_TT, "Other"])
    assert info.value.names == ("NoSuchFont", "Other")


def test_policy_substitute(
    monkeypatch: pytest.MonkeyPatch, fixture_fonts_dir: str, ttf_font_path: str
) -> None:
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
