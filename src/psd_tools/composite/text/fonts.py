"""PostScript-name font lookup and per-face metrics.

A :class:`FontMap` indexes every ``.ttf``/``.otf``/``.ttc``/``.otc`` face in
its directories by PostScript name and resolves the names a type layer uses.
FreeType supplies the names and unscaled metrics; HarfBuzz shapes.
"""

from __future__ import annotations

import os
import re
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from ctypes import byref, c_ulong
from dataclasses import dataclass
from typing import Literal

import freetype
import uharfbuzz as hb

from psd_tools.composite.text.constants import ASCENT_GLYPHS, FACE_CACHE_SIZE

How = Literal["exact", "loose", "system", "substitute", "missing"]
Policy = Literal["raise", "skip", "substitute"]

FONT_DIRS_ENV = "PSD_TOOLS_FONT_DIRS"
FONT_EXTENSIONS = (".ttf", ".otf", ".ttc", ".otc")
DEFAULT_FALLBACK = "MyriadPro-Regular"


class TextRenderError(Exception):
    """The text engine cannot render this layer."""


class MissingFontError(TextRenderError):
    """No face resolves one or more of the layer's PostScript names."""

    def __init__(self, names: Iterable[str]):
        self.names = tuple(names)
        super().__init__("missing font(s): " + ", ".join(self.names))


@dataclass(frozen=True)
class FontMetrics:
    """Unscaled metrics of one face, in font units unless noted."""

    upem: int
    #: ``horiBearingY`` of the first of ``ASCENT_GLYPHS`` the face has, else
    #: ``face.ascender``; ``ascent_glyph`` names the source (``"ascender"``
    #: for the final fallback).
    ascent_units: int
    ascent_glyph: str
    #: Centre of the underline relative to the baseline, as FreeType reports
    #: it: ``post``'s top edge minus half the thickness.
    underline_position: int
    underline_thickness: int
    #: The face carries an ``SVG `` table; its outlines render as silhouettes.
    has_svg: bool
    #: ``y_ppem`` of each embedded bitmap strike.
    strikes: frozenset[int]


@dataclass(frozen=True, eq=False)
class Face:
    """An open face: HarfBuzz font, FreeType face, metrics and file identity."""

    hb: hb.Font
    ft: freetype.Face
    metrics: FontMetrics
    key: tuple[str, int]


@dataclass(frozen=True, eq=False)
class Resolved:
    """One PostScript name's lookup result.

    ``face`` is ``None`` only under the ``skip`` policy, with ``how`` set to
    ``"missing"``.
    """

    face: Face | None
    how: How

    @property
    def path(self) -> str | None:
        """The resolved font file, or ``None`` when missing."""
        return None if self.face is None else self.face.key[0]


def norm(name: str) -> str:
    """Casefolded alphanumerics, the key for a loose match."""
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def _loose_keys(name: str) -> tuple[str, ...]:
    base = re.sub(r"-?Regular$", "", name)
    keys = (norm(name), norm(base))
    return keys if keys[1] != keys[0] else keys[:1]


def platform_font_dirs() -> list[str]:
    """The operating system's font directories, in search order."""
    if os.name == "nt":
        return [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")]
    home = os.path.expanduser("~")
    if os.uname().sysname == "Darwin":
        return [
            "/System/Library/Fonts",
            "/System/Library/Fonts/Supplemental",
            "/Library/Fonts",
            os.path.join(home, "Library", "Fonts"),
        ]
    return ["/usr/share/fonts", os.path.join(home, ".fonts")]


def _env_dirs() -> list[str]:
    value = os.environ.get(FONT_DIRS_ENV, "")
    return [d for d in value.split(os.pathsep) if d]


def _metrics(face: freetype.Face) -> FontMetrics:
    ascent, glyph = int(face.ascender), "ascender"
    for g in ASCENT_GLYPHS:
        if face.get_char_index(ord(g)):
            face.load_char(g, freetype.FT_LOAD_NO_SCALE)
            ascent, glyph = int(face.glyph.metrics.horiBearingY), g
            break
    length = c_ulong(0)
    tag = (ord("S") << 24) | (ord("V") << 16) | (ord("G") << 8) | ord(" ")
    has_svg = (
        freetype.FT_Load_Sfnt_Table(face._FT_Face, tag, 0, None, byref(length)) == 0
    )
    return FontMetrics(
        upem=int(face.units_per_EM),
        ascent_units=ascent,
        ascent_glyph=glyph,
        underline_position=int(face.underline_position),
        underline_thickness=int(face.underline_thickness),
        has_svg=has_svg,
        strikes=frozenset(round(s.y_ppem / 64) for s in face.available_sizes),
    )


def load_face(path: str, index: int = 0) -> Face:
    """Open one face with FreeType and HarfBuzz and read its metrics."""
    ft = freetype.Face(path, index)
    metrics = _metrics(ft)
    blob = hb.Blob.from_file_path(path)
    font = hb.Font(hb.Face(blob, index))
    font.scale = (metrics.upem, metrics.upem)
    return Face(hb=font, ft=ft, metrics=metrics, key=(path, index))


class _Index:
    """Faces under a list of directories, by exact and by loose name."""

    def __init__(self, dirs: Iterable[str]):
        self.exact: dict[str, tuple[str, int]] = {}
        self.loose: dict[str, tuple[str, int]] = {}
        for d in dirs:
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                path = os.path.join(d, fn)
                if not fn.lower().endswith(FONT_EXTENSIONS) or not os.path.isfile(path):
                    continue
                try:
                    self._add(path)
                except freetype.FT_Exception:
                    continue

    def _add(self, path: str) -> None:
        first = freetype.Face(path)
        for i in range(first.num_faces):
            face = first if i == 0 else freetype.Face(path, i)
            ps = (face.postscript_name or b"").decode("latin-1")
            family = (face.family_name or b"").decode("latin-1")
            style = (face.style_name or b"").decode("latin-1")
            if ps:
                self.exact.setdefault(ps, (path, i))
            for name in (ps, family + style):
                if name:
                    self.loose.setdefault(norm(name), (path, i))


class FontMap:
    """Resolves PostScript names to faces.

    Search order: ``dirs``, then ``PSD_TOOLS_FONT_DIRS``, then the platform
    directories. An exact PostScript name is tried before a loose match in
    the user directories, then both again in the platform directories;
    the first face in directory order wins. ``aliases`` map a name to a font
    file path or to another name and override everything.

    ``policy`` decides a name no face matches: ``"raise"`` raises
    :class:`MissingFontError`, ``"skip"`` resolves to a faceless
    :class:`Resolved`, ``"substitute"`` resolves the ``fallback`` name.
    """

    def __init__(
        self,
        dirs: Iterable[str | os.PathLike[str]] | None = None,
        *,
        aliases: Mapping[str, str | os.PathLike[str]] | None = None,
        policy: Policy = "raise",
        fallback: str = DEFAULT_FALLBACK,
    ):
        self.dirs = tuple(os.fspath(d) for d in dirs or ()) + tuple(_env_dirs())
        self.platform_dirs = tuple(platform_font_dirs())
        self.aliases = {k: os.fspath(v) for k, v in (aliases or {}).items()}
        self.policy: Policy = policy
        self.fallback = fallback
        self._user: _Index | None = None
        self._platform: _Index | None = None
        self._faces: OrderedDict[tuple[str, int], Face] = OrderedDict()

    def _indexes(self) -> tuple[_Index, _Index]:
        if self._user is None or self._platform is None:
            self._user = _Index(self.dirs)
            self._platform = _Index(self.platform_dirs)
        return self._user, self._platform

    def _face(self, key: tuple[str, int]) -> Face:
        face = self._faces.get(key)
        if face is None:
            face = load_face(*key)
            self._faces[key] = face
            while len(self._faces) > FACE_CACHE_SIZE:
                self._faces.popitem(last=False)
        else:
            self._faces.move_to_end(key)
        return face

    def _lookup(self, name: str) -> tuple[tuple[str, int], How] | None:
        alias = self.aliases.get(name)
        if alias is not None:
            if alias.lower().endswith(FONT_EXTENSIONS) or os.path.isfile(alias):
                return (alias, 0), "exact"
            name = alias
        user, platform = self._indexes()
        if name in user.exact:
            return user.exact[name], "exact"
        for key in _loose_keys(name):
            if key in user.loose:
                return user.loose[key], "loose"
        if name in platform.exact:
            return platform.exact[name], "system"
        for key in _loose_keys(name):
            if key in platform.loose:
                return platform.loose[key], "loose"
        return None

    def find(self, name: str) -> tuple[str, int, How] | None:
        """Where ``name`` would resolve: ``(path, index, how)`` or ``None``."""
        hit = self._lookup(name)
        return None if hit is None else (hit[0][0], hit[0][1], hit[1])

    def resolve(self, name: str) -> Resolved:
        """Resolve one PostScript name under this map's policy."""
        hit = self._lookup(name)
        if hit is not None:
            return Resolved(self._face(hit[0]), hit[1])
        if self.policy == "skip":
            return Resolved(None, "missing")
        if self.policy == "substitute":
            sub = self._lookup(self.fallback)
            if sub is not None:
                return Resolved(self._face(sub[0]), "substitute")
            raise MissingFontError((name, self.fallback))
        raise MissingFontError((name,))

    def resolve_all(self, names: Iterable[str]) -> dict[str, Resolved]:
        """Resolve every name; under ``raise`` the error lists all misses."""
        out: dict[str, Resolved] = {}
        missing: list[str] = []
        for name in dict.fromkeys(names):
            try:
                out[name] = self.resolve(name)
            except MissingFontError as ex:
                missing.extend(ex.names)
        if missing:
            raise MissingFontError(dict.fromkeys(missing))
        return out
