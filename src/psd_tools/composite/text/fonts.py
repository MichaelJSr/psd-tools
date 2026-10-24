"""PostScript-name font lookup and per-face metrics.

A :class:`FontMap` indexes every ``.ttf``/``.otf``/``.ttc``/``.otc`` face in
its directories by PostScript name and resolves the names a type layer uses.
FreeType supplies the names and metrics and runs TrueType instructions;
HarfBuzz shapes.
"""

from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from ctypes import POINTER, byref, c_ubyte, c_ulong, cast
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal

import freetype
import numpy as np
import uharfbuzz as hb

from psd_tools.composite.text.constants import (
    ASCENT_GLYPHS,
    ASCENT_PPEM,
    FACE_CACHE_SIZE,
)
from psd_tools.composite.text.svg import SvgTable

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
    #: The outline top of the first of ``ASCENT_GLYPHS`` the face has, as
    #: the font's instructions place it at ``ASCENT_PPEM`` for a TrueType
    #: face, else ``face.ascender``; ``ascent_glyph`` names the source
    #: (``"ascender"`` for the final fallback).
    ascent_units: float
    ascent_glyph: str
    #: Centre of the underline relative to the baseline, as FreeType reports
    #: it: ``post``'s top edge minus half the thickness.
    underline_position: int
    underline_thickness: int
    #: The face carries an ``SVG `` table of colour glyphs.
    has_svg: bool
    #: ``y_ppem`` of each embedded bitmap strike.
    strikes: frozenset[int]
    #: TrueType outlines (a ``glyf`` table), drawn through their instructions.
    truetype: bool = False


@dataclass(frozen=True, eq=False)
class Face:
    """An open face: HarfBuzz font, FreeType face, metrics and file identity.

    ``lock`` serialises use of ``ft``, whose size, transform and glyph slot
    are mutable state; shaping with ``hb`` needs no lock.
    """

    hb: hb.Font
    ft: freetype.Face
    metrics: FontMetrics
    key: tuple[str, int]
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _instructed: dict[int, np.ndarray | None] = field(default_factory=dict, repr=False)

    @cached_property
    def svg(self) -> SvgTable | None:
        """The face's ``SVG `` table, or ``None`` when it has none."""
        if not self.metrics.has_svg:
            return None
        with self.lock:
            data = sfnt_table(self.ft, b"SVG ")
        return None if data is None else SvgTable(data, self.metrics.upem)

    def instructed(self, gid: int) -> np.ndarray | None:
        """Glyph ``gid``'s points in font units, as Photoshop draws them.

        A TrueType glyph's ``(n, 2)`` outline points as the font's
        instructions leave them at one pixel per font unit; ``None`` when
        they move none, the face is not TrueType or the instructions fail.
        """
        if not self.metrics.truetype:
            return None
        with self.lock:
            if gid not in self._instructed:
                self._instructed[gid] = _moved(self.ft, gid)
            return self._instructed[gid]


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
    """The operating system's font directories, in search order.

    Each is indexed with its subdirectories.
    """
    if os.name == "nt":
        return [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")]
    home = os.path.expanduser("~")
    if os.uname().sysname == "Darwin":
        return [
            "/System/Library/Fonts",
            "/Library/Fonts",
            os.path.join(home, "Library", "Fonts"),
        ]
    data_home = os.environ.get("XDG_DATA_HOME") or os.path.join(home, ".local", "share")
    return [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.join(data_home, "fonts"),
        os.path.join(home, ".fonts"),
    ]


def _env_dirs() -> list[str]:
    value = os.environ.get(FONT_DIRS_ENV, "")
    return [d for d in value.split(os.pathsep) if d]


#: A glyph run through its TrueType instructions; bitmaps and the
#: auto-hinter are not used.
_INSTRUCTED = (
    freetype.FT_LOAD_DEFAULT | freetype.FT_LOAD_NO_BITMAP | freetype.FT_LOAD_NO_AUTOHINT
)
_IDENTITY = freetype.FT_Matrix(0x10000, 0, 0, 0x10000)


def _glyph_ymax(slot: freetype.GlyphSlot) -> int:
    # The outline's own top: the slot's ``horiBearingY`` can differ from it
    # for a CID-keyed CFF face.
    if slot.format == freetype.FT_GLYPH_FORMAT_OUTLINE and slot.outline.n_points:
        return int(slot.outline.get_bbox().yMax)
    return int(slot.metrics.horiBearingY)


def _has_table(face: freetype.Face, tag: bytes) -> bool:
    length = c_ulong(0)
    code = int.from_bytes(tag, "big")
    return freetype.FT_Load_Sfnt_Table(face._FT_Face, code, 0, None, byref(length)) == 0


def outline_points(slot: freetype.GlyphSlot) -> np.ndarray:
    """The slot outline's points: an ``(n, 2)`` view of FreeType's array."""
    outline = slot.outline._FT_Outline
    if not outline.n_points:
        return np.zeros((0, 2), np.int64)
    pointer = cast(outline.points, POINTER(freetype.FT_Pos))
    return np.ctypeslib.as_array(pointer, shape=(outline.n_points, 2))


def instructed_outline(
    face: freetype.Face, gid: int, ppem: int
) -> tuple[np.ndarray, float] | None:
    """Glyph ``gid`` run through the face's TrueType instructions at ``ppem``.

    Its points and its outline's exact top, in font units; ``None`` for a
    glyph with no outline. Each run is on a size of its own, so it does not
    depend on what the face ran before. Raises ``FT_Exception``.
    """
    size = freetype.FT_Size()
    error = freetype.FT_New_Size(face._FT_Face, byref(size))
    if error:
        raise freetype.FT_Exception(error)
    active = face._FT_Face.contents.size
    freetype.FT_Activate_Size(size)
    try:
        face.set_char_size(ppem * 64, ppem * 64, 72, 72)
        face.set_transform(_IDENTITY, freetype.FT_Vector(0, 0))
        face.load_glyph(gid, _INSTRUCTED)
        slot = face.glyph
        if slot.format != freetype.FT_GLYPH_FORMAT_OUTLINE or not slot.outline.n_points:
            return None
        scale = face.units_per_EM / (64 * ppem)
        points = outline_points(slot) * scale
        return points, slot.outline.get_bbox().yMax * scale
    finally:
        freetype.FT_Activate_Size(active)
        freetype.FT_Done_Size(size)


def _moved(face: freetype.Face, gid: int) -> np.ndarray | None:
    """Glyph ``gid``'s instructed points at one pixel per unit, if they move."""
    try:
        hit = instructed_outline(face, gid, face.units_per_EM)
        # An unscaled load is transformed too: by the identity set above.
        face.load_glyph(gid, freetype.FT_LOAD_NO_SCALE)
    except freetype.FT_Exception:
        return None
    if hit is None:
        return None
    points = hit[0]
    unscaled = outline_points(face.glyph)
    if points.shape != unscaled.shape or np.array_equal(points, unscaled):
        return None
    return points


def sfnt_table(face: freetype.Face, tag: bytes) -> bytes | None:
    """The raw bytes of the face's ``tag`` table, or ``None`` when it has none."""
    code = int.from_bytes(tag, "big")
    length = c_ulong(0)
    if freetype.FT_Load_Sfnt_Table(face._FT_Face, code, 0, None, byref(length)):
        return None
    buffer = (c_ubyte * length.value)()
    if freetype.FT_Load_Sfnt_Table(face._FT_Face, code, 0, buffer, byref(length)):
        return None
    return bytes(buffer)


def _metrics(face: freetype.Face) -> FontMetrics:
    truetype = _has_table(face, b"glyf")
    ascent: float = int(face.ascender)
    glyph = "ascender"
    for g in ASCENT_GLYPHS:
        gid = face.get_char_index(ord(g))
        if gid:
            face.load_char(g, freetype.FT_LOAD_NO_SCALE)
            ascent, glyph = _glyph_ymax(face.glyph), g
            if truetype:
                try:
                    hit = instructed_outline(face, gid, ASCENT_PPEM)
                except freetype.FT_Exception:
                    hit = None
                if hit is not None:
                    ascent = hit[1]
            break
    has_svg = _has_table(face, b"SVG ")
    return FontMetrics(
        upem=int(face.units_per_EM),
        ascent_units=ascent,
        ascent_glyph=glyph,
        underline_position=int(face.underline_position),
        underline_thickness=int(face.underline_thickness),
        has_svg=has_svg,
        strikes=frozenset(round(s.y_ppem / 64) for s in face.available_sizes),
        truetype=truetype,
    )


def load_face(path: str, index: int = 0) -> Face:
    """Open one face with FreeType and HarfBuzz and read its metrics."""
    ft = freetype.Face(path, index)
    metrics = _metrics(ft)
    blob = hb.Blob.from_file_path(path)
    font = hb.Font(hb.Face(blob, index))
    font.scale = (metrics.upem, metrics.upem)
    return Face(hb=font, ft=ft, metrics=metrics, key=(path, index))


def font_files(directory: str) -> list[str]:
    """Font files under ``directory`` and its subdirectories, in walk order.

    Files sort before subdirectories at each level; symbolic links to
    directories are not followed.
    """
    out: list[str] = []
    for root, subdirs, files in os.walk(directory):
        subdirs.sort()
        out += [
            os.path.join(root, fn)
            for fn in sorted(files)
            if fn.lower().endswith(FONT_EXTENSIONS)
        ]
    return out


class _Index:
    """Faces under a list of directories, by exact and by loose name."""

    def __init__(self, dirs: Iterable[str]):
        self.exact: dict[str, tuple[str, int]] = {}
        self.loose: dict[str, tuple[str, int]] = {}
        for d in dirs:
            for path in font_files(d):
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
        self._lock = threading.Lock()

    def _indexes(self) -> tuple[_Index, _Index]:
        with self._lock:
            if self._user is None or self._platform is None:
                self._user = _Index(self.dirs)
                self._platform = _Index(self.platform_dirs)
            return self._user, self._platform

    def _face(self, key: tuple[str, int]) -> Face:
        with self._lock:
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
            # An alias with a font extension names a file, present or not.
            if alias.lower().endswith(FONT_EXTENSIONS) or os.path.isfile(alias):
                return ((alias, 0), "exact") if os.path.isfile(alias) else None
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
        """Where ``name`` would resolve: ``(path, index, how)`` or ``None``.

        The file is not opened; one that has gone or cannot be read still
        counts as missing when resolved.
        """
        hit = self._lookup(name)
        return None if hit is None else (hit[0][0], hit[0][1], hit[1])

    def _open(self, name: str) -> tuple[Face, How] | None:
        """The open face ``name`` resolves to, or ``None`` when none does."""
        hit = self._lookup(name)
        if hit is None:
            return None
        try:
            return self._face(hit[0]), hit[1]
        except (freetype.FT_Exception, hb.HarfBuzzError, OSError):
            # Indexed or aliased, but removed or unreadable since.
            return None

    def resolve(self, name: str) -> Resolved:
        """Resolve one PostScript name under this map's policy.

        A name whose file is missing or cannot be opened is a missing font.
        """
        hit = self._open(name)
        if hit is not None:
            return Resolved(*hit)
        if self.policy == "skip":
            return Resolved(None, "missing")
        if self.policy == "substitute":
            sub = self._open(self.fallback)
            if sub is not None:
                return Resolved(sub[0], "substitute")
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
