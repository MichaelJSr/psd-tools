"""OpenType ``SVG `` colour glyphs, drawn with numpy.

Covers static SVG: ``path``, ``rect``, ``circle``, ``ellipse``, ``line``,
``polyline`` and ``polygon`` in groups with ``transform`` and ``opacity``,
filled with a solid colour or a linear or radial gradient, rectangles also
stroked, and embedded PNG or JPEG ``image`` elements. A glyph whose document
uses anything else (``use``, clipping, masks, filters, text, a stroke on
another shape) is unsupported: :meth:`SvgTable.drawing` returns ``None`` and
the caller draws its outline instead.

Coordinates are font units, y down, with the glyph origin at ``(0, 0)``.
"""

from __future__ import annotations

import base64
import io
import math
import re
import struct
import threading
import xml.etree.ElementTree as ET
import zlib
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from psd_tools.composite.scanline import fill_coverage, flatten_cubics

#: ``(a, b, c, d, e, f)``: ``x' = a x + c y + e``, ``y' = b x + d y + f``.
Affine = tuple[float, float, float, float, float, float]
RGBA = tuple[float, float, float, float]

IDENTITY: Affine = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
_PARSED_DOCS = 64
#: Bytes a compressed document may inflate to.
_MAX_DOCUMENT = 1 << 26
_XLINK = "{http://www.w3.org/1999/xlink}href"
_SHAPES = ("path", "rect", "circle", "ellipse", "line", "polyline", "polygon")
#: Elements that draw nothing themselves and need no support.
_INERT = ("title", "desc", "metadata", "style")
_PAINT_KEYS = (
    "fill",
    "fill-opacity",
    "fill-rule",
    "stroke",
    "stroke-width",
    "stroke-opacity",
)
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_COMMAND = re.compile(r"([MmLlHhVvCcSsQqTtAaZz])|" + _NUMBER.pattern)
_NAMED = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "red": (255, 0, 0),
    "lime": (0, 255, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "aqua": (0, 255, 255),
    "magenta": (255, 0, 255),
    "fuchsia": (255, 0, 255),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "silver": (192, 192, 192),
    "maroon": (128, 0, 0),
    "olive": (128, 128, 0),
    "navy": (0, 0, 128),
    "purple": (128, 0, 128),
    "teal": (0, 128, 128),
    "orange": (255, 165, 0),
}


class Unsupported(Exception):
    """The document uses SVG this renderer does not draw."""


def compose(m: Affine, n: Affine) -> Affine:
    """``m`` after ``n``."""
    a, b, c, d, e, f = m
    p, q, r, s, t, u = n
    return (
        a * p + c * q,
        b * p + d * q,
        a * r + c * s,
        b * r + d * s,
        a * t + c * u + e,
        b * t + d * u + f,
    )


def _invert(m: Affine) -> Affine:
    a, b, c, d, e, f = m
    det = a * d - b * c
    if not det:
        raise Unsupported("singular transform")
    return (
        d / det,
        -b / det,
        -c / det,
        a / det,
        (c * f - d * e) / det,
        (b * e - a * f) / det,
    )


def _apply(m: Affine, pts: np.ndarray) -> np.ndarray:
    a, b, c, d, e, f = m
    return np.column_stack(
        [a * pts[:, 0] + c * pts[:, 1] + e, b * pts[:, 0] + d * pts[:, 1] + f]
    )


# -- paints ----------------------------------------------------------------


@dataclass(frozen=True)
class Gradient:
    """A linear (``radius`` is ``None``) or radial gradient.

    Geometry is in gradient space: ``transform`` maps it to the shape's user
    space, through the shape's bounding box first when ``bbox_units``.
    """

    start: tuple[float, float]  #: linear x1, y1; radial focus
    end: tuple[float, float]  #: linear x2, y2; radial centre
    radius: float | None
    offsets: tuple[float, ...]
    colors: tuple[RGBA, ...]
    spread: str
    bbox_units: bool
    transform: Affine


Paint = RGBA | Gradient | None


@dataclass(frozen=True)
class Shape:
    """A filled region: contours of ``(x, y)`` vertices in user space."""

    curves: tuple[np.ndarray, ...]  #: ``(n, 4, 2)`` cubic segments per contour
    transform: Affine
    paint: Paint
    opacity: float
    nonzero: bool
    bbox: tuple[float, float, float, float]  #: of the geometry, user space


@dataclass(frozen=True)
class Picture:
    """An image: ``transform`` maps its pixel grid to user space."""

    #: Straight ``float32`` HxWx4: a transparent pixel keeps its stored colour.
    rgba: np.ndarray
    transform: Affine
    opacity: float


Item = Shape | Picture


@dataclass(frozen=True)
class Drawing:
    """A parsed glyph: what to paint, in order."""

    items: tuple[Item, ...]


# -- the SVG table ---------------------------------------------------------


class SvgTable:
    """The ``SVG `` table of one face: its documents by glyph id, parsed lazily."""

    def __init__(self, data: bytes):
        self._data = data
        self._index: list[tuple[int, int, int, int]] = []
        self._firsts: list[int] = []
        self._docs: OrderedDict[tuple[int, int], ET.Element | None] = OrderedDict()
        self._drawings: OrderedDict[tuple[int, RGBA], Drawing | None] = OrderedDict()
        self._lock = threading.Lock()
        if len(data) < 10:
            return
        (offset,) = struct.unpack_from(">I", data, 2)
        if offset + 2 > len(data):
            return
        (count,) = struct.unpack_from(">H", data, offset)
        for i in range(count):
            at = offset + 2 + 12 * i
            if at + 12 > len(data):
                break
            first, last, doc, length = struct.unpack_from(">HHII", data, at)
            self._index.append((first, last, offset + doc, length))
        self._index.sort()
        self._firsts = [entry[0] for entry in self._index]

    def __contains__(self, gid: int) -> bool:
        return self._find(gid) is not None

    def _find(self, gid: int) -> tuple[int, int] | None:
        k = bisect_right(self._firsts, gid) - 1
        if k < 0:
            return None
        first, last, start, length = self._index[k]
        return (start, length) if gid <= last else None

    def document(self, gid: int) -> bytes | None:
        """The raw (decompressed) document holding ``gid``."""
        where = self._find(gid)
        if where is None:
            return None
        start, length = where
        doc = self._data[start : start + length]
        if doc[:2] == b"\x1f\x8b":
            inflate = zlib.decompressobj(47)
            doc = inflate.decompress(doc, _MAX_DOCUMENT)
            if inflate.unconsumed_tail:
                return None
        return doc

    def _root(self, gid: int) -> ET.Element | None:
        where = self._find(gid)
        if where is None:
            return None
        with self._lock:
            if where in self._docs:
                self._docs.move_to_end(where)
                return self._docs[where]
        doc = self.document(gid)
        root: ET.Element | None = None
        # An entity can expand without bound; glyph documents need none.
        if doc is not None and b"<!ENTITY" not in doc and b"<!DOCTYPE" not in doc:
            try:
                root = ET.fromstring(doc)
            except ET.ParseError:
                root = None
        with self._lock:
            self._docs[where] = root
            while len(self._docs) > _PARSED_DOCS:
                self._docs.popitem(last=False)
        return root

    def drawing(self, gid: int, context: RGBA = (0.0, 0.0, 0.0, 1.0)) -> Drawing | None:
        """What glyph ``gid`` paints, or ``None`` when it has no supported document.

        ``context`` stands in for ``currentColor``, ``context-fill`` and
        ``context-stroke``: the run's fill.
        """
        key = (gid, context)
        with self._lock:
            if key in self._drawings:
                self._drawings.move_to_end(key)
                return self._drawings[key]
        root = self._root(gid)
        drawing = None
        if root is not None:
            try:
                drawing = _Parser(root, gid, context).drawing()
            except (Unsupported, ValueError, KeyError, IndexError, OSError):
                drawing = None
        with self._lock:
            self._drawings[key] = drawing
            while len(self._drawings) > _PARSED_DOCS:
                self._drawings.popitem(last=False)
        return drawing


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _painters(el: ET.Element) -> int:
    """Shapes and images at or under ``el``."""
    return sum(1 for node in el.iter() if _local(node.tag) in (*_SHAPES, "image"))


def _style(el: ET.Element) -> dict[str, str]:
    out = {k: v for k, v in el.attrib.items() if not k.startswith("{")}
    for part in out.pop("style", "").split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _length(value: str | None, default: float = 0.0) -> float:
    if value is None or not value.strip():
        return default
    v = value.strip()
    if v.endswith("px"):
        v = v[:-2]
    if not _NUMBER.fullmatch(v):
        raise Unsupported(f"length {value!r}")
    return float(v)


def _fraction(value: str | None, default: float) -> float:
    """A gradient coordinate or offset: a number or a percentage."""
    if value is None or not value.strip():
        return default
    v = value.strip()
    if v.endswith("%"):
        return float(v[:-1]) / 100
    return _length(v)


def parse_color(value: str, context: RGBA) -> RGBA | None:
    """An SVG colour as straight RGBA in 0..1; ``None`` for ``none``."""
    v = value.strip().lower()
    if v in ("none", "transparent"):
        return None
    if v in ("currentcolor", "context-fill", "context-stroke"):
        return context
    if v.startswith("#"):
        h = v[1:]
        if len(h) in (3, 4):
            h = "".join(ch * 2 for ch in h)
        if len(h) not in (6, 8) or not re.fullmatch(r"[0-9a-f]+", h):
            raise Unsupported(f"colour {value!r}")
        channels = [int(h[i : i + 2], 16) / 255 for i in range(0, len(h), 2)]
        return (
            channels[0],
            channels[1],
            channels[2],
            channels[3] if len(h) == 8 else 1.0,
        )
    m = re.fullmatch(r"rgba?\(([^)]*)\)", v)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        rgb = [
            float(p[:-1]) / 100 if p.endswith("%") else float(p) / 255
            for p in parts[:3]
        ]
        alpha = float(parts[3]) if len(parts) > 3 else 1.0
        return (rgb[0], rgb[1], rgb[2], alpha)
    if v in _NAMED:
        r, g, b = _NAMED[v]
        return (r / 255, g / 255, b / 255, 1.0)
    raise Unsupported(f"colour {value!r}")


def parse_transform(value: str | None) -> Affine:
    """An SVG ``transform`` attribute as an affine."""
    m = IDENTITY
    if not value:
        return m
    for name, args in re.findall(r"([a-zA-Z]+)\s*\(([^)]*)\)", value):
        a = [float(x) for x in _NUMBER.findall(args)]
        if name == "matrix" and len(a) == 6:
            t: Affine = (a[0], a[1], a[2], a[3], a[4], a[5])
        elif name == "translate" and a:
            t = (1.0, 0.0, 0.0, 1.0, a[0], a[1] if len(a) > 1 else 0.0)
        elif name == "scale" and a:
            t = (a[0], 0.0, 0.0, a[1] if len(a) > 1 else a[0], 0.0, 0.0)
        elif name == "rotate" and a:
            r = math.radians(a[0])
            cos, sin = math.cos(r), math.sin(r)
            t = (cos, sin, -sin, cos, 0.0, 0.0)
            if len(a) == 3:
                t = compose(
                    (1.0, 0.0, 0.0, 1.0, a[1], a[2]),
                    compose(t, (1.0, 0.0, 0.0, 1.0, -a[1], -a[2])),
                )
        elif name == "skewX" and a:
            t = (1.0, 0.0, math.tan(math.radians(a[0])), 1.0, 0.0, 0.0)
        elif name == "skewY" and a:
            t = (1.0, math.tan(math.radians(a[0])), 0.0, 1.0, 0.0, 0.0)
        else:
            raise Unsupported(f"transform {name}")
        m = compose(m, t)
    return m


# -- geometry --------------------------------------------------------------


def _line(p: tuple[float, float], q: tuple[float, float]) -> list[float]:
    return [p[0], p[1], p[0], p[1], q[0], q[1], q[0], q[1]]


def _quad(
    p: tuple[float, float], c: tuple[float, float], q: tuple[float, float]
) -> list[float]:
    return [
        p[0],
        p[1],
        p[0] + 2 / 3 * (c[0] - p[0]),
        p[1] + 2 / 3 * (c[1] - p[1]),
        q[0] + 2 / 3 * (c[0] - q[0]),
        q[1] + 2 / 3 * (c[1] - q[1]),
        q[0],
        q[1],
    ]


def _arc(
    p: tuple[float, float],
    rx: float,
    ry: float,
    phi: float,
    large: bool,
    sweep: bool,
    q: tuple[float, float],
) -> list[list[float]]:
    """An SVG elliptical arc as cubics (SVG 1.1 implementation notes F.6)."""
    if p == q:
        return []
    rx, ry = abs(rx), abs(ry)
    if not rx or not ry:
        return [_line(p, q)]
    cos, sin = math.cos(math.radians(phi)), math.sin(math.radians(phi))
    dx, dy = (p[0] - q[0]) / 2, (p[1] - q[1]) / 2
    x1, y1 = cos * dx + sin * dy, -sin * dx + cos * dy
    scale = x1 * x1 / (rx * rx) + y1 * y1 / (ry * ry)
    if scale > 1:
        rx, ry = rx * math.sqrt(scale), ry * math.sqrt(scale)
    num = rx * rx * ry * ry - rx * rx * y1 * y1 - ry * ry * x1 * x1
    den = rx * rx * y1 * y1 + ry * ry * x1 * x1
    k = math.sqrt(max(num, 0.0) / den) if den else 0.0
    if large == sweep:
        k = -k
    cx1, cy1 = k * rx * y1 / ry, -k * ry * x1 / rx
    cx = cos * cx1 - sin * cy1 + (p[0] + q[0]) / 2
    cy = sin * cx1 + cos * cy1 + (p[1] + q[1]) / 2

    def angle(ux: float, uy: float, vx: float, vy: float) -> float:
        return math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)

    t0 = angle(1, 0, (x1 - cx1) / rx, (y1 - cy1) / ry)
    dt = angle((x1 - cx1) / rx, (y1 - cy1) / ry, (-x1 - cx1) / rx, (-y1 - cy1) / ry)
    if not sweep and dt > 0:
        dt -= 2 * math.pi
    elif sweep and dt < 0:
        dt += 2 * math.pi
    n = max(1, math.ceil(abs(dt) / (math.pi / 2) - 1e-9))
    step = dt / n
    h = 4 / 3 * math.tan(step / 4)
    out = []

    def point(t: float) -> tuple[float, float]:
        x, y = rx * math.cos(t), ry * math.sin(t)
        return cos * x - sin * y + cx, sin * x + cos * y + cy

    def deriv(t: float) -> tuple[float, float]:
        x, y = -rx * math.sin(t), ry * math.cos(t)
        return cos * x - sin * y, sin * x + cos * y

    for i in range(n):
        a, b = t0 + i * step, t0 + (i + 1) * step
        pa, pb = point(a), point(b)
        da, db = deriv(a), deriv(b)
        if i == n - 1:
            pb = q
        out.append(
            [
                pa[0],
                pa[1],
                pa[0] + h * da[0],
                pa[1] + h * da[1],
                pb[0] - h * db[0],
                pb[1] - h * db[1],
                pb[0],
                pb[1],
            ]
        )
    return out


def _tokens(d: str) -> Iterator[str]:
    for m in _COMMAND.finditer(d):
        yield m.group(0)


def parse_path(d: str) -> list[np.ndarray]:
    """Path data as contours of cubic segments, ``(n, 4, 2)`` each."""
    toks = list(_tokens(d))
    i = 0
    contours: list[np.ndarray] = []
    segs: list[list[float]] = []
    cur = start = (0.0, 0.0)
    last_c: tuple[float, float] | None = None
    last_q: tuple[float, float] | None = None
    cmd = ""

    def num() -> float:
        nonlocal i
        if i >= len(toks) or toks[i].isalpha():
            raise Unsupported("path data")
        i += 1
        return float(toks[i - 1])

    def close() -> None:
        nonlocal segs
        if segs:
            if cur != start:
                segs.append(_line(cur, start))
            contours.append(np.array(segs, np.float64).reshape(-1, 4, 2))
        segs = []

    while i < len(toks):
        if toks[i].isalpha():
            cmd = toks[i]
            i += 1
        elif not cmd:
            raise Unsupported("path data")
        rel = cmd.islower()
        c = cmd.upper()
        ox, oy = cur if rel else (0.0, 0.0)
        qc: tuple[float, float] | None = None
        cc: tuple[float, float] | None = None
        if c == "Z":
            close()
            cur = start
            cmd = ""
        elif c == "M":
            close()
            cur = start = (ox + num(), oy + num())
            cmd = "l" if rel else "L"
        elif c in "LHV":
            if c == "L":
                p = (ox + num(), oy + num())
            elif c == "H":
                p = (ox + num(), cur[1])
            else:
                p = (cur[0], oy + num())
            segs.append(_line(cur, p))
            cur = p
        elif c in "CS":
            if c == "C":
                c1 = (ox + num(), oy + num())
            else:
                c1 = (2 * cur[0] - last_c[0], 2 * cur[1] - last_c[1]) if last_c else cur
            c2 = (ox + num(), oy + num())
            p = (ox + num(), oy + num())
            segs.append([cur[0], cur[1], c1[0], c1[1], c2[0], c2[1], p[0], p[1]])
            cc, cur = c2, p
        elif c in "QT":
            if c == "Q":
                q1 = (ox + num(), oy + num())
            else:
                q1 = (2 * cur[0] - last_q[0], 2 * cur[1] - last_q[1]) if last_q else cur
            p = (ox + num(), oy + num())
            segs.append(_quad(cur, q1, p))
            qc, cur = q1, p
        elif c == "A":
            rx, ry, phi = num(), num(), num()
            large, sweep = num() != 0, num() != 0
            p = (ox + num(), oy + num())
            segs.extend(_arc(cur, rx, ry, phi, large, sweep, p))
            cur = p
        last_c, last_q = cc, qc
    close()
    return contours


def _ellipse(cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    k = 4 / 3 * (math.sqrt(2) - 1)
    pts = [
        ((cx + rx, cy), (cx + rx, cy + k * ry), (cx + k * rx, cy + ry), (cx, cy + ry)),
        ((cx, cy + ry), (cx - k * rx, cy + ry), (cx - rx, cy + k * ry), (cx - rx, cy)),
        ((cx - rx, cy), (cx - rx, cy - k * ry), (cx - k * rx, cy - ry), (cx, cy - ry)),
        ((cx, cy - ry), (cx + k * rx, cy - ry), (cx + rx, cy - k * ry), (cx + rx, cy)),
    ]
    return np.array(pts, np.float64)


def rounded_rect(
    x: float, y: float, w: float, h: float, rx: float, ry: float
) -> np.ndarray:
    """A rectangle with elliptical corners as one clockwise (y down) contour."""
    rx, ry = min(max(rx, 0.0), w / 2), min(max(ry, 0.0), h / 2)
    if not rx or not ry:
        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        return np.array(
            [_line(corners[i], corners[(i + 1) % 4]) for i in range(4)], np.float64
        ).reshape(-1, 4, 2)
    k = 4 / 3 * (math.sqrt(2) - 1)
    r, b = x + w, y + h
    segs = [
        _line((x + rx, y), (r - rx, y)),
        [r - rx, y, r - rx + k * rx, y, r, y + ry - k * ry, r, y + ry],
        _line((r, y + ry), (r, b - ry)),
        [r, b - ry, r, b - ry + k * ry, r - rx + k * rx, b, r - rx, b],
        _line((r - rx, b), (x + rx, b)),
        [x + rx, b, x + rx - k * rx, b, x, b - ry + k * ry, x, b - ry],
        _line((x, b - ry), (x, y + ry)),
        [x, y + ry, x, y + ry - k * ry, x + rx - k * rx, y, x + rx, y],
    ]
    segs = [s for s in segs if (s[0], s[1]) != (s[6], s[7])]
    return np.array(segs, np.float64).reshape(-1, 4, 2)


def _rect_ring(
    rect: tuple[float, float, float, float, float, float], width: float
) -> tuple[np.ndarray, ...]:
    """The stroke of a rectangle: its outline grown and shrunk by half ``width``."""
    x, y, w, h, rx, ry = rect
    half = width / 2
    outer = rounded_rect(
        x - half,
        y - half,
        w + width,
        h + width,
        rx + half if rx else 0.0,
        ry + half if ry else 0.0,
    )
    if w <= width or h <= width:
        return (outer,)
    inner = rounded_rect(
        x + half,
        y + half,
        w - width,
        h - width,
        max(rx - half, 0.0),
        max(ry - half, 0.0),
    )
    # Reversed, the inner outline is a hole under either fill rule.
    return outer, inner[::-1, ::-1]


def _points(value: str) -> list[tuple[float, float]]:
    v = [float(x) for x in _NUMBER.findall(value)]
    return list(zip(v[0::2], v[1::2], strict=False))


def _bbox(contours: list[np.ndarray]) -> tuple[float, float, float, float]:
    pts = np.concatenate([c.reshape(-1, 2) for c in contours])
    return (
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    )


# -- parsing ---------------------------------------------------------------


class _Parser:
    def __init__(self, root: ET.Element, gid: int, context: RGBA):
        self.context = context
        self.ids = {el.get("id"): el for el in root.iter() if el.get("id")}
        target = self.ids.get(f"glyph{gid}")
        self.root = root
        self.target = target if target is not None else root
        self.items: list[Item] = []

    def drawing(self) -> Drawing:
        if _local(self.root.tag) != "svg":
            raise Unsupported("root")
        if self.root.get("viewBox"):
            raise Unsupported("viewBox")
        # Ancestors of the glyph element pass on their transform and paint.
        chain = self._chain(self.root, self.target) or [self.root]
        m, style, opacity = IDENTITY, self._inherited({}, self.root), 1.0
        for el in chain[1:-1] if len(chain) > 1 else []:
            m = compose(m, parse_transform(el.get("transform")))
            style = self._inherited(style, el)
            opacity *= _length(_style(el).get("opacity"), 1.0)
        self._walk(self.target, m, style, opacity)
        return Drawing(tuple(self.items))

    def _chain(self, node: ET.Element, target: ET.Element) -> list[ET.Element] | None:
        if node is target:
            return [node]
        for child in node:
            path = self._chain(child, target)
            if path is not None:
                return [node, *path]
        return None

    @staticmethod
    def _inherited(parent: dict[str, str], el: ET.Element) -> dict[str, str]:
        style = dict(parent)
        own = _style(el)
        for k in _PAINT_KEYS:
            if k in own:
                style[k] = own[k]
        return style

    def _walk(
        self, el: ET.Element, m: Affine, style: dict[str, str], opacity: float
    ) -> None:
        tag = _local(el.tag)
        own = _style(el)
        if own.get("display") == "none" or own.get("visibility") == "hidden":
            return
        for k in ("clip-path", "mask", "filter"):
            if own.get(k, "none") != "none":
                raise Unsupported(k)
        m = compose(m, parse_transform(own.get("transform")))
        style = self._inherited(style, el)
        own_opacity = _length(own.get("opacity"), 1.0)
        if tag == "svg" and el is not self.root:
            raise Unsupported("nested svg")
        if tag in ("svg", "g", "a") and own_opacity < 1 and _painters(el) > 1:
            # A group's opacity applies to the group composited as a whole.
            raise Unsupported("group opacity")
        opacity *= own_opacity
        if tag in ("svg", "g", "a"):
            for child in el:
                self._walk(child, m, style, opacity)
        elif tag in _SHAPES:
            self._shape(tag, own, m, style, opacity)
        elif tag == "image":
            self._image(el, own, m, opacity)
        elif tag in ("defs", "linearGradient", "radialGradient", *_INERT):
            return
        else:
            raise Unsupported(tag)

    def _shape(
        self,
        tag: str,
        a: dict[str, str],
        m: Affine,
        style: dict[str, str],
        opacity: float,
    ) -> None:
        stroke = style.get("stroke", "none")
        width = _length(style.get("stroke-width"), 1.0)
        stroked = stroke != "none" and width > 0
        rect: tuple[float, float, float, float, float, float] | None = None
        if tag == "rect":
            w, h = _length(a.get("width")), _length(a.get("height"))
            if w <= 0 or h <= 0:
                return
            rx_attr, ry_attr = a.get("rx"), a.get("ry")
            rx = _length(rx_attr if rx_attr is not None else ry_attr)
            ry = _length(ry_attr if ry_attr is not None else rx_attr)
            rect = (_length(a.get("x")), _length(a.get("y")), w, h, rx, ry)
            contours = [rounded_rect(*rect)]
        elif tag == "path":
            contours = parse_path(a.get("d", ""))
        elif tag in ("circle", "ellipse"):
            cx, cy = _length(a.get("cx")), _length(a.get("cy"))
            if tag == "circle":
                radii = (_length(a.get("r")),) * 2
            else:
                radii = (_length(a.get("rx")), _length(a.get("ry")))
            if min(radii) <= 0:
                return
            contours = [_ellipse(cx, cy, *radii)]
        elif tag in ("polyline", "polygon"):
            pts = _points(a.get("points", ""))
            if len(pts) < 2:
                return
            segs = [_line(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
            contours = [np.array(segs, np.float64).reshape(-1, 4, 2)]
        else:  # line: a stroke only
            if stroked:
                raise Unsupported("stroked line")
            return
        contours = [c for c in contours if len(c)]
        if not contours:
            return
        bbox = _bbox(contours)
        fill = self._paint(style.get("fill", "black"), bbox)
        fill_opacity = _length(style.get("fill-opacity"), 1.0)
        nonzero = style.get("fill-rule", "nonzero") != "evenodd"
        if fill is not None:
            self.items.append(
                Shape(tuple(contours), m, fill, opacity * fill_opacity, nonzero, bbox)
            )
        if not stroked:
            return
        if rect is None:
            raise Unsupported(f"stroked {tag}")
        if _length(style.get("stroke-opacity"), 1.0) != 1.0:
            raise Unsupported("stroke-opacity")
        paint = self._paint(stroke, bbox)
        if paint is not None:
            self.items.append(
                Shape(_rect_ring(rect, width), m, paint, opacity, True, bbox)
            )

    def _paint(self, value: str, bbox: tuple[float, float, float, float]) -> Paint:
        v = value.strip()
        ref = re.fullmatch(r"url\(\s*#([^)\s]+)\s*\)(?:\s+(.*))?", v)
        if ref is None:
            return parse_color(v, self.context)
        el = self.ids.get(ref.group(1))
        if el is None:
            return parse_color(ref.group(2), self.context) if ref.group(2) else None
        return self._gradient(el)

    def _gradient(self, el: ET.Element) -> Gradient:
        tag = _local(el.tag)
        if tag not in ("linearGradient", "radialGradient"):
            raise Unsupported(f"paint server {tag}")
        # xlink:href inherits attributes and stops from another gradient.
        chain = [el]
        while True:
            href = chain[-1].get(_XLINK) or chain[-1].get("href")
            if not href or not href.startswith("#") or len(chain) > 8:
                break
            nxt = self.ids.get(href[1:])
            if nxt is None:
                break
            chain.append(nxt)

        def attr(name: str) -> str | None:
            for g in chain:
                if g.get(name) is not None:
                    return g.get(name)
            return None

        stops = next(
            (list(g) for g in chain if any(_local(s.tag) == "stop" for s in g)), []
        )
        offsets: list[float] = []
        colors: list[RGBA] = []
        for s in stops:
            if _local(s.tag) != "stop":
                continue
            st = _style(s)
            off = min(max(_fraction(st.get("offset"), 0.0), 0.0), 1.0)
            if offsets:
                off = max(off, offsets[-1])
            c = parse_color(st.get("stop-color", "black"), self.context) or (
                0.0,
                0.0,
                0.0,
                0.0,
            )
            alpha = c[3] * _length(st.get("stop-opacity"), 1.0)
            offsets.append(off)
            colors.append((c[0], c[1], c[2], alpha))
        if not offsets:
            raise Unsupported("gradient without stops")
        bbox_units = attr("gradientUnits") != "userSpaceOnUse"
        spread = attr("spreadMethod") or "pad"
        transform = parse_transform(attr("gradientTransform"))
        if tag == "linearGradient":
            return Gradient(
                (_fraction(attr("x1"), 0.0), _fraction(attr("y1"), 0.0)),
                (_fraction(attr("x2"), 1.0), _fraction(attr("y2"), 0.0)),
                None,
                tuple(offsets),
                tuple(colors),
                spread,
                bbox_units,
                transform,
            )
        cx, cy = _fraction(attr("cx"), 0.5), _fraction(attr("cy"), 0.5)
        fx, fy = _fraction(attr("fx"), cx), _fraction(attr("fy"), cy)
        r = _fraction(attr("r"), 0.5)
        return Gradient(
            (fx, fy),
            (cx, cy),
            r,
            tuple(offsets),
            tuple(colors),
            spread,
            bbox_units,
            transform,
        )

    def _image(
        self, el: ET.Element, a: dict[str, str], m: Affine, opacity: float
    ) -> None:
        href = el.get(_XLINK) or el.get("href") or ""
        hit = re.match(r"data:image/(png|jpeg|jpg);base64,", href.strip())
        if hit is None:
            raise Unsupported("image source")
        from PIL import Image  # noqa: PLC0415

        raw = base64.b64decode(href.strip()[hit.end() :])
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            rgba = np.asarray(im.convert("RGBA"), dtype=np.float32) / 255
        ih, iw = rgba.shape[:2]
        x, y = _length(a.get("x")), _length(a.get("y"))
        w, h = _length(a.get("width"), float(iw)), _length(a.get("height"), float(ih))
        if w <= 0 or h <= 0 or not iw or not ih:
            return
        sx, sy = w / iw, h / ih
        par = (a.get("preserveAspectRatio") or "xMidYMid meet").split()
        if par[0] != "none":
            s = min(sx, sy) if (len(par) < 2 or par[1] == "meet") else max(sx, sy)
            if par[0] != "xMidYMid" or (len(par) > 1 and par[1] == "slice"):
                if abs(sx - sy) > 1e-9 * max(sx, sy):
                    raise Unsupported("preserveAspectRatio")
            x += (w - iw * s) / 2
            y += (h - ih * s) / 2
            sx = sy = s
        place = compose(m, (sx, 0.0, 0.0, sy, x, y))
        self.items.append(Picture(rgba, place, opacity))


# -- rasterization ---------------------------------------------------------


def _device_contours(shape: Shape, m: Affine) -> list[np.ndarray]:
    full = compose(m, shape.transform)
    out = []
    for c in shape.curves:
        pts = _apply(full, c.reshape(-1, 2)).reshape(-1, 4, 2)
        out.append(flatten_cubics(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]))
    return out


def _gradient_t(g: Gradient, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient parameter at gradient-space points."""
    if g.radius is None:
        (x1, y1), (x2, y2) = g.start, g.end
        dx, dy = x2 - x1, y2 - y1
        n = dx * dx + dy * dy
        if not n:
            return np.ones_like(x)
        return ((x - x1) * dx + (y - y1) * dy) / n
    (fx, fy), (cx, cy), r = g.start, g.end, g.radius
    if r <= 0:
        return np.ones_like(x)
    # Solve |p - f - t (c - f)|... the focal form: the circle of radius t r
    # about f + t (c - f) passes through the point.
    px, py = x - fx, y - fy
    ex, ey = cx - fx, cy - fy
    a = ex * ex + ey * ey - r * r
    b = px * ex + py * ey
    c = px * px + py * py
    if abs(a) < 1e-12:
        return np.where(b != 0, c / (2 * np.where(b != 0, b, 1.0)), 0.0)
    disc = np.sqrt(np.maximum(b * b - a * c, 0.0))
    return (b - disc) / a if a < 0 else c / np.maximum(b + disc, 1e-12)


def _spread(t: np.ndarray, how: str) -> np.ndarray:
    if how == "repeat":
        return t - np.floor(t)
    if how == "reflect":
        u = np.mod(t, 2.0)
        return np.where(u > 1, 2 - u, u)
    return np.clip(t, 0.0, 1.0)


def _paint_rgba(
    paint: Paint, shape: Shape, m: Affine, x0: int, y0: int, w: int, h: int
) -> np.ndarray:
    """Straight RGBA per pixel centre of a ``w`` x ``h`` window at ``(x0, y0)``."""
    if paint is None:
        return np.zeros((h, w, 4), np.float32)
    if not isinstance(paint, Gradient):
        return np.broadcast_to(np.array(paint, np.float32), (h, w, 4))
    g = paint
    to_user = g.transform
    if g.bbox_units:
        bx0, by0, bx1, by1 = shape.bbox
        if bx1 <= bx0 or by1 <= by0:
            return np.zeros((h, w, 4), np.float32)
        to_user = compose((bx1 - bx0, 0.0, 0.0, by1 - by0, bx0, by0), to_user)
    inv = _invert(compose(compose(m, shape.transform), to_user))
    ys, xs = np.mgrid[0:h, 0:w]
    px, py = xs + x0 + 0.5, ys + y0 + 0.5
    a, b, c, d, e, f = inv
    gx, gy = a * px + c * py + e, b * px + d * py + f
    t = _spread(_gradient_t(g, gx, gy), g.spread)
    offs = np.array(g.offsets)
    cols = np.array(g.colors, np.float64)
    # Interpolate premultiplied so a stop's colour does not bleed through its alpha.
    pre = cols.copy()
    pre[:, :3] *= pre[:, 3:]
    if len(offs) == 1:
        out = np.broadcast_to(pre[0], (h, w, 4)).copy()
    else:
        out = np.stack([np.interp(t, offs, pre[:, k]) for k in range(4)], axis=-1)
    alpha = out[..., 3:]
    out[..., :3] = np.where(alpha > 0, out[..., :3] / np.where(alpha > 0, alpha, 1), 0)
    return out.astype(np.float32)


#: An axis-aligned image is drawn into whole pixels, as Photoshop draws it:
#: its left and top edges move to ``floor(edge + bias)``, its size rounds
#: up, and source pixel ``i`` is sampled at ``i`` rather than ``i + 1/2``.
IMAGE_SNAP_LEFT = 0.25
IMAGE_SNAP_TOP = 0.1


def _area_weights(n_out: int, n_in: int, scale: float, offset: float) -> np.ndarray:
    """``(n_out, n_in)`` share of each source pixel in each output pixel.

    ``out = scale * in + offset``; a weight is the overlap of the two
    pixels in output units, so a downscale averages and an upscale holds.
    """
    lo = (np.arange(n_out) - offset) / scale
    hi = lo + 1 / scale
    src = np.arange(n_in, dtype=np.float64)
    overlap = np.minimum(hi[:, None], src[None, :] + 1) - np.maximum(lo[:, None], src)
    return np.clip(overlap, 0.0, None) * scale


def _snapped(
    full: Affine, iw: int, ih: int
) -> tuple[Affine, tuple[int, int, int, int]]:
    """The placement of an axis-aligned image snapped to whole pixels, and its rect."""
    a, _, _, d, e, f = full
    left, top = math.floor(e + IMAGE_SNAP_LEFT), math.floor(f + IMAGE_SNAP_TOP)
    w = max(1, math.ceil(a * iw - 1e-6))
    h = max(1, math.ceil(d * ih - 1e-6))
    sx, sy = w / iw, h / ih
    return (sx, 0.0, 0.0, sy, left - sx / 2, top - sy / 2), (
        left,
        top,
        left + w,
        top + h,
    )


def _axis_aligned(m: Affine) -> bool:
    return abs(m[1]) < 1e-9 and abs(m[2]) < 1e-9 and m[0] > 0 and m[3] > 0


def picture_layer(
    pic: Picture, m: Affine, x0: int, y0: int, w: int, h: int
) -> np.ndarray:
    """``pic`` resampled onto a ``w`` x ``h`` window at ``(x0, y0)``, premultiplied.

    Colour and alpha are resampled apart, as Photoshop does, so the colour
    of transparent pixels (black, as a rule) darkens a soft edge.
    """
    full = compose(m, pic.transform)
    src = pic.rgba.astype(np.float64)
    ih, iw = src.shape[:2]
    if _axis_aligned(full):
        place, (rx0, ry0, rx1, ry1) = _snapped(full, iw, ih)
        a, _, _, d, e, f = place
        wx = _area_weights(w, iw, a, e - x0)
        wy = _area_weights(h, ih, d, f - y0)
        # The snapped rectangle clips what the half-pixel shift moves out.
        wx[(np.arange(w) + x0 < rx0) | (np.arange(w) + x0 >= rx1)] = 0
        wy[(np.arange(h) + y0 < ry0) | (np.arange(h) + y0 >= ry1)] = 0
        rows = np.tensordot(wy, src, axes=(1, 0))
        layer = np.matmul(rows.transpose(0, 2, 1), wx.T).transpose(0, 2, 1)
    else:
        # Bilinear from pixel centres; no corpus glyph is rotated or skewed.
        p, q, r, s, t, u = _invert(full)
        ys, xs = np.mgrid[0:h, 0:w]
        px, py = xs + x0 + 0.5, ys + y0 + 0.5
        sx, sy = p * px + r * py + t - 0.5, q * px + s * py + u - 0.5
        jx, jy = np.floor(sx).astype(np.int64), np.floor(sy).astype(np.int64)
        fx, fy = sx - jx, sy - jy
        layer = np.zeros((h, w, 4))
        for dy, wy_ in ((0, 1 - fy), (1, fy)):
            for dx, wx_ in ((0, 1 - fx), (1, fx)):
                yy, xx = jy + dy, jx + dx
                ok = (yy >= 0) & (yy < ih) & (xx >= 0) & (xx < iw)
                vals = np.zeros((h, w, 4))
                vals[ok] = src[yy[ok], xx[ok]]
                layer += vals * (wy_ * wx_)[..., None]
    layer = np.clip(layer, 0.0, 1.0).astype(np.float32)
    layer[..., 3] *= pic.opacity
    layer[..., :3] *= layer[..., 3:]
    return layer


def device_bbox(drawing: Drawing, m: Affine) -> tuple[int, int, int, int] | None:
    """Integer pixel bounds of ``drawing`` under ``m``."""
    boxes = []
    for item in drawing.items:
        if isinstance(item, Shape):
            pts = np.concatenate([c.reshape(-1, 2) for c in item.curves])
            pts = _apply(compose(m, item.transform), pts)
        else:
            ih, iw = item.rgba.shape[:2]
            full = compose(m, item.transform)
            if _axis_aligned(full):
                boxes.append(_snapped(full, iw, ih)[1])
                continue
            corners = np.array([[0, 0], [iw, 0], [0, ih], [iw, ih]], np.float64)
            pts = _apply(full, corners)
        boxes.append(
            (
                math.floor(pts[:, 0].min()),
                math.floor(pts[:, 1].min()),
                math.ceil(pts[:, 0].max()),
                math.ceil(pts[:, 1].max()),
            )
        )
    if not boxes:
        return None
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def to_linear(c: np.ndarray) -> np.ndarray:
    """sRGB-encoded values to linear light."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(c: np.ndarray) -> np.ndarray:
    """Linear-light values to sRGB encoding."""
    return np.where(
        c <= 0.0031308, c * 12.92, 1.055 * np.maximum(c, 0.0) ** (1 / 2.4) - 0.055
    )


def _encode(out: np.ndarray, decode: bool) -> None:
    """Premultiplied ``out`` between sRGB encoding and linear light, in place."""
    alpha = out[..., 3:]
    ink = alpha[..., 0] > 0
    color = out[..., :3][ink] / alpha[ink]
    color = to_linear(color) if decode else to_srgb(np.clip(color, 0.0, 1.0))
    out[..., :3][ink] = color * alpha[ink]


def _sampled_alpha(
    drawing: Drawing, m: Affine, x0: int, y0: int, w: int, h: int, grid: int
) -> np.ndarray:
    """The shapes' joint alpha from a ``grid`` x ``grid`` point sample per pixel."""
    n = grid
    hi: Affine = (
        m[0] * n,
        m[1] * n,
        m[2] * n,
        m[3] * n,
        (m[4] - x0) * n,
        (m[5] - y0) * n,
    )
    keep = np.ones((h * n, w * n), np.float32)
    for item in drawing.items:
        if not isinstance(item, Shape) or item.paint is None:
            continue
        # A cell's area coverage passes one half where its centre is inside.
        inside = fill_coverage(
            _device_contours(item, hi), w * n, h * n, nonzero=item.nonzero
        )
        alpha = _paint_rgba(item.paint, item, m, x0, y0, w, h)[..., 3] * item.opacity
        alpha = np.repeat(np.repeat(alpha, n, axis=0), n, axis=1)
        keep *= 1 - (inside >= 0.5) * alpha
    return 1 - keep.reshape(h, n, w, n).mean(axis=(1, 3))


def rasterize(
    drawing: Drawing, m: Affine, alpha_grid: int = 0
) -> tuple[int, int, np.ndarray]:
    """``(left, top, premultiplied float32 HxWx4)`` of ``drawing`` under ``m``.

    ``m`` maps the glyph's font units to device pixels. Paints are
    interpolated in sRGB and composited in linear light, as Photoshop does.
    With ``alpha_grid`` the shapes' alpha comes from that many point samples
    a side per pixel, as Photoshop's Crisp, Strong and Smooth text has it,
    while their colours keep the exact coverage.
    """
    box = device_bbox(drawing, m)
    if box is None or box[2] <= box[0] or box[3] <= box[1]:
        return 0, 0, np.zeros((0, 0, 4), np.float32)
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    out = np.zeros((h, w, 4), np.float32)
    pictures = np.ones((h, w, 1), np.float32)
    for item in drawing.items:
        if isinstance(item, Picture):
            layer = picture_layer(item, m, x0, y0, w, h)
            _encode(layer, decode=True)
            out *= 1 - layer[..., 3:]
            out += layer
            pictures *= 1 - layer[..., 3:]
            continue
        contours = [c - (x0, y0) for c in _device_contours(item, m)]
        cov = fill_coverage(contours, w, h, nonzero=item.nonzero)
        if not cov.any():
            continue
        rgba = _paint_rgba(item.paint, item, m, x0, y0, w, h)
        alpha = (cov * item.opacity)[..., None] * rgba[..., 3:]
        out *= 1 - alpha
        out[..., :3] += alpha * to_linear(rgba[..., :3])
        out[..., 3:] += alpha
    if alpha_grid > 1 and any(isinstance(i, Shape) for i in drawing.items):
        shapes = _sampled_alpha(drawing, m, x0, y0, w, h, alpha_grid)[..., None]
        alpha = 1 - (1 - shapes) * pictures
        exact = out[..., 3:]
        color = np.where(exact > 0, out[..., :3] / np.maximum(exact, 1e-12), 0.0)
        out = np.concatenate([color * alpha, alpha], axis=-1).astype(np.float32)
    _encode(out, decode=False)
    return x0, y0, out
