"""The text layer as data: text, styled runs, paragraphs, shape and transform.

Built from a layer's EngineData; no font or raster dependency.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from psd_tools.composite.text.constants import AUTO_LEADING, SUB_POSITION, SUB_SIZE
from psd_tools.constants import Resource

if TYPE_CHECKING:
    from psd_tools.api.layers import TypeLayer

#: ``(xx, xy, yx, yy, tx, ty)`` as :attr:`TypeLayer.transform` orders it.
Transform = tuple[float, float, float, float, float, float]
RGB = tuple[float, float, float]

#: The type tool's ``AntA`` enum -> ``EngineDict.AntiAlias`` code.
_ANTIALIAS = {b"Anno": 0, b"AnCr": 1, b"AnSt": 2, b"AnSm": 3, b"antiAliasSharp": 4}
#: The mode the engine is calibrated for, when a layer states none.
_DEFAULT_ANTIALIAS = 2
_BLACK: RGB = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Style:
    """A run's character style after the Normal sheet's defaults apply."""

    font: str  #: PostScript name from ``FontSet``
    size: float  #: ``FontSize`` in pt
    baseline: int  #: ``FontBaseline``: 0 normal, 1 superscript, 2 subscript
    tracking: float  #: ``Tracking`` in thousandths of an em
    auto_kerning: bool
    ligatures: bool
    horizontal_scale: float
    vertical_scale: float
    caps: int  #: ``FontCaps``: 0 normal, 1 small caps, 2 all caps
    faux_bold: bool
    faux_italic: bool
    underline: bool
    strikethrough: bool
    baseline_shift: float  #: pt, positive is up
    fill: RGB  #: straight RGB 0..1
    leading: float  #: pt, used when ``auto_leading`` is false
    auto_leading: bool


@dataclass(frozen=True)
class Run:
    """``text[start:end]`` shares one :class:`Style`; ``str`` indices."""

    start: int
    end: int
    style: Style


@dataclass(frozen=True)
class Paragraph:
    """``text[start:end]`` including its paragraph terminator; ``str`` indices."""

    start: int
    end: int
    justification: int  #: 0 left, 1 right, 2 center, 3-5 justify last left/right/center, 6 justify all


@dataclass(frozen=True)
class TextModel:
    """Everything layout needs from a type layer."""

    text: str
    runs: tuple[Run, ...]
    paragraphs: tuple[Paragraph, ...]
    shape: int  #: ``ShapeType``: 0 point, 1 box
    box: (
        tuple[float, float, float, float] | None
    )  #: ``BoxBounds`` (L, T, R, B) as stored, box text only
    antialias: int  #: ``AntA``: 0 none, 1 crisp, 2 strong, 3 smooth, 4 sharp
    transform: Transform
    warnings: tuple[str, ...]
    resolution: float = 72.0  #: document ppi
    sub_size: float = SUB_SIZE  #: ``ResourceDict`` SubscriptSize/SuperscriptSize
    sub_position: float = SUB_POSITION
    auto_leading: float = AUTO_LEADING  #: ``ResourceDict`` auto-leading ratio

    @classmethod
    def from_layer(cls, layer: TypeLayer) -> TextModel:
        """Read a :class:`TypeLayer`'s EngineData."""
        engine = layer.engine_dict
        resources = layer.resource_dict
        text = _text(layer)
        offsets = utf16_to_str(text)
        warnings: list[str] = []

        font_set = [str(_value(f["Name"])) for f in _get(resources, "FontSet", ())]
        sheets = _get(resources, "StyleSheetSet", None)
        normal_index = int(_value(_get(resources, "TheNormalStyleSheet", 0)))
        normal = _nth(sheets, normal_index, "StyleSheetData")
        psheets = _get(resources, "ParagraphSheetSet", None)
        pnormal_index = int(_value(_get(resources, "TheNormalParagraphSheet", 0)))
        pnormal = _nth(psheets, pnormal_index, "Properties")

        runs = _runs(text, offsets, _get(engine, "StyleRun", None), normal, font_set)
        paragraphs = _paragraphs(
            text, offsets, _get(engine, "ParagraphRun", None), pnormal
        )
        if any(p.justification == 6 for p in paragraphs):
            warnings.append("justification_6")

        shape, box = _shape(engine)
        if _warp_style(layer) not in (None, b"warpNone"):
            warnings.append("warp")
        xx, xy, yx, yy, tx, ty = layer.transform

        return cls(
            text=text,
            runs=tuple(runs),
            paragraphs=tuple(paragraphs),
            shape=shape,
            box=box,
            antialias=_antialias(layer, engine),
            transform=(
                float(xx),
                float(xy),
                float(yx),
                float(yy),
                float(tx),
                float(ty),
            ),
            warnings=tuple(warnings),
            resolution=_resolution(layer),
            sub_size=float(_value(_get(resources, "SubscriptSize", SUB_SIZE))),
            sub_position=float(
                _value(_get(resources, "SubscriptPosition", SUB_POSITION))
            ),
            auto_leading=float(_value(_get(pnormal, "AutoLeading", AUTO_LEADING))),
        )


def utf16_to_str(text: str) -> list[int]:
    """``str`` index of every UTF-16 code unit offset; the last entry is ``len(text)``.

    Run lengths in EngineData count UTF-16 code units, so an astral character
    takes two of them.
    """
    out: list[int] = []
    for i, ch in enumerate(text):
        out.append(i)
        if ord(ch) > 0xFFFF:
            out.append(i)
    out.append(len(text))
    return out


def _value(item: Any) -> Any:
    return getattr(item, "value", item)


def _get(mapping: Any, key: str, default: Any) -> Any:
    if mapping is None:
        return default
    value = mapping.get(key)
    return default if value is None else value


def _nth(items: Any, index: int, key: str) -> Any:
    if items is None or not 0 <= index < len(items):
        return None
    return _get(items[index], key, None)


def _text(layer: TypeLayer) -> str:
    editor = _get(layer.engine_dict, "Editor", None)
    item = _get(editor, "Text", None)
    return str(_value(item)) if item is not None else layer.text


def _lookup(data: Any, normal: Any, key: str, default: Any) -> Any:
    for sheet in (data, normal):
        if sheet is not None:
            value = sheet.get(key)
            if value is not None:
                return _value(value)
    return default


def _fill(data: Any, normal: Any) -> RGB:
    color = _lookup(data, normal, "FillColor", None)
    values = _get(color, "Values", None)
    if values is None or len(values) < 4:
        return _BLACK
    # ARGB; Photoshop quantises to 8 bits, no colour management.
    r, g, b = (round(float(_value(v)) * 255) / 255 for v in values[1:4])
    return (r, g, b)


def _style(data: Any, normal: Any, font_set: Sequence[str]) -> Style:
    index = int(_lookup(data, normal, "Font", 0))
    if not 0 <= index < len(font_set):
        index = 0
    return Style(
        font=font_set[index] if font_set else "",
        size=float(_lookup(data, normal, "FontSize", 12.0)),
        baseline=int(_lookup(data, normal, "FontBaseline", 0)),
        tracking=float(_lookup(data, normal, "Tracking", 0)),
        auto_kerning=bool(_lookup(data, normal, "AutoKerning", True)),
        ligatures=bool(_lookup(data, normal, "Ligatures", True)),
        horizontal_scale=float(_lookup(data, normal, "HorizontalScale", 1.0)),
        vertical_scale=float(_lookup(data, normal, "VerticalScale", 1.0)),
        caps=int(_lookup(data, normal, "FontCaps", 0)),
        faux_bold=bool(_lookup(data, normal, "FauxBold", False)),
        faux_italic=bool(_lookup(data, normal, "FauxItalic", False)),
        underline=bool(_lookup(data, normal, "Underline", False)),
        strikethrough=bool(_lookup(data, normal, "Strikethrough", False)),
        baseline_shift=float(_lookup(data, normal, "BaselineShift", 0.0)),
        fill=_fill(data, normal),
        leading=float(_lookup(data, normal, "Leading", 0.0)),
        auto_leading=bool(_lookup(data, normal, "AutoLeading", True)),
    )


def _spans(text: str, offsets: Sequence[int], run: Any) -> list[tuple[int, int, Any]]:
    """``(start, end, run data)`` in ``str`` indices for a ``StyleRun``/``ParagraphRun``."""
    last = len(offsets) - 1
    out: list[tuple[int, int, Any]] = []
    unit = 0
    lengths = _get(run, "RunLengthArray", ())
    for data, length in zip(_get(run, "RunArray", ()), lengths, strict=False):
        n = int(_value(length))
        start, end = offsets[min(unit, last)], offsets[min(unit + n, last)]
        unit += n
        if end > start:
            out.append((start, end, data))
    return out


def _runs(
    text: str, offsets: Sequence[int], style_run: Any, normal: Any, font_set: list[str]
) -> list[Run]:
    runs: list[Run] = []
    for start, end, data in _spans(text, offsets, style_run):
        sheet = _get(_get(data, "StyleSheet", None), "StyleSheetData", None)
        runs.append(Run(start, end, _style(sheet, normal, font_set)))
    # Text the runs do not reach keeps the last style (or the Normal sheet).
    covered = runs[-1].end if runs else 0
    if covered < len(text) or not runs:
        style = runs[-1].style if runs else _style(None, normal, font_set)
        runs.append(Run(covered, len(text), style))
    return runs


def _paragraphs(
    text: str, offsets: Sequence[int], paragraph_run: Any, pnormal: Any
) -> list[Paragraph]:
    """Paragraphs are cut at CR; each takes the ``ParagraphRun`` covering its start."""
    spans = _spans(text, offsets, paragraph_run)
    starts = [s for s, _, _ in spans]
    default = int(_value(_get(pnormal, "Justification", 0)))
    out: list[Paragraph] = []
    start = 0
    for i, ch in enumerate(text):
        if ch == "\r":
            out.append(Paragraph(start, i + 1, 0))
            start = i + 1
    if start < len(text) or not out:
        out.append(Paragraph(start, len(text), 0))
    result: list[Paragraph] = []
    for p in out:
        justification = default
        k = bisect_right(starts, p.start) - 1
        if k >= 0:
            props = _get(_get(spans[k][2], "ParagraphSheet", None), "Properties", None)
            justification = int(_value(_get(props, "Justification", default)))
        result.append(Paragraph(p.start, p.end, justification))
    return result


def _shape(engine: Any) -> tuple[int, tuple[float, float, float, float] | None]:
    children = _get(
        _get(_get(engine, "Rendered", None), "Shapes", None), "Children", None
    )
    if not children:
        return 0, None
    child = children[0]
    cookie = _get(_get(child, "Cookie", None), "Photoshop", None)
    shape = int(_value(_get(child, "ShapeType", _get(cookie, "ShapeType", 0))))
    bounds = _get(cookie, "BoxBounds", None)
    box = None
    if shape == 1 and bounds is not None and len(bounds) == 4:
        left, top, right, bottom = (float(_value(v)) for v in bounds)
        box = (left, top, right, bottom)
    return shape, box


def _antialias(layer: TypeLayer, engine: Any) -> int:
    enum = getattr(layer._data.text_data.get(b"AntA"), "enum", None)
    code = _ANTIALIAS.get(enum) if isinstance(enum, bytes) else None
    if code is None:
        code = int(_value(_get(engine, "AntiAlias", _DEFAULT_ANTIALIAS)))
    return code


def _warp_style(layer: TypeLayer) -> bytes | None:
    warp = layer.warp
    if warp is None:
        return None
    return getattr(warp.get(b"warpStyle"), "enum", None)


def _resolution(layer: TypeLayer) -> float:
    try:
        info = layer._psd.image_resources.get_data(Resource.RESOLUTION_INFO)
    except AttributeError:
        return 72.0
    if info is None or info.horizontal <= 0:
        return 72.0
    ppi = info.horizontal / 65536  # 16.16 fixed point
    return ppi * 2.54 if info.horizontal_unit == 2 else ppi
