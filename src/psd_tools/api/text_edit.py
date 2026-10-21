"""Edits to a type layer's text and character styles in its EngineData.

The public functions take ``str`` indices; EngineData's run lengths count
UTF-16 code units, so an astral character takes two of them. Editing never
renders and needs no optional dependency: the stored pixels stay as they
are until :py:meth:`TypeLayer.rasterize
<psd_tools.api.layers.TypeLayer.rasterize>` replaces them.
"""

from __future__ import annotations

import copy
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from psd_tools.psd import engine_data

if TYPE_CHECKING:
    from psd_tools.api.layers import TypeLayer

__all__ = ["set_run_style", "set_span_style", "set_text", "utf16_prefix"]

Keep = Literal["runs", "single"]
#: ``(unit start, unit end, run data)``
Span = tuple[int, int, Any]


def utf16_prefix(text: str) -> list[int]:
    """UTF-16 code-unit offset of every ``str`` index; the last entry is the total."""
    out = [0]
    for c in text:
        out.append(out[-1] + (2 if ord(c) > 0xFFFF else 1))
    return out


def set_text(layer: TypeLayer, text: str, keep: Keep = "runs") -> None:
    r"""Replace the layer's text.

    ``keep="runs"`` keeps the style runs of the unchanged prefix and suffix;
    replaced text takes the style of the first character it replaces and
    inserted text continues the run before it. Paragraphs are rebuilt from
    ``\r``, a new one inheriting the paragraph the edit started in.
    ``keep="single"`` collapses the styles to the first run and the
    paragraphs to the first paragraph. The last paragraph's ``\r`` is
    appended, so a ``text`` ending in one gains an empty paragraph. Setting
    the text the layer already has with ``keep="runs"`` changes nothing.
    """
    old = _editor_text(layer)
    new = text + "\r"
    if new == old and keep == "runs":
        return
    old_u, new_u = utf16_prefix(old), utf16_prefix(new)
    n_old, n_new = old_u[-1], new_u[-1]
    p = 0
    while p < min(len(old), len(new)) and old[p] == new[p]:
        p += 1
    s = 0
    while s < min(len(old), len(new)) - p and old[-1 - s] == new[-1 - s]:
        s += 1
    prefix = old_u[p]
    suffix = n_old - old_u[len(old) - s]
    shift = n_new - n_old

    # Both sections are read and checked before either is written, so a
    # layer the edit cannot handle is left as it was.
    style = _section(layer, "StyleRun")
    spans = _spans(style, n_old)
    paragraphs = _section(layer, "ParagraphRun")
    pspans = _spans(paragraphs, n_old)
    if keep == "single":
        runs = [(0, n_new, spans[0][2])]
    else:
        replaced = n_old - prefix - suffix > 0
        at = _data_at(spans, prefix if replaced or prefix == 0 else prefix - 1)
        runs = [(u0, min(u1, prefix), d) for u0, u1, d in spans if u0 < prefix]
        middle = n_new - prefix - suffix
        if middle > 0:
            runs.append((prefix, prefix + middle, at))
        runs += [
            (max(u0, n_old - suffix) + shift, u1 + shift, d)
            for u0, u1, d in spans
            if u1 > n_old - suffix
        ]
    _write_spans(style, runs, merge=True)

    def old_unit(u: int) -> int:
        if u < prefix:
            return u
        if u >= n_new - suffix:
            return u - shift
        return prefix

    out: list[Span] = []
    start = 0
    for i, c in enumerate(new):
        if c == "\r":
            end = new_u[i + 1]
            data = (
                pspans[0][2] if keep == "single" else _data_at(pspans, old_unit(start))
            )
            out.append((start, end, data))
            start = end
    _write_spans(paragraphs, out, merge=False)

    _set_stored_text(layer, new)
    _touch(layer)


def set_span_style(
    layer: TypeLayer,
    start: int,
    end: int,
    *,
    fill: tuple[float, float, float] | None = None,
    **keys: Any,
) -> None:
    """Set character style keys on ``text[start:end]``.

    ``keys`` are ``StyleSheetData`` names with plain Python values
    (``FauxBold=True``, ``Tracking=50``, ``FontSize=20.0``); ``Font`` takes
    a PostScript name already in the layer's ``FontSet``. ``fill`` is an RGB
    triple in 0..1. Runs are split at the ends of the span and equal
    neighbours merged.
    """
    text = _editor_text(layer)
    if not 0 <= start <= end <= len(text):
        raise IndexError(
            f"span {start}:{end} outside the text of {len(text)} characters"
        )
    units = utf16_prefix(text)
    a, b = units[start], units[end]
    style = _section(layer, "StyleRun")
    spans = _spans(style, units[-1])
    values = _values(layer, fill, keys)
    if b <= a or not values:
        return
    out: list[Span] = []
    for u0, u1, data in spans:
        for s0, s1 in ((u0, min(u1, a)), (max(u0, a), min(u1, b)), (max(u0, b), u1)):
            if s1 <= s0:
                continue
            part = data
            if s0 >= a and s1 <= b:
                part = copy.deepcopy(data)
                sheet = _sheet(part)
                for key, value in values.items():
                    sheet[key] = value
            out.append((s0, s1, part))
    _write_spans(style, out, merge=True)
    _touch(layer)


def set_run_style(
    layer: TypeLayer,
    index: int,
    *,
    fill: tuple[float, float, float] | None = None,
    **keys: Any,
) -> None:
    """:func:`set_span_style` over style run ``index``."""
    text = _editor_text(layer)
    units = utf16_prefix(text)
    spans = _spans(_section(layer, "StyleRun"), units[-1])
    if not 0 <= index < len(spans):
        raise IndexError(f"run {index} of {len(spans)}")
    u0, u1, _ = spans[index]
    set_span_style(
        layer, _str_index(units, u0), _str_index(units, u1), fill=fill, **keys
    )


def _editor_text(layer: TypeLayer) -> str:
    editor = layer.engine_dict.get("Editor")
    item = editor.get("Text") if isinstance(editor, engine_data.Dict) else None
    if item is None:
        raise ValueError("the layer's EngineData has no Editor.Text")
    return str(item.value)


def _section(layer: TypeLayer, key: str) -> engine_data.Dict:
    section = layer.engine_dict.get(key)
    if not isinstance(section, engine_data.Dict) or "RunArray" not in section:
        raise ValueError(f"the layer's EngineData has no {key}")
    return section


def _spans(section: engine_data.Dict, total: int) -> list[Span]:
    """Return the runs as unit spans covering exactly ``total`` units."""
    out: list[Span] = []
    unit = 0
    lengths = section.get("RunLengthArray") or []
    for data, length in zip(section["RunArray"], lengths, strict=False):
        n = int(length.value)
        if n > 0 and unit < total:
            out.append((unit, min(unit + n, total), data))
        unit += n
    if not out:
        raise ValueError("the layer's EngineData has no runs")
    if out[-1][1] < total:
        out[-1] = (out[-1][0], total, out[-1][2])
    return out


def _data_at(spans: Sequence[Span], unit: int) -> Any:
    """Return the run data covering ``unit``, the last run's past the end."""
    starts = [u0 for u0, _, _ in spans]
    return spans[max(0, bisect_right(starts, unit) - 1)][2]


def _write_spans(section: engine_data.Dict, spans: Sequence[Span], merge: bool) -> None:
    runs: list[Span] = []
    for u0, u1, data in spans:
        if u1 <= u0:
            continue
        if merge and runs and (runs[-1][2] is data or runs[-1][2] == data):
            runs[-1] = (runs[-1][0], u1, runs[-1][2])
        else:
            runs.append((u0, u1, data))
    datas: list[Any] = [d for _, _, d in runs]
    counts: list[Any] = [engine_data.Integer(u1 - u0) for u0, u1, _ in runs]
    section["RunArray"] = engine_data.List(datas)
    section["RunLengthArray"] = engine_data.List(counts)


def _str_index(units: Sequence[int], unit: int) -> int:
    return bisect_right(units, unit) - 1


def _sheet(data: Any) -> engine_data.Dict:
    sheet = data.get("StyleSheet")
    if not isinstance(sheet, engine_data.Dict):
        sheet = data["StyleSheet"] = engine_data.Dict()
    inner = sheet.get("StyleSheetData")
    if not isinstance(inner, engine_data.Dict):
        inner = sheet["StyleSheetData"] = engine_data.Dict()
    return inner


def _values(
    layer: TypeLayer, fill: tuple[float, float, float] | None, keys: Mapping[str, Any]
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in keys.items():
        if key == "Font" and isinstance(value, str):
            names = [
                str(f["Name"].value) for f in layer.resource_dict.get("FontSet", [])
            ]
            if value not in names:
                raise ValueError(
                    f"font {value!r} is not in the layer's FontSet {names}"
                )
            value = names.index(value)
        values[key] = _element(value)
    if fill is not None:
        r, g, b = (float(v) for v in fill)
        color = engine_data.Dict()
        color["Type"] = engine_data.Integer(1)
        argb: list[Any] = [engine_data.Float(v) for v in (1.0, r, g, b)]
        color["Values"] = engine_data.List(argb)
        values["FillColor"] = color
    return values


def _element(value: Any) -> Any:
    """Return an EngineData element for a plain Python value."""
    if isinstance(value, engine_data.Dict | engine_data.List):
        return value
    if hasattr(value, "value") and not isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, bool):
        return engine_data.Bool(value)
    if isinstance(value, int):
        return engine_data.Integer(value)
    if isinstance(value, float):
        return engine_data.Float(value)
    if isinstance(value, str):
        return engine_data.String(value)
    if isinstance(value, Mapping):
        out = engine_data.Dict()
        for k, v in value.items():
            out[k] = _element(v)
        return out
    if isinstance(value, list | tuple):
        return engine_data.List([_element(v) for v in value])
    raise TypeError(f"cannot store {type(value).__name__} in EngineData")


def _set_stored_text(layer: TypeLayer, text: str) -> None:
    r"""Write ``text`` (with its final ``\r``) to ``Editor.Text`` and ``Txt``."""
    layer.engine_dict["Editor"]["Text"].value = text
    item = layer._data.text_data.get(b"Txt ")
    if item is not None:
        old = item.value if isinstance(item.value, str) else ""
        item.value = text[:-1] + ("\x00" if old.endswith("\x00") else "")


def _touch(layer: TypeLayer) -> None:
    layer.__dict__.pop("_typesetting", None)
    layer._psd.mark_updated()
