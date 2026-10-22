"""Edits to a type layer's text and character styles in its EngineData.

The public functions take ``str`` indices; EngineData's run lengths count
UTF-16 code units, so an astral character takes two of them. Editing never
renders and needs no optional dependency: the stored pixels stay as they
are until :py:meth:`TypeLayer.rasterize
<psd_tools.api.layers.TypeLayer.rasterize>` replaces them.

A layer's text is stored twice: in its own TySh EngineData and in its story
in the document's Txt2 block (the layer's ``TextIndex`` is the story's
index). Photoshop reads the story and rewrites TySh from it on save, so the
edits here change both; :func:`sync_story` brings back a story that an
older version left behind.
"""

from __future__ import annotations

import copy
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from psd_tools.constants import Tag
from psd_tools.psd import engine_data

if TYPE_CHECKING:
    from psd_tools.api.layers import TypeLayer

__all__ = [
    "add_font",
    "set_run_style",
    "set_span_style",
    "set_text",
    "story_text",
    "sync_story",
    "utf16_prefix",
]

Keep = Literal["runs", "single"]
#: ``(unit start, unit end, run data)``
Span = tuple[int, int, Any]


@dataclass(frozen=True)
class _Spelling:
    """Txt2 key names: Photoshop CC numbers them, older versions name them."""

    objects: str
    stories: str
    model: str
    view: str
    text: str
    paragraphs: str
    styles: str
    runs: str
    data: str
    length: str
    sheet: str
    features: str
    rendered: str
    font_set: str
    font_name: tuple[str, ...]
    paint: tuple[str, str, str, str]
    numbered: bool
    # Document resources and the story's references into them.
    resources: str
    items: str
    item: str
    display: str
    name: str
    style_sheets: tuple[str, str, str, str]  #: set, sheet, parent, features
    paragraph_sheets: tuple[str, str, str, str]
    frame_set: str
    frames: str
    strikes: str
    frame_strike: tuple[str, str, str]  #: tag key, tag, frame index key


_NUMBERED = _Spelling(
    objects="1",
    stories="1",
    model="0",
    view="1",
    text="0",
    paragraphs="5",
    styles="6",
    runs="0",
    data="0",
    length="1",
    sheet="0",
    features="6",
    rendered="1",
    font_set="1",
    font_name=("0", "0", "0"),
    paint=("99", "0", "0", "1"),
    numbered=True,
    resources="0",
    items="0",
    item="0",
    display="1",
    name="0",
    style_sheets=("5", "0", "5", "6"),
    paragraph_sheets=("6", "0", "6", "5"),
    frame_set="8",
    frames="0",
    strikes="2",
    frame_strike=("99", "F", "10"),
)
_NAMED = _Spelling(
    objects="DocumentObjects",
    stories="TextObjects",
    model="Model",
    view="View",
    text="Text",
    paragraphs="ParagraphRun",
    styles="StyleRun",
    runs="RunArray",
    data="RunData",
    length="Length",
    sheet="StyleSheet",
    features="Features",
    rendered="RenderedData",
    font_set="FontSet",
    font_name=("Resource", "Identifier", "Name"),
    paint=("StreamTag", "Color", "Type", "Values"),
    numbered=False,
    resources="DocumentResources",
    items="Resources",
    item="Resource",
    display="DisplayList",
    name="Name",
    style_sheets=("StyleSheetSet", "StyleSheet", "Parent", "Features"),
    paragraph_sheets=("ParagraphSheetSet", "ParagraphSheet", "Parent", "Features"),
    frame_set="TextFrameSet",
    frames="Frames",
    strikes="Strikes",
    frame_strike=("StreamTag", "FrameStrike", "Frame"),
)

#: What ends a paragraph run: Photoshop ends one at a forced line break too.
_PARAGRAPH_ENDS = "\r\x03"

#: StyleSheetData keys a story feature mirrors under the same name, with
#: Photoshop CC's number for it.
_FEATURES = {
    "Font": 0,
    "FontSize": 1,
    "FauxBold": 2,
    "FauxItalic": 3,
    "AutoLeading": 4,
    "Leading": 5,
    "HorizontalScale": 6,
    "VerticalScale": 7,
    "Tracking": 8,
    "BaselineShift": 9,
    "FontCaps": 12,
    "FontBaseline": 13,
    "Ligatures": 18,
    "BaselineDirection": 35,
    "Tsume": 36,
    "StyleRunAlignment": 37,
    "Language": 38,
    "NoBreak": 52,
    "FillColor": 53,
    "StrokeColor": 54,
}


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
    inserted text continues the run before it. Paragraph runs are rebuilt
    to end at each ``\r`` and each ``\x03`` (a forced line break, which also
    ends one in Photoshop's files), a new one inheriting the paragraph the
    edit started in.
    ``keep="single"`` collapses the styles to the first run and the
    paragraphs to the first paragraph. The last paragraph's ``\r`` is
    appended, so a ``text`` ending in one gains an empty paragraph. Setting
    the text the layer already has with ``keep="runs"`` changes nothing.

    The layer's Txt2 story gets the same edit, spliced around what changed
    from its own text; its kern and alternate-glyph marks are not carried
    onto new characters.
    """
    old = _editor_text(layer)
    new = text + "\r"
    story = _story(layer)
    same = new == old and keep == "runs"
    if same and (story is None or story.text == new):
        return

    # Both blocks are read and checked before either is written, so a layer
    # the edit cannot handle is left as it was.
    writes: list[Callable[[], None]] = []
    if not same:
        edit = _Splice(old, new, keep)
        style = _section(layer, "StyleRun")
        paragraphs = _section(layer, "ParagraphRun")
        runs = edit.styles(_spans(style, edit.n_old))
        pars = edit.paragraphs(_spans(paragraphs, edit.n_old))

        def write_tysh() -> None:
            _write_spans(style, runs, merge=True)
            _write_spans(paragraphs, pars, merge=False)
            _set_stored_text(layer, new)

        writes.append(write_tysh)
    if story is not None:
        writes.append(story.respan(new, keep))
    for write in writes:
        write()
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
    a PostScript name in the layer's ``FontSet``, which :func:`add_font`
    extends. ``fill`` is an RGB triple in 0..1. Runs are split at the ends
    of the span and equal neighbours merged.

    With a Txt2 story the key must have a story feature (``_FEATURES``),
    else ``ValueError``: Photoshop would undo a TySh-only style.
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
    story = _story(layer)
    write_story = (
        story.restyle(layer, text, a, b, fill, keys) if story is not None else None
    )
    _write_spans(style, _restyled(spans, a, b, values, _sheet), merge=True)
    if write_story is not None:
        write_story()
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


def story_text(layer: TypeLayer) -> str | None:
    r"""Return the text of the layer's Txt2 story, final ``\r`` included, or None."""
    story = _story(layer)
    return None if story is None else story.text


def sync_story(layer: TypeLayer) -> bool:
    """Bring the layer's Txt2 story to the layer's text; True when it changed.

    The story's text is replaced and its runs spliced as :func:`set_text`
    does around what differs; its styles stay its own.
    """
    story = _story(layer)
    text = _editor_text(layer)
    if story is None or story.text == text:
        return False
    story.respan(text, "runs")()
    _touch(layer)
    return True


class _Splice:
    """Where ``old`` and ``new`` differ, in UTF-16 units, and the runs after."""

    def __init__(self, old: str, new: str, keep: Keep) -> None:
        old_u, new_u = utf16_prefix(old), utf16_prefix(new)
        p = 0
        while p < min(len(old), len(new)) and old[p] == new[p]:
            p += 1
        s = 0
        while s < min(len(old), len(new)) - p and old[-1 - s] == new[-1 - s]:
            s += 1
        self.new, self.new_u, self.keep = new, new_u, keep
        self.n_old, self.n_new = old_u[-1], new_u[-1]
        self.prefix = old_u[p]
        self.suffix = self.n_old - old_u[len(old) - s]
        self.shift = self.n_new - self.n_old

    def _around(self, spans: Sequence[Span], middle_data: Any) -> list[Span]:
        """Keep the prefix and suffix runs; give the middle ``middle_data``."""
        prefix, suffix, n_old = self.prefix, self.suffix, self.n_old
        runs = [(u0, min(u1, prefix), d) for u0, u1, d in spans if u0 < prefix]
        middle = self.n_new - prefix - suffix
        if middle > 0:
            runs.append((prefix, prefix + middle, middle_data))
        runs += [
            (max(u0, n_old - suffix) + self.shift, u1 + self.shift, d)
            for u0, u1, d in spans
            if u1 > n_old - suffix
        ]
        return runs

    def styles(self, spans: Sequence[Span]) -> list[Span]:
        if self.keep == "single":
            return [(0, self.n_new, spans[0][2])]
        replaced = self.n_old - self.prefix - self.suffix > 0
        prefix = self.prefix
        at = _data_at(spans, prefix if replaced or prefix == 0 else prefix - 1)
        return self._around(spans, at)

    def marks(self, spans: Sequence[Span], empty: Any) -> list[Span]:
        """Per-character marks (kerns, alternate glyphs): new text has none."""
        if self.keep == "single":
            return [(0, self.n_new, empty)]
        return self._around(spans, empty)

    def paragraphs(self, spans: Sequence[Span]) -> list[Span]:
        out: list[Span] = []
        start = 0
        for i, c in enumerate(self.new):
            if c in _PARAGRAPH_ENDS:
                end = self.new_u[i + 1]
                data = (
                    spans[0][2]
                    if self.keep == "single"
                    else _data_at(spans, self._old_unit(start))
                )
                out.append((start, end, data))
                start = end
        return out

    def _old_unit(self, u: int) -> int:
        if u < self.prefix:
            return u
        if u >= self.n_new - self.suffix:
            return u - self.shift
        return self.prefix


class _Story:
    """A type layer's story in the document's Txt2 block."""

    def __init__(
        self, root: engine_data.Dict, story: engine_data.Dict, keys: _Spelling
    ) -> None:
        self.root, self.story, self.k = root, story, keys
        self.model = story[keys.model]

    @property
    def text(self) -> str:
        return str(self.model[self.k.text].value)

    def _is_runs(self, section: Any) -> bool:
        runs = (
            section.get(self.k.runs) if isinstance(section, engine_data.Dict) else None
        )
        return isinstance(runs, engine_data.List) and all(
            isinstance(run, engine_data.Dict) and self.k.length in run for run in runs
        )

    def _spans(self, section: engine_data.Dict, total: int) -> list[Span]:
        out: list[Span] = []
        unit = 0
        for run in section[self.k.runs]:
            n = int(run[self.k.length].value)
            if n > 0 and unit < total:
                out.append((unit, min(unit + n, total), run.get(self.k.data)))
            unit += n
        if not out:
            raise ValueError("the layer's Txt2 story has no runs")
        if out[-1][1] < total:
            out[-1] = (out[-1][0], total, out[-1][2])
        return out

    def _write(
        self, section: engine_data.Dict, spans: Sequence[Span], merge: bool
    ) -> None:
        runs: list[Span] = []
        for u0, u1, data in spans:
            if u1 <= u0:
                continue
            if merge and runs and (runs[-1][2] is data or runs[-1][2] == data):
                runs[-1] = (runs[-1][0], u1, runs[-1][2])
            else:
                runs.append((u0, u1, data))
        out: list[Any] = []
        for u0, u1, data in runs:
            run = engine_data.Dict()
            run[self.k.data] = data
            run[self.k.length] = engine_data.Integer(u1 - u0)
            out.append(run)
        section[self.k.runs] = engine_data.List(out)

    def respan(self, new: str, keep: Keep) -> Callable[[], None]:
        """Check the story can take ``new``; return the write that sets it."""
        k = self.k
        edit = _Splice(self.text, new, keep)
        plans: list[tuple[engine_data.Dict, list[Span], bool]] = []
        for key, section in self.model.items():
            if not self._is_runs(section):
                continue
            spans = self._spans(section, edit.n_old)
            name = str(key.value)
            if name == k.styles:
                plans.append((section, edit.styles(spans), True))
            elif name == k.paragraphs:
                plans.append((section, edit.paragraphs(spans), False))
            else:
                plans.append((section, edit.marks(spans, engine_data.Dict()), True))
        for name in (k.styles, k.paragraphs):
            if not self._is_runs(self.model.get(name)):
                raise ValueError(f"the layer's Txt2 story has no {name} runs")
        view = self.story.get(k.view)
        rendered = view.get(k.rendered) if isinstance(view, engine_data.Dict) else None
        if isinstance(rendered, engine_data.Dict) and self._is_runs(rendered):
            # Lines per paragraph: a new paragraph takes the edited one's count.
            spans = self._spans(rendered, edit.n_old)
            plans.append((rendered, edit.paragraphs(spans), False))

        def write() -> None:
            for section, spans, merge in plans:
                self._write(section, spans, merge)
            self.model[k.text].value = new

        return write

    def restyle(
        self,
        layer: TypeLayer,
        text: str,
        a: int,
        b: int,
        fill: tuple[float, float, float] | None,
        keys: Mapping[str, Any],
    ) -> Callable[[], None]:
        """Check the story can take the span style; return the write that sets it."""
        if self.text != text:
            raise ValueError(
                "the layer's Txt2 story holds other text: sync_story() first"
            )
        k = self.k
        values: dict[str, Any] = {}
        for key, value in keys.items():
            number = _FEATURES.get(key)
            if number is None:
                raise ValueError(f"{key} has no Txt2 story feature")
            if key == "Font":
                value = self._font_index(layer, value)
            values[str(number) if k.numbered else key] = _element(value)
        if fill is not None:
            values[str(_FEATURES["FillColor"]) if k.numbered else "FillColor"] = (
                self._paint(fill)
            )
        section = self.model[k.styles]
        spans = self._spans(section, utf16_prefix(text)[-1])

        def features(data: Any) -> engine_data.Dict:
            sheet = data.get(k.sheet)
            if not isinstance(sheet, engine_data.Dict):
                sheet = data[k.sheet] = engine_data.Dict()
            inner = sheet.get(k.features)
            if not isinstance(inner, engine_data.Dict):
                inner = sheet[k.features] = engine_data.Dict()
            return inner

        out = _restyled(spans, a, b, values, features)
        return lambda: self._write(section, out, merge=True)

    def _font_index(self, layer: TypeLayer, value: Any) -> int:
        """Return the story's FontSet index of a ``Font`` name or TySh index."""
        if isinstance(value, str):
            name = value
        else:
            name = _font_names(_font_set(layer.resource_dict))[int(value)]
        names = _story_font_names(self.root, self.k)
        if name not in names:
            raise ValueError(
                f"font {name!r} is not in the Txt2 FontSet {names}; "
                "add it with add_font()"
            )
        return names.index(name)

    def font_adder(
        self, name: str, font_type: int, script: int, synthetic: int
    ) -> Callable[[], None] | None:
        """Return the write that adds face ``name`` to Txt2; None when listed."""
        fonts = _resource_items(self.root, self.k, self.k.font_set)
        if name in _story_font_names(self.root, self.k):
            return None
        k = self.k
        item_key, identifier_key, name_key = k.font_name
        identifier = engine_data.Dict()
        identifier[name_key] = engine_data.String(name)
        if k.numbered:
            # Script and Synthetic are written only when set, as Photoshop does.
            for key, value in (("1", script), ("2", font_type), ("3", synthetic)):
                if value or key == "2":
                    identifier[key] = engine_data.Integer(int(value))
        else:
            identifier["Type"] = engine_data.Integer(int(font_type))
        resource = engine_data.Dict()
        resource[k.paint[0]] = engine_data.Property("CoolTypeFont")
        resource[identifier_key] = identifier
        entry = engine_data.Dict()
        entry[item_key] = resource
        return lambda: fonts.append(entry)

    def _paint(self, fill: tuple[float, float, float]) -> engine_data.Dict:
        tag, color_key, type_key, values_key = self.k.paint
        r, g, b = (float(v) for v in fill)
        color = engine_data.Dict()
        color[type_key] = engine_data.Integer(1)
        argb: list[Any] = [engine_data.Float(v) for v in (1.0, r, g, b)]
        color[values_key] = engine_data.List(argb)
        paint = engine_data.Dict()
        paint[tag] = engine_data.Property("SimplePaint")
        paint[color_key] = color
        return paint


def _story(layer: TypeLayer) -> _Story | None:
    """Return the layer's Txt2 story; None without Txt2 or a TextIndex."""
    psd = getattr(layer, "_psd", None)
    if psd is None:
        return None
    index = layer._data.text_data.get(b"TextIndex")
    txt2 = None if index is None else _txt2(psd)
    if txt2 is None:
        return None
    root, keys, stories = txt2
    i = int(getattr(index, "value", index))
    story = _nth_story(stories, i, keys)
    for other in psd.descendants():  # a copied layer keeps its source's index
        if other is not layer and getattr(other, "kind", None) == "type":
            j = other._data.text_data.get(b"TextIndex")
            if j is not None and int(getattr(j, "value", j)) == i:
                raise ValueError(f"TextIndex {i} is also {other.name!r}'s")
    return _Story(root, story, keys)


def _txt2(psd: Any) -> tuple[engine_data.Dict, _Spelling, engine_data.List] | None:
    """Return the document's Txt2 root, its spelling and its stories, or None."""
    blocks = _document_blocks(psd)
    if blocks is None or Tag.TEXT_ENGINE_DATA not in blocks:
        return None
    root = blocks.get_data(Tag.TEXT_ENGINE_DATA)
    for keys in (_NUMBERED, _NAMED):
        objects = root.get(keys.objects) if isinstance(root, engine_data.Dict) else None
        stories = (
            objects.get(keys.stories) if isinstance(objects, engine_data.Dict) else None
        )
        if isinstance(stories, engine_data.List):
            return root, keys, stories
    raise ValueError("the document's Txt2 block has no text objects")


def _document_blocks(psd: Any) -> Any:
    record = getattr(psd, "_record", None)
    info = getattr(record, "layer_and_mask_information", None)
    return getattr(info, "tagged_blocks", None)


def _nth_story(stories: engine_data.List, i: int, keys: _Spelling) -> engine_data.Dict:
    if not 0 <= i < len(stories):
        raise ValueError(f"TextIndex {i} outside the {len(stories)} Txt2 stories")
    story = stories[i]
    model = story.get(keys.model) if isinstance(story, engine_data.Dict) else None
    if not isinstance(model, engine_data.Dict) or keys.text not in model:
        raise ValueError(f"Txt2 story {i} has no text")
    return story


def _resource_items(
    root: engine_data.Dict, keys: _Spelling, set_key: str
) -> engine_data.List:
    """Return the ``Resources`` list of one of the document's resource sets."""
    node = root.get(keys.resources)
    node = node.get(set_key) if isinstance(node, engine_data.Dict) else None
    node = node.get(keys.items) if isinstance(node, engine_data.Dict) else None
    if not isinstance(node, engine_data.List):
        raise ValueError(f"the document's Txt2 block has no {set_key} resources")
    return node


def _story_font_names(root: engine_data.Dict, keys: _Spelling) -> list[str]:
    names = []
    for font in _resource_items(root, keys, keys.font_set):
        for key in keys.font_name:
            font = font[key]
        names.append(str(font.value))
    return names


def _restyled(
    spans: Sequence[Span],
    a: int,
    b: int,
    values: Mapping[str, Any],
    target: Callable[[Any], Any],
) -> list[Span]:
    """Split the runs at ``a`` and ``b``; set ``values`` on copies inside."""
    out: list[Span] = []
    for u0, u1, data in spans:
        for s0, s1 in ((u0, min(u1, a)), (max(u0, a), min(u1, b)), (max(u0, b), u1)):
            if s1 <= s0:
                continue
            part = data
            if s0 >= a and s1 <= b:
                part = copy.deepcopy(data)
                sheet = target(part)
                for key, value in values.items():
                    sheet[key] = value
            out.append((s0, s1, part))
    return out


def add_font(
    layer: TypeLayer,
    name: str,
    *,
    font_type: int = 1,
    script: int = 0,
    synthetic: int = 0,
) -> int:
    """Add the face PostScript ``name`` to the layer's ``FontSet``.

    Returns the face's index, the value a run's ``Font`` key holds. A face
    already listed keeps its entry. ``font_type`` is Photoshop's outline
    flag, ``1`` for TrueType and ``0`` for PostScript (CFF or Type 1);
    ``script`` and ``synthetic`` are stored as given, ``0`` being Roman
    and none. The face is appended, so no run changes font. The document
    resources in the layer's EngineData list it too, and so does the
    FontSet of the document's Txt2 block when the layer has a story.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("a font needs a PostScript name")
    fonts = _font_set(layer.resource_dict)
    if fonts is None:
        raise ValueError("the layer's EngineData has no FontSet")
    story = _story(layer)
    add_to_story = (
        story.font_adder(name, font_type, script, synthetic) if story else None
    )
    if add_to_story is not None:
        add_to_story()
    names = _font_names(fonts)
    if name in names:
        return names.index(name)
    entry = _font_entry(name, font_type, script, synthetic)
    shared = _font_set(layer.document_resources)
    if shared is not None and name not in _font_names(shared):
        shared.append(copy.deepcopy(entry))
    fonts.append(entry)
    layer.__dict__.pop("_typesetting", None)
    return len(fonts) - 1


def _font_set(resources: engine_data.Dict) -> engine_data.List | None:
    fonts = resources.get("FontSet")
    return fonts if isinstance(fonts, engine_data.List) else None


def _font_names(fonts: engine_data.List | None) -> list[str]:
    return [str(f["Name"].value) for f in fonts or ()]


def _font_entry(
    name: str, font_type: int, script: int, synthetic: int
) -> engine_data.Dict:
    entry = engine_data.Dict()
    entry["Name"] = engine_data.String(name)
    entry["Script"] = engine_data.Integer(int(script))
    entry["FontType"] = engine_data.Integer(int(font_type))
    entry["Synthetic"] = engine_data.Integer(int(synthetic))
    return entry


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
            names = _font_names(_font_set(layer.resource_dict))
            if value not in names:
                raise ValueError(
                    f"font {value!r} is not in the layer's FontSet {names}; "
                    "add it with add_font()"
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
    # TextIndex stays: it names the story the edit wrote, which Photoshop reads.
    layer.__dict__.pop("_typesetting", None)
    layer._psd.mark_updated()
