import copy
from types import SimpleNamespace
from typing import Any, cast

import pytest

pytest.importorskip("freetype")
pytest.importorskip("uharfbuzz")

from psd_tools import PSDImage
from psd_tools.api.layers import TypeLayer
from psd_tools.composite.text import model as model_module
from psd_tools.composite.text.layout import frame
from psd_tools.composite.text.model import TextModel, utf16_to_str
from psd_tools.psd import engine_data

from ...utils import full_name


def _type_layer(path: str) -> TypeLayer:
    psd = PSDImage.open(full_name(path))
    return next(x for x in psd.descendants() if isinstance(x, TypeLayer))


@pytest.fixture(scope="module")
def text_layer() -> TypeLayer:
    return _type_layer("text.psd")


@pytest.fixture(scope="module")
def box_layer() -> TypeLayer:
    return _type_layer("layers/type-layer.psd")


class _Layer:
    """The attributes :meth:`TextModel.from_layer` reads, without a PSD."""

    def __init__(
        self,
        engine_dict: Any,
        resource_dict: Any,
        *,
        anta: bytes | None = None,
        warp: dict[bytes, Any] | None = None,
    ) -> None:
        self.engine_dict = engine_dict
        self.resource_dict = resource_dict
        self.transform = (2.0, 0.0, 0.0, 2.0, 10.0, 20.0)
        text_data = {b"AntA": SimpleNamespace(enum=anta)} if anta else {}
        self._data = SimpleNamespace(text_data=text_data)
        self.warp = warp
        self.text = ""
        self._psd = None


def _list(items: list[Any]) -> engine_data.List:
    out = engine_data.List()
    out.extend(items)
    return out


def _fake(
    layer: TypeLayer,
    text: str,
    run_lengths: list[int],
    paragraph_lengths: list[int],
    justifications: list[int] | None = None,
    **kwargs: Any,
) -> TypeLayer:
    """A copy of ``layer``'s EngineData with new text and run lengths."""
    engine = copy.deepcopy(layer.engine_dict)
    engine["Editor"]["Text"] = engine_data.String(text)
    style = engine["StyleRun"]
    run = style["RunArray"][0]
    style["RunArray"] = _list([copy.deepcopy(run) for _ in run_lengths])
    style["RunLengthArray"] = _list([engine_data.Integer(n) for n in run_lengths])
    for k, data in enumerate(style["RunArray"]):
        data["StyleSheet"]["StyleSheetData"]["FontSize"] = engine_data.Float(10.0 + k)
    para = engine["ParagraphRun"]
    first = para["RunArray"][0]
    para["RunArray"] = _list([copy.deepcopy(first) for _ in paragraph_lengths])
    para["RunLengthArray"] = _list([engine_data.Integer(n) for n in paragraph_lengths])
    for k, data in enumerate(para["RunArray"]):
        if justifications:
            data["ParagraphSheet"]["Properties"]["Justification"] = engine_data.Integer(
                justifications[k]
            )
    return cast(TypeLayer, _Layer(engine, layer.resource_dict, **kwargs))


def test_point_text_model(text_layer: TypeLayer) -> None:
    m = TextModel.from_layer(text_layer)
    assert m.text == "Line 1\rLine 2\rLine 3 and text\r"
    assert (m.shape, m.box, m.antialias) == (0, None, 3)
    assert m.transform == pytest.approx((1.0, 0.0, 0.0, 1.0, 83.8125, 119.72265625))
    assert m.warnings == ()
    assert (m.sub_size, m.sub_position, m.auto_leading) == (
        0.583,
        0.333,
        1.2,
    )
    assert len(m.runs) == 1
    run = m.runs[0]
    assert (run.start, run.end) == (0, len(m.text))
    assert (run.style.font, run.style.size, run.style.baseline) == ("ArialMT", 13.0, 0)
    assert run.style.auto_leading
    assert [(p.start, p.end, p.justification) for p in m.paragraphs] == [
        (0, 7, 0),
        (7, 14, 0),
        (14, 30, 0),
    ]


def test_box_text_model(box_layer: TypeLayer) -> None:
    m = TextModel.from_layer(box_layer)
    assert m.text == "A\r"
    assert m.shape == 1
    assert m.box == pytest.approx((0.0, 0.0, 32.0, 28.02122))
    assert m.antialias == 2
    style = m.runs[0].style
    # The run has no FontBaseline: the Normal sheet's value applies.
    assert (style.font, style.size, style.baseline, style.tracking) == (
        "ArialMT",
        30.0,
        0,
        15.0,
    )
    assert style.fill == (1.0, 0.0, 0.0)
    assert style.baseline_shift == pytest.approx(-0.59346)
    assert m.paragraphs == (model_module.Paragraph(0, 2, 2),)


def test_utf16_to_str() -> None:
    assert utf16_to_str("") == [0]
    assert utf16_to_str("ab") == [0, 1, 2]
    assert utf16_to_str("a\U0001f550b") == [0, 1, 1, 2, 3]


def test_astral_run_boundaries(text_layer: TypeLayer) -> None:
    # Run and paragraph lengths count UTF-16 units: the emoji takes two.
    text = "A\U0001f550B\rC\r"
    m = TextModel.from_layer(_fake(text_layer, text, [3, 4], [5, 2]))
    assert [(r.start, r.end) for r in m.runs] == [(0, 2), (2, 6)]
    assert [(p.start, p.end) for p in m.paragraphs] == [(0, 4), (4, 6)]
    assert [r.style.size for r in m.runs] == [10.0, 11.0]


def test_paragraphs_from_text(text_layer: TypeLayer) -> None:
    # A forced break stays inside its paragraph; an interior empty paragraph
    # is kept; text without a final terminator still ends a paragraph.
    text = "A\x03B\r\rC"
    m = TextModel.from_layer(
        _fake(text_layer, text, [6], [4, 1, 1], justifications=[2, 3, 6])
    )
    assert [(p.start, p.end, p.justification) for p in m.paragraphs] == [
        (0, 4, 2),
        (4, 5, 3),
        (5, 6, 6),
    ]
    assert "justification_6" in m.warnings


def test_runs_cover_the_text(text_layer: TypeLayer) -> None:
    # Short run lengths: the last style extends to the end of the text.
    m = TextModel.from_layer(_fake(text_layer, "ABCD\r", [2], [5]))
    assert [(r.start, r.end) for r in m.runs] == [(0, 2), (2, 5)]
    assert m.runs[1].style == m.runs[0].style


def test_default_chain(text_layer: TypeLayer) -> None:
    normal = text_layer.resource_dict["StyleSheetSet"][0]["StyleSheetData"]
    font_set = ["Zero", "One"]
    # Run value wins, then the Normal sheet, then the hard default.
    run = engine_data.Dict()
    run["FontSize"] = engine_data.Float(40.0)
    style = model_module._style(run, normal, font_set)
    assert (style.font, style.size) == ("One", 40.0)
    assert style.auto_leading and style.ligatures and style.auto_kerning
    bare = model_module._style(None, None, font_set)
    assert (bare.font, bare.size, bare.baseline, bare.tracking) == (
        "Zero",
        12.0,
        0,
        0.0,
    )
    assert (bare.horizontal_scale, bare.vertical_scale, bare.caps) == (1.0, 1.0, 0)
    assert (bare.leading, bare.auto_leading, bare.baseline_shift) == (0.0, True, 0.0)
    assert bare.fill == (0.0, 0.0, 0.0)
    assert not (bare.faux_bold or bare.faux_italic or bare.underline)


def test_fill_is_quantised(text_layer: TypeLayer) -> None:
    run = engine_data.Dict()
    color = engine_data.Dict()
    color["Values"] = _list([engine_data.Float(v) for v in (1.0, 0.5, 0.98822, 0.001)])
    run["FillColor"] = color
    style = model_module._style(run, None, ["F"])
    assert style.fill == (128 / 255, 252 / 255, 0.0)


def test_absolute_frame_box(box_layer: TypeLayer) -> None:
    engine = copy.deepcopy(box_layer.engine_dict)
    cookie = engine["Rendered"]["Shapes"]["Children"][0]["Cookie"]["Photoshop"]
    cookie["BoxBounds"] = _list(
        [engine_data.Float(v) for v in (14.0, 1364.0, 1040.9, 1873.0)]
    )
    layer = cast(TypeLayer, _Layer(engine, box_layer.resource_dict, anta=b"AnCr"))
    m = TextModel.from_layer(layer)
    assert m.box == pytest.approx((14.0, 1364.0, 1040.9, 1873.0))
    # Stored in an absolute frame: the box's origin is the text-space origin.
    assert frame(m) == pytest.approx((0.0, 0.0, 1026.9, 509.0))
    relative = TextModel.from_layer(box_layer)
    assert frame(relative) == pytest.approx((0.0, 0.0, 32.0, 28.02122))


def test_antialias(text_layer: TypeLayer) -> None:
    engine = copy.deepcopy(text_layer.engine_dict)
    resources = text_layer.resource_dict
    for enum, code in ((b"Anno", 0), (b"AnCr", 1), (b"AnSt", 2), (b"AnSm", 3)):
        layer = cast(TypeLayer, _Layer(engine, resources, anta=enum))
        assert TextModel.from_layer(layer).antialias == code
    layer = cast(TypeLayer, _Layer(engine, resources, anta=b"antiAliasSharp"))
    assert TextModel.from_layer(layer).antialias == 4
    # Without the descriptor enum the EngineDict value stands in.
    engine["AntiAlias"] = engine_data.Integer(1)
    assert (
        TextModel.from_layer(cast(TypeLayer, _Layer(engine, resources))).antialias == 1
    )
    del engine["AntiAlias"]
    assert (
        TextModel.from_layer(cast(TypeLayer, _Layer(engine, resources))).antialias == 2
    )


def test_platform_antialias_falls_back(text_layer: TypeLayer) -> None:
    # Photoshop writes the platform mode to both the enum and the EngineDict.
    engine = copy.deepcopy(text_layer.engine_dict)
    engine["AntiAlias"] = engine_data.Integer(6)
    resources = text_layer.resource_dict
    for enum in (b"antiAliasPlatformLCD", b"antiAliasPlatformGray"):
        m = TextModel.from_layer(cast(TypeLayer, _Layer(engine, resources, anta=enum)))
        assert m.antialias == 2 and "antialias_fallback" in m.warnings
    m = TextModel.from_layer(cast(TypeLayer, _Layer(engine, resources)))
    assert m.antialias == 2 and "antialias_fallback" in m.warnings
    engine["AntiAlias"] = engine_data.Integer(3)
    m = TextModel.from_layer(cast(TypeLayer, _Layer(engine, resources)))
    assert m.antialias == 3 and "antialias_fallback" not in m.warnings


def test_warp_warning(text_layer: TypeLayer) -> None:
    warp = {b"warpStyle": SimpleNamespace(enum=b"warpArc")}
    layer = cast(
        TypeLayer, _Layer(text_layer.engine_dict, text_layer.resource_dict, warp=warp)
    )
    assert TextModel.from_layer(layer).warnings == ("warp",)
    plain = {b"warpStyle": SimpleNamespace(enum=b"warpNone")}
    layer = cast(
        TypeLayer, _Layer(text_layer.engine_dict, text_layer.resource_dict, warp=plain)
    )
    assert TextModel.from_layer(layer).warnings == ()


def test_resource_constants(text_layer: TypeLayer) -> None:
    resources = copy.deepcopy(text_layer.resource_dict)
    resources["SubscriptSize"] = engine_data.Float(0.6)
    resources["SubscriptPosition"] = engine_data.Float(0.3)
    resources["ParagraphSheetSet"][0]["Properties"]["AutoLeading"] = engine_data.Float(
        1.5
    )
    m = TextModel.from_layer(cast(TypeLayer, _Layer(text_layer.engine_dict, resources)))
    assert (m.sub_size, m.sub_position, m.auto_leading) == (0.6, 0.3, 1.5)
    assert m.transform == (2.0, 0.0, 0.0, 2.0, 10.0, 20.0)
