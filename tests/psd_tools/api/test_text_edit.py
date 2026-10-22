import io
from typing import cast

import pytest

from psd_tools import PSDImage
from psd_tools.api.layers import TypeLayer
from psd_tools.api.text_edit import set_span_style, set_text, utf16_prefix
from psd_tools.psd import engine_data

from ..utils import full_name

TEXT = "Line 1\rLine 2\rLine 3 and text"


def open_layer(name: str = "text.psd") -> TypeLayer:
    psd = PSDImage.open(full_name(name))
    return next(x for x in psd.descendants() if isinstance(x, TypeLayer))


def lengths(layer: TypeLayer, key: str = "StyleRun") -> list[int]:
    return [int(x.value) for x in layer.engine_dict[key]["RunLengthArray"]]


def fills(layer: TypeLayer) -> list[tuple[float, ...]]:
    out = []
    for run in layer.engine_dict["StyleRun"]["RunArray"]:
        color = run["StyleSheet"]["StyleSheetData"]["FillColor"]["Values"]
        out.append(tuple(float(v.value) for v in color)[1:])
    return out


def test_utf16_prefix() -> None:
    assert utf16_prefix("") == [0]
    assert utf16_prefix("a\U0001f600b") == [0, 1, 3, 4]


def test_fixture_state() -> None:
    layer = open_layer()
    assert layer.text == TEXT
    assert lengths(layer) == [30]
    assert lengths(layer, "ParagraphRun") == [7, 7, 16]


def test_text_setter() -> None:
    layer = open_layer()
    layer.text = "Hello"
    assert layer.text == "Hello"
    assert layer.engine_dict["Editor"]["Text"].value == "Hello\r"
    assert layer._data.text_data[b"Txt "].value == "Hello\x00"
    assert lengths(layer) == [6]
    assert lengths(layer, "ParagraphRun") == [6]
    assert layer._psd.is_updated()
    assert layer.typesetting.text == "Hello"


def test_set_text_without_paragraph_runs_changes_nothing() -> None:
    layer = open_layer()
    del layer.engine_dict["ParagraphRun"]
    before = lengths(layer)
    with pytest.raises(ValueError, match="ParagraphRun"):
        layer.text = "Hello"
    assert layer.text == TEXT
    assert lengths(layer) == before
    assert not layer._psd.is_updated()


def test_set_text_keeps_prefix_and_suffix_runs() -> None:
    layer = open_layer()
    layer.set_span_style(0, 6, fill=(1, 0, 0))
    assert lengths(layer) == [6, 24]
    layer.set_text("Line 1\rLine TWO\rLine 3 and text")
    # "Line 1\rLine " (12 units) keeps both runs; "TWO" takes the style at
    # the edit start; the suffix keeps the black run.
    assert lengths(layer) == [6, 26]
    assert fills(layer) == [(1.0, 0.0, 0.0), (0.0, 0.0, 0.0)]
    assert lengths(layer, "ParagraphRun") == [7, 9, 16]


def test_insertion_continues_the_preceding_run() -> None:
    layer = open_layer()
    layer.set_span_style(7, 13, fill=(0, 1, 0))  # "Line 2"
    assert lengths(layer) == [7, 6, 17]
    layer.set_text("Line 1\rLine 2 and more\rLine 3 and text")
    assert lengths(layer) == [7, 15, 17]
    assert fills(layer)[1] == (0.0, 1.0, 0.0)


def test_replacement_takes_the_replaced_style() -> None:
    layer = open_layer()
    layer.set_span_style(7, 13, fill=(0, 1, 0))  # "Line 2"
    layer.set_text("Line 1\rSecond\rLine 3 and text")
    assert lengths(layer) == [7, 6, 17]
    assert fills(layer)[1] == (0.0, 1.0, 0.0)
    layer.set_text("Line 1\rLine 3 and text")
    assert lengths(layer) == [23]


def test_set_text_with_astral_character() -> None:
    layer = open_layer()
    layer.set_text("Line 1 \U0001f600\rLine 2\rLine 3 and text")
    assert sum(lengths(layer)) == utf16_prefix(layer.text + "\r")[-1] == 33
    assert lengths(layer, "ParagraphRun") == [10, 7, 16]
    # A span after the emoji is addressed in str indices: "Line 2".
    layer.set_span_style(9, 15, fill=(0, 0, 1))
    assert lengths(layer) == [10, 6, 17]


def test_paragraphs_inherit_the_edited_one() -> None:
    layer = open_layer()
    paragraphs = layer.engine_dict["ParagraphRun"]["RunArray"]
    paragraphs[1]["ParagraphSheet"]["Properties"]["Justification"] = (
        engine_data.Integer(2)
    )
    layer.set_text("Line 1\rLine 2a\rLine 2b\rLine 3 and text")
    runs = layer.engine_dict["ParagraphRun"]["RunArray"]
    assert lengths(layer, "ParagraphRun") == [7, 8, 8, 16]
    just = [int(r["ParagraphSheet"]["Properties"]["Justification"].value) for r in runs]
    assert just == [0, 2, 2, 0]


def test_set_text_same_text_changes_nothing() -> None:
    layer = open_layer()
    layer.set_span_style(0, 4, fill=(1, 0, 0))
    psd = cast(PSDImage, layer._psd)
    before = io.BytesIO()
    psd.save(before)
    psd._updated = False
    layer.set_text(layer.text)
    assert not psd.is_updated()
    after = io.BytesIO()
    psd.save(after)
    assert after.getvalue() == before.getvalue()


def test_set_text_single() -> None:
    layer = open_layer()
    layer.set_span_style(0, 4, fill=(1, 0, 0), FauxBold=True)
    layer.set_text("One\rTwo", keep="single")
    assert lengths(layer) == [8]
    assert fills(layer) == [(1.0, 0.0, 0.0)]
    assert lengths(layer, "ParagraphRun") == [4, 4]


def test_span_split_and_merge() -> None:
    layer = open_layer()
    layer.set_span_style(2, 4, fill=(1, 0, 0))
    assert lengths(layer) == [2, 2, 26]
    layer.set_span_style(2, 4, fill=(0, 0, 0))
    assert lengths(layer) == [30]
    layer.set_span_style(0, 0, fill=(1, 0, 0))
    assert lengths(layer) == [30]


def test_span_keys_and_types() -> None:
    layer = open_layer()
    layer.set_span_style(
        0, 4, FauxBold=True, Tracking=50, FontSize=20.0, Font="ArialMT"
    )
    data = layer.engine_dict["StyleRun"]["RunArray"][0]["StyleSheet"]["StyleSheetData"]
    assert isinstance(data["FauxBold"], engine_data.Bool) and data["FauxBold"].value
    assert (
        isinstance(data["Tracking"], engine_data.Integer)
        and data["Tracking"].value == 50
    )
    assert (
        isinstance(data["FontSize"], engine_data.Float)
        and data["FontSize"].value == 20.0
    )
    assert data["Font"].value == 0
    other = layer.engine_dict["StyleRun"]["RunArray"][1]["StyleSheet"]["StyleSheetData"]
    assert "FauxBold" not in other or not other["FauxBold"].value
    with pytest.raises(ValueError, match="FontSet"):
        layer.set_span_style(0, 4, Font="NoSuchFont-Bold")
    with pytest.raises(IndexError):
        layer.set_span_style(0, 99)


def test_set_run_style() -> None:
    layer = open_layer()
    layer.set_span_style(0, 6, fill=(1, 0, 0))
    layer.set_run_style(1, Underline=True)
    assert lengths(layer) == [6, 24]
    runs = layer.engine_dict["StyleRun"]["RunArray"]
    assert runs[1]["StyleSheet"]["StyleSheetData"]["Underline"].value
    assert "Underline" not in runs[0]["StyleSheet"]["StyleSheetData"] or not (
        runs[0]["StyleSheet"]["StyleSheetData"]["Underline"].value
    )
    with pytest.raises(IndexError):
        layer.set_run_style(5)


def test_box_layer_edit() -> None:
    layer = open_layer("layers/type-layer.psd")
    assert layer.text == "A"
    set_text(layer, "AB")
    assert lengths(layer) == [3]
    set_span_style(layer, 1, 2, fill=(0, 0, 1))
    assert lengths(layer) == [1, 1, 1]
    assert layer.text == "AB"


def test_edit_round_trips_through_save() -> None:
    layer = open_layer()
    layer.set_span_style(0, 6, fill=(1, 0, 0), FauxBold=True)
    layer.set_text("Line ONE\rLine 2\rLine 3 and text")
    buf = io.BytesIO()
    cast(PSDImage, layer._psd).save(buf)
    buf.seek(0)
    again = next(
        x for x in PSDImage.open(buf).descendants() if isinstance(x, TypeLayer)
    )
    assert again.text == "Line ONE\rLine 2\rLine 3 and text"
    assert lengths(again) == lengths(layer)
    assert fills(again) == fills(layer)
    assert lengths(again, "ParagraphRun") == [9, 7, 16]
    # The stored pixels are Photoshop's until rasterize().
    assert again.bbox == layer.bbox == (84, 110, 169, 151)


def text_index(layer: TypeLayer) -> object:
    return layer._data.text_data.get(b"TextIndex")


@pytest.mark.parametrize(
    "edit",
    [
        lambda layer: layer.set_text("Gain 3"),
        lambda layer: layer.set_span_style(0, 2, FauxBold=True),
        lambda layer: layer.set_run_style(0, Tracking=20),
    ],
)
def test_an_edit_unlinks_the_layer_from_the_document_text(edit: object) -> None:
    psd = PSDImage.open(full_name("layer-edit.psd"))
    ability, name = (
        next(x for x in psd.descendants() if x.name == n) for n in ("Ability", "Name")
    )
    assert isinstance(ability, TypeLayer) and isinstance(name, TypeLayer)
    assert (text_index(ability), text_index(name)) == (1, 0)
    edit(ability)  # type: ignore[operator]
    assert text_index(ability) is None
    assert text_index(name) == 0
    buf = io.BytesIO()
    psd.save(buf, update_preview=False)
    buf.seek(0)
    again = PSDImage.open(buf)
    indices = {
        x.name: text_index(x) for x in again.descendants() if isinstance(x, TypeLayer)
    }
    assert indices == {"Ability": None, "Name": 0}


def test_an_unchanged_text_keeps_the_link() -> None:
    psd = PSDImage.open(full_name("layer-edit.psd"))
    ability = next(x for x in psd.descendants() if x.name == "Ability")
    assert isinstance(ability, TypeLayer)
    ability.set_text(ability.text)
    ability.set_span_style(1, 1, FauxBold=True)
    assert text_index(ability) == 1
