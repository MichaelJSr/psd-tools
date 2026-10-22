import copy
import io
from typing import Any, cast

import pytest

from psd_tools import PSDImage
from psd_tools.api.layers import TypeLayer
from psd_tools.api.text_edit import set_span_style, set_text, utf16_prefix
from psd_tools.constants import Tag
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
    assert layer.story_text == "Hello\r"  # the named-key Txt2 story follows


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
    layer.set_run_style(1, FauxBold=True)
    assert lengths(layer) == [6, 24]
    runs = layer.engine_dict["StyleRun"]["RunArray"]
    assert runs[1]["StyleSheet"]["StyleSheetData"]["FauxBold"].value
    assert not runs[0]["StyleSheet"]["StyleSheetData"]["FauxBold"].value
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
    assert again.story_text == "Line ONE\rLine 2\rLine 3 and text\r"
    # The stored pixels are Photoshop's until rasterize().
    assert again.bbox == layer.bbox == (84, 110, 169, 151)


# The Txt2 story. text-story.psd is Photoshop 2023's (numbered Txt2 keys): a
# point "Name" and a box "Ability" whose "Frozen" is blue.
STORY = "text-story.psd"
ABILITY = "(1CD) Attacks inflict 1T Frozen.\rLine two."
BLUE = (0.0, 0.50196, 1.0)


def story_layer(name: str, path: str = STORY) -> TypeLayer:
    psd = PSDImage.open(full_name(path))
    return next(
        x for x in psd.descendants() if isinstance(x, TypeLayer) and x.name == name
    )


def txt2(layer: TypeLayer) -> Any:
    psd = cast(PSDImage, layer._psd)
    return psd._record.layer_and_mask_information.tagged_blocks.get_data(
        Tag.TEXT_ENGINE_DATA
    )


def story(layer: TypeLayer) -> Any:
    return txt2(layer)["1"]["1"][int(layer._data.text_data[b"TextIndex"])]


def story_runs(layer: TypeLayer, key: str = "6", view: bool = False) -> list[int]:
    node = story(layer)["1" if view else "0"][key]["0"]
    return [int(run["1"].value) for run in node]


def story_fills(layer: TypeLayer) -> list[tuple[float, ...]]:
    out = []
    for run in story(layer)["0"]["6"]["0"]:
        paint = run["0"]["0"]["6"]["53"]["0"]["1"]
        out.append(tuple(float(v.value) for v in paint)[1:])
    return out


def raw(element: Any) -> bytes:
    buf = io.BytesIO()
    element.write(buf)
    return buf.getvalue()


def saved(layer: TypeLayer) -> PSDImage:
    buf = io.BytesIO()
    cast(PSDImage, layer._psd).save(buf)
    buf.seek(0)
    return PSDImage.open(buf)


def test_story_fixture_state() -> None:
    for name, text in (("Name", "B.F.B"), ("Ability", ABILITY)):
        layer = story_layer(name)
        assert layer.text == text and layer.story_text == text + "\r"
    ability = story_layer("Ability")
    assert story_runs(ability) == lengths(ability) == [25, 6, 12]
    assert story_fills(ability)[1] == fills(ability)[1] == pytest.approx(BLUE)
    assert story_runs(ability, "5") == [33, 10]
    assert story_runs(ability, "1", view=True) == [33, 10]  # lines per paragraph


def test_name_edit_writes_both_blocks_and_nothing_else() -> None:
    layer = story_layer("Name")
    before_psd = PSDImage.open(full_name(STORY))
    old = txt2(story_layer("Name"))
    layer.set_text("B.F.B.")
    assert layer.story_text == "B.F.B.\r"
    assert story_runs(layer) == story_runs(layer, "5") == lengths(layer) == [7]
    again = saved(layer)
    name = next(
        x for x in again.descendants() if isinstance(x, TypeLayer) and x.name == "Name"
    )
    assert name.text == "B.F.B." and name.story_text == "B.F.B.\r"
    # Txt2 parses unchanged but for the edited story's text and run lengths;
    # its view (Photoshop's glyph layout, rebuilt on load) stays as it was.
    new = txt2(name)
    assert raw(new["0"]) == raw(old["0"])
    assert [raw(new["1"][k]) for k in ("0", "2", "3")] == [
        raw(old["1"][k]) for k in ("0", "2", "3")
    ]
    index = int(name._data.text_data[b"TextIndex"])
    for i, (a, b) in enumerate(zip(old["1"]["1"], new["1"]["1"], strict=True)):
        if i != index:
            assert raw(a) == raw(b)
    assert raw(new["1"]["1"][index]["1"]) == raw(old["1"]["1"][index]["1"])
    model = new["1"]["1"][index]["0"]
    model["0"].value = "B.F.B\r"
    for key in ("5", "6"):
        model[key]["0"][0]["1"] = engine_data.Integer(6)
    assert raw(model) == raw(old["1"]["1"][index]["0"])
    # The other layers' records are untouched.
    others = [x for x in before_psd.descendants() if x.name != "Name"]
    for x in others:
        y = next(z for z in again.descendants() if z.name == x.name)
        assert raw(x._record) == raw(y._record)


def test_story_runs_follow_a_replacement() -> None:
    layer = story_layer("Ability")
    layer.set_text("(1CD) Attacks inflict 1T Burning.\rLine two.")
    # "Burning" takes the replaced word's blue in both blocks.
    assert story_runs(layer) == lengths(layer) == [25, 7, 12]
    assert story_fills(layer)[1] == fills(layer)[1] == pytest.approx(BLUE)
    assert story_runs(layer, "5") == story_runs(layer, "1", view=True) == [34, 10]


def test_story_paragraphs_follow_a_new_paragraph() -> None:
    layer = story_layer("Ability")
    layer.set_text(ABILITY + "\rLine three.")
    assert story_runs(layer, "5") == lengths(layer, "ParagraphRun") == [33, 10, 12]
    assert story_runs(layer, "1", view=True) == [33, 10, 12]
    lines = [int(r["0"]["1"].value) for r in story(layer)["1"]["1"]["0"]]
    assert lines == [1, 1, 1]


def test_span_style_writes_the_story_fill() -> None:
    layer = story_layer("Ability")
    layer.set_span_style(33, 37, fill=(1, 0, 0))  # "Line"
    assert story_runs(layer) == lengths(layer) == [25, 6, 2, 4, 6]
    assert story_fills(layer) == fills(layer)
    assert story_fills(layer)[3] == (1.0, 0.0, 0.0)
    layer.set_span_style(33, 37, fill=(0, 0, 0))  # back: the runs merge again
    assert story_runs(layer) == lengths(layer) == [25, 6, 12]
    again = saved(layer)
    ability = next(x for x in again.descendants() if x.name == "Ability")
    assert story_runs(cast(TypeLayer, ability)) == [25, 6, 12]


def test_span_style_keys_map_to_story_features() -> None:
    layer = story_layer("Name")
    layer.set_span_style(0, 2, FontSize=30.0, FauxBold=True, Font="AdobeInvisFont")
    features = story(layer)["0"]["6"]["0"][0]["0"]["0"]["6"]
    assert float(features["1"].value) == 30.0 and features["2"].value
    names = [str(f["0"]["0"]["0"].value) for f in txt2(layer)["0"]["1"]["0"]]
    assert names[int(features["0"].value)] == "AdobeInvisFont"
    # The story's own FontSet decides the index, not the layer's.
    fonts = txt2(layer)["0"]["1"]["0"]
    fonts.reverse()
    layer.set_span_style(0, 2, Font="ArialMT")
    features = story(layer)["0"]["6"]["0"][0]["0"]["0"]["6"]
    assert int(features["0"].value) == len(fonts) - 1


def test_style_key_without_a_story_feature_refuses() -> None:
    layer = story_layer("Ability")
    psd = cast(PSDImage, layer._psd)
    before = raw(txt2(layer)), raw(layer._record)
    with pytest.raises(ValueError, match="Underline has no Txt2 story feature"):
        layer.set_span_style(0, 4, Underline=True)
    assert (raw(txt2(layer)), raw(layer._record)) == before
    assert not psd.is_updated()


def test_sync_story_brings_a_stale_story_back() -> None:
    fresh = story_layer("Ability")
    fresh.set_text("(1CD) Attacks inflict 2T Frozen.\rLine two.")
    want = raw(story(fresh))
    # An older version's edit: TySh changed, the story did not.
    layer = story_layer("Ability")
    stale = copy.deepcopy(story(layer))
    layer.set_text("(1CD) Attacks inflict 2T Frozen.\rLine two.")
    txt2(layer)["1"]["1"][int(layer._data.text_data[b"TextIndex"])] = stale
    assert layer.story_text == ABILITY + "\r"
    with pytest.raises(ValueError, match="sync_story"):
        layer.set_span_style(0, 4, fill=(1, 0, 0))
    assert layer.sync_story()
    assert raw(story(layer)) == want
    assert not layer.sync_story()


def test_same_text_heals_a_stale_story() -> None:
    layer = story_layer("Name")
    stale = copy.deepcopy(story(layer))
    layer.set_text("B.F.B.")
    txt2(layer)["1"]["1"][int(layer._data.text_data[b"TextIndex"])] = stale
    record = raw(layer._record)
    layer.set_text("B.F.B.")
    assert layer.story_text == "B.F.B.\r"
    assert raw(layer._record) == record  # TySh was right already


def test_story_marks_stay_on_their_characters() -> None:
    layer = story_layer("Ability")
    model = story(layer)["0"]
    marks = engine_data.Dict()  # a KernRun: a manual kern on the first "\r"
    runs = []
    for data, n in ((engine_data.Dict(), 32), (None, 1), (engine_data.Dict(), 10)):
        run = engine_data.Dict()
        if data is None:
            data = engine_data.Dict()
            data["0"] = engine_data.Integer(0)
        run["0"] = data
        run["1"] = engine_data.Integer(n)
        runs.append(run)
    marks["0"] = engine_data.List(runs)
    model["8"] = marks
    layer.set_text("(1CD) Attacks inflict 1T Frozen now.\rLine two.")
    got = [(int(r["1"].value), len(r["0"])) for r in model["8"]["0"]]
    assert got == [(36, 0), (1, 1), (10, 0)]  # the new " now" has no kern


def test_set_text_without_a_txt2_block() -> None:
    layer = story_layer("Name")
    blocks = cast(PSDImage, layer._psd)._record.layer_and_mask_information.tagged_blocks
    del blocks[Tag.TEXT_ENGINE_DATA]
    assert layer.story_text is None and not layer.sync_story()
    layer.set_text("B.F.B.")
    layer.set_span_style(0, 1, Underline=True)  # TySh only: nothing to mirror
    assert layer.text == "B.F.B."


def test_story_index_out_of_range_refuses() -> None:
    layer = story_layer("Name")
    layer._data.text_data[b"TextIndex"].value = 7
    with pytest.raises(ValueError, match="TextIndex 7"):
        layer.set_text("B.F.B.")
    assert layer.text == "B.F.B"
