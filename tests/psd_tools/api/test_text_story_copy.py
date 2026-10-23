"""A copied or moved type layer keeps a Txt2 story of its own.

text-story.psd (Photoshop 2023, numbered Txt2 keys) holds a box "Ability"
(story 0, frame 1) and a point "Name" (story 1, frame 0); layer-edit.psd
holds "Ability" (story 1) and "Name" (story 0) in its "Text" group.
"""

import io
from typing import Any, cast

import pytest

from psd_tools import PSDImage
from psd_tools.api.layers import Group, TypeLayer
from psd_tools.constants import Tag

from ..utils import full_name


def _open(path: str) -> PSDImage:
    return PSDImage.open(full_name(path))


def _type(group: Any, name: str) -> TypeLayer:
    layer = next(x for x in group.descendants() if x.name == name)
    assert isinstance(layer, TypeLayer)
    return layer


def _txt2(psd: PSDImage) -> Any:
    blocks = psd._record.layer_and_mask_information.tagged_blocks
    assert blocks is not None
    return blocks.get_data(Tag.TEXT_ENGINE_DATA)


def _stories(psd: PSDImage) -> Any:
    return _txt2(psd)["1"]["1"]


def _frames(psd: PSDImage) -> Any:
    return _txt2(psd)["0"]["8"]["0"]


def _fonts(psd: PSDImage) -> list[str]:
    return [str(f["0"]["0"]["0"].value) for f in _txt2(psd)["0"]["1"]["0"]]


def _index(layer: TypeLayer) -> int:
    return int(layer._data.text_data[b"TextIndex"])


def _story(layer: TypeLayer) -> Any:
    return _stories(cast(PSDImage, layer._psd))[_index(layer)]


def _frame_of(layer: TypeLayer) -> int:
    return int(_story(layer)["1"]["0"][0]["0"].value)


def _run_fonts(layer: TypeLayer) -> list[str]:
    names = _fonts(cast(PSDImage, layer._psd))
    runs = _story(layer)["0"]["6"]["0"]
    return [names[int(run["0"]["0"]["6"]["0"].value)] for run in runs]


def _raw(element: Any) -> bytes:
    buf = io.BytesIO()
    element.write(buf)
    return buf.getvalue()


def _reopen(psd: PSDImage) -> PSDImage:
    buf = io.BytesIO()
    psd.save(buf)
    buf.seek(0)
    return PSDImage.open(buf)


def _check(psd: PSDImage) -> None:
    """Every type layer has a story of its own, with its text and own frame."""
    layers = [x for x in psd.descendants() if isinstance(x, TypeLayer)]
    indices = [_index(x) for x in layers]
    assert len(set(indices)) == len(indices)
    assert all(0 <= i < len(_stories(psd)) for i in indices)
    frames = [_frame_of(x) for x in layers]
    assert len(set(frames)) == len(frames)
    assert all(0 <= i < len(_frames(psd)) for i in frames)
    for layer in layers:
        assert layer.story_text == layer.text + "\r"
        assert not layer.sync_story()


def test_duplicate_gets_its_own_story() -> None:
    psd = _open("text-story.psd")
    name = _type(psd, "Name")
    before = _raw(_story(name))
    copy = name.duplicate()
    assert isinstance(copy, TypeLayer)
    assert (_index(name), _index(copy)) == (1, 2)
    assert (_frame_of(name), _frame_of(copy)) == (0, 2)
    assert _raw(_frames(psd)[2]) == _raw(_frames(psd)[0])
    copy.set_text("Copy")  # its own story: no other layer shares the index
    assert _raw(_story(name)) == before
    again = _reopen(psd)
    _check(again)
    texts = {_index(x): x.text for x in again.descendants() if isinstance(x, TypeLayer)}
    assert texts == {
        0: "(1CD) Attacks inflict 1T Frozen.\rLine two.",
        1: "B.F.B",
        2: "Copy",
    }


def test_copy_into_another_document_then_into_a_group() -> None:
    source, dest = _open("text-story.psd"), _open("layer-edit.psd")
    name, own = _type(source, "Name"), _type(dest, "Name")
    source_bytes = _raw(_txt2(source))
    copy = name.copy_to(dest)
    assert _index(copy) == len(_stories(dest)) - 1 == 2
    assert _raw(_story(copy)["0"]) == _raw(_story(name)["0"])
    assert _raw(_frames(dest)[_frame_of(copy)]) == _raw(_frames(source)[0])
    assert _run_fonts(copy) == _run_fonts(name)
    assert _raw(_txt2(source)) == source_bytes  # the source keeps its story
    # A move inside the document keeps the index and the story.
    copy.move_to(cast(Group, next(x for x in dest if x.name == "Stats Bars")))
    assert _index(copy) == 2
    again = _reopen(dest)
    _check(again)
    moved = _type(next(x for x in again if x.name == "Stats Bars"), "Name")
    assert (_index(moved), moved.story_text) == (2, "B.F.B\r")
    assert _run_fonts(moved) == _run_fonts(name)
    copy.set_text("B.F.B.")  # and it edits as its own
    assert copy.story_text == "B.F.B.\r"
    assert own.story_text == own.text + "\r" != copy.story_text


def test_copy_remaps_fonts_and_adds_missing_ones() -> None:
    # type-layer.psd lists AdobeInvisFont second, text-story.psd fourth.
    source, dest = _open("text-story.psd"), _open("layers/type-layer.psd")
    name = _type(source, "Name")
    name.add_font("stoogedings-Regular")
    name.set_span_style(0, 1, Font="AdobeInvisFont")
    name.set_span_style(1, 2, Font="stoogedings-Regular")
    want = _run_fonts(name)
    assert want[:2] == ["AdobeInvisFont", "stoogedings-Regular"]
    fonts = _fonts(dest)
    copy = name.copy_to(dest)
    assert _run_fonts(copy) == want
    assert _fonts(dest) == fonts + ["stoogedings-Regular"]
    _check(_reopen(dest))


def test_copy_maps_sheets_by_name() -> None:
    source, dest = _open("text-story.psd"), _open("layer-edit.psd")
    sheets = _txt2(dest)["0"]["5"]
    sheets["0"][1]["0"]["0"].value = "Other"  # the destination has no "None"
    name = _type(source, "Name")
    copy = name.copy_to(dest)
    assert [str(s["0"]["0"].value) for s in sheets["0"]] == [
        "Normal RGB",
        "Other",
        "None",
    ]
    assert [int(d["0"].value) for d in sheets["1"]] == [0, 1, 2]
    assert int(_story(copy)["0"]["6"]["0"][0]["0"]["0"]["5"].value) == 2
    assert int(sheets["0"][2]["0"]["5"].value) == 0  # its parent: Normal RGB
    _check(_reopen(dest))


def test_copy_into_a_document_without_txt2() -> None:
    source, dest = _open("text-story.psd"), _open("group.psd")
    ability = _type(source, "Ability")
    copy = ability.copy_to(dest)
    assert (_index(copy), len(_stories(dest)), _frame_of(copy)) == (0, 1, 0)
    assert len(_frames(dest)) == 1
    assert _raw(_frames(dest)[0]) == _raw(_frames(source)[1])  # its box
    again = _reopen(dest)
    _check(again)
    assert _type(again, "Ability").text == ability.text


def test_copy_of_a_group_carries_each_story() -> None:
    source, dest = _open("layer-edit.psd"), _open("text-story.psd")
    text = next(x for x in source if x.name == "Text")
    copy = cast(Group, text.copy_to(dest))
    copies = [x for x in copy.descendants() if isinstance(x, TypeLayer)]
    assert sorted(_index(x) for x in copies) == [2, 3]
    _check(_reopen(dest))


def test_moving_into_another_document_carries_the_story() -> None:
    source, dest = _open("text-story.psd"), _open("layer-edit.psd")
    name = _type(source, "Name")
    group = cast(Group, next(x for x in dest if x.name == "Text"))
    with pytest.warns(DeprecationWarning):
        group.append(name)
    assert name._psd is dest and _index(name) == 2
    _check(_reopen(dest))


def test_copy_between_numbered_and_named_txt2_refuses() -> None:
    source, dest = _open("text.psd"), _open("text-story.psd")
    layer = next(x for x in source.descendants() if isinstance(x, TypeLayer))
    before = _raw(_txt2(dest)), len(list(dest.descendants()))
    with pytest.raises(ValueError, match="numbered keys"):
        layer.copy_to(dest)
    assert (_raw(_txt2(dest)), len(list(dest.descendants()))) == before
