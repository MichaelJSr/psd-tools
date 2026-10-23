"""``Layer.copy_effects_from()`` and ``Layer.clear_effects()``."""

import io
from typing import Any

import pytest

from psd_tools.api.layers import Layer
from psd_tools.api.psd_image import PSDImage
from psd_tools.constants import Tag
from psd_tools.psd.descriptor import String
from psd_tools.terminology import Enum, Key

from ..utils import full_name

EFFECT_TAGS = (
    Tag.OBJECT_BASED_EFFECTS_LAYER_INFO,
    Tag.OBJECT_BASED_EFFECTS_LAYER_INFO_V0,
    Tag.OBJECT_BASED_EFFECTS_LAYER_INFO_V1,
    Tag.EFFECTS_LAYER,
)
PATTERN_TAGS = (Tag.PATTERNS1, Tag.PATTERNS2, Tag.PATTERNS3)


def _card() -> PSDImage:
    return PSDImage.open(full_name("layer-edit.psd"))


def _named(psd: PSDImage, name: str) -> Layer:
    return next(layer for layer in psd.descendants() if layer.name == name)


def _tags(layer: Layer) -> list[bytes]:
    return [bytes(tag) for tag in layer.tagged_blocks if tag in EFFECT_TAGS]


def _names(layer: Layer) -> list[str]:
    return [effect.name for effect in layer.effects]


def _patterns(psd: PSDImage) -> list[Any]:
    if psd.tagged_blocks is None:
        return []
    return [p for tag in PATTERN_TAGS for p in psd.tagged_blocks.get_data(tag) or []]


def test_the_effects_are_copied() -> None:
    psd = _card()
    shown, hidden = _named(psd, "1"), _named(psd, "0")
    assert _names(shown) == ["ColorOverlay"] and _names(hidden) == []

    hidden.copy_effects_from(shown)

    assert _names(hidden) == ["ColorOverlay"] == _names(shown)
    assert _tags(hidden) == _tags(shown) == [b"lfx2", b"lrFX"]
    for tag in _tags(shown):
        assert hidden.tagged_blocks[tag] == shown.tagged_blocks[tag]
        assert hidden.tagged_blocks[tag] is not shown.tagged_blocks[tag]
    assert psd.is_updated()


def test_the_copy_is_deep() -> None:
    psd = _card()
    shown, hidden = _named(psd, "1"), _named(psd, "0")
    hidden.copy_effects_from(shown)
    descriptor = hidden.effects[0].descriptor
    descriptor[Key.Enabled] = False
    assert shown.effects[0].enabled
    assert not hidden.effects[0].enabled


def test_the_copy_replaces_every_block_of_the_target() -> None:
    psd = _card()
    bar, text = _named(psd, "hp"), _named(psd, "Text")
    assert _tags(bar) == [b"lfx2", b"lrFX"]
    assert _tags(text) == [b"lfxs", b"lrFX"]

    bar.copy_effects_from(text)

    # lfx2 goes; lrFX is replaced where it stands, lfxs appended.
    assert _tags(bar) == [b"lrFX", b"lfxs"]
    assert _names(bar) == _names(text) == ["DropShadow", "Stroke"]


def test_an_existing_block_is_replaced_in_place() -> None:
    psd = _card()
    shown, bar = _named(psd, "1"), _named(psd, "hp")
    order = list(bar.tagged_blocks)
    bar.copy_effects_from(shown)
    assert list(bar.tagged_blocks) == order
    assert _names(bar) == ["ColorOverlay"]


def test_a_source_without_effects_clears_the_target() -> None:
    psd = _card()
    shown, hidden = _named(psd, "1"), _named(psd, "0")
    shown.copy_effects_from(hidden)
    assert _tags(shown) == [] and _names(shown) == []
    assert psd.is_updated()


def test_clear_effects() -> None:
    psd = _card()
    text = _named(psd, "Text")
    text.clear_effects()
    assert _tags(text) == []
    assert not text.has_effects()
    assert psd.is_updated()


@pytest.mark.parametrize("edit", ["clear", "copy none", "copy self"])
def test_a_layer_left_as_it_was_keeps_the_preview_current(edit: str) -> None:
    psd = _card()
    hidden, shown = _named(psd, "0"), _named(psd, "1")
    if edit == "clear":
        hidden.clear_effects()
    elif edit == "copy none":
        hidden.copy_effects_from(_named(psd, "2"))
    else:
        shown.copy_effects_from(shown)
        assert _names(shown) == ["ColorOverlay"]
    assert not psd.is_updated()


def test_the_source_must_be_a_layer() -> None:
    psd = _card()
    with pytest.raises(TypeError, match="layer"):
        _named(psd, "0").copy_effects_from(psd)  # type: ignore[arg-type]


def test_effects_copied_from_another_document_carry_their_pattern() -> None:
    source = PSDImage.open(full_name("patterns.psd"))
    layer = _named(source, "Rectangle 1 copy 6")
    assert _names(layer) == ["PatternOverlay"]
    dest = _card()
    target = _named(dest, "hp")
    known = len(_patterns(dest))

    target.copy_effects_from(layer)

    pattern = target.tagged_blocks.get_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO)[
        b"patternFill"
    ][Enum.Pattern]
    ref = pattern[Key.ID]
    assert isinstance(ref, String)
    assert len(_patterns(dest)) == known + 1
    assert (
        dest._get_pattern(ref.value.rstrip("\x00"))
        == _patterns(source)[
            [p.pattern_id for p in _patterns(source)].index(ref.value.rstrip("\x00"))
        ]
    )


def test_copied_effects_round_trip_through_save() -> None:
    psd = _card()
    _named(psd, "0").copy_effects_from(_named(psd, "1"))
    _named(psd, "1").clear_effects()
    buffer = io.BytesIO()
    psd.save(buffer, update_preview=False)
    buffer.seek(0)
    again = PSDImage.open(buffer)
    assert _names(_named(again, "0")) == ["ColorOverlay"]
    assert _names(_named(again, "1")) == []
    assert (
        _named(again, "0").tagged_blocks[Tag.OBJECT_BASED_EFFECTS_LAYER_INFO]
        == (_named(psd, "0").tagged_blocks[Tag.OBJECT_BASED_EFFECTS_LAYER_INFO])
    )
