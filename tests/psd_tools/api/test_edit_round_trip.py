"""Structural edits survive ``save(update_preview=False)`` and reopening.

``layer-edit.psd`` is a Photoshop-written 48x64 card: ``Text`` (DropShadow,
Stroke) holding the type layers ``Ability`` and ``Name``; ``Stats Bars`` with
``Bars`` (``hp`` stroked, ``def``) and ``Darks/Hp Dark``, three exposure
digits ``0``-``2`` with masks, ``1`` visible and colour-overlaid.
"""

import io
from collections.abc import Callable
from typing import Any, cast

import pytest

from psd_tools.api.layers import Group, Layer, TypeLayer
from psd_tools.api.psd_image import PSDImage
from psd_tools.constants import Tag

from ..utils import full_name

FIXTURE = "layer-edit.psd"


def _open() -> PSDImage:
    return PSDImage.open(full_name(FIXTURE))


def _find(psd: PSDImage, path: str) -> Any:
    node: Any = psd
    for part in path.split("/"):
        node = next(child for child in node if child.name == part)
    return node


def _named(psd: PSDImage, name: str) -> Any:
    return next(layer for layer in psd.descendants() if layer.name == name)


def _bytes(psd: PSDImage, layer: Layer) -> tuple[bytes, ...]:
    """The layer's record, channels and closing record as written."""
    version = psd._record.header.version
    out = []
    records = [layer._record]
    if isinstance(layer, Group) and layer._bounding_record is not None:
        records.append(layer._bounding_record)
    for record in records:
        buffer = io.BytesIO()
        record.write(buffer, version=version)
        out.append(buffer.getvalue())
    out.append(
        b"".join(bytes([c.compression]) + (c.data or b"") for c in layer._channels)
    )
    return tuple(out)


def _fingerprint(psd: PSDImage) -> dict[int, tuple[bytes, ...]]:
    return {layer.layer_id: _bytes(psd, layer) for layer in psd.descendants()}


def _tree(group: Any, depth: int = 0) -> list[tuple[Any, ...]]:
    rows: list[tuple[Any, ...]] = []
    for layer in group:
        rows.append(
            (
                depth,
                layer.kind,
                layer.name,
                layer.visible,
                layer.opacity,
                layer.bbox,
                layer.mask.bbox if layer.mask else None,
                [effect.name for effect in layer.effects],
                layer.text if isinstance(layer, TypeLayer) else None,
            )
        )
        if isinstance(layer, Group):
            rows += _tree(layer, depth + 1)
    return rows


def _reopen(psd: PSDImage) -> PSDImage:
    buffer = io.BytesIO()
    psd.save(buffer, update_preview=False)
    buffer.seek(0)
    return PSDImage.open(buffer)


def _rename(psd: PSDImage) -> set[int]:
    layer = _named(psd, "Name")
    layer.name = "Title"
    return {layer.layer_id}


def _toggle_digit(psd: PSDImage) -> set[int]:
    shown = _find(psd, "Stats Bars/Darks/Hp Dark/1")
    hidden = _find(psd, "Stats Bars/Darks/Hp Dark/2")
    hidden.copy_effects_from(shown)
    shown.clear_effects()
    shown.visible, hidden.visible = False, True
    return {shown.layer_id, hidden.layer_id}


def _move(psd: PSDImage) -> set[int]:
    ability = _named(psd, "Ability")
    ability.move_to(psd, index=1)
    return {ability.layer_id}


def _remove(psd: PSDImage) -> set[int]:
    digit = _find(psd, "Stats Bars/Darks/Hp Dark/2")
    digit.parent.remove(digit)
    return {digit.layer_id}


def _group(psd: PSDImage) -> set[int]:
    stats = _find(psd, "Stats Bars")
    bars = _find(psd, "Stats Bars/Bars")
    holder = stats.create_group(name="Holder")
    holder.move_to(stats, index=0)
    bars.move_to(holder)
    return {bars.layer_id}


def _template_row(psd: PSDImage) -> set[int]:
    template = _open()
    darks = _find(psd, "Stats Bars/Darks")
    row = _find(template, "Stats Bars/Darks/Hp Dark").copy_to(darks, index=0)
    row.name = "Def Dark"
    for digit in row:
        digit.mask.top += 10
    return set()


def _move_mask(psd: PSDImage) -> set[int]:
    digit = _find(psd, "Stats Bars/Darks/Hp Dark/0")
    digit.mask.offset = (digit.mask.left + 3, digit.mask.top - 2)
    return {digit.layer_id}


def _font(psd: PSDImage) -> set[int]:
    ability = _named(psd, "Ability")
    ability.add_font("stoogedings-Regular")
    ability.set_span_style(0, 4, Font="stoogedings-Regular")
    return {ability.layer_id}


def _text(psd: PSDImage) -> set[int]:
    ability = _named(psd, "Ability")
    ability.set_text("Gain 3")
    return {ability.layer_id}


def _foreign_type(psd: PSDImage) -> set[int]:
    source = PSDImage.open(full_name("layers/type-layer.psd"))[0]
    source.copy_to(_find(psd, "Text"))
    return set()


EDITS: dict[str, Callable[[PSDImage], set[int]]] = {
    "rename": _rename,
    "toggle digit": _toggle_digit,
    "move": _move,
    "remove": _remove,
    "group": _group,
    "template row": _template_row,
    "move mask": _move_mask,
    "font": _font,
    "text": _text,
    "foreign type layer": _foreign_type,
}


def _check(psd: PSDImage, edit: Callable[[PSDImage], set[int]]) -> PSDImage:
    before = _fingerprint(psd)
    touched = edit(psd)
    edited, tree = _fingerprint(psd), _tree(psd)
    again = _reopen(psd)
    assert _tree(again) == tree
    reopened = _fingerprint(again)
    assert reopened == edited
    for layer_id, data in before.items():
        if layer_id not in touched and layer_id in reopened:
            assert reopened[layer_id] == data, layer_id
    return again


@pytest.mark.parametrize("name", EDITS)
def test_an_edit_round_trips(name: str) -> None:
    _check(_open(), EDITS[name])


def test_edits_saved_one_after_another_round_trip() -> None:
    psd = _open()
    for edit in EDITS.values():
        psd = _check(psd, edit)
    assert [layer.name for layer in _find(psd, "Text")] == ["Title", "A"]
    assert [layer.name for layer in _find(psd, "Stats Bars/Darks")] == [
        "Def Dark",
        "Hp Dark",
    ]


def test_an_unedited_document_saves_byte_for_byte() -> None:
    with open(full_name(FIXTURE), "rb") as f:
        data = f.read()
    buffer = io.BytesIO()
    _open().save(buffer, update_preview=False)
    assert buffer.getvalue() == data


def test_a_rename_leaves_the_preview_current() -> None:
    psd = _open()
    _rename(psd)
    assert not psd.is_updated()


def test_the_fixture_links_its_type_layers_to_the_document_text() -> None:
    psd = _open()
    assert psd.tagged_blocks is not None
    assert Tag.TEXT_ENGINE_DATA in psd.tagged_blocks
    indices = {
        layer.name: cast(TypeLayer, layer)._data.text_data.get(b"TextIndex")
        for layer in psd.descendants()
        if isinstance(layer, TypeLayer)
    }
    assert indices == {"Ability": 1, "Name": 0}
