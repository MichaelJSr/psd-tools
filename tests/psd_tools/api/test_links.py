"""Links read and written as repository paths."""

import io

import numpy as np
import pytest

from psd_tools.api.layers import SmartObjectLayer
from psd_tools.api.links import LINK_DATE, RepoTree, link_content_id
from psd_tools.api.psd_image import PSDImage
from psd_tools.constants import Tag
from psd_tools.psd.descriptor import String

from ..utils import full_name

CARD = "Creatures/Kirby/3 Stars/Card.psd"


def _layer(psd: PSDImage, name: str) -> SmartObjectLayer:
    layer = next(x for x in psd.descendants() if x.name == name)
    assert isinstance(layer, SmartObjectLayer)
    return layer


def _external(**paths: str) -> SmartObjectLayer:
    """placedLayer.psd's linked PNG with only the given path strings."""
    psd = PSDImage.open(full_name("placedLayer.psd"))
    layer = _layer(psd, "linked-png")
    linked_file = layer.smart_object._data.linked_file  # type: ignore[union-attr]
    for stored in (b"fullPath", b"relPath", b"originalPath"):
        linked_file.pop(stored, None)
    for key, value in paths.items():
        linked_file[key.encode()] = String(value + "\x00")
    return layer


@pytest.mark.parametrize(
    ("paths", "expected"),
    [
        ({"relPath": "../../../Types/Stars/x.png"}, "Types/Stars/x.png"),
        # Militron_(No_Armor): an absolute URL in relPath
        (
            {
                "relPath": "file:///C//Users/Zoey/Documents/GitHub/TTSCardMaker/Types/x.png"
            },
            "Types/x.png",
        ),
        # Tomorrow_I'll
        (
            {
                "relPath": "file:////Users/m/Desktop/TTSCardMaker/Types/Fragile Equip.png"
            },
            "Types/Fragile Equip.png",
        ),
        (
            {"originalPath": "A:\\Chris\\TTSCardMaker\\Types\\Stars\\x.png"},
            "Types/Stars/x.png",
        ),
        # relPath climbing above the root falls through to fullPath
        (
            {
                "relPath": "../../../../x.png",
                "fullPath": "/TTSCardMaker/MDW/Linked/x.png",
            },
            "MDW/Linked/x.png",
        ),
        ({"relPath": "../../../../Types/x.png"}, None),
    ],
)
def test_link_target_reads_the_stored_strings(
    paths: dict[str, str], expected: str | None
) -> None:
    assert _external(**paths).smart_object.link_target(CARD) == expected


def test_link_target_matches_the_tree_blind_to_case() -> None:
    so = _external(relPath="../../../types/STARS/x.PNG").smart_object
    tree = RepoTree(["Types/Stars/x.png", "Types/y.png"])
    assert so.link_target(CARD, tree) == "Types/Stars/x.png"
    assert so.link_target(CARD, ["Types/y.png"]) is None
    assert so.link_target() is None  # a relative path needs the PSD's


def test_link_writes_photoshops_record_and_keeps_the_pixels() -> None:
    pytest.importorskip("aggdraw")
    data = open(full_name("placedLayer.psd"), "rb").read()
    psd = PSDImage.open(io.BytesIO(data))  # opened from bytes: no file path
    psd.repo_path = CARD
    layer = _layer(psd, "embedded-png")
    before = np.asarray(layer.composite())
    unique_id = layer.smart_object.unique_id
    corners = (1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 1.0, 4.0)
    layer.smart_object.link(
        "MDW/Linked/Card - art.png", 17272, blob="b1", pixels=(9, 8), corners=corners
    )
    saved = io.BytesIO()
    psd.save(saved, update_preview=False)

    back = PSDImage.open(io.BytesIO(saved.getvalue()))
    back.repo_path = CARD
    layer = _layer(back, "embedded-png")
    so = layer.smart_object
    assert (so.kind, so.unique_id, so.filesize) == ("external", unique_id, 17272)
    assert so.link_target() == "MDW/Linked/Card - art.png"
    assert so.content_id == link_content_id("b1", CARD, unique_id)
    item = so._data
    assert item is not None and item.version == 8 and item.timestamp == LINK_DATE
    assert str(item.linked_file[b"relPath"].value) == (
        "../../../MDW/Linked/Card - art.png\x00"
    )
    blocks = layer._record.tagged_blocks
    assert Tag.SMART_OBJECT_LAYER_DATA2 in blocks and Tag.PLACED_LAYER2 not in blocks
    assert so.transform_box == corners
    size = blocks.get_data(Tag.SMART_OBJECT_LAYER_DATA2).data[b"Sz  "]
    assert (size[b"Wdth"].value, size[b"Hght"].value) == (9.0, 8.0)
    assert back.tagged_blocks is not None
    assert Tag.LINKED_LAYER2 not in back.tagged_blocks  # no orphaned item
    assert np.array_equal(np.asarray(layer.composite()), before)
    with pytest.raises(ValueError, match="Card - art.png"):
        so.data
    assert so.read(lambda path: path.encode()) == b"MDW/Linked/Card - art.png"

    again = PSDImage.open(io.BytesIO(data))
    again.repo_path = CARD
    _layer(again, "embedded-png").smart_object.link(
        "MDW/Linked/Card - art.png", 17272, blob="b1", pixels=(9, 8), corners=corners
    )
    repeat = io.BytesIO()
    again.save(repeat, update_preview=False)
    assert repeat.getvalue() == saved.getvalue()


def test_relink_rewrites_the_paths_and_needs_the_psd_path() -> None:
    layer = _external(relPath="file:///C//Users/Zoey/TTSCardMaker/Types/x.png")
    so = layer.smart_object
    with pytest.raises(ValueError, match="repo_path"):
        so.relink()
    so.relink(psd_path=CARD, blob="b2")
    item = so._data
    assert item is not None and item.version == 8 and item.timestamp == LINK_DATE
    assert list(item.linked_file) == [
        b"descVersion",
        b"Nm  ",
        b"fullPath",
        b"originalPath",
        b"relPath",
    ]
    assert so.link_target(CARD) == "Types/x.png"
    assert str(item.linked_file[b"relPath"].value) == "../../../Types/x.png\x00"
    assert so.content_id == link_content_id("b2", CARD, so.unique_id)


def test_a_copied_link_is_rebased_on_the_destination() -> None:
    source = PSDImage.open(full_name("placedLayer.psd"))
    layer = _layer(source, "linked-png")
    dest = PSDImage.new("RGB", source.size)
    with pytest.raises(ValueError, match="repo_path"):
        layer.copy_to(dest)
    assert dest.tagged_blocks is None or not dest.tagged_blocks.get_data(
        Tag.LINKED_LAYER_EXTERNAL
    )
    source.repo_path = "MDW/Templates/Template.psd"
    dest.repo_path = CARD
    copy = layer.copy_to(dest)
    assert isinstance(copy, SmartObjectLayer)
    assert copy.smart_object.link_target() == "MDW/Templates/linked-layer.png"
    assert layer.smart_object.link_target() == "MDW/Templates/linked-layer.png"
