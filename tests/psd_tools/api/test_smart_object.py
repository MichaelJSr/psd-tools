import io
import logging
import os
import struct
import tracemalloc
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from psd_tools.api.layers import SmartObjectLayer
from psd_tools.api.psd_image import PSDImage
from psd_tools.api.smart_object import SmartObject, _has_filters
from psd_tools.constants import LinkedLayerType, Tag
from psd_tools.psd.descriptor import Bool, String
from psd_tools.psd.filter_effects import FilterEffect
from psd_tools.psd.linked_layer import LinkedLayer, LinkedLayers

from ..smart_filters import enable_filters, feid_item, paired_item, store_items
from ..utils import full_name

logger = logging.getLogger(__name__)


@pytest.fixture
def embedded_object() -> Iterator[SmartObject]:
    psdimage = PSDImage.open(full_name("placedLayer.psd"))
    layer = psdimage[3]
    assert isinstance(layer, SmartObjectLayer)
    yield layer.smart_object


@pytest.fixture
def external_object() -> Iterator[SmartObject]:
    psdimage = PSDImage.open(full_name("placedLayer.psd"))
    layer = psdimage[1]
    assert isinstance(layer, SmartObjectLayer)
    yield layer.smart_object


@pytest.fixture
def linked_layer_png() -> Iterator[bytes]:
    with open(full_name("linked-layer.png"), "rb") as f:
        yield f.read()


def test_smart_object_data(
    embedded_object: SmartObject, linked_layer_png: bytes
) -> None:
    assert embedded_object.kind in "data"
    assert embedded_object.filename == "linked-layer.png"
    assert embedded_object.unique_id == "5a96c404-ab9c-1177-97ef-96ca454b82b7"
    assert embedded_object.warp
    assert embedded_object.filesize == 17272
    assert embedded_object.data == linked_layer_png


def test_smart_object_external(external_object: SmartObject) -> None:
    assert external_object.kind in "external"
    assert external_object.filename == "linked-layer.png"
    assert external_object.unique_id == "5a96c402-ab9c-1177-97ef-96ca454b82b7"
    assert external_object.warp
    assert external_object.filesize == 17272


def test_smart_filter_accessors_without_filters(
    embedded_object: SmartObject,
) -> None:
    assert embedded_object.filters is None
    assert embedded_object.has_enabled_filters is False
    assert embedded_object.filter_mask_enabled is False
    assert embedded_object.filter_mask_extends_white is True
    assert embedded_object._paired_effect() is None
    assert embedded_object.filter_source() is None
    assert embedded_object.filter_mask() is None


@dataclass
class FilteredObject:
    psd: PSDImage
    layer: SmartObjectLayer
    item: FilterEffect
    source: np.ndarray  # uint8 (h, w, 3)
    alpha: np.ndarray  # uint8 (h, w)
    mask: np.ndarray  # uint8 (h + 2, w), one row past the bbox at each end


def _filtered_object(version: int = 2) -> FilteredObject:
    """layers/smartobject-layer.psd with random FEid planes and a taller mask."""
    psd = PSDImage.open(full_name("layers/smartobject-layer.psd"))
    layer = psd[0]
    assert isinstance(layer, SmartObjectLayer)
    x0, y0, x1, y1 = layer.bbox
    h, w = y1 - y0, x1 - x0
    rng = np.random.default_rng(version)
    source = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    alpha = rng.integers(0, 256, (h, w), dtype=np.uint8)
    mask = rng.integers(0, 256, (h + 2, w), dtype=np.uint8)
    mask_box = (x0, y0 - 1, x1, y1 + 1)
    item = feid_item(layer.bbox, source, alpha, mask, mask_box, version)
    store_items(psd, [item])
    return FilteredObject(psd, layer, item, source, alpha, mask)


@pytest.mark.parametrize("version", [1, 2])
def test_smart_filter_accessors_read_the_document_block(version: int) -> None:
    """*version* 2 writes 4-byte RLE row counts, as Photoshop does; 1 writes 2-byte."""
    filtered = _filtered_object(version)
    layer = filtered.layer
    so = layer.smart_object
    fx = enable_filters(layer, extends_white=False)
    entry = fx[b"filterFXList"][0]
    entry[b"enab"] = Bool(False)

    assert so.filters is fx
    assert so.has_enabled_filters is False
    entry[b"enab"] = Bool(True)
    assert so.has_enabled_filters is True
    assert so.filter_mask_enabled is True
    assert so.filter_mask_extends_white is False
    assert paired_item(so) is filtered.item
    stored = so.filter_source()
    assert stored is not None
    assert stored[0] == layer.bbox
    assert np.array_equal(stored[1], filtered.source)
    assert stored[2] is not None and np.array_equal(stored[2], filtered.alpha)
    painted = so.filter_mask()
    assert painted is not None
    x0, y0, x1, y1 = layer.bbox
    assert painted[0] == (x0, y0 - 1, x1, y1 + 1)
    assert np.array_equal(painted[1], filtered.mask)


def test_open_finds_smart_filters_without_parsing_others() -> None:
    psd = PSDImage.open(full_name("layers/smartobject-layer.psd"))
    layer = psd[0]
    assert isinstance(layer, SmartObjectLayer)
    assert layer.tagged_blocks[Tag.SMART_OBJECT_LAYER_DATA1]._held is not None
    assert psd._opened_records[id(layer._record)][1] is None
    enable_filters(layer)
    assert _has_filters(layer.tagged_blocks)  # parsed: read as before
    out = io.BytesIO()
    psd.save(out)
    psd = PSDImage.open(io.BytesIO(out.getvalue()))
    assert psd._opened_records[id(psd[0]._record)][1] == 0


def test_smart_filter_item_is_named_after_the_placed_id() -> None:
    """An item named after the layer's placed ID is its own, wherever it lies.

    Photoshop moves an item with its layer.
    """
    filtered = _filtered_object()
    layer = filtered.layer
    so = layer.smart_object
    enable_filters(layer)
    assert so.placed_id == "1b723783-5c1a-5f48-a260-e6925c5c68df"
    assert so.filter_item is None
    decoy = feid_item(
        layer.bbox, filtered.source, filtered.alpha, filtered.mask[1:-1], uuid="x"
    )
    filtered.item.uuid = so.placed_id
    store_items(filtered.psd, [decoy, filtered.item])
    assert so.filter_item is filtered.item
    assert paired_item(so) is filtered.item
    layer.left, layer.top = layer.left + 7, layer.top + 3
    top, left, bottom, right = filtered.item.rectangle
    filtered.item.rectangle = (top + 3, left + 7, bottom + 3, right + 7)
    stored = so.filter_source()
    assert stored is not None and stored[0] == layer.bbox
    assert np.array_equal(stored[1], filtered.source)


def _objects(psd: PSDImage) -> list[SmartObjectLayer]:
    return [layer for layer in psd if isinstance(layer, SmartObjectLayer)]


def _saved(psd: PSDImage) -> PSDImage:
    buf = io.BytesIO()
    psd.save(buf, update_preview=False)
    buf.seek(0)
    return PSDImage.open(buf)


def _with_items() -> PSDImage:
    """placedLayer.psd as saved with an FEid item named after each smart object."""
    psd = PSDImage.open(full_name("placedLayer.psd"))
    plane = np.zeros((1, 1), np.uint8)
    source = np.zeros((1, 1, 3), np.uint8)
    ids = [layer.smart_object.placed_id or "" for layer in _objects(psd)]
    items = [
        feid_item((0, 0, 1, 1), source, plane, plane, uuid=placed) for placed in ids
    ]
    store_items(psd, items)
    return _saved(psd)


def _named(psd: PSDImage) -> list[list[str]]:
    """The IDs in the lnk2, lnkE and FEid blocks."""
    blocks = psd.tagged_blocks
    assert blocks is not None
    tags = (Tag.LINKED_LAYER2, Tag.LINKED_LAYER_EXTERNAL, Tag.FILTER_EFFECTS2)
    return [[item.uuid.strip("\x00") for item in blocks.get_data(tag)] for tag in tags]


def test_an_edited_save_drops_what_no_layer_names() -> None:
    """Photoshop 2023 refuses a file holding any of them."""
    psd = _with_items()
    external, kept, embedded = _objects(psd)
    psd.remove(external)
    psd.remove(embedded)
    so = kept.smart_object
    assert _named(_saved(psd)) == [[], [so.unique_id], [so.placed_id]]


def test_a_moved_layer_keeps_its_data() -> None:
    psd = _with_items()
    named = _named(psd)
    _objects(psd)[2].move_to(psd, index=1)
    assert psd.is_updated()
    assert _named(_saved(psd)) == named


def test_a_smart_object_without_an_id_keeps_every_item() -> None:
    psd = _with_items()
    named = _named(psd)
    gone, unknown, _ = _objects(psd)
    config = unknown.smart_object._config
    assert config is not None
    del config.data[b"Idnt"], config.data[b"placed"]
    psd.remove(gone)
    assert _named(_saved(psd)) == named


def _placed(psd: PSDImage) -> list[str]:
    """Each smart object's stored placed ID, NULs kept."""
    configs = [layer.smart_object._config for layer in _objects(psd)]
    return [str(c.data[b"placed"].value) for c in configs if c is not None]


def test_an_edited_save_ends_a_named_placed_id_with_a_nul() -> None:
    """Photoshop 2023 refuses an FEid item naming an ID without it; others stay."""
    psd = _with_items()
    free, made = (layer.smart_object._config for layer in _objects(psd)[1:])
    assert free is not None and made is not None and psd.tagged_blocks is not None
    free.data[b"placed"] = String("free")
    made.data[b"placed"] = String("made")
    psd.tagged_blocks.get_data(Tag.FILTER_EFFECTS2)[2].uuid = "made"
    kept = _placed(_saved(psd))
    assert kept[1:] == ["free", "made"]
    psd.mark_updated()
    saved = _saved(psd)
    assert _placed(saved) == [kept[0], "free", "made\x00"]
    assert _named(saved)[2] == [kept[0].strip("\x00"), "made"]


def test_an_unedited_save_keeps_an_item_no_layer_names() -> None:
    psd = _with_items()
    assert psd.tagged_blocks is not None
    psd.tagged_blocks.get_data(Tag.FILTER_EFFECTS2)[0].uuid = "x"
    named = _named(psd)
    assert named[2][0] == "x" and _named(_saved(psd)) == named


def test_smart_filter_flags_that_are_not_booleans_read_as_the_defaults() -> None:
    filtered = _filtered_object()
    fx = enable_filters(filtered.layer, extends_white=False)
    fx[b"filterMaskEnable"] = String("on")
    fx[b"filterMaskExtendWithWhite"] = String("off")
    so = filtered.layer.smart_object
    assert so.filter_mask_enabled is False
    assert so.filter_mask_extends_white is True


def test_smart_filter_planes_of_another_depth_are_not_read() -> None:
    filtered = _filtered_object()
    layer = filtered.layer
    enable_filters(layer)
    filtered.item.depth = 16
    assert layer.smart_object.filter_source() is None
    assert layer.smart_object.filter_mask() is None


def test_smart_filter_decodes_charge_what_the_caller_holds() -> None:
    filtered = _filtered_object()
    layer = filtered.layer
    enable_filters(layer)
    so = layer.smart_object
    budget = 64 * layer.width * layer.height
    filtered.psd.max_alloc_bytes = budget
    assert so.filter_mask() is not None and so.filter_source() is not None
    with pytest.raises(ValueError, match="budget"):
        so._filter_mask(held=budget)
    with pytest.raises(ValueError, match="budget"):
        so._filter_source(held=budget)


def _make_external_smart_object(
    full_path: str,
    rel_path: str,
    embedded_filename: str = "linked.png",
) -> SmartObject:
    """Build a SmartObject whose _data mimics an EXTERNAL-kind LinkedLayer."""
    linked_file = {
        b"fullPath": MagicMock(value=full_path),
        b"relPath": MagicMock(value=rel_path),
    }
    data_mock = MagicMock()
    data_mock.kind = LinkedLayerType.EXTERNAL
    data_mock.filename = embedded_filename
    data_mock.linked_file = linked_file

    config_mock = MagicMock()
    config_mock.data = {b"Idnt": MagicMock(value="test-uuid\x00")}

    so = object.__new__(SmartObject)
    so._data = data_mock
    so._config = config_mock
    so._placed_layer = None
    return so


# ---------------------------------------------------------------------------
# Security tests — GHSA-r6c8-3pw3-m54g
# ---------------------------------------------------------------------------


class TestExternalReadConfinement:
    """A crafted PSD cannot read arbitrary files via an absolute ``fullPath``.

    Regression test for GHSA-r6c8-3pw3-m54g.
    """

    def test_data_ignores_malicious_absolute_fullpath(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """.data never reads a linked file, so it cannot disclose one."""
        secret = tmp_path / "secret.txt"
        secret.write_bytes(b"TOP-SECRET")
        workdir = tmp_path / "work"
        workdir.mkdir()
        monkeypatch.chdir(workdir)
        so = _make_external_smart_object(str(secret), "missing.png")
        so._layer = MagicMock(_psd=MagicMock(repo_path=None))
        with pytest.raises(ValueError, match="places nowhere"):
            _ = so.data


def _document_with_embedded_payload(payload: bytes) -> bytes:
    """A minimal v1 document whose lnk2 block is dominated by *payload*.

    The layer section is empty and the document-level tagged blocks hold one
    embedded smart object, mirroring how Photoshop files place them.
    """
    entry = LinkedLayer(kind=LinkedLayerType.DATA, version=4, data=payload)
    body = LinkedLayers([entry]).tobytes()  # type: ignore[list-item]
    block = b"8BIM" + Tag.LINKED_LAYER2.value + struct.pack(">I", len(body)) + body
    header = b"8BPS" + struct.pack(">H", 1) + b"\0" * 6
    header += struct.pack(">HIIHH", 3, 4, 4, 8, 3)
    section = struct.pack(">I", 0) + block
    layers = struct.pack(">I", len(section)) + section
    image = struct.pack(">H", 0) + b"\x80" * 48
    return header + struct.pack(">I", 0) * 2 + layers + image


def test_open_of_a_dominant_embedded_payload_holds_one_copy(tmp_path: Path) -> None:
    """Opening a document dominated by one embedded payload.

    The payload must not be materialized more than once (#899).
    """
    payload = os.urandom(6 * 1024 * 1024)
    document = _document_with_embedded_payload(payload)
    path = tmp_path / "forged.psd"
    path.write_bytes(document)

    tracemalloc.start()
    try:
        with open(path, "rb") as f:
            psd = PSDImage.open(f)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    # The retained copy plus slack: a second full-size buffer fails this.
    ceiling = len(payload) + max(2 * 1024 * 1024, len(payload) // 8)
    assert peak <= ceiling

    blocks = psd.tagged_blocks
    assert blocks is not None
    entries = blocks.get_data(Tag.LINKED_LAYER2)
    assert entries is not None
    assert [entry.data for entry in entries] == [payload]

    buffer = io.BytesIO()
    psd.save(buffer)
    assert buffer.getvalue() == document
