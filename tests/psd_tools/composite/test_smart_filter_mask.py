"""A smart object's filter mask blends its filtered raster with the FEid source."""

import importlib
import io
import logging
import tracemalloc
import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from psd_tools import PSDImage
from psd_tools.api.layers import GroupMixin, Layer, SmartObjectLayer
from psd_tools.api.smart_object import SmartObject
from psd_tools.api.utils import get_color_channels
from psd_tools.composite.composite import (
    _UNFILTER_PLANES,
    Compositor,
    _blend_unfiltered,
    _unfilter_under_mask,
    composite,
)
from psd_tools.compression import PSDDecompressionWarning
from psd_tools.constants import Compression, Tag
from psd_tools.psd.descriptor import Bool, Descriptor
from psd_tools.psd.filter_effects import FilterEffectExtra

from ..smart_filters import enable_filters, feid_item, store_items
from ..utils import full_name

pytest.importorskip("scipy")

composite_module = importlib.import_module("psd_tools.composite.composite")

# numpy's cast buffers and tracemalloc's own bookkeeping, flat in the pixel
# count and well under one plane of the blend test.
_TRACE_SLACK = 64 * 1024


def _smart_objects(psd: PSDImage) -> list[SmartObjectLayer]:
    return [layer for layer in psd.descendants() if isinstance(layer, SmartObjectLayer)]


def _rgba(layer: SmartObjectLayer) -> np.ndarray:
    image = layer.composite(force=True)
    assert image is not None
    return np.asarray(image.convert("RGBA")).astype(int)


def _pixels(layer: SmartObjectLayer) -> tuple[np.ndarray, np.ndarray]:
    """The layer's stored colour (h, w, 3) and alpha (h, w, 1) in [0, 1]."""
    color, shape = layer.numpy("color"), layer.numpy("shape")
    assert color is not None and shape is not None
    return color, shape


def _bytes(values: np.ndarray) -> np.ndarray:
    return (values * 255).round().astype(np.uint8)


@dataclass
class Filtered:
    psd: PSDImage
    layer: SmartObjectLayer
    stored: np.ndarray  # uint8 (h, w, 3), the filtered raster
    source: np.ndarray  # uint8 (h, w, 3), the FEid source
    shape: np.ndarray  # float (h, w), the layer's alpha
    opaque: np.ndarray  # bool (h, w)
    half: int
    fx: Descriptor

    def halves(self) -> tuple[np.ndarray, np.ndarray]:
        """The opaque pixels left of ``half`` and right of it."""
        left = self.opaque.copy()
        left[:, self.half :] = False
        return left, self.opaque & ~left


def _filtered(version: int = 2, alpha: np.ndarray | None = None) -> Filtered:
    """layers/smartobject-layer.psd with a mask white on its left half."""
    psd = PSDImage.open(full_name("layers/smartobject-layer.psd"))
    layer = _smart_objects(psd)[0]
    color, shape = _pixels(layer)
    stored = _bytes(color)
    source = 255 - stored
    h, w = stored.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mask[:, : w // 2] = 255
    if alpha is None:
        alpha = _bytes(shape[..., 0])
    fx = enable_filters(layer)
    store_items(psd, [feid_item(layer.bbox, source, alpha, mask, version=version)])
    opaque = shape[..., 0] > 0.999
    return Filtered(psd, layer, stored, source, shape[..., 0], opaque, w // 2, fx)


@pytest.fixture
def filtered() -> Filtered:
    return _filtered()


def _blend_layer(side: int, budget: int | str = "unlimited") -> SmartObjectLayer:
    """A stand-in for :py:func:`_blend_unfiltered`'s layer, on ``(0, 0, side, side)``."""
    return cast(
        SmartObjectLayer,
        SimpleNamespace(
            bbox=(0, 0, side, side),
            name="blend",
            _psd=SimpleNamespace(_max_alloc_bytes=budget),
        ),
    )


def test_the_mask_picks_the_filtered_raster_where_white_and_the_source_where_black(
    filtered: Filtered,
) -> None:
    out = _rgba(filtered.layer)
    left, right = filtered.halves()
    assert np.abs(out[..., :3] - filtered.stored)[left].max() == 0
    assert np.abs(out[..., :3] - filtered.source)[right].max() <= 1


def test_a_disabled_mask_leaves_the_filtered_raster(filtered: Filtered) -> None:
    filtered.fx[b"filterMaskEnable"] = Bool(False)
    out = _rgba(filtered.layer)[..., :3]
    assert np.abs(out - filtered.stored)[filtered.opaque].max() == 0


def test_disabled_filters_leave_the_stored_raster(filtered: Filtered) -> None:
    filtered.fx[b"enab"] = Bool(False)
    out = _rgba(filtered.layer)[..., :3]
    assert np.abs(out - filtered.stored)[filtered.opaque].max() == 0


def test_the_filter_effect_survives_a_save(filtered: Filtered) -> None:
    buf = io.BytesIO()
    filtered.psd.save(buf)
    buf.seek(0)
    layer = _smart_objects(PSDImage.open(buf))[0]
    assert layer.smart_object.filter_effect() is not None
    _, right = filtered.halves()
    assert np.abs(_rgba(layer)[..., :3] - filtered.source)[right].max() <= 1


def test_the_accessors_read_the_block(filtered: Filtered) -> None:
    so = filtered.layer.smart_object
    assert so.has_enabled_filters and so.filter_mask_enabled
    source = so.filter_source()
    assert source is not None
    bbox, color, alpha = source
    assert bbox == filtered.layer.bbox and color.shape == filtered.stored.shape
    assert alpha is not None
    mask = so.filter_mask()
    assert mask is not None
    assert mask[0] == filtered.layer.bbox
    assert mask[1][0, 0] == 255 and mask[1][0, -1] == 0


@pytest.mark.parametrize("in_layer_order", [True, False])
def test_layers_with_disjoint_rects_pair_with_their_own_items(
    in_layer_order: bool,
) -> None:
    psd = PSDImage.open(full_name("blend-modes/rgb-blend-modes.psd"))
    layers = _smart_objects(psd)[:2]
    items = []
    for layer in layers:
        enable_filters(layer)
        h, w = layer.height, layer.width
        zeros = np.zeros((h, w), np.uint8)
        items.append(feid_item(layer.bbox, np.zeros((h, w, 3), np.uint8), zeros, zeros))
    store_items(psd, items if in_layer_order else items[::-1])
    for layer, item in zip(layers, items, strict=True):
        assert layer.smart_object.filter_effect() is item


def test_an_item_that_does_not_contain_the_bbox_is_not_used() -> None:
    psd = PSDImage.open(full_name("layers/smartobject-layer.psd"))
    layer = _smart_objects(psd)[0]
    stored = _rgba(layer)
    enable_filters(layer)
    x0, y0, x1, y1 = layer.bbox
    h, w = y1 - y0, x1 - x0 - 1
    zeros = np.zeros((h, w), np.uint8)
    item = feid_item((x0 + 1, y0, x1, y1), np.zeros((h, w, 3), np.uint8), zeros, zeros)
    store_items(psd, [item])
    assert layer.smart_object.filter_effect() is None
    assert np.array_equal(_rgba(layer), stored)


def _reopened(psd: PSDImage) -> PSDImage:
    buf = io.BytesIO()
    psd.save(buf)
    buf.seek(0)
    return PSDImage.open(buf)


def test_a_layer_keeps_its_item_after_an_earlier_filtered_layer_is_removed() -> None:
    """Item 0 has a black mask, item 1 a white one; the edit leaves FEid as is."""
    psd = PSDImage.open(full_name("blend-modes/rgb-blend-modes.psd"))
    canvas = (0, 0, psd.width, psd.height)
    size = (psd.height, psd.width)
    source = np.zeros((*size, 3), np.uint8)
    alpha = np.full(size, 255, np.uint8)
    items = []
    for index, layer in enumerate(_smart_objects(psd)[:2]):
        enable_filters(layer)
        mask = np.full(size, 255 * index, np.uint8)
        items.append(feid_item(canvas, source, alpha, mask, uuid=str(index)))
    store_items(psd, items)
    psd = _reopened(psd)
    first, second = _smart_objects(psd)[:2]
    stored = np.asarray(second.composite(force=True))
    parent = first.parent
    assert isinstance(parent, GroupMixin)
    parent.remove(first)
    item = second.smart_object.filter_effect()
    assert item is not None and item.uuid == "1"
    assert np.array_equal(np.asarray(second.composite(force=True)), stored)


def test_the_mask_and_the_source_move_with_the_layer(filtered: Filtered) -> None:
    """The FEid rects stay where the layer was opened."""
    layer = filtered.layer
    layer.left += 3
    layer.top -= 2
    so = layer.smart_object
    source, mask = so.filter_source(), so.filter_mask()
    assert source is not None and mask is not None
    assert source[0] == mask[0] == layer.bbox
    out = _rgba(layer)[..., :3]
    left, right = filtered.halves()
    assert np.abs(out - filtered.stored)[left].max() == 0
    assert np.abs(out - filtered.source)[right].max() <= 1


def test_a_duplicate_pairs_as_its_source_does(filtered: Filtered) -> None:
    copy = filtered.layer.duplicate()
    item = filtered.layer.smart_object.filter_effect()
    assert item is not None and copy.smart_object.filter_effect() is item
    copy.left += 4
    assert np.array_equal(_rgba(copy), _rgba(filtered.layer))


@pytest.mark.parametrize("move", [False, True])
def test_a_layer_from_another_document_keeps_its_stored_pixels(move: bool) -> None:
    """The other document's item contains the layer's bbox but is not its own."""
    origin, target = _filtered(), _filtered()
    if move:
        with pytest.warns(DeprecationWarning):
            target.psd.append(origin.layer)
        layer = origin.layer
    else:
        layer = origin.layer.copy_to(target.psd)
    assert layer.smart_object.filter_effect() is None
    out = _rgba(layer)[..., :3]
    assert np.abs(out - origin.stored)[origin.opaque].max() == 0


@pytest.mark.parametrize("larger", [True, False])
def test_the_mask_is_placed_by_document_position(larger: bool) -> None:
    """White left of ``split``: a rect past the bbox, or one that extends black.

    The second rect ends at ``split``.
    """
    psd = PSDImage.open(full_name("layers/smartobject-layer.psd"))
    layer = _smart_objects(psd)[0]
    color, shape = _pixels(layer)
    stored = _bytes(color)
    opaque = shape[..., 0] > 0.999
    source = 255 - stored
    x0, y0, x1, y1 = layer.bbox
    split = x0 + (x1 - x0) // 2
    mask_box = (x0 - 3, y0 - 2, x1 + 4, y1 + 5) if larger else (x0, y0, split, y1)
    mask = np.zeros((mask_box[3] - mask_box[1], mask_box[2] - mask_box[0]), np.uint8)
    mask[:, : split - mask_box[0]] = 255
    enable_filters(layer, extends_white=False)
    alpha = np.full(stored.shape[:2], 255, np.uint8)
    store_items(psd, [feid_item(layer.bbox, source, alpha, mask, mask_box=mask_box)])
    out = _rgba(layer)[..., :3]
    columns = np.arange(x0, x1)[None, :]
    assert np.abs(out - stored)[opaque & (columns < split)].max() == 0
    assert np.abs(out - source)[opaque & (columns >= split)].max() <= 1


def test_a_semi_transparent_source_lerps_alpha_and_premultiplied_colour() -> None:
    psd = PSDImage.open(full_name("layers/smartobject-layer.psd"))
    layer = _smart_objects(psd)[0]
    stored, a_f = _pixels(layer)
    h, w = stored.shape[:2]
    source = np.full((h, w, 3), (40, 200, 120), np.uint8)
    a_u = np.full((h, w), 100, np.uint8)
    mask = np.zeros((h, w), np.uint8)
    mask[:, : w // 3] = 255
    mask[:, w // 3 : 2 * w // 3] = 128
    enable_filters(layer)
    store_items(psd, [feid_item(layer.bbox, source, a_u, mask)])
    m = (mask / 255.0)[..., None]
    alpha = m * a_f + (1 - m) * (a_u / 255.0)[..., None]
    premul = m * a_f * stored + (1 - m) * (a_u / 255.0)[..., None] * (source / 255.0)
    color = premul / np.maximum(alpha, 1e-9)
    out = _rgba(layer)
    assert np.abs(out[..., 3] - alpha[..., 0] * 255).max() <= 1
    visible = alpha[..., 0] > 0.25
    assert np.abs(out[..., :3] - color * 255)[visible].max() <= 1


@pytest.mark.parametrize(
    "filename",
    ["blend-modes/cmyk-blend-modes.psd", "blend-modes/gray-blend-modes.psd"],
)
def test_a_document_that_is_not_rgb_keeps_the_stored_pixels(
    filename: str, caplog: pytest.LogCaptureFixture
) -> None:
    psd = PSDImage.open(full_name(filename))
    layer = _smart_objects(psd)[0]
    before = np.asarray(layer.composite(force=True))
    h, w = layer.height, layer.width
    planes = np.zeros((h, w, get_color_channels(psd)), np.uint8)
    zeros = np.zeros((h, w), np.uint8)
    enable_filters(layer)
    store_items(psd, [feid_item(layer.bbox, planes, zeros, zeros)])
    assert layer.smart_object.filter_source() is not None
    with caplog.at_level(logging.WARNING):
        after = np.asarray(layer.composite(force=True))
    assert np.array_equal(before, after)
    assert "Smart filter mask" not in caplog.text


def test_a_two_byte_row_table_decodes_as_the_four_byte_one() -> None:
    four, two = _filtered(version=2), _filtered(version=1)
    assert np.array_equal(_rgba(two.layer), _rgba(four.layer))


def test_row_counts_that_fit_neither_width_keep_the_stored_pixels(
    filtered: Filtered, caplog: pytest.LogCaptureFixture
) -> None:
    item = filtered.layer.smart_object.filter_effect()
    assert item is not None and isinstance(item.extra, FilterEffectExtra)
    item.extra.data = item.extra.data[:-1]
    with caplog.at_level(logging.WARNING):
        out = _rgba(filtered.layer)[..., :3]
    assert np.abs(out - filtered.stored)[filtered.opaque].max() == 0
    assert "Smart filter mask" in caplog.text


@pytest.mark.parametrize(
    "compression", [Compression.ZIP, Compression.ZIP_WITH_PREDICTION]
)
def test_a_plane_that_does_not_decode_keeps_the_stored_pixels(
    filtered: Filtered, compression: Compression, caplog: pytest.LogCaptureFixture
) -> None:
    """Rather than the black plane decompress() puts in its place."""
    item = filtered.layer.smart_object.filter_effect()
    assert item is not None and isinstance(item.extra, FilterEffectExtra)
    item.extra.compression = int(compression)
    item.extra.data = b"\x78\x9c" + b"\xff" * 20
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with caplog.at_level(logging.WARNING):
            out = _rgba(filtered.layer)[..., :3]
    assert np.abs(out - filtered.stored)[filtered.opaque].max() == 0
    assert "Smart filter mask" in caplog.text
    assert not [w for w in caught if issubclass(w.category, PSDDecompressionWarning)]


def test_the_mask_is_blended_once_for_the_layer_and_its_effects(
    filtered: Filtered, monkeypatch: pytest.MonkeyPatch
) -> None:
    effects = PSDImage.open(full_name("layer_effects.psd"))
    glow = next(layer for layer in effects.descendants() if layer.name == "Outer Glow")
    block = glow.tagged_blocks.get_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO)
    filtered.layer.tagged_blocks.set_data(Tag.OBJECT_BASED_EFFECTS_LAYER_INFO, block)
    reads: list[Layer] = []
    blends: list[Layer] = []
    read, blend = Compositor._read_object, _blend_unfiltered

    def counted_read(self: Compositor, layer: Layer) -> tuple:
        reads.append(layer)
        return read(self, layer)

    def counted_blend(layer: SmartObjectLayer, *args: object) -> tuple:
        blends.append(layer)
        return blend(layer, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(Compositor, "_read_object", counted_read)
    monkeypatch.setattr(composite_module, "_blend_unfiltered", counted_blend)
    composite(filtered.psd)
    assert reads.count(filtered.layer) > 1
    assert blends == [filtered.layer]


def test_a_source_over_the_allocation_budget_keeps_the_stored_pixels(
    caplog: pytest.LogCaptureFixture,
) -> None:
    psd = PSDImage.open(full_name("layers/smartobject-layer.psd"))
    layer = _smart_objects(psd)[0]
    stored = _rgba(layer)
    enable_filters(layer)
    x0, y0, x1, y1 = layer.bbox
    side = 2000
    zeros = np.zeros((side, side), np.uint8)
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    item = feid_item(
        (0, 0, side, side), np.zeros((side, side, 3), np.uint8), zeros, zeros
    )
    item.extra = feid_item(layer.bbox, mask[..., None], mask, mask).extra
    store_items(psd, [item])
    psd.max_alloc_bytes = 4 * side * side
    with pytest.raises(ValueError, match="budget"):
        layer.smart_object.filter_source()
    with caplog.at_level(logging.WARNING):
        assert np.array_equal(_rgba(layer), stored)
    assert "Smart filter mask" in caplog.text


def test_the_blend_stays_within_its_allocation_estimate() -> None:
    side = 512
    layer = _blend_layer(side)
    rng = np.random.default_rng(0)
    color = rng.random((side, side, 3), dtype=np.float32)
    shape = rng.random((side, side, 1), dtype=np.float32)
    source = (
        (0, 0, side, side),
        rng.integers(0, 256, (side, side, 3), dtype=np.uint8),
        rng.integers(0, 256, (side, side), dtype=np.uint8),
    )
    weight = np.full((side, side, 1), 0.5, np.float32)
    tracemalloc.start()
    try:
        _blend_unfiltered(layer, color, shape, weight, source)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak <= _UNFILTER_PLANES * side * side * 4 + _TRACE_SLACK


def test_a_blend_over_the_allocation_budget_keeps_the_stored_pixels(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The budget covers the plane decodes but not the blend."""
    psd = PSDImage.open(full_name("layers/smartobject-layer.psd"))
    layer = _smart_objects(psd)[0]
    h, w = layer.height, layer.width
    enable_filters(layer)
    opaque = np.full((h, w), 255, np.uint8)
    mask = np.zeros((h, w), np.uint8)
    store_items(
        psd, [feid_item(layer.bbox, np.zeros((h, w, 3), np.uint8), opaque, mask)]
    )
    color, shape = _pixels(layer)
    psd.max_alloc_bytes = 40 * h * w
    assert layer.smart_object.filter_source() is not None
    with caplog.at_level(logging.WARNING):
        out = _unfilter_under_mask(layer, color, shape)
    assert out[0] is color and out[1] is shape
    assert "budget" in caplog.text


def test_the_filtered_pixels_stand_outside_the_stored_source() -> None:
    side = 8
    rng = np.random.default_rng(1)
    color = rng.random((side, side, 3), dtype=np.float32)
    shape = rng.random((side, side, 1), dtype=np.float32)
    half = side // 2
    source = (
        (0, 0, half, side),
        np.full((side, half, 3), 200, np.uint8),
        np.full((side, half), 255, np.uint8),
    )
    weight = np.zeros((side, side, 1), np.float32)
    out, alpha = _blend_unfiltered(_blend_layer(side), color, shape, weight, source)
    np.testing.assert_allclose(out[:, half:], color[:, half:], atol=1e-6)
    np.testing.assert_allclose(alpha[:, half:], shape[:, half:], atol=1e-6)
    np.testing.assert_allclose(out[:, :half], 200 / 255, atol=1e-6)
    np.testing.assert_allclose(alpha[:, :half], 1, atol=1e-6)


def test_an_all_white_mask_does_not_decode_the_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    filtered = _filtered()
    item = filtered.layer.smart_object.filter_effect()
    assert item is not None
    h, w = filtered.stored.shape[:2]
    white = np.full((h, w), 255, np.uint8)
    item.extra = feid_item(filtered.layer.bbox, white[..., None], white, white).extra
    calls: list[int] = []

    def spy(self: SmartObject, held: int = 0) -> None:
        calls.append(held)

    monkeypatch.setattr(SmartObject, "_filter_source", spy)
    out = _rgba(filtered.layer)[..., :3]
    assert calls == []
    assert np.abs(out - filtered.stored)[filtered.opaque].max() == 0
