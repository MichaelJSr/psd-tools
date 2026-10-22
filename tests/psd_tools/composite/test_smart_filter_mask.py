"""A smart object's filter mask blends its filtered raster with the FEid source."""

import io
import logging
import tracemalloc
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from psd_tools import PSDImage
from psd_tools.api.layers import GroupMixin, SmartObjectLayer
from psd_tools.api.utils import get_color_channels
from psd_tools.composite.composite import _UNFILTER_PLANES, _blend_unfiltered
from psd_tools.compression import compress
from psd_tools.constants import Compression, Tag
from psd_tools.psd.descriptor import Bool, Descriptor, List
from psd_tools.psd.filter_effects import (
    FilterEffect,
    FilterEffectChannel,
    FilterEffectExtra,
)

from ..utils import full_name

pytest.importorskip("scipy")

MAX_CHANNELS = 24

# numpy's cast buffers and tracemalloc's own bookkeeping, flat in the pixel
# count and well under one plane of the blend test.
_TRACE_SLACK = 64 * 1024

Box = tuple[int, int, int, int]


def _plane(array: np.ndarray, version: int = 2) -> FilterEffectChannel:
    """An RLE plane; *version* 2 writes 4-byte row counts, 1 writes 2-byte ones."""
    h, w = array.shape
    data = compress(array.tobytes(), Compression.RLE, w, h, 8, version)
    return FilterEffectChannel(
        is_written=1, compression=int(Compression.RLE), data=data
    )


def _item(
    box: Box,
    source: np.ndarray,
    alpha: np.ndarray,
    mask: np.ndarray,
    mask_box: Box | None = None,
    version: int = 2,
) -> FilterEffect:
    """An FEid item: *source* and *alpha* on *box*, *mask* on *mask_box* (or *box*)."""
    x0, y0, x1, y1 = box
    channels = [_plane(source[..., i], version) for i in range(source.shape[-1])]
    unwritten = MAX_CHANNELS + 1 - len(channels)
    channels += [FilterEffectChannel(is_written=0) for _ in range(unwritten)]
    channels.append(_plane(alpha, version))
    mx0, my0, mx1, my1 = mask_box or box
    plane = _plane(mask, version)
    extra = FilterEffectExtra(
        is_written=1,
        rectangle=[my0, mx0, my1, mx1],  # type: ignore[list-item]
        compression=int(Compression.RLE),
        data=plane.data,
    )
    return FilterEffect(
        uuid="0",
        version=1,
        rectangle=(y0, x0, y1, x1),
        depth=8,
        max_channels=MAX_CHANNELS,
        channels=channels,
        extra=extra,
    )


def _enable(
    layer: SmartObjectLayer, mask: bool = True, extends_white: bool = True
) -> Descriptor:
    """Give *layer* one enabled smart filter and return its ``filterFX``."""
    fx = Descriptor(classID=b"null")
    fx[b"enab"] = Bool(True)
    fx[b"filterMaskEnable"] = Bool(mask)
    fx[b"filterMaskExtendWithWhite"] = Bool(extends_white)
    entry = Descriptor(classID=b"null")
    entry[b"enab"] = Bool(True)
    entries = List()
    entries.append(entry)
    fx[b"filterFXList"] = entries
    config = layer.smart_object._config
    assert config is not None
    config.data[b"filterFX"] = fx
    return fx


def _store(psd: PSDImage, items: list[FilterEffect]) -> None:
    assert psd.tagged_blocks is not None
    psd.tagged_blocks.set_data(Tag.FILTER_EFFECTS2, version=1, items=items)


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
    fx = _enable(layer)
    _store(psd, [_item(layer.bbox, source, alpha, mask, version=version)])
    opaque = shape[..., 0] > 0.999
    return Filtered(psd, layer, stored, source, shape[..., 0], opaque, w // 2, fx)


@pytest.fixture
def filtered() -> Filtered:
    return _filtered()


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
        _enable(layer)
        h, w = layer.height, layer.width
        zeros = np.zeros((h, w), np.uint8)
        items.append(_item(layer.bbox, np.zeros((h, w, 3), np.uint8), zeros, zeros))
    _store(psd, items if in_layer_order else items[::-1])
    for layer, item in zip(layers, items, strict=True):
        assert layer.smart_object.filter_effect() is item


def test_an_item_that_does_not_contain_the_bbox_is_not_used() -> None:
    psd = PSDImage.open(full_name("layers/smartobject-layer.psd"))
    layer = _smart_objects(psd)[0]
    stored = _rgba(layer)
    _enable(layer)
    x0, y0, x1, y1 = layer.bbox
    h, w = y1 - y0, x1 - x0 - 1
    zeros = np.zeros((h, w), np.uint8)
    item = _item((x0 + 1, y0, x1, y1), np.zeros((h, w, 3), np.uint8), zeros, zeros)
    _store(psd, [item])
    assert layer.smart_object.filter_effect() is None
    assert np.array_equal(_rgba(layer), stored)


def test_the_pairing_follows_the_layer_order_after_an_edit() -> None:
    psd = PSDImage.open(full_name("blend-modes/rgb-blend-modes.psd"))
    first_layer, second_layer = _smart_objects(psd)[:2]
    canvas = (0, 0, psd.width, psd.height)
    zeros = np.zeros((psd.height, psd.width), np.uint8)
    source = np.zeros((psd.height, psd.width, 3), np.uint8)
    items = [_item(canvas, source, zeros, zeros) for _ in range(2)]
    for layer in (first_layer, second_layer):
        _enable(layer)
    _store(psd, items)
    assert second_layer.smart_object.filter_effect() is items[1]
    parent = first_layer.parent
    assert isinstance(parent, GroupMixin)
    parent.remove(first_layer)
    assert second_layer.smart_object.filter_effect() is items[0]


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
    _enable(layer, extends_white=False)
    alpha = np.full(stored.shape[:2], 255, np.uint8)
    _store(psd, [_item(layer.bbox, source, alpha, mask, mask_box=mask_box)])
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
    _enable(layer)
    _store(psd, [_item(layer.bbox, source, a_u, mask)])
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
    _enable(layer)
    _store(psd, [_item(layer.bbox, planes, zeros, zeros)])
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


def test_a_source_over_the_allocation_budget_keeps_the_stored_pixels(
    caplog: pytest.LogCaptureFixture,
) -> None:
    psd = PSDImage.open(full_name("layers/smartobject-layer.psd"))
    layer = _smart_objects(psd)[0]
    stored = _rgba(layer)
    _enable(layer)
    x0, y0, x1, y1 = layer.bbox
    side = 2000
    zeros = np.zeros((side, side), np.uint8)
    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    item = _item((0, 0, side, side), np.zeros((side, side, 3), np.uint8), zeros, zeros)
    item.extra = _item(layer.bbox, mask[..., None], mask, mask).extra
    _store(psd, [item])
    psd.max_alloc_bytes = 4 * side * side
    with pytest.raises(ValueError, match="budget"):
        layer.smart_object.filter_source()
    with caplog.at_level(logging.WARNING):
        assert np.array_equal(_rgba(layer), stored)
    assert "Smart filter mask" in caplog.text


def test_the_blend_stays_within_its_allocation_estimate() -> None:
    side = 512
    layer = cast(
        SmartObjectLayer,
        SimpleNamespace(
            bbox=(0, 0, side, side),
            name="blend",
            _psd=SimpleNamespace(_max_alloc_bytes="unlimited"),
        ),
    )
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
