"""A smart object's filter mask blends its filtered raster with the FEid source."""

import io

import numpy as np
import pytest

from psd_tools import PSDImage
from psd_tools.constants import Compression, Tag
from psd_tools.psd.descriptor import Bool, Descriptor, List
from psd_tools.psd.filter_effects import FilterEffect, FilterEffectChannel, FilterEffectExtra

from ..utils import full_name

pytest.importorskip("scipy")

MAX_CHANNELS = 24


def _plane(array: np.ndarray) -> FilterEffectChannel:
    h, w = array.shape
    from psd_tools.compression import compress

    data = compress(array.tobytes(), Compression.RLE, w, h, 8, 2)
    return FilterEffectChannel(is_written=1, compression=int(Compression.RLE), data=data)


def _with_filter_mask(psd: PSDImage, layer, source: np.ndarray, alpha: np.ndarray, mask: np.ndarray):
    """Attach smart filters to *layer* whose FEid source is *source* under *mask*."""
    from psd_tools.compression import compress

    x0, y0, x1, y1 = layer.bbox
    h, w = y1 - y0, x1 - x0
    channels = [_plane(source[..., i]) for i in range(3)]
    channels += [FilterEffectChannel(is_written=0) for _ in range(MAX_CHANNELS - 2)]
    channels.append(_plane(alpha))
    extra = FilterEffectExtra(
        is_written=1,
        rectangle=[y0, x0, y1, x1],
        compression=int(Compression.RLE),
        data=compress(mask.tobytes(), Compression.RLE, w, h, 8, 2),
    )
    item = FilterEffect(
        uuid="0",
        version=1,
        rectangle=(y0, x0, y1, x1),
        depth=8,
        max_channels=MAX_CHANNELS,
        channels=channels,
        extra=extra,
    )
    psd.tagged_blocks.set_data(Tag.FILTER_EFFECTS2, version=1, items=[item])
    fx = Descriptor(classID=b"null")
    fx[b"enab"] = Bool(True)
    fx[b"filterMaskEnable"] = Bool(True)
    fx[b"filterMaskExtendWithWhite"] = Bool(True)
    entry = Descriptor(classID=b"null")
    entry[b"enab"] = Bool(True)
    fx[b"filterFXList"] = List([entry])
    layer.smart_object._config.data[b"filterFX"] = fx
    return fx


@pytest.fixture
def filtered():
    psd = PSDImage.open(full_name("layers/smartobject-layer.psd"))
    layer = next(l for l in psd.descendants() if l.kind == "smartobject")
    stored = (layer.numpy("color") * 255).round().astype(np.uint8)
    shape = layer.numpy("shape")[..., 0]
    source = 255 - stored
    h, w = stored.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    mask[:, : w // 2] = 255
    fx = _with_filter_mask(psd, layer, source, (shape * 255).round().astype(np.uint8), mask)
    return psd, layer, stored, source, shape > 0.999, w // 2, fx


def _rgb(layer) -> np.ndarray:
    return np.asarray(layer.composite(force=True).convert("RGBA")).astype(int)[..., :3]


def test_the_mask_picks_the_filtered_raster_where_white_and_the_source_where_black(filtered):
    psd, layer, stored, source, opaque, half, fx = filtered
    out = _rgb(layer)
    left = opaque.copy()
    left[:, half:] = False
    right = opaque & ~left
    assert np.abs(out - stored)[left].max() == 0
    assert np.abs(out - source)[right].max() <= 1


def test_a_disabled_mask_leaves_the_filtered_raster(filtered):
    psd, layer, stored, source, opaque, half, fx = filtered
    fx[b"filterMaskEnable"] = Bool(False)
    assert np.abs(_rgb(layer) - stored)[opaque].max() == 0


def test_disabled_filters_leave_the_stored_raster(filtered):
    psd, layer, stored, source, opaque, half, fx = filtered
    fx[b"enab"] = Bool(False)
    assert np.abs(_rgb(layer) - stored)[opaque].max() == 0


def test_the_filter_effect_survives_a_save(filtered):
    psd, layer, stored, source, opaque, half, fx = filtered
    buf = io.BytesIO()
    psd.save(buf)
    buf.seek(0)
    reopened = PSDImage.open(buf)
    layer2 = next(l for l in reopened.descendants() if l.kind == "smartobject")
    assert layer2.smart_object.filter_effect() is not None
    out = _rgb(layer2)
    assert np.abs(out - source)[opaque & (np.arange(out.shape[1]) >= half)[None, :]].max() <= 1


def test_the_accessors_read_the_block(filtered):
    psd, layer, stored, source, opaque, half, fx = filtered
    so = layer.smart_object
    assert so.has_enabled_filters and so.filter_mask_enabled
    bbox, color, alpha = so.filter_source()
    assert bbox == layer.bbox and color.shape == stored.shape and alpha is not None
    mbox, mask = so.filter_mask()
    assert mbox == layer.bbox and mask[0, 0] == 255 and mask[0, -1] == 0
