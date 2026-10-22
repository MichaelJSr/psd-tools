"""Builders for a document's smart filter data (``filterFX`` and ``FEid``)."""

import numpy as np

from psd_tools.api.layers import SmartObjectLayer
from psd_tools.api.psd_image import PSDImage
from psd_tools.compression import compress
from psd_tools.constants import Compression, Tag
from psd_tools.psd.descriptor import Bool, Descriptor, List
from psd_tools.psd.filter_effects import (
    FilterEffect,
    FilterEffectChannel,
    FilterEffectExtra,
)

MAX_CHANNELS = 24

Box = tuple[int, int, int, int]


def rle_plane(array: np.ndarray, version: int = 2) -> FilterEffectChannel:
    """An RLE plane; *version* 2 writes 4-byte row counts, 1 writes 2-byte ones."""
    h, w = array.shape
    data = compress(array.tobytes(), Compression.RLE, w, h, 8, version)
    return FilterEffectChannel(
        is_written=1, compression=int(Compression.RLE), data=data
    )


def feid_item(
    box: Box,
    source: np.ndarray,
    alpha: np.ndarray,
    mask: np.ndarray,
    mask_box: Box | None = None,
    version: int = 2,
    uuid: str = "0",
) -> FilterEffect:
    """An FEid item: *source* and *alpha* on *box*, *mask* on *mask_box* (or *box*)."""
    x0, y0, x1, y1 = box
    channels = [rle_plane(source[..., i], version) for i in range(source.shape[-1])]
    unwritten = MAX_CHANNELS + 1 - len(channels)
    channels += [FilterEffectChannel(is_written=0) for _ in range(unwritten)]
    channels.append(rle_plane(alpha, version))
    mx0, my0, mx1, my1 = mask_box or box
    extra = FilterEffectExtra(
        is_written=1,
        rectangle=[my0, mx0, my1, mx1],  # type: ignore[list-item]
        compression=int(Compression.RLE),
        data=rle_plane(mask, version).data,
    )
    return FilterEffect(
        uuid=uuid,
        version=1,
        rectangle=(y0, x0, y1, x1),
        depth=8,
        max_channels=MAX_CHANNELS,
        channels=channels,
        extra=extra,
    )


def enable_filters(
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


def store_items(psd: PSDImage, items: list[FilterEffect]) -> None:
    """Set the document's FEid block to *items*."""
    assert psd.tagged_blocks is not None
    psd.tagged_blocks.set_data(Tag.FILTER_EFFECTS2, version=1, items=items)
