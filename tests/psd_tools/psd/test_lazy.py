"""Lazy reads: payloads stay in the file until used, and read the same."""

import copy
import io
import os
from typing import IO, cast

import numpy as np
import pytest

from psd_tools import PSDImage
from psd_tools.api.layers import SmartObjectLayer
from psd_tools.constants import Tag
from psd_tools.psd import PSD
from psd_tools.psd.lazy import Span
from psd_tools.psd.parse_limits import ParseLimitError, ParseLimits
from psd_tools.range_reader import RangeReader

from ..utils import all_files, full_name


class Recorder:
    """A fetch over bytes that records the ranges it served (exact windows)."""

    def __init__(self, data: bytes):
        self.data = data
        self.ranges: list[tuple[int, int]] = []

    def __call__(self, start: int, stop: int) -> tuple[bytes, int]:
        self.ranges.append((start, stop))
        return self.data[start:stop], len(self.data)

    def touched(self, start: int, stop: int) -> bool:
        return any(a < stop and start < b for a, b in self.ranges)


def exact_reader(data: bytes) -> tuple[RangeReader, Recorder]:
    """A reader fetching exactly what is read (1-byte windows)."""
    recorder = Recorder(data)
    return RangeReader(recorder, chunk=1, max_chunk=1, head=1), recorder


def _read(filename: str, lazy: bool) -> tuple[PSD, bytes]:
    with open(filename, "rb") as f:
        data = f.read()
    return PSD.read(io.BytesIO(data), lazy=lazy), data


def _open(reader: RangeReader) -> PSDImage:
    return PSDImage.open(cast(IO[bytes], reader), lazy=True)


def _payloads(psd: PSD) -> list[Span]:
    """The deferred payloads' spans: channels, image data, lazy blocks."""
    spans: list[Span] = []
    for _, channels in psd._iter_layers():
        spans.extend(c._span for c in channels if c._span is not None)
    if psd.image_data._span is not None:
        spans.append(psd.image_data._span)
    blocks = psd.layer_and_mask_information.tagged_blocks
    for block in blocks.values() if blocks is not None else ():
        if block._span is not None:
            spans.append(block._span[0])
    return spans


@pytest.mark.parametrize("filename", all_files())
def test_lazy_read_equals_eager(filename: str) -> None:
    eager, _ = _read(filename, lazy=False)
    lazy, _ = _read(filename, lazy=True)
    assert lazy.tobytes() == eager.tobytes()
    assert lazy == eager


@pytest.mark.parametrize(
    "filename", ["layers/pixel-layer.psd", "layers/smartobject-layer.psd"]
)
def test_lazy_open_fetches_no_payload(filename: str) -> None:
    with open(full_name(filename), "rb") as f:
        data = f.read()
    reader, recorder = exact_reader(data)
    psd = _open(reader)
    spans = _payloads(psd._record)
    assert spans
    assert not any(recorder.touched(s.offset, s.offset + s.length) for s in spans)
    assert any(layer.has_pixels() for layer in psd.descendants())
    assert not any(recorder.touched(s.offset, s.offset + s.length) for s in spans)


def test_lazy_layer_pixels_equal_eager() -> None:
    filename = full_name("layers/pixel-layer.psd")
    eager = PSDImage.open(filename)
    with open(filename, "rb") as f:
        reader, recorder = exact_reader(f.read())
    lazy = _open(reader)
    before = len(recorder.ranges)
    for a, b in zip(eager.descendants(), lazy.descendants(), strict=True):
        if a.has_pixels():
            x, y = a.numpy(), b.numpy()
            assert x is not None and y is not None and np.array_equal(x, y)
    assert len(recorder.ranges) > before
    assert np.array_equal(eager.numpy(), lazy.numpy())


def test_lazy_smart_object_block() -> None:
    filename = full_name("layers/smartobject-layer.psd")
    eager = PSDImage.open(filename)
    with open(filename, "rb") as f:
        reader, recorder = exact_reader(f.read())
    lazy = _open(reader)
    blocks = lazy._record.layer_and_mask_information.tagged_blocks
    assert blocks is not None and blocks[Tag.LINKED_LAYER2]._span is not None
    span = blocks[Tag.LINKED_LAYER2]._span[0]
    assert not recorder.touched(span.offset, span.offset + span.length)
    a, b = (
        next(x for x in psd.descendants() if isinstance(x, SmartObjectLayer))
        for psd in (eager, lazy)
    )
    assert isinstance(a, SmartObjectLayer) and isinstance(b, SmartObjectLayer)
    assert b.smart_object.data == a.smart_object.data
    assert recorder.touched(span.offset, span.offset + span.length)


def test_lazy_duplicate_shares_the_file() -> None:
    filename = full_name("layers/pixel-layer.psd")
    with open(filename, "rb") as f:
        reader, _ = exact_reader(f.read())
    lazy = _open(reader)
    layer = next(layer for layer in lazy.descendants() if layer.has_pixels())
    channels = copy.deepcopy(layer._channels)
    assert channels[0]._span is layer._channels[0]._span
    assert channels == layer._channels


def test_lazy_needs_an_open_file() -> None:
    with pytest.raises(ValueError):
        PSDImage.open(full_name("layers/pixel-layer.psd"), lazy=True)


def test_lazy_payloads_keep_the_parse_budget() -> None:
    filename = full_name("layers/pixel-layer.psd")
    with open(filename, "rb") as f:
        data = f.read()
    limits = ParseLimits(max_total_bytes=len(data) // 2)
    with pytest.raises(ParseLimitError):
        PSD.read(io.BytesIO(data), parse_limits=limits)
    with pytest.raises(ParseLimitError):
        PSD.read(io.BytesIO(data), parse_limits=limits, lazy=True)


def test_lazy_blocks_in_a_block_copy_stay_eager() -> None:
    filename = full_name("16bit5x5.psd")
    if not os.path.exists(filename):
        pytest.skip("no 16-bit fixture")
    lazy, _ = _read(filename, lazy=True)
    info = lazy._get_layer_info()
    assert info is not None
    assert all(
        c._span is None for channels in info.channel_image_data for c in channels
    )
