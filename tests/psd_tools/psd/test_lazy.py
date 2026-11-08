"""Lazy reads: payloads stay in the file until used, and read the same."""

import copy
import io
import os
import pickle
from typing import IO, cast

import numpy as np
import pytest

from psd_tools import PSDImage
from psd_tools.api._resources import _descriptors
from psd_tools.api.layers import SmartObjectLayer
from psd_tools.constants import Tag
from psd_tools.psd import PSD
from psd_tools.psd.lazy import Span, load
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


def test_layer_blocks_read_later_keep_the_parse_budget() -> None:
    """An open holds a layer's blocks unparsed; reading them spends its budget."""
    with open(full_name("layers/type-layer.psd"), "rb") as f:
        data = f.read()
    psd = PSD.read(io.BytesIO(data), parse_limits=ParseLimits(max_objects=1000))
    info = psd.layer_and_mask_information.layer_info
    assert info is not None and info.layer_records is not None
    with pytest.raises(ParseLimitError, match="max_objects"):
        for record in info.layer_records:
            for block in record.tagged_blocks.values():
                load(block)


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


def test_lazy_walkers_read_the_payloads() -> None:
    """Field walkers (``_find``, the resource transfer's) see what an eager read has."""
    eager, _ = _read(full_name("layers/smartobject-layer.psd"), lazy=False)
    lazy, _ = _read(full_name("layers/smartobject-layer.psd"), lazy=True)
    assert _payloads(lazy)
    found = [list(psd._find(lambda e: isinstance(e, bytes))) for psd in (eager, lazy)]
    assert found[1] == found[0]
    assert list(_descriptors(lazy)) == list(_descriptors(eager))
    assert not _payloads(lazy)


def test_lazy_payload_is_checked_when_read() -> None:
    """A corrupt channel compression fails an eager open, and a lazy one only when read."""
    lazy, data = _read(full_name("layers/pixel-layer.psd"), lazy=True)
    span = next(
        c._span
        for _, channels in lazy._iter_layers()
        for c in channels
        if c._span is not None
    )
    bad = data[: span.offset] + b"\xff\xff" + data[span.offset + 2 :]
    with pytest.raises(ValueError):
        PSD.read(io.BytesIO(bad))
    lazy = PSD.read(io.BytesIO(bad), lazy=True)
    channel = next(
        c
        for _, channels in lazy._iter_layers()
        for c in channels
        if c._span is not None and c._span.offset == span.offset
    )
    with pytest.raises(ValueError):
        channel.data


@pytest.mark.parametrize(
    "filename", ["layers/pixel-layer.psd", "layers/type-layer.psd"]
)
def test_bytesio_open_defers_in_memory(filename: str) -> None:
    """A BytesIO open defers payloads, keeps no channel read, saves as read, pickles."""
    with open(full_name(filename), "rb") as f:
        raw = f.read()
    eager = PSDImage(PSD.read(io.BytesIO(raw)))
    expected = io.BytesIO()
    eager.save(expected)
    buf = io.BytesIO(raw)
    psd = PSDImage.open(buf)
    assert _payloads(psd._record)
    buf.seek(0)
    buf.write(b"\0" * len(raw))  # the caller's buffer is not the source
    unread = [c for _, cs in psd._record._iter_layers() for c in cs if c._span]
    assert unread
    for layer, like in zip(psd.descendants(), eager.descendants(), strict=True):
        a, b = layer.numpy(), like.numpy()
        assert (a is None) == (b is None)
        assert a is None or b is None or a.tobytes() == b.tobytes()
    assert all(c._span is not None for c in unread)  # read, none kept
    out = io.BytesIO()
    psd.save(out)
    assert out.getvalue() == expected.getvalue()
    again = pickle.loads(pickle.dumps(psd))
    assert again.numpy().tobytes() == eager.numpy().tobytes()
