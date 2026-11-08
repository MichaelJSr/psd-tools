import random
import struct
from typing import Any

import pytest

import psd_tools.compression as codecs
import psd_tools.compression.rle as rle

from .test_compression import EDGE_CASE_1, RAW_IMAGE_3x3_8bit

try:
    import psd_tools.compression._rle as _rle  # type: ignore[import-not-found]
except ImportError:
    _rle = None

compiled = pytest.mark.skipif(_rle is None, reason="compiled RLE not built")


@compiled
def test_identical() -> None:
    size = len(RAW_IMAGE_3x3_8bit)
    encoded = rle.encode(RAW_IMAGE_3x3_8bit)
    encoded_c = _rle.encode(RAW_IMAGE_3x3_8bit)
    assert encoded == encoded_c
    decoded = rle.decode(encoded, size)
    decoded_c = _rle.decode(encoded_c, size)
    assert decoded == RAW_IMAGE_3x3_8bit
    assert decoded_c == RAW_IMAGE_3x3_8bit

    size = len(EDGE_CASE_1)
    encoded = rle.encode(EDGE_CASE_1)
    encoded_c = _rle.encode(EDGE_CASE_1)
    assert encoded == encoded_c
    decoded = rle.decode(encoded, size)
    decoded_c = _rle.decode(encoded_c, size)
    assert decoded == EDGE_CASE_1
    assert decoded_c == EDGE_CASE_1


@pytest.mark.parametrize(
    ("mod, data, size, expected"),
    [
        # 0xfd = repeat-run: 256-253=3, so repeat 4× byte 0x01.
        # size=3 → overflow clipped: b'\x01\x01\x01'
        (rle, b"\xfd\x01", 3, b"\x01\x01\x01"),
        pytest.param(_rle, b"\xfd\x01", 3, b"\x01\x01\x01", marks=compiled),
        # size=5 → 4 real bytes + 1 zero-padded: b'\x01\x01\x01\x01\x00'
        (rle, b"\xfd\x01", 5, b"\x01\x01\x01\x01\x00"),
        pytest.param(_rle, b"\xfd\x01", 5, b"\x01\x01\x01\x01\x00", marks=compiled),
        # 0x02 = copy-run: copy 3 bytes (0x01 0x02 0x03).
        # size=2 → overflow clipped: b'\x01\x02'
        (rle, b"\x02\x01\x02\x03", 2, b"\x01\x02"),
        pytest.param(_rle, b"\x02\x01\x02\x03", 2, b"\x01\x02", marks=compiled),
        # size=4 → 3 real bytes + 1 zero-padded: b'\x01\x02\x03\x00'
        (rle, b"\x02\x01\x02\x03", 4, b"\x01\x02\x03\x00"),
        pytest.param(_rle, b"\x02\x01\x02\x03", 4, b"\x01\x02\x03\x00", marks=compiled),
    ],
)
def test_tolerant_decode(mod: Any, data: bytes, size: int, expected: bytes) -> None:
    # The decoder must never raise; it clips overflow runs and zero-pads short output.
    result = mod.decode(data, size)
    assert result == expected
    assert len(result) == size


@compiled
@pytest.mark.parametrize("version", [1, 2])
def test_rows_decode_in_one_call(monkeypatch: pytest.MonkeyPatch, version: int) -> None:
    """``decode_rows`` is the per-row join, runs cut short by the data included."""
    monkeypatch.setattr(codecs, "rle_impl", _rle)
    rng = random.Random(version)
    for _ in range(400):
        height, width = rng.randint(1, 9), rng.randint(1, 40)
        counts = [rng.randint(0, 30) for _ in range(height)]
        body = bytes(rng.getrandbits(8) for _ in range(rng.randint(0, sum(counts) + 8)))
        data = struct.pack(">%d%s" % (height, "HI"[version - 1]), *counts) + body
        with monkeypatch.context() as patch:
            patch.setattr(_rle, "decode", None)
            batched = codecs.decode_rle(data, width, height, 8, version)
        with monkeypatch.context() as patch:
            patch.setattr(codecs, "rle_impl", rle)
            assert codecs.decode_rle(data, width, height, 8, version) == batched
        assert len(batched) == height * width


@compiled
def test_rows_decode_packets_of_every_length() -> None:
    """Short and long runs and literals, rows clipped or cut short, the data's end."""
    rng = random.Random(0)
    for _ in range(300):
        width = rng.choice([1, 2, 8, 15, 16, 17, 31, 64, 300])
        rows = []
        for _ in range(rng.randint(1, 5)):
            raw = bytearray()
            while len(raw) < width + 20:
                n = rng.randint(1, 140)
                raw += rng.randbytes(1) * n if rng.random() < 0.5 else rng.randbytes(n)
            rows.append(rle.encode(bytes(raw[: rng.randint(0, width + 20)])))
        counts = [len(row) for row in rows]
        data = b"".join(rows)
        data = data[: rng.randint(len(data) // 2, len(data))]
        expected, pos = [], 0
        for count in counts:
            expected.append(rle.decode(data[pos : pos + count], width))
            pos = min(pos + count, len(data))
        assert _rle.decode_rows(data, 0, counts, width) == b"".join(expected)
