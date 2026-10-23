"""RangeReader: a seekable file over fetched ranges, and over HTTP."""

import http.server
import io
import random
import re
import threading
from collections.abc import Iterator
from typing import IO, cast

import numpy as np
import pytest

from psd_tools import PSDImage
from psd_tools.range_reader import RangeReader

from .utils import full_name

BIG = full_name("blend-modes/rgb-blend-modes.psd")


def local_fetch(data: bytes, log: list | None = None):
    def fetch(start: int, stop: int) -> tuple[bytes, int]:
        if log is not None:
            log.append((start, stop))
        return data[start:stop], len(data)

    return fetch


def test_reads_like_a_file() -> None:
    data = bytes(random.Random(1).randrange(256) for _ in range(5000))
    reader = RangeReader(local_fetch(data), chunk=64, max_chunk=512, head=64)
    model = io.BytesIO(data)
    rng = random.Random(2)
    for _ in range(300):
        if rng.random() < 0.3:
            offset = rng.randrange(-100, 5100)
            whence = rng.choice((io.SEEK_SET, io.SEEK_CUR, io.SEEK_END))
            if (offset if whence == io.SEEK_SET else 0) < 0:
                continue
            try:
                expected = model.seek(offset, whence)
            except (OSError, ValueError):
                with pytest.raises((OSError, ValueError)):
                    reader.seek(offset, whence)
                continue
            assert reader.seek(offset, whence) == expected
        else:
            size = rng.randrange(0, 700)
            assert reader.read(size) == model.read(size)
        assert reader.tell() == model.tell()
    reader.seek(0)
    assert reader.read() == data


def test_windows_grow_while_sequential_and_reset_after_a_seek() -> None:
    data = bytes(10000)
    log: list = []
    reader = RangeReader(local_fetch(data, log), chunk=100, max_chunk=400, head=50)
    for _ in range(10):
        reader.read(100)
    assert log == [(0, 50), (50, 250), (250, 650), (650, 1050)]
    reader.seek(5000)
    reader.read(10)
    assert log[-1] == (5000, 5100)
    assert reader.requests == len(log) and reader.fetched == sum(b - a for a, b in log)


def test_cached_bytes_are_not_fetched_again() -> None:
    data = bytes(range(256)) * 4
    log: list = []
    reader = RangeReader(local_fetch(data, log), chunk=64, head=64)
    reader.seek(500)
    reader.read(10)
    reader.seek(0)
    assert reader.read() == data
    assert sum(b - a for a, b in log) == len(data)


def test_a_short_or_changed_reply_raises() -> None:
    data = bytes(1000)
    reader = RangeReader(
        lambda a, b: (data[a : b - 1], len(data)) if a else (data[a:b], len(data)),
        chunk=10,
        head=10,
    )
    reader.seek(500)
    with pytest.raises(OSError):
        reader.read(5)
    reader = RangeReader(
        lambda a, b: (data[a:b], len(data) + (a > 0)), chunk=10, head=10
    )
    reader.seek(500)
    with pytest.raises(OSError):
        reader.read(5)


class _Handler(http.server.BaseHTTPRequestHandler):
    data = b""
    ranges = True
    log: list = []
    url = ""

    def do_GET(self) -> None:
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        type(self).log.append(self.headers.get("Range"))
        if match is None or not self.ranges:
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.data)))
            self.end_headers()
            self.wfile.write(self.data)
            return
        start, end = int(match.group(1)), min(int(match.group(2)), len(self.data) - 1)
        body = self.data[start : end + 1]
        self.send_response(206)
        self.send_header(
            "Content-Range", "bytes %d-%d/%d" % (start, end, len(self.data))
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def server() -> Iterator[type[_Handler]]:
    with open(BIG, "rb") as f:
        handler: type[_Handler] = type(
            "Handler", (_Handler,), {"data": f.read(), "log": []}
        )
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    handler.url = "http://127.0.0.1:%d/card.psd" % httpd.server_address[1]
    try:
        yield handler
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join()


def test_lazy_open_over_http(server: type[_Handler]) -> None:
    eager = PSDImage.open(BIG)
    reader = RangeReader.from_url(
        server.url, chunk=1 << 12, max_chunk=1 << 16, head=1 << 12
    )
    lazy = PSDImage.open(cast(IO[bytes], reader), lazy=True)
    assert [layer.name for layer in lazy.descendants()] == [
        layer.name for layer in eager.descendants()
    ]
    assert reader.fetched < len(server.data) // 4
    assert reader.requests == len(server.log)
    layer = next(layer for layer in eager.descendants() if layer.has_pixels())
    twin = next(twin for twin in lazy.descendants() if twin.layer_id == layer.layer_id)
    x, y = twin.numpy(), layer.numpy()
    assert x is not None and y is not None and np.array_equal(x, y)


def test_a_server_without_ranges_raises(server: type[_Handler]) -> None:
    server.ranges = False
    with pytest.raises(OSError):
        RangeReader.from_url(server.url)
