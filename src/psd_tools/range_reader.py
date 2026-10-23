"""
A read-only, seekable file over byte ranges fetched on demand.

With a lazy open, a PSD read over HTTP costs its layer records, not its
pixels::

    from psd_tools import PSDImage
    from psd_tools.range_reader import RangeReader

    reader = RangeReader.from_url("https://example.com/card.psd")
    psd = PSDImage.open(reader, lazy=True)
    print(reader.requests, reader.fetched)  # what it cost so far

The first fetch takes ``head`` bytes (a card's header, resources and layer
records in one request). Each later miss fetches a window: ``chunk`` bytes
after a seek, twice the last window while reads run on from the last fetch (up
to ``max_chunk``), so a sequential section costs a few requests and a skipped
payload none. Fetched bytes are kept.
"""

import io
import re
import threading
import urllib.request
from bisect import bisect_right
from collections.abc import Callable
from typing import Any

#: ``fetch(start, stop)`` returns the bytes ``[start, stop)`` and the file's size.
Fetch = Callable[[int, int], tuple[bytes, int]]

_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


class RangeReader(io.RawIOBase):
    """
    A file over ``fetch``.

    The first ``head`` bytes are fetched now: they give the size.

    :param fetch: ``fetch(start, stop) -> (bytes, size)``, exact or raising.
    :param chunk: the window after a seek.
    :param max_chunk: the largest window a sequential read grows to.
    :param head: the first fetch.
    """

    def __init__(
        self,
        fetch: Fetch,
        chunk: int = 1 << 16,
        max_chunk: int = 1 << 22,
        head: int = 1 << 18,
    ):
        super().__init__()
        if chunk <= 0 or max_chunk < chunk or head <= 0:
            raise ValueError("need 0 < chunk <= max_chunk and 0 < head")
        self._fetch = fetch
        self.chunk, self.max_chunk = chunk, max_chunk
        self._starts: list[int] = []
        self._blocks: list[bytes] = []
        self._lock = threading.Lock()
        self._pos = 0
        self._window = chunk
        self._last_end = -1
        self.requests = 0  #: fetches made
        self.fetched = 0  #: bytes fetched
        self.size = -1
        data, self.size = self._get(0, head, known=False)
        self._insert(0, data)

    @classmethod
    def from_url(
        cls,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> "RangeReader":
        """Read ``url`` by HTTP Range requests (the server must answer 206)."""
        return cls(http_fetch(url, headers, timeout), **kwargs)

    def _get(self, start: int, stop: int, known: bool = True) -> tuple[bytes, int]:
        if known:
            stop = min(stop, self.size)
        data, size = self._fetch(start, stop)
        self.requests += 1
        self.fetched += len(data)
        if known and size != self.size:
            raise OSError("file size changed: %d -> %d" % (self.size, size))
        if len(data) != min(stop, size) - start:
            raise OSError(
                "short range: %d bytes at %d, expected %d"
                % (len(data), start, min(stop, size) - start)
            )
        self._last_end = start + len(data)
        return data, size

    def _insert(self, start: int, data: bytes) -> None:
        index = bisect_right(self._starts, start)
        self._starts.insert(index, start)
        self._blocks.insert(index, data)

    def _range(self, start: int, stop: int) -> bytes:
        """Bytes ``[start, stop)``, fetching the gaps."""
        out = []
        pos = start
        while pos < stop:
            index = bisect_right(self._starts, pos) - 1
            if index >= 0 and pos < self._starts[index] + len(self._blocks[index]):
                offset = pos - self._starts[index]
                piece = self._blocks[index][offset : offset + stop - pos]
                out.append(piece)
                pos += len(piece)
                continue
            following = (
                self._starts[index + 1] if index + 1 < len(self._starts) else self.size
            )
            self._window = (
                min(self._window * 2, self.max_chunk)
                if pos == self._last_end
                else self.chunk
            )
            data, _ = self._get(pos, min(max(stop, pos + self._window), following))
            self._insert(pos, data)
        return b"".join(out)

    def readable(self) -> bool:
        """Return True."""
        return True

    def seekable(self) -> bool:
        """Return True."""
        return True

    def tell(self) -> int:
        """Return the position."""
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Move the position; nothing is fetched until a read."""
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self._pos + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError("invalid whence: %r" % whence)
        if position < 0:
            raise OSError("negative seek position %d" % position)
        self._pos = position
        return position

    def read(self, size: int | None = -1) -> bytes:
        """Read up to ``size`` bytes (all when negative), fetching the misses."""
        stop = (
            self.size if size is None or size < 0 else min(self._pos + size, self.size)
        )
        if stop <= self._pos:
            return b""
        with self._lock:
            data = self._range(self._pos, stop)
        self._pos = stop
        return data

    def readinto(self, buffer: Any) -> int:
        """Read into ``buffer``; return the count."""
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def http_fetch(
    url: str, headers: dict[str, str] | None = None, timeout: float = 30.0
) -> Fetch:
    """Make a ``fetch`` of HTTP Range requests with urllib; a 200 (no ranges) raises."""

    def fetch(start: int, stop: int) -> tuple[bytes, int]:
        request = urllib.request.Request(
            url, headers={**(headers or {}), "Range": "bytes=%d-%d" % (start, stop - 1)}
        )
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            if reply.status != 206:
                raise OSError("no range reply from %s (HTTP %d)" % (url, reply.status))
            match = _CONTENT_RANGE.fullmatch(reply.headers.get("Content-Range", ""))
            if match is None or int(match.group(1)) != start:
                raise OSError("bad Content-Range from %s" % url)
            return reply.read(), int(match.group(3))

    return fetch
