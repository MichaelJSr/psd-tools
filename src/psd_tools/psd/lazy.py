"""
Lazy reads: pixel payloads left in the file until something reads them.

``PSD.read(fp, lazy=True)`` (and ``PSDImage.open(fp, lazy=True)``) parses the
structure as usual but skips the layers' channel data and the merged image
data. Each skipped payload is a :py:class:`Span` of the file, read at first
use of ``ChannelData.data`` or ``ImageData.data``. The file must stay open as
long as the document is used.

This pays off when reading the file is expensive, e.g. over HTTP range
requests (:py:class:`~psd_tools.range_reader.RangeReader`): opening a card
then fetches its layer records, not its pixels.

Payloads inside tagged blocks (``Lr16``/``Lr32`` in 16- and 32-bit files) are
parsed from a copy of the block and stay eager.
"""

import struct
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import IO, Any

from psd_tools.constants import Compression
from psd_tools.psd.bin_utils import _BoundedReader, _remaining_bytes
from psd_tools.psd.parse_limits import consume_bytes


class _Source:
    """The document's file and the lock that serializes its seek-then-read."""

    __slots__ = ("fp", "lock")

    def __init__(self, fp: IO[bytes]):
        self.fp = fp
        self.lock = threading.Lock()


class Span:
    """``length`` bytes at ``offset`` of the document's file, read on demand."""

    __slots__ = ("_source", "offset", "length")

    def __init__(self, source: _Source, offset: int, length: int):
        self._source = source
        self.offset = offset
        self.length = length

    def read(self) -> bytes:
        """Read the bytes; raise OSError when the file has fewer."""
        with self._source.lock:
            fp = self._source.fp
            fp.seek(self.offset)
            data = fp.read(self.length)
        if len(data) != self.length:
            raise OSError(
                "Failed to read data section: read=%d, expected=%d."
                % (len(data), self.length)
            )
        return data

    def __copy__(self) -> "Span":
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> "Span":
        return self  # immutable, and the file is shared, not copied

    def __repr__(self) -> str:
        return "Span(offset=%d, length=%d)" % (self.offset, self.length)


_lazy: ContextVar[_Source | None] = ContextVar("psd_lazy_source", default=None)


@contextmanager
def lazy_reads(fp: IO[bytes]) -> Iterator[None]:
    """Defer payload reads from ``fp`` while the block runs."""
    token = _lazy.set(_Source(fp))
    try:
        yield
    finally:
        _lazy.reset(token)


def defer(fp: IO[bytes], length: int) -> Span | None:
    """
    Skip ``length`` bytes of ``fp`` and return their span.

    Only when lazy reads are on and ``fp`` reads the document's own file (not
    a copy of a block); else None, and the caller reads the payload now. The
    bytes count against the parse budget as a read would.
    """
    source = _lazy.get()
    if source is None:
        return None
    root = fp
    while isinstance(root, _BoundedReader):
        root = root._fp
    if root is not source.fp:
        return None
    if length < 0 or length > _remaining_bytes(fp):
        raise OSError("Invalid data section size: %d" % length)
    consume_bytes(length)
    offset = fp.tell()
    fp.seek(length, 1)
    return Span(source, offset, length)


class LazyPayload:
    """
    Compression and data of a channel or of the image data.

    A lazy read leaves both in ``_span`` (the 2-byte compression, then the
    data) and reads them on first access. The class defines the
    ``_compression``, ``_data`` and ``_span`` fields.
    """

    _compression: Compression
    _data: bytes
    _span: Span | None

    def _load(self) -> None:
        span = self._span
        if span is not None:
            raw = span.read()
            compression = Compression(struct.unpack(">H", raw[:2])[0])
            self._compression, self._data, self._span = compression, raw[2:], None

    @property
    def compression(self) -> Compression:
        """Compression type. See :py:class:`~psd_tools.constants.Compression`."""
        self._load()
        return self._compression

    @compression.setter
    def compression(self, value: Compression) -> None:
        self._load()
        self._compression = value

    @property
    def data(self) -> bytes:
        """Data, as compressed with ``compression``."""
        self._load()
        return self._data

    @data.setter
    def data(self, value: bytes) -> None:
        self._load()
        self._data = value

    @property
    def _data_length(self) -> int:
        """Length of the data, unread when lazy."""
        return self._span.length - 2 if self._span is not None else len(self._data)

    def __eq__(self, other: object) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        assert isinstance(other, LazyPayload)
        return (self.compression, self.data) == (other.compression, other.data)

    __hash__ = None  # type: ignore[assignment]
