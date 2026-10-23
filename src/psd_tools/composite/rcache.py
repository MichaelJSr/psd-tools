"""Content-keyed render cache shared across composite passes (research prototype).

Opt-in: ``with RenderCache() as rc: psd.composite(...)``. Off, the compositor
pays one ContextVar read per pass. Keys hash what a layer subtree *is* now
(record bytes, channel bytes, children), never ``id()``, so an edit between
passes changes the key instead of leaving a stale entry. Values are stored
read-only: a caller that writes into one raises instead of corrupting it.
"""

from __future__ import annotations

import hashlib
import io
from collections import Counter, OrderedDict
from contextvars import ContextVar, Token
from typing import Any

import numpy as np

_ACTIVE: ContextVar[RenderCache | None] = ContextVar(
    "psd_tools_render_cache", default=None
)


def active() -> RenderCache | None:
    return _ACTIVE.get()


class RenderCache:
    """LRU of compositor intermediates, bounded in bytes."""

    def __init__(self, max_bytes: int = 1 << 30) -> None:
        self.max_bytes = max_bytes
        self._store: OrderedDict[tuple, tuple[np.ndarray | None, ...]] = OrderedDict()
        self._bytes = 0
        self.hits: Counter[str] = Counter()
        self.misses: Counter[str] = Counter()
        self._token: Token | None = None

    def __enter__(self) -> RenderCache:
        self._token = _ACTIVE.set(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        assert self._token is not None
        _ACTIVE.reset(self._token)
        self._token = None

    def get(self, key: tuple) -> tuple[np.ndarray | None, ...] | None:
        value = self._store.get(key)
        if value is None:
            self.misses[key[0]] += 1
            return None
        self._store.move_to_end(key)
        self.hits[key[0]] += 1
        return value

    def peek(self, key: tuple) -> tuple[np.ndarray | None, ...] | None:
        """``get`` without counting a hit or a miss."""
        value = self._store.get(key)
        if value is not None:
            self._store.move_to_end(key)
        return value

    def put(self, key: tuple, value: tuple[np.ndarray | None, ...]) -> None:
        size = 0
        for array in value:
            if array is not None:
                array.flags.writeable = False
                size += array.nbytes
        if size > self.max_bytes:
            return
        old = self._store.pop(key, None)
        if old is not None:
            self._bytes -= sum(a.nbytes for a in old if a is not None)
        self._store[key] = value
        self._bytes += size
        while self._bytes > self.max_bytes:
            _, evicted = self._store.popitem(last=False)
            self._bytes -= sum(a.nbytes for a in evicted if a is not None)

    @property
    def nbytes(self) -> int:
        return self._bytes


def _record_bytes(record: Any) -> bytes:
    buffer = io.BytesIO()
    record.write(buffer)
    return buffer.getvalue()


def layer_key(layer: Any, memo: dict[int, bytes]) -> bytes:
    """Hash of the layer's subtree as it is now: records, channels, children."""
    known = memo.get(id(layer))
    if known is not None:
        return known
    digest = hashlib.blake2b(digest_size=16)
    digest.update(_record_bytes(layer._record))
    for channel in layer._channels:
        digest.update(channel.data)
    if layer.is_group():
        bounding = getattr(layer, "_bounding_record", None)
        if bounding is not None:
            digest.update(_record_bytes(bounding))
        for child in layer:
            digest.update(layer_key(child, memo))
    key = memo[id(layer)] = digest.digest()
    return key


def doc_key(psd: Any) -> bytes:
    """What every layer's render may read from the document: header, the
    resources effects consult, and the document-level tagged blocks (patterns,
    linked smart-object data, filter effects).
    """
    from psd_tools.constants import Resource

    digest = hashlib.blake2b(digest_size=16)
    digest.update(_record_bytes(psd._record.header))
    for resource in (
        Resource.ICC_PROFILE,
        Resource.GLOBAL_ANGLE,
        Resource.GLOBAL_ALTITUDE,
    ):
        if resource in psd.image_resources:
            digest.update(repr(psd.image_resources.get_data(resource)).encode())
    blocks = psd.tagged_blocks
    if blocks is not None:
        # Only the blocks a render reads: Txt2 changes with every text edit
        # and is never composited.
        for tag in _RENDERED_DOCUMENT_BLOCKS:
            if tag in blocks:
                digest.update(tag.value)
                digest.update(_record_bytes(blocks[tag]))
    return digest.digest()


def _rendered_blocks() -> tuple:
    from psd_tools.constants import Tag

    return (
        Tag.PATTERNS1,
        Tag.PATTERNS2,
        Tag.PATTERNS3,
        Tag.LINKED_LAYER1,
        Tag.LINKED_LAYER2,
        Tag.LINKED_LAYER3,
        Tag.LINKED_LAYER_EXTERNAL,
        Tag.FILTER_EFFECTS1,
        Tag.FILTER_EFFECTS2,
        Tag.FILTER_EFFECTS3,
    )


_RENDERED_DOCUMENT_BLOCKS = _rendered_blocks()
