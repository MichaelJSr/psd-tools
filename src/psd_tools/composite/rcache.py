"""Render cache shared across composite passes.

Opt-in::

    with RenderCache():
        psd.composite(viewport=region)
        edited.composite(viewport=region)  # resumes below the edit

A pass stores the compositor state after each top-level layer below the
first whose text blend it reads off the preview, and the group alphas and
stroke bands it draws. Keys hash what a layer subtree *is* now
(record bytes, channel bytes, children), never ``id()``, so an edit between
passes changes the key and nothing needs invalidating. Stored arrays are
read-only. With no cache active a pass pays one ContextVar read.

``PSD_RCACHE_VERIFY=1`` renders every cached pass a second time without the
cache and raises :py:class:`CacheMismatch` unless the two are equal.
"""

from __future__ import annotations

import hashlib
import io
import os
from collections import Counter, OrderedDict
from collections.abc import Hashable
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any

import numpy as np

from psd_tools.api.text_edit import stylistic_alternates
from psd_tools.constants import Resource, Tag

if TYPE_CHECKING:
    from psd_tools.api.layers import Layer
    from psd_tools.api.psd_image import PSDImage

_ACTIVE: ContextVar[RenderCache | None] = ContextVar(
    "psd_tools_render_cache", default=None
)

Entry = tuple[np.ndarray, ...]


class CacheMismatch(AssertionError):
    """A cached render differs from the uncached one."""


def active() -> RenderCache | None:
    """The cache of the innermost ``with RenderCache()`` block, if any."""
    return _ACTIVE.get()


def suspended() -> Token[RenderCache | None]:
    """Turn the cache off until ``_ACTIVE.reset()`` of the returned token."""
    return _ACTIVE.set(None)


class RenderCache:
    """LRU of compositor intermediates, bounded in bytes.

    Not thread-safe: give each thread its own.

    :param max_bytes: The size the stored arrays are kept within.
    :param verify: Check every cached pass against an uncached one; by
        default whether ``PSD_RCACHE_VERIFY`` is ``1``.
    """

    def __init__(self, max_bytes: int = 512 << 20, verify: bool | None = None) -> None:
        self.max_bytes = max_bytes
        self.verify = (
            os.environ.get("PSD_RCACHE_VERIFY") == "1" if verify is None else verify
        )
        self.hits: Counter[str] = Counter()
        self.misses: Counter[str] = Counter()
        self._store: OrderedDict[Hashable, Entry] = OrderedDict()
        self._bytes = 0
        self._tokens: list[Token[RenderCache | None]] = []

    def __enter__(self) -> RenderCache:
        self._tokens.append(_ACTIVE.set(self))
        return self

    def __exit__(self, *exc: object) -> None:
        _ACTIVE.reset(self._tokens.pop())

    def __contains__(self, key: object) -> bool:
        return key in self._store

    def get(self, key: tuple[str, Any]) -> Entry | None:
        """The entry, counted as a hit or a miss of its kind ``key[0]``."""
        value = self._store.get(key)
        if value is None:
            self.misses[key[0]] += 1
            return None
        self._store.move_to_end(key)
        self.hits[key[0]] += 1
        return value

    def put(self, key: tuple[str, Any], value: Entry) -> None:
        """Store the arrays, read-only, evicting the least recently used."""
        for array in value:
            array.flags.writeable = False
        size = _nbytes(value)
        if size > self.max_bytes:
            return
        old = self._store.pop(key, None)
        if old is not None:
            self._bytes -= _nbytes(old)
        self._store[key] = value
        self._bytes += size
        while self._bytes > self.max_bytes:
            _, evicted = self._store.popitem(last=False)
            self._bytes -= _nbytes(evicted)

    def clear(self) -> None:
        """Drop every entry."""
        self._store.clear()
        self._bytes = 0

    @property
    def nbytes(self) -> int:
        """The bytes stored, an array shared by two entries counted twice."""
        return self._bytes


def _nbytes(value: Entry) -> int:
    return sum(array.nbytes for array in value)


def check(plain: Entry, cached: Entry, what: object) -> None:
    """Raise :py:class:`CacheMismatch` unless the two renders are equal."""
    for name, a, b in zip(("color", "shape", "alpha"), plain, cached, strict=True):
        if a.shape != b.shape or not np.array_equal(a, b, equal_nan=True):
            raise CacheMismatch(f"render cache changed the {name} of {what}")


def _record_bytes(record: Any, version: int) -> bytes:
    buffer = io.BytesIO()
    record.write(buffer, version=version)
    return buffer.getvalue()


def layer_key(layer: Layer, memo: dict[int, bytes]) -> bytes:
    """Hash of the layer's subtree as it is now: records, channels, children."""
    known = memo.get(id(layer))
    if known is not None:
        return known
    version = layer._psd.version
    digest = hashlib.blake2b(digest_size=16)
    digest.update(_record_bytes(layer._record, version))
    for channel in layer._channels:
        digest.update(channel.data)
    if layer.kind == "type":  # Stylistic Alternates live in Txt2 only
        digest.update(repr(stylistic_alternates(layer)).encode())  # type: ignore[arg-type]
    if layer.is_group():
        bounding = getattr(layer, "_bounding_record", None)
        if bounding is not None:
            digest.update(_record_bytes(bounding, version))
        for child in layer:  # type: ignore[attr-defined]
            digest.update(layer_key(child, memo))
    key = memo[id(layer)] = digest.digest()
    return key


# The document-level blocks a render reads. Never Txt2: it changes with every
# text edit and is not composited.
_DOCUMENT_BLOCKS = (
    Tag.PATTERNS1,
    Tag.PATTERNS2,
    Tag.PATTERNS3,
    Tag.LINKED_LAYER1,
    Tag.LINKED_LAYER2,
    Tag.LINKED_LAYER3,
    Tag.FILTER_EFFECTS1,
    Tag.FILTER_EFFECTS2,
    Tag.FILTER_EFFECTS3,
)
_DOCUMENT_RESOURCES = (
    Resource.ICC_PROFILE,
    Resource.GLOBAL_ANGLE,
    Resource.GLOBAL_ALTITUDE,
)


def doc_key(psd: PSDImage) -> bytes | None:
    """What every layer's render may read from the document.

    None when the document links an external file, which no key can follow.
    Photoshop writes an empty ``lnkE`` block into most documents; that links
    nothing.
    """
    blocks = psd.tagged_blocks
    if blocks is not None and blocks.get_data(Tag.LINKED_LAYER_EXTERNAL):
        return None
    record = psd._record
    version = record.header.version
    digest = hashlib.blake2b(digest_size=16)
    digest.update(_record_bytes(record.header, version))
    digest.update(_record_bytes(record.color_mode_data, version))
    digest.update(repr(psd._max_alloc_bytes).encode())
    for resource in _DOCUMENT_RESOURCES:
        if resource in psd.image_resources:
            digest.update(_record_bytes(psd.image_resources[resource], version))
    if blocks is not None:
        for tag in _DOCUMENT_BLOCKS:
            if tag in blocks:
                digest.update(_record_bytes(blocks[tag], version))
    return digest.digest()
