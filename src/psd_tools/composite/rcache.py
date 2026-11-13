"""Render cache shared across composite passes.

Opt-in::

    with RenderCache():
        psd.composite(viewport=region)
        edited.composite(viewport=region)  # resumes below the edit

A pass stores the compositor state after each top-level layer below the
first whose text blend it reads off the preview, and the group alphas and
stroke bands it draws. Keys hash what a layer subtree *is* now
(record bytes, channel bytes, children), never ``id()``, so an edit between
passes changes the key and nothing needs invalidating. A payload that cannot
change (channel bytes, a block as read, a linked file) is hashed once per
cache, and a large one with the bytes of one hashed (another open of the
file) is compared instead; records are hashed on every pass. Stored arrays
are read-only. With no cache active a pass pays one ContextVar read.

Inside ``with frozen(psd):``, a promise that ``psd`` is only read, its layer
and document keys are worked out once for every pass instead.

``PSD_RCACHE_VERIFY=1`` renders every cached pass a second time without the
cache and raises :py:class:`CacheMismatch` unless the two are equal, and keys
a frozen document again when its scope ends.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import io
import os
import weakref
from collections import Counter, OrderedDict
from collections.abc import Callable, Hashable, Iterator
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Any

import attrs
import numpy as np

from psd_tools.api.text_edit import stylistic_alternates
from psd_tools.constants import Compression, Resource, Tag
from psd_tools.psd.filter_effects import (
    FilterEffectChannel,
    FilterEffectExtra,
    FilterEffects,
)
from psd_tools.psd.linked_layer import LinkedLayers

if TYPE_CHECKING:
    from psd_tools.api.layers import Layer
    from psd_tools.api.psd_image import PSDImage
    from psd_tools.psd.tagged_blocks import TaggedBlock

_ACTIVE: ContextVar[RenderCache | None] = ContextVar(
    "psd_tools_render_cache", default=None
)

Entry = tuple[np.ndarray, ...]
Box = tuple[int, int, int, int]


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

    Not thread-safe: give each thread its own. It also keeps the digest of
    each immutable payload a key hashed, and the payload, while the channel
    or block that held it lives (up to 512 MB).

    :param max_bytes: The size the stored arrays are kept within.
    :param verify: Check every cached pass against an uncached one; by
        default whether ``PSD_RCACHE_VERIFY`` is ``1``.
    """

    def __init__(self, max_bytes: int = 256 << 20, verify: bool | None = None) -> None:
        self.max_bytes = max_bytes
        self.verify = (
            os.environ.get("PSD_RCACHE_VERIFY") == "1" if verify is None else verify
        )
        self.hits: Counter[str] = Counter()
        self.misses: Counter[str] = Counter()
        self._store: OrderedDict[Hashable, Entry] = OrderedDict()
        self._held: dict[int, list[Any]] = {}  # id: [array, entries holding it]
        self._boxes: dict[Hashable, list[Box]] = {}  # key: boxes put_boxed() stored
        self._bytes = 0
        self._tokens: list[Token[RenderCache | None]] = []
        # Payload digests: mark -> (the objects hashed, kept so ids stay unique,
        # digest, a weakref to their owner that drops the entry, size, its
        # _same key or None).
        self._digests: dict[Hashable, tuple[Any, bytes, Any, int, Any]] = {}
        self._pinned = 0
        # Large pinned payloads by (len, head, tail): [(mark, payload, digest)].
        self._same: dict[tuple[int, bytes, bytes], list[tuple[Any, Any, bytes]]] = {}

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
        if _nbytes(value) > self.max_bytes:
            return
        old = self._store.pop(key, None)
        if old is not None:
            self._drop(old)
        self._store[key] = value
        for array in value:
            held = self._held.get(id(array))
            if held is None:
                self._held[id(array)] = [array, 1]
                self._bytes += array.nbytes
            else:
                held[1] += 1
        while self._bytes > self.max_bytes:
            _, evicted = self._store.popitem(last=False)
            self._drop(evicted)

    def get_around(self, key: tuple, box: Box) -> tuple[Box, Entry] | None:
        """A :py:meth:`put_boxed` entry on a box around ``box``, and that box.

        Counted as one hit or miss of its kind ``key[0]``.
        """
        x0, y0, x1, y1 = box
        boxes = self._boxes.get(key)
        if boxes:
            for held in boxes:
                if held[0] <= x0 and held[1] <= y0 and x1 <= held[2] and y1 <= held[3]:
                    value = self._store.get((*key, held))
                    if value is not None:
                        self._store.move_to_end((*key, held))
                        self.hits[key[0]] += 1
                        return held, value
            boxes[:] = [held for held in boxes if (*key, held) in self._store]
        self.misses[key[0]] += 1
        return None

    def put_boxed(self, key: tuple, box: Box, value: Entry) -> None:
        """:py:meth:`put` under ``key`` and ``box``, found by :py:meth:`get_around`."""
        self.put((*key, box), value)
        if (*key, box) in self._store:
            boxes = self._boxes.setdefault(key, [])
            if box not in boxes:
                boxes.append(box)

    def _drop(self, value: Entry) -> None:
        for array in value:
            held = self._held[id(array)]
            held[1] -= 1
            if not held[1]:
                del self._held[id(array)]
                self._bytes -= array.nbytes

    def clear(self) -> None:
        """Drop every entry."""
        self._store.clear()
        self._held.clear()
        self._boxes.clear()
        self._bytes = 0
        self._digests.clear()
        self._same.clear()
        self._pinned = 0

    @property
    def nbytes(self) -> int:
        """The bytes stored, an array shared by entries counted once."""
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


# Payload bytes the digests may keep alive; past it they start over.
_PINNED_MAX = 512 << 20
# From this size a payload is first compared with those of its length hashed.
_SAME_MIN = 64 << 10
_SAME_STEP = 1 << 20


def _same_key(data: bytes | memoryview) -> tuple[int, bytes, bytes] | None:
    if type(data) is memoryview and type(data.obj) is not bytes:
        return None  # over a buffer that may change
    return (len(data), bytes(data[:64]), bytes(data[-64:]))


def _equal(a: bytes | memoryview, b: bytes | memoryview) -> bool:
    # memcmp's speed; memoryview == unpacks each byte.
    x, y = np.frombuffer(a, np.uint8), np.frombuffer(b, np.uint8)
    return x.size == y.size and all(
        np.array_equal(x[i : i + _SAME_STEP], y[i : i + _SAME_STEP])
        for i in range(0, x.size, _SAME_STEP)
    )


def _digest(
    rc: RenderCache | None,
    mark: Hashable,
    pin: Any,
    size: int,
    payload: Callable[[], bytes | memoryview],
    owner: Any,
) -> bytes:
    """sha256 of ``payload()``, kept under ``mark`` with ``pin`` while ``owner`` lives.

    A large payload with the bytes of one hashed before (another open of the
    file) takes its digest.
    """
    if rc is None:
        return hashlib.sha256(payload()).digest()
    known = rc._digests.get(mark)
    if known is not None:
        return known[1]
    data = payload()
    same = _same_key(data) if size >= _SAME_MIN else None
    digest = None
    if same is not None:  # two at most: past them, sha256
        candidates = rc._same.get(same, ())[:2]
        digest = next((d for _, b, d in candidates if _equal(data, b)), None)
    if digest is None:
        digest = hashlib.sha256(data).digest()
    try:
        tie = weakref.ref(owner, functools.partial(_forget, weakref.ref(rc), mark))
    except TypeError:
        return digest
    if rc._pinned + size > _PINNED_MAX:
        rc._digests.clear()
        rc._same.clear()
        rc._pinned = 0
    if data is not pin and type(data) is not memoryview:
        same = None  # bytes made for the hash: kept by nothing else
    rc._digests[mark] = (pin, digest, tie, size, same)
    if same is not None:
        rc._same.setdefault(same, []).append((mark, data, digest))
    rc._pinned += size
    return digest


def _forget(cache: weakref.ref[RenderCache], mark: Hashable, _: object) -> None:
    rc = cache()
    entry = None if rc is None else rc._digests.pop(mark, None)
    if rc is not None and entry is not None:
        rc._pinned -= entry[3]
        same = entry[4]
        if same is not None:
            kept = [item for item in rc._same.get(same, ()) if item[0] != mark]
            if kept:
                rc._same[same] = kept
            else:
                rc._same.pop(same, None)


def _bytes_digest(rc: RenderCache | None, data: Any, owner: Any) -> bytes:
    if type(data) is bytes:
        return _digest(rc, id(data), data, len(data), lambda: data, owner)
    return hashlib.sha256(data).digest()  # mutable: whole every time


def _channel_digest(channel: Any, rc: RenderCache | None) -> bytes:
    """Digest of ``channel.data``; an unread one hashed in place."""
    span = channel._span
    if span is not None:
        whole = span._source.data
        end = span.offset + span.length
        if whole is not None and end <= len(whole):

            def unread() -> memoryview:
                # A bad compression raises, as reading ``data`` does.
                Compression(int.from_bytes(whole[span.offset : span.offset + 2]))
                return memoryview(whole)[span.offset + 2 : end]

            return _digest(rc, id(span), span, span.length, unread, channel)
    return _bytes_digest(rc, channel.data, channel)


def layer_key(layer: Layer, memo: dict[int, bytes]) -> bytes:
    """Hash of the layer's subtree as it is now: records, channels, children.

    In a :py:func:`frozen` document, the one worked out in its scope.
    """
    known = memo.get(id(layer))
    if known is not None:
        return known
    psd = layer._psd
    state = psd._frozen_keys
    if state is not None and state.owner == id(psd):
        memo = state.keys
    return _layer_key(layer, memo, _ACTIVE.get())


class _Frozen:
    __slots__ = ("owner", "keys", "doc", "depth", "verify")

    def __init__(self, owner: int) -> None:
        self.owner = owner  # id() of its document: a copy's is not its own
        self.keys: dict[int, bytes] = {}
        self.doc: bytes | None = None
        self.depth = 0
        self.verify = False


@contextlib.contextmanager
def frozen(psd: PSDImage, verify: bool | None = None) -> Iterator[PSDImage]:
    """A scope in which the caller promises ``psd`` is only read.

    Each layer's :py:func:`layer_key` and the :py:func:`doc_key` are worked
    out once in it and kept for every pass, instead of written and hashed
    again in each. A change inside the scope is not seen: a pass would reuse
    what the document drew before it. Scopes nest; open and close those of a
    document in one thread.

    :param verify: Key the document again at the outermost exit and raise
        :py:class:`CacheMismatch` if anything moved; by default whether
        ``PSD_RCACHE_VERIFY`` is ``1``.
    """
    state = psd._frozen_keys
    if state is None or state.owner != id(psd):
        state = psd._frozen_keys = _Frozen(id(psd))
    state.depth += 1
    state.verify |= (
        os.environ.get("PSD_RCACHE_VERIFY") == "1" if verify is None else verify
    )
    try:
        yield psd
    finally:
        state.depth -= 1
        if not state.depth:
            psd._frozen_keys = None
    if not state.depth and state.verify:
        _check_frozen(psd, state)


def _check_frozen(psd: PSDImage, state: _Frozen) -> None:
    fresh: dict[int, bytes] = {}
    rc = _ACTIVE.get()
    for layer in psd.descendants():
        key = state.keys.get(id(layer))
        if key is not None and _layer_key(layer, fresh, rc) != key:
            raise CacheMismatch(f"{layer!r} changed inside rcache.frozen()")
    if state.doc is not None and _doc_key(psd) != state.doc:
        raise CacheMismatch("the document changed inside rcache.frozen()")


def _layer_key(layer: Layer, memo: dict[int, bytes], rc: RenderCache | None) -> bytes:
    known = memo.get(id(layer))
    if known is not None:
        return known
    version = layer._psd.version
    # sha256 runs on the CPU's hash instructions: 3x blake2b on an M-series.
    digest = hashlib.sha256()
    digest.update(_record_bytes(layer._record, version))
    for channel in layer._channels:
        digest.update(_channel_digest(channel, rc))
    if layer.kind == "type":  # Stylistic Alternates live in Txt2 only
        digest.update(repr(stylistic_alternates(layer)).encode())  # type: ignore[arg-type]
    if layer.is_group():
        bounding = getattr(layer, "_bounding_record", None)
        if bounding is not None:
            digest.update(_record_bytes(bounding, version))
        for child in layer:  # type: ignore[attr-defined]
            digest.update(_layer_key(child, memo, rc))
    key = memo[id(layer)] = digest.digest()[:16]
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
    Tag.LINKED_LAYER_EXTERNAL,
    Tag.FILTER_EFFECTS1,
    Tag.FILTER_EFFECTS2,
    Tag.FILTER_EFFECTS3,
)
_DOCUMENT_RESOURCES = (
    Resource.ICC_PROFILE,
    Resource.GLOBAL_ANGLE,
    Resource.GLOBAL_ALTITUDE,
)


def _written(data: Any, version: int) -> bytes:
    """A parsed block's bytes inside its length, as ``TaggedBlock.write`` writes them."""
    buffer = io.BytesIO()
    data.write(buffer, padding=4, version=version)
    return buffer.getvalue()


class _Unsplit(Exception):
    pass


def _emptied(item: Any, payloads: list[bytes]) -> Any:
    body = item.data
    if body is None:
        return item
    if type(body) is not bytes:
        raise _Unsplit
    payloads.append(body)
    return attrs.evolve(item, data=b"")


def _split(data: Any) -> tuple[Any, tuple[bytes, ...]] | None:
    """``data`` with its linked files or smart filter pixels emptied, and those."""
    payloads: list[bytes] = []
    try:
        if type(data) is LinkedLayers:
            items = [_emptied(item, payloads) for item in data]
        elif type(data) is FilterEffects:
            items = []
            for effect in data:
                parts = {}
                if effect.channels is not None:
                    parts["channels"] = [
                        _emptied(c, payloads) if type(c) is FilterEffectChannel else c
                        for c in effect.channels
                    ]
                if type(effect.extra) is FilterEffectExtra:
                    parts["extra"] = _emptied(effect.extra, payloads)
                items.append(attrs.evolve(effect, **parts))
        else:
            return None
        return attrs.evolve(data, items=items), tuple(payloads)
    except (_Unsplit, AttributeError, TypeError, ValueError):
        return None


def _block_digest(block: TaggedBlock, version: int, rc: RenderCache | None) -> bytes:
    """The block's signature and key, then a digest of its bytes inside its length.

    The bytes as read (held, or unread) when it has them, else its data
    written: a parsed block writes the bytes it was read from.
    """
    key = block.key if isinstance(block.key, bytes) else block.key.value
    head = block.signature + key
    held = block._held
    if held is not None:
        return head + _bytes_digest(rc, held, block)
    if block._span is not None and block._span[1] == version:
        span = block._span[0]
        return head + _digest(rc, id(span), span, span.length, span.view, block)
    data = block.data
    if not hasattr(data, "write"):
        return head + _bytes_digest(rc, data, block)
    split = _split(data) if rc is not None else None
    if split is None:
        return head + hashlib.sha256(_written(data, version)).digest()
    # Its records written again, its large payloads known by identity.
    light, payloads = split
    mark = (type(data), _written(light, version), *map(id, payloads))
    size = sum(map(len, payloads))
    return head + _digest(
        rc, mark, payloads, size, lambda: _written(data, version), data
    )


def doc_key(psd: PSDImage) -> bytes | None:
    """What every layer's render may read from the document.

    A linked layer draws its stored pixels, never its file, so its ``lnkE``
    record is hashed like an embedded one. In a :py:func:`frozen` document,
    the one worked out in its scope.
    """
    state = psd._frozen_keys
    if state is None or state.owner != id(psd):
        return _doc_key(psd)
    if state.doc is None:
        state.doc = _doc_key(psd)
    return state.doc


def _doc_key(psd: PSDImage) -> bytes | None:
    blocks = psd.tagged_blocks
    record = psd._record
    version = record.header.version
    rc = _ACTIVE.get()
    digest = hashlib.sha256()
    digest.update(_record_bytes(record.header, version))
    digest.update(_record_bytes(record.color_mode_data, version))
    digest.update(repr(psd._max_alloc_bytes).encode())
    for resource in _DOCUMENT_RESOURCES:
        if resource in psd.image_resources:
            digest.update(_record_bytes(psd.image_resources[resource], version))
    if blocks is not None:
        for tag in _DOCUMENT_BLOCKS:
            if tag in blocks:
                digest.update(_block_digest(blocks[tag], version, rc))
    return digest.digest()[:16]
