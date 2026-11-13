"""The cross-pass render cache returns what an uncached pass computes."""

from __future__ import annotations

import contextlib
import gc
import hashlib
import io
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from PIL import Image

from psd_tools import PSDImage
from psd_tools.api.layers import Layer, PixelLayer, TypeLayer
from psd_tools.composite import composite, rcache
from psd_tools.composite.rcache import CacheMismatch, RenderCache, doc_key, layer_key
from psd_tools.constants import BlendMode, Tag
from psd_tools.psd.lazy import load

from ..smart_filters import feid_item, store_items
from ..utils import full_name

pytest.importorskip("scipy")

_FILES = [
    "layer_effects.psd",
    "effects/group-clips-child-stroke.psd",
    "effects/double-stroke-effects.psd",
    "effects/stroke-effects.psd",
]


def _render(
    psd: PSDImage | str,
    cache: RenderCache | None,
    edit: bool = False,
    viewport: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    if isinstance(psd, str):
        psd = PSDImage.open(full_name(psd))
    if edit:
        psd[len(psd) - 1].visible = not psd[len(psd) - 1].visible
    if cache is None:
        return np.concatenate(composite(psd, viewport=viewport), axis=2)
    with cache:
        return np.concatenate(composite(psd, viewport=viewport), axis=2)


@pytest.mark.parametrize("path", _FILES)
def test_a_cached_pass_is_bit_identical(path: str) -> None:
    cache = RenderCache()
    plain = _render(path, None)
    assert np.array_equal(_render(path, cache), plain)
    assert np.array_equal(_render(path, cache), plain)  # every entry a hit
    assert cache.hits["prefix"] == 1


@pytest.mark.parametrize("path", _FILES)
def test_an_edit_between_passes_is_not_served_stale(path: str) -> None:
    cache = RenderCache()
    _render(path, cache)
    assert np.array_equal(
        _render(path, cache, edit=True), _render(path, None, edit=True)
    )


def test_stored_values_are_read_only() -> None:
    cache = RenderCache()
    _render(_FILES[1], cache)
    assert cache._store
    for value in cache._store.values():
        for array in value:
            if array is not None:
                with pytest.raises(ValueError):
                    array[...] = 0


def test_no_cache_is_active_outside_the_block() -> None:
    cache = RenderCache()
    with cache:
        pass
    _render(_FILES[0], None)
    assert not cache._store


def test_a_clip_layer_edit_is_not_served_stale() -> None:
    # The clip layer is composited inside its base's apply, so no state is
    # stored between the two: it would be named by a key without the clip.
    path = "effects/double-stroke-effects.psd"
    cache = RenderCache()
    _render(path, cache)
    assert sum(cast(tuple, key)[0] == "prefix" for key in cache._store) == 1
    psd = PSDImage.open(full_name(path))
    clip = psd[1]
    assert clip.clipping
    clip.opacity = 128
    expected = _render(psd, None)
    assert np.array_equal(_render(psd, cache), expected)
    assert cache.hits["prefix"] == 0


def test_a_layer_that_starts_clipping_is_not_served_stale() -> None:
    # The state after the base was stored when it had no clip layers.
    psd = PSDImage.open(full_name("layer_effects.psd"))
    cache = RenderCache()
    _render(psd, cache)
    psd[3].clipping = True
    assert np.array_equal(_render(psd, cache), _render(psd, None))


def test_a_group_pass_follows_the_background() -> None:
    # Deep knockout inside a group knocks out to the document's Background.
    psd = PSDImage.open(full_name("effects/group-clips-child-stroke.psd"))
    group = psd[1]
    cache = RenderCache()
    with cache:
        composite(group)
    assert cache.misses["prefix"] == 1
    psd[0].visible = False
    with cache:
        composite(group)
    assert cache.misses["prefix"] == 2


def test_regions_share_group_alphas() -> None:
    path = "effects/group-stroke-off-canvas.psd"
    cache = RenderCache()
    psd = PSDImage.open(full_name(path))
    _render(psd, cache, viewport=(24, 0, 96, 32))
    assert cache.misses["group_alpha"] > 0
    misses = cache.misses["group_alpha"]
    plain = _render(psd, None, viewport=(44, 10, 52, 22))
    assert np.array_equal(_render(psd, cache, viewport=(44, 10, 52, 22)), plain)
    assert cache.misses["group_alpha"] == misses
    assert cache.hits["group_alpha"] > 0


def test_a_group_alpha_is_composited_on_the_box_asked_for() -> None:
    # The box an uncached pass composites, not the group's whole box.
    psd = PSDImage.open(full_name("effects/group-stroke-off-canvas.psd"))
    cache = RenderCache()
    for box in ((36, 8, 40, 12), (37, 9, 39, 11), (56, 8, 60, 24)):
        plain = _render(psd, None, viewport=box)
        assert np.array_equal(_render(psd, cache, viewport=box), plain)
    keys = [cast(tuple, key) for key in cache._store]
    boxes = [key[-1] for key in keys if key[0] == "group_alpha"]
    assert boxes == [(36, 8, 52, 24), (44, 8, 60, 24)]  # "Inside" is (36, 8, 60, 24)
    assert (cache.hits["group_alpha"], cache.misses["group_alpha"]) == (1, 2)


def test_a_boxed_entry_serves_boxes_inside_it_until_evicted() -> None:
    cache = RenderCache(max_bytes=2000)
    key, alpha = ("group_alpha", b"k"), np.zeros(250, np.float32)
    cache.put_boxed(key, (0, 0, 10, 10), (alpha,))
    hit = cache.get_around(key, (2, 2, 10, 5))
    assert hit is not None and hit[0] == (0, 0, 10, 10) and hit[1][0] is alpha
    assert cache.get_around(key, (2, 2, 11, 5)) is None
    cache.put(("prefix", 1), (np.zeros(400, np.float32),))  # evicts it
    assert cache.get_around(key, (2, 2, 3, 3)) is None
    assert not cache._boxes[key]
    cache.put_boxed(key, (0, 0, 1, 1), (alpha,))
    cache.clear()
    assert not cache._boxes
    nothing = RenderCache(max_bytes=0)
    nothing.put_boxed(key, (0, 0, 10, 10), (alpha,))
    assert nothing.get_around(key, (0, 0, 1, 1)) is None and not nothing._boxes


def test_verify_catches_a_wrong_entry() -> None:
    path = "effects/double-stroke-effects.psd"
    cache = RenderCache(verify=True)
    _render(path, cache)
    for key, value in list(cache._store.items()):
        if cast(tuple, key)[0] == "prefix":
            color = np.array(value[0])
            color[0, 0] = 0.5
            cache._store[key] = (color, *value[1:])
    with pytest.raises(CacheMismatch):
        _render(path, cache)


def test_verify_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PSD_RCACHE_VERIFY", "1")
    assert RenderCache().verify
    monkeypatch.delenv("PSD_RCACHE_VERIFY")
    assert not RenderCache().verify


def test_the_cache_stays_within_its_size() -> None:
    cache = RenderCache(max_bytes=64 << 10)
    for path in _FILES:
        _render(path, cache)
        assert cache.nbytes <= cache.max_bytes


def test_an_array_shared_by_entries_is_counted_once() -> None:
    cache = RenderCache(max_bytes=3000)
    a, b = np.zeros(250, np.float32), np.ones(250, np.float32)
    cache.put(("prefix", 1), (a, b))
    cache.put(("prefix", 2), (a, a))
    assert cache.nbytes == 2000
    cache.put(("prefix", 3), (np.zeros(250, np.float32),))
    cache.put(("prefix", 4), (np.zeros(250, np.float32),))  # evicts 1 only
    assert ("prefix", 1) not in cache and ("prefix", 2) in cache
    assert cache.nbytes == 3000


def test_a_checkpoint_holds_equal_planes_once() -> None:
    cache = RenderCache()
    plain = _render("hidden-groups.psd", None)
    assert np.array_equal(_render("hidden-groups.psd", cache), plain)
    states = [v for k, v in cache._store.items() if cast(tuple, k)[0] == "prefix"]
    assert len(states) > 1
    assert all(state[1] is state[3] for state in states)  # alpha, alpha_g
    assert len({id(a) for state in states for a in state}) < 2 * len(states)
    assert np.array_equal(_render("hidden-groups.psd", cache), plain)


def _story(psd: PSDImage) -> bytes:
    assert psd.tagged_blocks is not None
    buffer = io.BytesIO()
    psd.tagged_blocks[Tag.TEXT_ENGINE_DATA].write(buffer)
    return buffer.getvalue()


def test_a_text_edit_keeps_the_document_key() -> None:
    psd = PSDImage.open(full_name("layer_effects.psd"))
    before, story = doc_key(psd), _story(psd)
    assert before is not None
    text = next(layer for layer in psd.descendants() if isinstance(layer, TypeLayer))
    text.set_text("Bevel x", keep="runs")
    assert _story(psd) != story
    assert doc_key(psd) == before


def test_an_external_link_is_part_of_the_document_key() -> None:
    # A linked layer draws its stored pixels, so the cache stays on.
    psd = PSDImage.open(full_name("placedLayer.psd"))
    assert psd.tagged_blocks is not None
    before = doc_key(psd)
    assert before is not None
    psd.tagged_blocks.get_data(Tag.LINKED_LAYER_EXTERNAL)[0].filesize += 1
    assert doc_key(psd) not in (None, before)
    cache = RenderCache()
    _render(psd, cache)
    assert cache._store


def _keys(psd: PSDImage, cache: RenderCache | None) -> tuple[bytes | None, ...]:
    with cache or contextlib.nullcontext():
        return (doc_key(psd), *(layer_key(layer, {}) for layer in psd))


def _with_smart_filter(tmp_path: Path) -> Path:
    psd = PSDImage.open(full_name("layers/smartobject-layer.psd"))
    plane = np.full((4, 4), 200, np.uint8)
    store_items(psd, [feid_item((0, 0, 4, 4), plane[..., None], plane, plane)])
    path = tmp_path / "fx.psd"
    psd.save(path)
    return path


def test_keys_do_not_depend_on_how_a_payload_is_held(tmp_path: Path) -> None:
    # Path open: links parsed, FEid held. BytesIO: both unread, channels too.
    path = _with_smart_filter(tmp_path)
    eager = PSDImage.open(path)
    lazy = PSDImage.open(io.BytesIO(path.read_bytes()))
    want = _keys(eager, None)
    cache = RenderCache()
    for psd in (eager, lazy, eager, lazy):
        assert _keys(psd, cache) == want
        assert _keys(psd, None) == want
    for psd in (eager, lazy):
        assert psd.tagged_blocks is not None
        for block in psd.tagged_blocks.values():
            load(block)
        for layer in psd.descendants():
            for channel in layer._channels:
                load(channel)
        assert _keys(psd, cache) == want
        assert _keys(psd, RenderCache()) == want


def test_a_payload_swapped_under_the_cache_changes_the_key(tmp_path: Path) -> None:
    psd = PSDImage.open(io.BytesIO(_with_smart_filter(tmp_path).read_bytes()))
    assert psd.tagged_blocks is not None
    cache = RenderCache()
    before = _keys(psd, cache)
    item = psd.tagged_blocks.get_data(Tag.LINKED_LAYER2)[0]
    effect = psd.tagged_blocks.get_data(Tag.FILTER_EFFECTS2)[0]
    channel = psd[0]._channels[0]
    file, pixels, plane = item.data, effect.channels[0].data, channel.data
    for owner, name, value in (
        (item, "data", file),
        (effect.channels[0], "data", pixels),
        (channel, "data", plane),
    ):
        setattr(owner, name, bytes(reversed(value)) + b"\0")
        assert _keys(psd, cache) != before
        setattr(owner, name, bytes(bytearray(value)))  # equal, another object
        assert _keys(psd, cache) == before


@pytest.mark.parametrize("lazy", [False, True])
def test_the_digests_go_with_their_document(tmp_path: Path, lazy: bool) -> None:
    path = _with_smart_filter(tmp_path)
    psd = PSDImage.open(io.BytesIO(path.read_bytes()) if lazy else path)
    cache = RenderCache()
    _keys(psd, cache)
    assert cache._digests
    del psd
    gc.collect()
    assert not cache._digests and cache._pinned == 0


def test_a_reopened_file_takes_the_digests_by_its_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    psd = PSDImage.new("RGB", (320, 320))
    noise = np.random.default_rng(0).integers(0, 256, (320, 320, 3), np.uint8)
    psd.append(PixelLayer.frompil(Image.fromarray(noise, "RGB"), psd, "noise"))
    path = tmp_path / "noise.psd"
    psd.save(path)
    want = _keys(PSDImage.open(path), None)
    hashed: list[int] = []

    def counted(data: bytes | memoryview = b"") -> object:
        if len(data) >= rcache._SAME_MIN:
            hashed.append(len(data))
        return hashlib.sha256(data)

    monkeypatch.setattr(rcache, "hashlib", SimpleNamespace(sha256=counted))
    cache = RenderCache()
    # Path (eager), then BytesIO (unread spans) twice.
    docs = [PSDImage.open(path)]
    docs += [PSDImage.open(io.BytesIO(path.read_bytes())) for _ in range(2)]
    assert [_keys(doc, cache) for doc in docs] == [want] * 3
    assert len(hashed) == 3  # the noise channels, once each
    assert len(cache._same) == 3
    del docs
    gc.collect()
    assert not cache._same and not cache._digests and cache._pinned == 0


class _Owner:
    pass


def test_a_payload_over_a_mutable_buffer_is_never_matched() -> None:
    buffer = bytearray(rcache._SAME_MIN)
    owners = [_Owner(), _Owner()]
    cache = RenderCache()
    rcache._digest(cache, 1, buffer, len(buffer), lambda: memoryview(buffer), owners[0])
    buffer[len(buffer) // 2] = 1  # its head and tail as they were
    now = bytes(buffer)
    got = rcache._digest(cache, 2, now, len(now), lambda: now, owners[1])
    assert got == hashlib.sha256(now).digest()


_PROPERTY_FILES = [
    "blend-and-clipping.psd",
    "clipping-mask.psd",
    "clip-adjustment.psd",
    "hidden-groups.psd",
    "masks.psd",
    "layer_effects.psd",
    "effects/group-clips-child-stroke.psd",
    "effects/double-stroke-effects.psd",
    "effects/stroke-effects.psd",
    "effects/stroke-composite.psd",
    "effects/group-stroke-off-canvas.psd",
]
_MODES = [BlendMode.NORMAL, BlendMode.MULTIPLY, BlendMode.SCREEN, BlendMode.OVERLAY]


def _random_edit(rng: random.Random, psd: PSDImage) -> None:
    layer: Layer = rng.choice(list(psd.descendants()))
    edit = rng.choice(["visible", "opacity", "fill", "mode", "clip", "move"])
    if edit == "visible":
        layer.visible = not layer.visible
    elif edit == "opacity":
        layer.opacity = rng.randrange(256)
    elif edit == "fill":
        layer.fill_opacity = rng.randrange(256)
    elif edit == "mode" and not layer.is_group():
        layer.blend_mode = rng.choice(_MODES)
    elif edit == "clip":
        layer.clipping = not layer.clipping
    elif edit == "move" and layer.kind == "pixel":
        layer.offset = (layer.left + rng.randint(-3, 3), layer.top + rng.randint(-3, 3))


def _random_viewport(
    rng: random.Random, psd: PSDImage
) -> tuple[int, int, int, int] | None:
    if rng.random() < 0.5:
        return None
    x0, y0 = rng.randrange(psd.width), rng.randrange(psd.height)
    return (x0, y0, rng.randint(x0 + 1, psd.width), rng.randint(y0 + 1, psd.height))


@pytest.mark.parametrize("seed", range(12))
def test_random_edits_render_as_without_the_cache(seed: int) -> None:
    rng = random.Random(seed)
    psd = PSDImage.open(full_name(rng.choice(_PROPERTY_FILES)))
    cache = RenderCache()
    for _ in range(5):
        viewport = _random_viewport(rng, psd)
        with np.errstate(all="ignore"):
            plain = _render(psd, None, viewport=viewport)
            cached = _render(psd, cache, viewport=viewport)
        assert np.array_equal(cached, plain, equal_nan=True)
        _random_edit(rng, psd)


def _veiled_kept_raster() -> PSDImage:
    """``text-kept-raster.psd`` veiled over its name, previewed with it straight.

    The name only settles straight on the finished canvas.
    """
    psd = PSDImage.open(full_name("text-kept-raster.psd"))
    name = next(layer for layer in psd.descendants() if layer.name == "Suffocation")
    left, top, right, bottom = name.bbox
    veil = Image.new("RGBA", (right - left, bottom - top), (128, 128, 128, 96))
    psd.append(PixelLayer.frompil(veil, psd, "veil", top, left))
    straight, _, _ = composite(
        psd, force=True, text_preview=False, layer_text_gamma={name.layer_id: None}
    )
    image = Image.fromarray(np.round(straight[..., :3] * 255).astype(np.uint8), "RGB")
    buffer = io.BytesIO()
    psd.save(buffer, preview=image)
    buffer.seek(0)
    return PSDImage.open(buffer)


@pytest.mark.parametrize("pinned", [False, True])
def test_a_text_decision_is_made_again_in_every_pass(pinned: bool) -> None:
    """The preview's choice for a type layer is not in a checkpoint.

    Restored past the name, a pass would skip deciding it, and the name that
    settles straight on the finished canvas would stay in the gamma. A layer
    pinned by ID needs no decision, so the pass resumes past it.
    """
    psd = _veiled_kept_raster()
    gammas = {
        layer.layer_id: None
        for layer in psd.descendants()
        if pinned and isinstance(layer, TypeLayer)
    }
    plain = np.concatenate(composite(psd, force=True, layer_text_gamma=gammas), axis=2)
    cache = RenderCache()
    for _ in range(2):
        with cache:
            cached = composite(psd, force=True, layer_text_gamma=gammas)
        assert np.array_equal(np.concatenate(cached, axis=2), plain)
    assert cache.hits["prefix"] == (1 if pinned else 0)


def test_a_frozen_document_writes_each_record_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = "effects/group-clips-child-stroke.psd"
    psd = PSDImage.open(full_name(path))
    box = (0, 0, psd.width // 2, psd.height // 2)
    plain = [_render(path, None), _render(path, None, viewport=box)]
    writes: Counter[int] = Counter()
    write = rcache._record_bytes

    def counted(record: object, version: int) -> bytes:
        writes[id(record)] += 1
        return write(record, version)

    monkeypatch.setattr(rcache, "_record_bytes", counted)
    cache = RenderCache()
    with rcache.frozen(psd, verify=True):
        with rcache.frozen(psd):  # nested
            assert np.array_equal(_render(psd, cache), plain[0])
        assert np.array_equal(_render(psd, cache, viewport=box), plain[1])
        assert writes and max(writes.values()) == 1
    assert psd._frozen_keys is None
    writes.clear()
    _render(psd, cache)  # out of the scope: every pass again
    _render(psd, cache)
    assert min(writes.values()) >= 2


def test_a_change_in_a_frozen_document_raises_under_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PSD_RCACHE_VERIFY", "1")
    psd = PSDImage.open(full_name("layer_effects.psd"))
    with pytest.raises(CacheMismatch, match="frozen"):
        with RenderCache(verify=False), rcache.frozen(psd):
            composite(psd)
            psd[len(psd) - 1].opacity = 128
    assert psd._frozen_keys is None
