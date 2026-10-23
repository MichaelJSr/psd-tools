"""The cross-pass render cache returns what an uncached pass computes."""

from __future__ import annotations

import io
import random
from typing import cast

import numpy as np
import pytest

from psd_tools import PSDImage
from psd_tools.api.layers import Layer, TypeLayer
from psd_tools.composite import composite
from psd_tools.composite.rcache import CacheMismatch, RenderCache, doc_key
from psd_tools.constants import BlendMode, Tag
from psd_tools.psd.linked_layer import LinkedLayer, LinkedLayers

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


def test_an_external_link_turns_the_cache_off() -> None:
    psd = PSDImage.open(full_name("layer_effects.psd"))
    assert psd.tagged_blocks is not None
    psd.tagged_blocks.set_data(Tag.LINKED_LAYER_EXTERNAL, LinkedLayers())
    assert doc_key(psd) is not None  # an empty block links nothing
    links = LinkedLayers([LinkedLayer()])  # type: ignore[list-item]
    psd.tagged_blocks.set_data(Tag.LINKED_LAYER_EXTERNAL, links)
    assert doc_key(psd) is None
    cache = RenderCache()
    _render(psd, cache)
    assert not cache._store and not cache.misses


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
