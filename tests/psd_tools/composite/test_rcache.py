"""The cross-pass render cache returns what an uncached pass computes."""

from __future__ import annotations

import numpy as np
import pytest

from psd_tools import PSDImage
from psd_tools.composite import composite
from psd_tools.composite.rcache import RenderCache

from ..utils import full_name

pytest.importorskip("scipy")

_FILES = [
    "layer_effects.psd",
    "effects/group-clips-child-stroke.psd",
    "effects/double-stroke-effects.psd",
    "effects/stroke-effects.psd",
]


def _render(path: str, cache: RenderCache | None, edit: bool = False) -> np.ndarray:
    psd = PSDImage.open(full_name(path))
    if edit:
        psd[len(psd) - 1].visible = not psd[len(psd) - 1].visible
    if cache is None:
        return np.concatenate(composite(psd), axis=2)
    with cache:
        return np.concatenate(composite(psd), axis=2)


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
    value = next(iter(cache._store.values()))
    with pytest.raises(ValueError):
        value[0][...] = 0


def test_no_cache_is_active_outside_the_block() -> None:
    cache = RenderCache()
    with cache:
        pass
    _render(_FILES[0], None)
    assert not cache._store
