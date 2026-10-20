"""Rasterization: a :class:`Layout` under a transform to straight RGBA."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Literal, TypeVar

import numpy as np

from psd_tools.composite.text.constants import GLYPH_CACHE_SIZE, MAX_CANVAS_PIXELS
from psd_tools.composite.text.layout import Layout
from psd_tools.composite.text.model import Transform

#: ``(x0, y0, x1, y1)`` in document pixels.
Bbox = tuple[int, int, int, int]
T = TypeVar("T")


@dataclass(frozen=True)
class Options:
    """Rendering options; the defaults are the measured recipe."""

    #: Applies to the default :class:`FontMap`; an explicit map keeps its own policy.
    missing_font: Literal["raise", "skip", "substitute"] = "raise"
    #: Non-diagonal transforms: FreeType matrix, or axis-aligned then resampled.
    general_transform: Literal["matrix", "resample"] = "matrix"
    monochrome_emoji: bool = True
    max_canvas_pixels: int = MAX_CANVAS_PIXELS


class GlyphCache:
    """LRU of rendered glyph bitmaps, keyed by everything that shapes one."""

    def __init__(self, maxsize: int = GLYPH_CACHE_SIZE):
        self.maxsize = maxsize
        self._items: OrderedDict[Hashable, object] = OrderedDict()

    def __len__(self) -> int:
        return len(self._items)

    def get(self, key: Hashable, make: Callable[[], T]) -> T:
        """The cached value for ``key``, rendering it with ``make`` once."""
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key]  # type: ignore[return-value]
        value = make()
        self._items[key] = value
        while len(self._items) > self.maxsize:
            self._items.popitem(last=False)
        return value


def rasterize(
    layout: Layout,
    transform: Transform,
    antialias: int,
    options: Options,
    cache: GlyphCache | None = None,
) -> tuple[np.ndarray, Bbox]:
    """Render ``layout`` to ``float32`` HxWx4 straight RGBA and its bbox."""
    raise NotImplementedError
