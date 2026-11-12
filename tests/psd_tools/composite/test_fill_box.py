"""A fill drawn from its path is not cut at a box that truncates the path's."""

import numpy as np
import pytest

from psd_tools import PSDImage
from psd_tools.composite import paint
from psd_tools.composite.composite import Compositor

from ..utils import full_name


def _render(cut: tuple[int, int]) -> np.ndarray:
    """``vector-mask.psd`` with ``Shape 1``'s box cut short by ``cut`` (right, bottom).

    A shape with no pixels takes its box from the origination box, truncated
    to whole pixels, which can stop short of the path the same way.
    """
    psd = PSDImage.open(full_name("vector-mask.psd"))
    next(x for x in psd if x.name == "Background copy").visible = False
    layer = next(x for x in psd if x.name == "Shape 1")
    assert not layer.has_pixels()
    left, top, right, bottom = layer.bbox
    layer._bbox = (left, top, right - cut[0], bottom - cut[1])
    return np.asarray(psd.composite(ignore_preview=True))


def test_a_fill_without_pixels_is_drawn_where_its_path_is() -> None:
    assert np.array_equal(_render((2, 3)), _render((0, 0)))


def test_only_a_fill_without_pixels_grows_past_its_box() -> None:
    psd = PSDImage.open(full_name("layers-minimal/shape-layer.psd"))
    layer = next(x for x in psd.descendants() if x.name == "Polygon 1")
    viewport = (0, 0, psd.width, psd.height)
    left, top, right, bottom = layer.bbox
    layer._bbox = (left, top, right, bottom - 2)
    assert bottom <= Compositor._fill_box(layer, viewport)[3] <= psd.height
    stored = PSDImage.open(full_name("blend-modes/color-burn.psd"))
    pixel = next(x for x in stored.descendants() if x.name == "Ellipse 1")
    assert pixel.has_pixels() and pixel.has_vector_mask()
    assert Compositor._fill_box(pixel, (0, 0, 64, 64)) == pixel.bbox


@pytest.mark.parametrize("name", ["Polygon 2", "Ellipse 1"])
def test_a_fill_drawn_past_its_box_keeps_its_layout(name: str) -> None:
    psd = PSDImage.open(full_name("adjustment-fillers.psd"))
    layer = next(x for x in psd.descendants() if x.name == name)
    left, top, right, bottom = layer.bbox
    alone, _ = paint.create_fill(layer, layer.bbox)
    wider, _ = paint.create_fill(
        layer, (left - 3, top - 2, right + 4, bottom + 5), layer.bbox
    )
    assert alone is not None and wider is not None
    assert np.array_equal(wider[2 : 2 + bottom - top, 3 : 3 + right - left], alone)
