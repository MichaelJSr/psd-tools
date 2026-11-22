"""Smart filters drawn on a new picture of a placed layer (Card sync's redraw)."""

from collections.abc import Iterable
from typing import cast

import numpy as np
import pytest

from psd_tools import PSDImage
from psd_tools.api.layers import SmartObjectLayer
from psd_tools.psd.descriptor import (
    Bool,
    Descriptor,
    Double,
    Integer,
    List,
    RawData,
    UnitFloat,
)
from psd_tools.terminology import Unit

from ..smart_filters import enable_filters, feid_item, store_items
from ..utils import full_name


def _object() -> tuple[SmartObjectLayer, Descriptor]:
    """smartobject-layer.psd (32 x 32, Trnf the canvas) with one filter on."""
    psd = PSDImage.open(full_name("layers/smartobject-layer.psd"))
    layer = psd[0]
    assert isinstance(layer, SmartObjectLayer)
    fx = enable_filters(layer, mask=False)
    return layer, fx[b"filterFXList"][0]


def _puppet(
    orig: list, deformed: list, tris: list, pins: dict, dy: float = 0.0
) -> Descriptor:
    """A Puppet Warp run on the canvas quad *dy* px lower than the layer's."""
    fltr = Descriptor(classID=b"rigidTransform")
    for i, (x, y) in enumerate(((0, 0), (32, 0), (32, 32), (0, 32))):
        fltr[b"PuX%d" % i] = Double(x)
        fltr[b"PuY%d" % i] = Double(y + dy)
    shape = Descriptor()
    shape[b"originalVertexArray"] = RawData(np.asarray(orig, "<f4").tobytes())
    shape[b"deformedVertexArray"] = RawData(np.asarray(deformed, "<f4").tobytes())
    shape[b"indexArray"] = RawData(np.asarray(tris, "<i4").tobytes())
    shape[b"pinVertexIndices"] = _list(Integer(v) for v in pins)
    shape[b"PnDp"] = _list(Double(d) for d in pins.values())
    fltr[b"puppetShapeList"] = _list([shape])
    return fltr


def _list(values: Iterable[object]) -> List:
    out = List()
    for value in values:
        out.append(value)
    return out


def test_invert_drawn_and_other_stacks_refused() -> None:
    layer, entry = _object()
    so = layer.smart_object
    rgba = np.random.default_rng(0).random((4, 3, 4), dtype=np.float32)
    assert not so.can_draw_filters()  # a kind it does not draw
    with pytest.raises(ValueError):
        so.draw_filters(rgba, (1, 2, 4, 6))
    entry[b"filterID"] = Integer(int.from_bytes(b"Invr", "big"))
    assert so.can_draw_filters()
    out, box = so.draw_filters(rgba, (1, 2, 4, 6))
    assert box == (1, 2, 4, 6)
    assert np.array_equal(out[..., :3], 1 - rgba[..., :3])
    assert np.array_equal(out[..., 3], rgba[..., 3])
    blend = Descriptor()
    blend[b"Opct"] = Double(50.0)
    entry[b"blendOptions"] = blend
    assert not so.can_draw_filters()
    del entry[b"blendOptions"]
    fx = so.filters
    assert fx is not None
    second = Descriptor()
    fx[b"filterFXList"].append(second)
    assert not so.can_draw_filters()


def test_puppet_warp_moves_the_picture_on_the_moved_layer() -> None:
    """The mesh, drawn 10 px lower than the layer is now, moves it by (3, 2)."""
    layer, entry = _object()
    square = [(0, 10), (32, 10), (32, 42), (0, 42)]
    moved = [(x + 3, y + 2) for x, y in square]
    entry[b"Fltr"] = _puppet(square, moved, [(0, 1, 2), (0, 2, 3)], {0: 0.1}, 10)
    so = layer.smart_object
    assert so.can_draw_filters()
    rgba = np.random.default_rng(1).random((8, 10, 4), dtype=np.float32)
    rgba[..., 3] = np.maximum(rgba[..., 3], 0.1)
    for sampling in ("bilinear", "nearest"):
        out, box = so.draw_filters(rgba, (4, 6, 14, 14), sampling=sampling)
        assert box == (7, 8, 17, 16)
        np.testing.assert_allclose(out, rgba, atol=1e-6)


@pytest.mark.parametrize(("left", "right", "shown"), [(0.1, 0.9, 1), (0.9, 0.1, 0)])
def test_puppet_warp_fold_shows_the_deeper_pins_side(
    left: float, right: float, shown: int
) -> None:
    """A fold: the lower-left triangle flips over the other; Pin Depth decides."""
    layer, entry = _object()
    square = [(0, 0), (16, 0), (16, 16), (0, 16)]
    folded = [(0, 0), (16, 0), (16, 16), (16, -16)]
    entry[b"Fltr"] = _puppet(
        square, folded, [(0, 1, 2), (0, 2, 3)], {1: left, 3: right}
    )
    rgba = np.zeros((16, 16, 4), np.float32)
    rgba[..., 3] = 1
    y, x = np.mgrid[:16, :16]
    rgba[..., 0] = x > y  # red above the diagonal (vertex 1's side)
    rgba[..., 2] = x < y  # blue below it (vertex 3's)
    out, box = layer.smart_object.draw_filters(rgba, (0, 0, 16, 16), sampling="nearest")
    inside = out[12 - box[1], 14 - box[0]]  # (14, 12): in both triangles
    assert inside[0] == 1 - shown and inside[2] == shown


def _blur(cls: bytes, **values: float) -> Descriptor:
    fltr = Descriptor(classID=cls)
    for key, value in values.items():
        fltr[key.ljust(4).encode()] = (
            Integer(int(value)) if key == "Angl" else UnitFloat(value, Unit.Pixels)
        )
    return fltr


def test_gaussian_blur_drawn_under_a_painted_mask() -> None:
    """A Gaussian Blur under a painted mask is drawn: the mask does not refuse.

    Premultiplied, sigma = radius; edge pixels repeat at the canvas's edge
    (left) and fade inside it.
    """
    ndimage = pytest.importorskip("scipy.ndimage")
    layer, entry = _object()
    entry[b"Fltr"] = _blur(b"GsnB", Rds=2.0)
    so = layer.smart_object
    zeros = np.zeros((32, 32), np.uint8)
    fx = so.filters
    assert fx is not None
    fx[b"filterMaskEnable"] = Bool(True)
    store_items(
        cast(PSDImage, layer._psd),
        [feid_item((0, 0, 32, 32), np.zeros((32, 32, 3), np.uint8), zeros, zeros)],
    )
    assert so.can_draw_filters()
    rng = np.random.default_rng(2)
    rgba = rng.random((20, 12, 4), dtype=np.float32)
    out, box = so.draw_filters(rgba, (0, 6, 12, 26))
    pre = np.dstack([rgba[..., :3] * rgba[..., 3:], rgba[..., 3]])
    pre = np.pad(pre, ((7, 7), (0, 7), (0, 0)))  # 3 sigma + 1 inside the canvas
    pre = np.pad(pre, ((0, 0), (7, 0), (0, 0)), mode="edge")  # its left edge
    planes = [
        ndimage.gaussian_filter(pre[..., k], 2.0, mode="nearest", truncate=3.0)
        for k in range(4)
    ]
    want = np.dstack(planes)[:, 7:]  # canvas box (0, -1, 19, 33)
    shown = np.rint(want[..., 3] * 255) > 0
    rows, cols = np.flatnonzero(shown.any(1)), np.flatnonzero(shown.any(0))
    assert box == (cols[0], rows[0] - 1, cols[-1] + 1, rows[-1]) and cols[0] == 0
    want = want[rows[0] : rows[-1] + 1, : cols[-1] + 1]
    got = np.dstack([out[..., :3] * out[..., 3:], out[..., 3]])
    np.testing.assert_allclose(got, want, atol=1e-6)
    assert out[10, 0, 3] > 0.3 and out[0, 4, 3] < 0.01


def test_motion_blur_drawn_at_a_proven_distance() -> None:
    """A 6 px Motion Blur at angle 0: 7 taps, alpha (S + 2) // 7; others refused."""
    layer, entry = _object()
    so = layer.smart_object
    for angle, distance, drawn in ((0, 6, True), (0, 17, False), (90, 6, False)):
        entry[b"Fltr"] = _blur(b"MtnB", Angl=angle, Dstn=distance)
        assert so.can_draw_filters() == drawn
    entry[b"Fltr"] = _blur(b"MtnB", Angl=0, Dstn=6)
    dot = np.ones((1, 1, 4), np.float32)
    out, box = so.draw_filters(dot, (5, 5, 6, 6))
    assert box == (2, 5, 9, 6)
    assert np.array_equal(np.rint(out[..., 3] * 255), np.full((1, 7), 257 // 7))
    assert np.array_equal(out[..., :3], np.ones((1, 7, 3), np.float32))
