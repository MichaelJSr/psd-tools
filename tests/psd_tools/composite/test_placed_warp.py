"""A placed layer's custom warp drawn on a new picture of its file (Card sync's redraw)."""

import numpy as np
import pytest

from psd_tools import PSDImage
from psd_tools.api.layers import SmartObjectLayer
from psd_tools.api.smart_object import SmartObject
from psd_tools.psd.descriptor import (
    Descriptor,
    Double,
    Enumerated,
    ObjectArray,
    UnitFloats,
)
from psd_tools.terminology import Unit

from ..utils import full_name


def _warped(xs: list[float], ys: list[float]) -> SmartObject:
    """smartobject-layer.psd (32 x 32, Trnf the canvas) under a warpCustom of *xs*, *ys*."""
    psd = PSDImage.open(full_name("layers/smartobject-layer.psd"))
    layer = psd[0]
    assert isinstance(layer, SmartObjectLayer)
    so = layer.smart_object
    warp = so.warp
    assert isinstance(warp, Descriptor)
    warp[b"warpStyle"] = Enumerated(typeID=b"warpStyle", enum=b"warpCustom")
    mesh = ObjectArray(classID=b"customEnvelopeWarp")
    mesh[b"Hrzn"] = UnitFloats(unit=Unit.Pixels, values=[float(v) for v in xs])
    mesh[b"Vrtc"] = UnitFloats(unit=Unit.Pixels, values=[float(v) for v in ys])
    envelope = Descriptor(classID=b"customEnvelopeWarp")
    envelope[b"meshPoints"] = mesh
    warp[b"customEnvelopeWarp"] = envelope
    return so


def _grid(width: float, height: float) -> tuple[list[float], list[float]]:
    xs = [c * width / 3 for _ in range(4) for c in range(4)]
    ys = [r * height / 3 for r in range(4) for _ in range(4)]
    return xs, ys


def test_a_flat_patch_draws_the_file_as_it_is() -> None:
    """The control points' box goes onto the quad.

    A grid of any scale draws each pixel from itself, nearest.
    """
    rgba = np.random.default_rng(0).integers(1, 256, (32, 32, 4), dtype=np.uint8)
    for scale in (1.0, 0.5):
        so = _warped(*_grid(32 * scale, 32 * scale))
        assert so.can_draw_warp()
        out, box = so.draw_warp(rgba)
        assert box == (0, 0, 32, 32)
        assert np.abs(out * 255 - rgba).max() < 0.5


def test_a_sheared_patch_moves_the_rows() -> None:
    """Each row of points 3 px further right than the one above.

    The quad holds the points' box (32 + 9 px wide), so the bottom row lands
    9 * 32 / 41 px right of the top.
    """
    xs, ys = _grid(32, 32)
    so = _warped([x + 3 * (i // 4) for i, x in enumerate(xs)], ys)
    rgba = np.zeros((32, 32, 4), np.uint8)
    rgba[:, 0] = 255  # a line down the left edge
    out, box = so.draw_warp(rgba)
    lines = [box[0] + int(np.flatnonzero(out[y, :, 3] > 0.5)[0]) for y in (0, 31)]
    assert lines[0] == 0 and abs(lines[1] - 9 * 32 / 41) < 1


def test_other_warps_and_pictures_refused() -> None:
    so = _warped(*_grid(32, 32))
    with pytest.raises(ValueError):
        so.draw_warp(np.zeros((32, 33, 4), np.uint8))  # another size
    warp = so.warp
    assert isinstance(warp, Descriptor)
    warp[b"customEnvelopeWarp"][b"quiltSliceX"] = UnitFloats(
        unit=Unit.Pixels, values=[16.0]
    )
    assert not so.can_draw_warp()  # a split grid
    del warp[b"customEnvelopeWarp"][b"quiltSliceX"]
    warp[b"warpValue"] = Double(10.0)
    assert not so.can_draw_warp()  # a bend
    warp[b"warpValue"] = Double(0.0)
    warp[b"warpStyle"] = Enumerated(typeID=b"warpStyle", enum=b"warpArc")
    assert not so.can_draw_warp()  # a preset
    with pytest.raises(ValueError):
        so.draw_warp(np.zeros((32, 32, 4), np.uint8))
    so = _warped(*_grid(32, 32))
    clear = np.zeros((32, 32, 4), np.uint8)
    assert so.draw_warp(clear)[1] == (0, 0, 0, 0)
    with pytest.raises(ValueError):
        so.draw_warp(clear, (0, 0, 32, 0, 40, 32, 0, 32))  # no parallelogram
