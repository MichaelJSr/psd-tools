"""Smart filters drawn on a new picture of a placed layer (Card sync's redraw)."""

from collections.abc import Iterable
from typing import cast

import numpy as np
import pytest

from psd_tools import PSDImage
from psd_tools.api.layers import SmartObjectLayer
from psd_tools.composite import smart_filters as sf
from psd_tools.psd.descriptor import (
    Bool,
    Descriptor,
    Double,
    Enumerated,
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
    for angle, distance, drawn in (
        (0, 6, True),
        (0, 17, True),
        (0, 256, True),
        (0, 257, False),
        (0, 6.5, False),
        (90, 6, False),
    ):
        entry[b"Fltr"] = _blur(b"MtnB", Angl=angle, Dstn=distance)
        assert so.can_draw_filters() == drawn
    entry[b"Fltr"] = _blur(b"MtnB", Angl=0, Dstn=6)
    dot = np.ones((1, 1, 4), np.float32)
    out, box = so.draw_filters(dot, (5, 5, 6, 6))
    assert box == (2, 5, 9, 6)
    assert np.array_equal(np.rint(out[..., 3] * 255), np.full((1, 7), 257 // 7))
    assert np.array_equal(out[..., :3], np.ones((1, 7, 3), np.float32))


_DOTS = [(1, 1), (2, 3), (3, 2), (6, 5), (15, 8), (17, 10), (31, 16), (255, 128)]


@pytest.mark.parametrize("distance, alpha", [*_DOTS, (256, 130)])
def test_motion_blur_rounds_as_photoshop(distance: int, alpha: int) -> None:
    """*alpha* is the least a dot shows with: (a + c) // taps.

    c = taps // 2, less 1 unless taps is a power of 2 (Photoshop 2023 on 48
    distances, 1-256 px).
    """
    layer, entry = _object()
    entry[b"Fltr"] = _blur(b"MtnB", Angl=0, Dstn=distance)
    for a, shown in ((alpha, 1), (alpha - 1, 0)):
        dot = np.full((1, 1, 4), a / 255, np.float32)
        out, box = layer.smart_object.draw_filters(dot, (300, 5, 301, 6))
        assert np.rint(out[..., 3] * 255).max(initial=0) == shown
        assert box[2] - box[0] == shown * (distance + 1)


@pytest.mark.parametrize(
    "distance, dot, stored",
    [
        (6, (227, 78, 49, 33), (204, 51, 51, 5)),
        (17, (149, 102, 60, 53), (170, 85, 85, 3)),
    ],
)
def test_motion_blur_colours_as_photoshop(
    distance: int, dot: tuple, stored: tuple
) -> None:
    """A faint dot as Photoshop 2023 stores it: 8-bit premultiplied sums."""
    layer, entry = _object()
    entry[b"Fltr"] = _blur(b"MtnB", Angl=0, Dstn=distance)
    pixel = np.array(dot, np.float32).reshape(1, 1, 4) / 255
    out, _ = layer.smart_object.draw_filters(pixel, (5, 5, 6, 6))
    assert np.array_equal(np.rint(out * 255), np.full((1, distance + 1, 4), stored))


def _hue(hue: int, *ranges: Descriptor, colorize: bool = False) -> Descriptor:
    fltr = Descriptor(classID=b"HStr")
    fltr[b"Clrz"] = Bool(colorize)
    master = Descriptor(classID=b"Hst2")
    for key, value in ((b"H   ", hue), (b"Strt", 0), (b"Lght", 0)):
        master[key] = Integer(value)
    fltr[b"Adjs"] = _list([master, *ranges])
    return fltr


def _grey(tint: bool = False) -> Descriptor:
    """Black & White as on Godskin Apostle's fire."""
    fltr = Descriptor(classID=b"BanW")
    for key, value in zip(
        (b"Rd  ", b"Yllw", b"Grn ", b"Cyn ", b"Bl  ", b"Mgnt"),
        (-68, 87, 40, 60, 20, 55),
        strict=True,
    ):
        fltr[key] = Integer(value)
    fltr[b"useTint"] = Bool(tint)
    return fltr


def _selective(*hues: tuple, method: bytes = b"Rltv") -> Descriptor:
    """Selective Color: (hue, cyan, magenta, yellow, black) in %."""
    fltr = Descriptor(classID=b"SlcC")
    fltr[b"Mthd"] = Enumerated(b"CrcM", method)
    items = []
    for hue, *inks in hues:
        item = Descriptor(classID=b"ClrC")
        item[b"Clrs"] = Enumerated(b"Clrs", hue)
        for key, value in zip((b"Cyn ", b"Mgnt", b"Ylw ", b"Blck"), inks, strict=True):
            item[key] = UnitFloat(value, Unit.Percent)
        items.append(item)
    fltr[b"ClrC"] = _list(items)
    return fltr


_COLORS = [
    (200, 30, 100),
    (15, 250, 75),
    (255, 0, 130),
    (0, 5, 5),
    (90, 90, 240),
    (230, 180, 40),
    (60, 200, 220),
    (250, 245, 120),
]


# Photoshop 2023's 8-bit output, read off its all-colour lattice under each
# as a smart filter.
@pytest.mark.parametrize(
    ("fltr", "photoshop"),
    [
        (
            _hue(57),
            [(200, 121, 30), (15, 201, 250), (255, 112, 0), (0, 0, 5),
             (232, 90, 240), (99, 230, 40), (72, 60, 220), (132, 250, 120)],
        ),
        (
            _grey(),
            [(0, 0, 0), (121, 121, 121), (0, 0, 0), (3, 3, 3),
             (120, 120, 120), (128, 128, 128), (148, 148, 148), (225, 225, 225)],
        ),
        (
            _selective(
                (b"Ylws", 0, 0, 0, -100), (b"Grns", 0, 0, 0, 40), (b"Rds ", -100, 0, 0, 100)
            ),
            [(222, 18, 61), (5, 249, 26), (255, 0, 69), (0, 5, 5),
             (90, 90, 240), (249, 207, 150), (60, 200, 220), (253, 250, 184)],
        ),
        (
            _selective(
                (b"Rds ", 10, 25, -35, 0),
                (b"Ylws", -15, 45, 70, 0),
                (b"Grns", 5, -5, 33, 20),
                (b"Cyns", -80, 60, -20, -10),
                (b"Bls ", 100, 100, -100, 0),
                (b"Mgnt", 40, -60, 10, 30),
            ),
            [(186, 48, 103), (51, 249, 35), (255, 62, 124), (4, 5, 6),
             (37, 37, 249), (232, 158, 33), (143, 183, 228), (250, 243, 75)],
        ),
    ],
)  # fmt: skip
def test_colour_filters_drawn_as_photoshop(
    fltr: Descriptor, photoshop: list[tuple[int, int, int]]
) -> None:
    """Within a level of Photoshop's on each colour; alpha and box kept."""
    layer, entry = _object()
    entry[b"Fltr"] = fltr
    so = layer.smart_object
    assert so.can_draw_filters()
    rgba = np.full((1, len(_COLORS), 4), 0.5, np.float32)
    rgba[0, :, :3] = np.array(_COLORS) / 255
    out, box = so.draw_filters(rgba, (2, 3, 10, 4))
    assert box == (2, 3, 10, 4)
    assert np.array_equal(out[..., 3], rgba[..., 3])
    assert np.abs(np.rint(out[0, :, :3] * 255) - np.array(photoshop)).max() <= 1


def test_colour_filters_past_the_proven_settings_refused() -> None:
    layer, entry = _object()
    so = layer.smart_object
    reds = Descriptor(classID=b"Hst2")
    reds[b"LclR"] = Integer(1)
    for fltr in (
        _hue(30, colorize=True),
        _hue(30, reds),
        _grey(tint=True),
        _selective((b"Rds ", 10, 0, 0, 0), method=b"Absl"),
        _selective((b"Rds ", 10, 0, 0, 0), (b"Whts", 0, 0, 0, 20)),
    ):
        entry[b"Fltr"] = fltr
        assert not so.can_draw_filters()


def _mosaic(cell: float) -> Descriptor:
    fltr = Descriptor(classID=b"Msc ")
    fltr[b"ClSz"] = UnitFloat(cell, Unit.Pixels)
    return fltr


def test_mosaic_cells_on_the_canvas_grid() -> None:
    """Cells of 4 px on the canvas's grid, premultiplied means.

    Each the mean of its 16 pixels (outside the picture clear). Cells across
    the canvas's edge, or a quad above or left of it (the layer's or a new
    one), refused.
    """
    layer, entry = _object()
    so = layer.smart_object
    entry[b"Fltr"] = _mosaic(4)
    assert so.can_draw_filters()
    rgba = np.random.default_rng(4).random((5, 8, 4), dtype=np.float32)
    rgba[..., 3] = np.maximum(rgba[..., 3], 0.1)
    out, box = so.draw_filters(rgba, (5, 6, 13, 11))
    assert box == (4, 4, 16, 12)
    pre = np.zeros((12, 16, 4))
    pre[6:11, 5:13, :3] = rgba[..., :3] * rgba[..., 3:]
    pre[6:11, 5:13, 3] = rgba[..., 3]
    got = np.dstack([out[..., :3] * out[..., 3:], out[..., 3]])
    for y in range(4, 12):
        for x in range(4, 16):
            cell = pre[y - y % 4 : y - y % 4 + 4, x - x % 4 : x - x % 4 + 4]
            np.testing.assert_allclose(got[y - 4, x - 4], cell.mean((0, 1)), atol=1e-6)
    above = (2.0, -4.0, 34.0, -4.0, 34.0, 28.0, 2.0, 28.0)
    assert not so.can_draw_filters(above)
    assert so.can_draw_filters([v + (4 if i % 2 else 0) for i, v in enumerate(above)])
    with pytest.raises(ValueError):
        so.draw_filters(rgba, (5, 6, 13, 11), above)
    entry[b"Fltr"] = _mosaic(6)  # 32 px: no whole cells
    assert not so.can_draw_filters()
    entry[b"Fltr"] = _mosaic(4)
    placed = so._placed_layer
    assert placed is not None
    placed.transform = (-1.0, 0.0, 32.0, 0.0, 32.0, 32.0, -1.0, 32.0)
    assert not so.can_draw_filters()


def _perspective(
    vertices: list, warped: list, quads: int = 1, mode: bytes = b"warp"
) -> Descriptor:
    """A Perspective Warp run on the canvas quad 10 px lower than the layer's."""
    fltr = Descriptor(classID=b"perspectiveWarpTransform")
    fltr[b"Md  "] = Enumerated(b"perspectiveWarpMode", mode)
    for key, points in ((b"vertices", vertices), (b"warpedVertices", warped)):
        found = []
        for x, y in points:
            point = Descriptor(classID=b"Pnt ")
            point[b"Hrzn"], point[b"Vrtc"] = Double(x), Double(y + 10)
            found.append(point)
        fltr[key] = _list(found)
    quad = Descriptor(classID=b"perspectiveWarpQuad")
    quad[b"indices"] = _list(Integer(i) for i in range(4))
    fltr[b"quads"] = _list([quad] * quads)
    for i, (x, y) in enumerate(((0, 0), (32, 0), (32, 32), (0, 32))):
        fltr[b"PuX%d" % i], fltr[b"PuY%d" % i] = Double(x), Double(y + 10)
    return fltr


def test_perspective_warp_redraws_its_stored_pixels() -> None:
    """A one-quad warp draws its stored FEid source onto its stored pixels.

    A move by (3, 2); on a quad twice the size round its centre (a file of
    twice the pixels) the vertices follow it, a move of (6, 4). A trapezoid
    projects the square's centre where its diagonals cross. Layout mode or
    two quads refused.
    """
    layer, entry = _object()
    so = layer.smart_object
    square = [(4, 4), (28, 4), (28, 28), (4, 28)]
    entry[b"Fltr"] = _perspective(square, [(x + 3, y + 2) for x, y in square])
    assert so.can_draw_filters()
    stored = np.rint(np.asarray(layer.numpy(), np.float64) * 255).astype(np.uint8)
    x0, y0, x1, y1 = layer.bbox
    h, w = stored.shape[:2]
    source = (x0 - 3, y0 - 2, x1, y1)  # holds the layer's box: it pairs
    plane = np.zeros((h + 2, w + 3, 4), np.uint8)
    plane[:h, :w] = stored
    zeros = np.zeros((h + 2, w + 3), np.uint8)
    item = feid_item(source, plane[..., :3], plane[..., 3], zeros)
    store_items(cast(PSDImage, layer._psd), [item])
    found = so.filter_source()
    assert found is not None and found[0] == source and found[2] is not None
    picture = np.dstack([found[1], found[2]]).astype(np.float32) / 255
    out, box = so.draw_filters(picture, source)
    assert box == (x0, y0, x1, y1)
    ink = stored[..., 3] > 0
    assert np.array_equal(np.rint(out[..., 3] * 255), stored[..., 3])
    assert np.array_equal(np.rint(out[ink, :3] * 255), stored[ink, :3])
    double = (-16.0, -16.0, 48.0, -16.0, 48.0, 48.0, -16.0, 48.0)
    assert so.can_draw_filters(double)
    moved, where = so.draw_filters(picture, source, double)
    assert where == (x0 + 3, y0 + 2, x1 + 3, y1 + 2)
    np.testing.assert_allclose(moved, out, atol=1e-6)
    trapezoid = [(10, 4), (22, 4), (30, 28), (2, 28)]
    entry[b"Fltr"] = _perspective(square, trapezoid)
    dot = np.ones((2, 2, 4), np.float32)
    out, box = so.draw_filters(dot, (15, 15, 17, 17))  # the square's centre
    ys, xs = np.mgrid[box[1] : box[3], box[0] : box[2]] + 0.5
    weight = out[..., 3] / out[..., 3].sum()
    # the diagonals cross at (16, 11.2); the corners' mean is (16, 16)
    assert abs((xs * weight).sum() - 16) < 0.1 and abs((ys * weight).sum() - 11.2) < 0.1
    for fltr in (
        _perspective(square, trapezoid, quads=2),
        _perspective(square, trapezoid, mode=b"layout"),
    ):
        entry[b"Fltr"] = fltr
        assert not so.can_draw_filters()


# Polevaulter Zombie's and Magnet-shroom's star (1983 x 1307 px) on their turned quads
_STARS = (
    [894.008, 18.753, 1058.317, 20.92, 1057.308, 128.824, 892.999, 126.947],
    [973.19, 22.12, 1180.08, 24.643, 1178.714, 160.65, 971.822, 158.587],
)


def _turned(angle: float = 0.3) -> list[float]:
    """The square a filter of :py:func:`_perspective` ran on, turned round its centre."""
    c, s = np.cos(angle), np.sin(angle)
    out = []
    for x, y in ((0, 0), (32, 0), (32, 32), (0, 32)):
        out += [16 + c * (x - 16) - s * (y - 16), 26 + s * (x - 16) + c * (y - 16)]
    return out


def test_place_averages_a_big_reduction() -> None:
    """A 1 px checkerboard placed at 1/12 on a turned quad is its mean grey.

    A mip pyramid, as Photoshop places it; uint8 or float alike, the box
    round the quad.
    """
    h, w = 288, 384
    board = np.zeros((h, w, 4), np.uint8)
    board[..., 3] = 255
    board[np.add.outer(np.arange(h), np.arange(w)) % 2 == 0] = 255
    quad = [v / 12 + 40 for v in (0, 0, w, 0, w, h, 0, h)]
    c, s = np.cos(0.4), np.sin(0.4)
    for i in range(0, 8, 2):
        x, y = quad[i] - 56, quad[i + 1] - 52
        quad[i], quad[i + 1] = 56 + c * x - s * y, 52 + s * x + c * y
    out, box = sf.place(board, quad)
    again, where = sf.place(board.astype(np.float32) / 255, quad)
    assert where == box and np.allclose(again, out, atol=1e-6)
    inside = out[..., 3] > 0.999
    assert inside.sum() > 500 and np.abs(out[inside, :3] - 0.5).max() < 1e-6
    for k in (0, 1):  # the quad's bounds, its tips' faint pixels trimmed
        assert (
            abs(box[k] - min(quad[k::2])) < 1 and abs(box[k + 2] - max(quad[k::2])) < 1
        )


def test_placeable_at_the_stars_scales_only() -> None:
    """The proven scales and quads only.

    The stars' quads (levels 2.47, 2.14), not twice or half their size, nor a
    quad far from a parallelogram.
    """
    size = (1983.0, 1307.0)
    for quad in _STARS:
        assert sf.placeable(size, quad)
        cx, cy = sum(quad[0::2]) / 4, sum(quad[1::2]) / 4
        for k in (2.0, 0.5):
            scaled = [
                (cx if i % 2 == 0 else cy) + k * (v - (cx if i % 2 == 0 else cy))
                for i, v in enumerate(quad)
            ]
            assert not sf.placeable(size, scaled)
        bent = list(quad)
        bent[4] += 20
        assert not sf.placeable(size, bent)
    assert not sf.placeable(size, [0.0] * 8)


def test_perspective_warp_on_a_turned_quad_at_a_proven_scale() -> None:
    """On a turned quad the warp is drawn for a file placed as proven.

    12 times its pixels, not 3 times or the object's own. A quad no affine
    image of the one it ran on is refused (Photoshop follows by the affine of
    three corners). Uprights take any size.
    """
    layer, entry = _object()
    so = layer.smart_object
    square = [(4, 4), (28, 4), (28, 28), (4, 28)]
    entry[b"Fltr"] = _perspective(square, square)
    turned = _turned()
    assert so.can_draw_filters(turned, (384, 384))
    assert not so.can_draw_filters(turned, (96, 96)) and not so.can_draw_filters(turned)
    bent = list(turned)
    bent[4] += 0.5
    assert not so.can_draw_filters(bent, (384, 384))
    assert so.can_draw_filters(None, (384, 384)) and so.can_draw_filters()
    picture, box = so.place_picture(np.full((384, 384, 4), 255, np.uint8), turned)
    out, where = so.draw_filters(picture, box, turned, size=(384, 384))
    a, b = (np.dstack([p[..., :3] * p[..., 3:], p[..., 3]]) for p in (out, picture))
    assert where == box and np.allclose(a, b, atol=1e-6)  # vertices unmoved
    with pytest.raises(ValueError):
        so.draw_filters(picture, box, turned)


def test_warps_kept_on_a_trimmed_quad() -> None:
    """A trim's quad (a padded file, its ink where it was): the warps' maps kept.

    A Perspective Warp's follow (10 px up) baked into its vertices, ``PuX0``..
    the new quad, the values' types kept; a Puppet Warp's ``PuX0``.. its
    map's preimage of the new quad. Each draws the canvas as before. A
    Perspective Warp's quad no affine image of the one it ran on, or a
    Puppet Warp's new quad no parallelogram, refused, nothing written.
    """
    layer, entry = _object()
    so = layer.smart_object
    padded = [-2.0, -3.0, 35.0, -3.0, 35.0, 33.0, -2.0, 33.0]
    rgba = np.random.default_rng(2).random((12, 14, 4), dtype=np.float32)
    rgba[..., 3] = np.maximum(rgba[..., 3], 0.1)
    box = (9, 8, 23, 20)
    square = [(4, 4), (28, 4), (28, 28), (4, 28)]
    entry[b"Fltr"] = fltr = _perspective(square, [(6, 3), (27, 6), (29, 27), (3, 26)])
    fltr[b"vertices"][0][b"Hrzn"] = UnitFloat(unit=Unit.Pixels, value=4.0)
    before, where = so.draw_filters(rgba, box)
    assert so.can_keep_filters(padded)
    so.keep_filters(padded)
    assert [fltr[k].value for k in sf._PU] == padded
    first = fltr[b"vertices"][0]
    assert isinstance(first[b"Hrzn"], UnitFloat) and first[b"Hrzn"].unit == Unit.Pixels
    assert (first[b"Hrzn"].value, first[b"Vrtc"].value) == (4.0, 4.0)
    after, box2 = so.draw_filters(rgba, box, padded)
    assert box2 == where and np.allclose(after, before, atol=1e-6)
    fltr[b"PuX1"] = Double(30.0)
    assert not so.can_keep_filters(padded)
    with pytest.raises(ValueError):
        so.keep_filters(padded)
    assert fltr[b"PuY0"].value == -3.0

    mesh = [(0, 10), (32, 10), (32, 42), (0, 42)]
    bent = [(3, 12), (33, 11), (30, 40), (2, 43)]
    entry[b"Fltr"] = fltr = _puppet(mesh, bent, [(0, 1, 2), (0, 2, 3)], {0: 0.1}, 10)
    before, where = so.draw_filters(rgba, box, sampling="bilinear")
    so.keep_filters(padded)
    assert np.allclose(
        [fltr[k].value for k in sf._PU],
        [v + 10 * (i % 2) for i, v in enumerate(padded)],
    )
    after, box2 = so.draw_filters(rgba, box, padded, sampling="bilinear")
    assert box2 == where and np.allclose(after, before, atol=1e-6)
    trapezoid = [0.0, 0.0, 32.0, 0.0, 40.0, 32.0, -8.0, 32.0]
    assert not so.can_keep_filters(trapezoid)
