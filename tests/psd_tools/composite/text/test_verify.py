import numpy as np
import pytest

pytest.importorskip("freetype")
pytest.importorskip("uharfbuzz")

from psd_tools import PSDImage
from psd_tools.api.layers import TypeLayer
from psd_tools.composite.text.constants import ANY_ALPHA, BAD_DIFF, IOU_COVER
from psd_tools.composite.text.verify import fidelity, stored_raster

from ...utils import full_name


def square(size: int = 6, side: int = 4, fill: tuple[float, float, float] = (0, 0, 0)):
    """A ``size`` x ``size`` straight RGBA raster with an opaque ``side`` square."""
    rgba = np.zeros((size, size, 4), np.float32)
    rgba[1 : 1 + side, 1 : 1 + side, :3] = fill
    rgba[1 : 1 + side, 1 : 1 + side, 3] = 1.0
    return rgba


def test_constants_match_the_harness() -> None:
    assert IOU_COVER == 0.5
    assert BAD_DIFF == 24 / 255
    assert ANY_ALPHA == 0.1


def test_identical_rasters() -> None:
    a = square()
    f = fidelity(a, (10, 20, 16, 26), a.copy(), (10, 20, 16, 26))
    assert f.status == "ok"
    assert (f.iou, f.bad_pct, f.mae) == (1.0, 0.0, 0.0)
    assert (f.cdx, f.cdy) == (0.0, 0.0)
    assert (f.dx0, f.dy0, f.dx1, f.dy1) == (0, 0, 0, 0)


def test_shift_by_one_pixel_through_the_bbox() -> None:
    a = square()
    f = fidelity(a, (0, 0, 6, 6), a, (1, 0, 7, 6))
    # 4x4 squares overlapping on 3x4: intersection 12, union 20.
    assert f.iou == pytest.approx(12 / 20)
    assert f.cdx == pytest.approx(1.0)
    assert f.cdy == pytest.approx(0.0)
    assert (f.dx0, f.dy0, f.dx1, f.dy1) == (1, 0, 1, 0)
    # The uncovered columns on both sides are bad: 8 of 20 pixels with ink.
    assert f.bad_pct == pytest.approx(100 * 8 / 20)
    assert f.mae == pytest.approx(8 / 20)


def test_colour_difference_alone() -> None:
    a = square(fill=(0, 0, 0))
    b = square(fill=(1, 0, 0))
    f = fidelity(a, (0, 0, 6, 6), b, (0, 0, 6, 6))
    assert f.iou == 1.0
    assert f.mae == 0.0
    assert f.bad_pct == 100.0


def test_small_alpha_difference_is_not_bad() -> None:
    a = square()
    b = a.copy()
    b[1:5, 1:5, 3] = 1.0 - BAD_DIFF + 1e-4
    assert fidelity(a, (0, 0, 6, 6), b, (0, 0, 6, 6)).bad_pct == 0.0
    b[1:5, 1:5, 3] = 1.0 - BAD_DIFF - 1e-4
    assert fidelity(a, (0, 0, 6, 6), b, (0, 0, 6, 6)).bad_pct == 100.0


def test_premultiplied_comparison() -> None:
    # Half-transparent white against opaque grey: premultiplied they agree.
    a = square(fill=(1, 1, 1))
    a[..., 3] *= 0.5
    b = square(fill=(0.5, 0.5, 0.5))
    f = fidelity(a, (0, 0, 6, 6), b, (0, 0, 6, 6))
    assert f.mae == pytest.approx(0.5)
    assert f.iou == 0.0  # the half-transparent square is below IOU_COVER
    assert f.bad_pct == 100.0


def test_empty_render() -> None:
    a = square()
    f = fidelity(a, (0, 0, 6, 6), None, None)
    assert f.status == "empty"
    assert f.iou == 0.0
    assert f.bad_pct == 100.0
    assert f.mae == pytest.approx(1.0)
    assert f.cdx is None and f.dx0 is None
    g = fidelity(a, (0, 0, 6, 6), np.zeros((0, 0, 4), np.float32), (3, 3, 3, 3))
    assert g.status == "empty" and g.iou == 0.0
    assert fidelity(a, (0, 0, 6, 6), None, None, status="missing").status == "missing"


def test_nothing_on_either_side() -> None:
    f = fidelity(None, None, None, None)
    assert f.status == "empty"
    assert (f.iou, f.bad_pct, f.mae) == (1.0, 0.0, 0.0)


def test_any_alpha_mask_excludes_faint_pixels() -> None:
    a = square()
    b = a.copy()
    b[0, 0, 3] = ANY_ALPHA / 2  # faint pixel outside the square
    f = fidelity(a, (0, 0, 6, 6), b, (0, 0, 6, 6))
    assert f.bad_pct == 0.0 and f.mae == 0.0
    b[0, 0, 3] = ANY_ALPHA * 2
    f = fidelity(a, (0, 0, 6, 6), b, (0, 0, 6, 6))
    assert f.bad_pct == pytest.approx(100 / 17)
    assert f.dx0 == -1 and f.dy0 == -1


@pytest.mark.parametrize("mode", ["CMYK", "LAB", "L"])
def test_stored_raster_in_other_colour_modes(mode: str) -> None:
    psd = PSDImage.open(full_name("text.psd"))
    layer = next(x for x in psd.descendants() if isinstance(x, TypeLayer))
    rgb, bbox = stored_raster(layer)
    copied = layer.copy_to(PSDImage.new(mode, psd.size))
    rgba, copied_bbox = stored_raster(copied)
    assert rgb is not None and rgba is not None
    assert copied_bbox == bbox and rgba.shape == rgb.shape
    # The alpha is the stored channel, not the layer rectangle.
    np.testing.assert_array_equal(rgba[..., 3], rgb[..., 3])
    ink = rgb[..., 3] > 0.5
    assert np.abs(rgba[..., :3][ink] - rgb[..., :3][ink]).max() < 0.1
    assert fidelity(rgb, bbox, rgba, copied_bbox).bad_pct < 1.0


def test_fidelity_allocation_budget() -> None:
    a = square()
    far = (100_000, 0, 100_006, 6)
    with pytest.raises(ValueError, match="per axis"):
        fidelity(a, (0, 0, 6, 6), a, far, max_alloc_bytes="unlimited")
    with pytest.raises(ValueError):
        fidelity(a, (0, 0, 6, 6), a, (10_000, 0, 10_006, 6), max_alloc_bytes=1_000)
    f = fidelity(a, (0, 0, 6, 6), a, (10_000, 0, 10_006, 6), max_alloc_bytes=10**7)
    assert f.iou == 0.0


def test_stored_raster_of_a_type_layer() -> None:
    psd = PSDImage.open(full_name("text.psd"))
    layer = next(x for x in psd.descendants() if isinstance(x, TypeLayer))
    rgba, bbox = stored_raster(layer)
    assert rgba is not None and bbox is not None
    assert bbox == tuple(layer.bbox)
    assert rgba.shape == (bbox[3] - bbox[1], bbox[2] - bbox[0], 4)
    assert rgba.dtype == np.float32
    assert rgba.min() >= 0.0 and rgba.max() <= 1.0
    assert rgba[..., 3].max() == 1.0
