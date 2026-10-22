"""Where a pattern overlay's tiling starts, how it turns, and how it scales."""

import math

import numpy as np
import pytest

from psd_tools import PSDImage
from psd_tools.api import numpy_io
from psd_tools.api.layers import Layer
from psd_tools.composite import paint
from psd_tools.composite.composite import _draw_pattern_overlay, _style_bbox
from psd_tools.constants import Tag
from psd_tools.psd.descriptor import Bool, Descriptor, Double

from ..utils import full_name

Box = tuple[int, int, int, int]


def _overlay(
    filename: str = "fill_adjustments.psd", name: str = "Rectangle 2"
) -> tuple[PSDImage, Layer, Descriptor]:
    psd = PSDImage.open(full_name(filename))
    layer = next(x for x in psd.descendants() if x.name == name)
    (effect,) = layer.effects.find("patternoverlay")
    return psd, layer, effect.descriptor


def _panel(psd: PSDImage, desc: Descriptor) -> np.ndarray:
    pattern = psd._get_pattern(desc[b"Ptrn"][b"Idnt"].value.rstrip("\x00"))
    assert pattern is not None
    return numpy_io.get_pattern(pattern)[:, :, :3]


def _tiling_from(
    psd: PSDImage, desc: Descriptor, origin: tuple[int, int], box: Box
) -> np.ndarray:
    """The unscaled panel tiled across ``box`` with its first pixel at ``origin``."""
    panel = _panel(psd, desc)
    rows = (np.arange(box[1], box[3]) - origin[1]) % panel.shape[0]
    columns = (np.arange(box[0], box[2]) - origin[0]) % panel.shape[1]
    return panel[rows][:, columns]


def _drawn(layer: Layer, desc: Descriptor) -> np.ndarray:
    box = _style_bbox(layer)
    fill, _ = _draw_pattern_overlay(layer, desc, 3, box, box)
    assert fill is not None
    return fill


@pytest.mark.parametrize(
    ("point", "origin"),
    [
        (None, (-4, 0)),  # as Photoshop saved it, 131 px left of the box
        ((150.4, 20.6), (150, 21)),
        ((-300.0, 75.0), (-300, 75)),
    ],
)
def test_the_tiling_starts_at_the_reference_point(
    point: tuple[float, float] | None, origin: tuple[int, int]
) -> None:
    psd, layer, desc = _overlay()
    if point is not None:
        layer.reference_point = point
    box = _style_bbox(layer)
    assert (box[0], box[1]) != origin
    assert np.array_equal(_drawn(layer, desc), _tiling_from(psd, desc, origin, box))


def test_an_unscaled_tiling_matches_the_sampler() -> None:
    psd, _, desc = _overlay()
    panel = _panel(psd, desc)
    viewport = (3, 5, 103, 75)
    exact = paint._sample_tiling(panel, viewport, (-7, 11), 1.0)
    assert np.array_equal(exact, _tiling_from(psd, desc, (-7, 11), viewport))
    sampled = paint._sample_tiling(panel, viewport, (-7, 11), 1.0 + 1e-9)
    assert np.allclose(exact, sampled, atol=1e-5)


def test_moving_the_layer_moves_its_overlay() -> None:
    psd, layer, desc = _overlay("layer_effects.psd", "Pattern Overlay")
    before = _drawn(layer, desc)
    layer.left += 7
    layer.top += 5
    after = _drawn(layer, desc)
    assert np.array_equal(after, before)


def test_the_phase_moves_the_tiling_from_the_reference_point() -> None:
    psd, layer, desc = _overlay()
    desc[b"phase"][b"Hrzn"] = Double(-37.0)
    desc[b"phase"][b"Vrtc"] = Double(-11.0)
    box = _style_bbox(layer)
    expected = _tiling_from(psd, desc, (-4 - 37, 0 - 11), box)
    assert np.array_equal(_drawn(layer, desc), expected)


def test_unaligned_the_phase_is_measured_from_the_document_origin() -> None:
    psd, layer, desc = _overlay()
    desc[b"Algn"] = Bool(False)
    desc[b"phase"][b"Hrzn"] = Double(-23.0)
    desc[b"phase"][b"Vrtc"] = Double(-9.0)
    layer.reference_point = (60.0, 70.0)
    box = _style_bbox(layer)
    assert np.array_equal(_drawn(layer, desc), _tiling_from(psd, desc, (-23, -9), box))


def test_without_a_reference_point_the_tiling_starts_at_the_box() -> None:
    psd, layer, desc = _overlay()
    del layer.tagged_blocks[Tag.REFERENCE_POINT]
    box = _style_bbox(layer)
    expected = _tiling_from(psd, desc, (box[0], box[1]), box)
    assert np.array_equal(_drawn(layer, desc), expected)


def test_pattern_origin_rounds_to_the_nearest_pixel() -> None:
    _, _, desc = _overlay()
    desc[b"phase"][b"Hrzn"] = Double(-15.886954442438153)
    desc[b"phase"][b"Vrtc"] = Double(-13.962771225790817)
    point = (32.88695444243815, 22.962771225790817)
    assert paint.pattern_origin(desc, (0, 0, 1, 1), point) == (17, 9)


@pytest.mark.parametrize(
    ("filename", "name", "bound"),
    [
        # The adjustment layers above it shift every pixel of the preview.
        ("fill_adjustments.psd", "Rectangle 2", 8.0),
        ("transparency/fill-opacity.psd", "A", 0.5),
        # 200% scale, 1 px lines.
        ("patterns.psd", "Rectangle 1 copy 7", 1.5),
        ("patterns.psd", "Rectangle 1 copy 8", 1.5),
        ("patterns.psd", "Rectangle 1 copy 9", 1.5),
    ],
)
def test_a_pattern_overlay_matches_photoshops_preview(
    filename: str, name: str, bound: float
) -> None:
    """Mean absolute difference over the layer's box, in 8-bit levels."""
    psd, layer, _ = _overlay(filename, name)
    stored = psd.topil()
    assert stored is not None
    preview = np.asarray(stored.convert("RGB"), dtype=np.float64)
    rendered = np.asarray(psd.composite(force=True).convert("RGB"), dtype=np.float64)
    left, top, right, bottom = (max(0, x) for x in layer.bbox)
    region = (slice(top, bottom), slice(left, right))
    assert np.abs(rendered[region] - preview[region]).mean() <= bound


@pytest.mark.parametrize(
    ("angle", "turns"), [(90.0, 1), (180.0, 2), (-90.0, 3), (450.0, 1), (45.0, 0)]
)
def test_a_right_angle_turns_the_tiles_about_the_origin(
    angle: float, turns: int
) -> None:
    psd, _, desc = _overlay()
    desc[b"Angl"] = Double(angle)
    viewport = (3, 5, 203, 155)
    drawn, _ = paint.draw_pattern_fill(viewport, psd, desc, origin=(3, 5))
    panel = np.rot90(_panel(psd, desc), turns)
    reps = (
        math.ceil(150 / panel.shape[0]),
        math.ceil(200 / panel.shape[1]),
        1,
    )
    assert drawn is not None
    assert np.array_equal(drawn, np.tile(panel, reps)[:150, :200])


def test_a_non_finite_pattern_angle_is_rejected() -> None:
    psd, _, desc = _overlay()
    desc[b"Angl"] = Double(float("inf"))
    with pytest.raises(ValueError, match="not finite"):
        paint.draw_pattern_fill(psd.viewbox, psd, desc)


def test_an_enlarged_pattern_keeps_its_samples_on_the_origin() -> None:
    """Fitted on ``patterns.psd``: a 1 px line at 200% is one dark pixel and two halves."""
    panel = np.array([[0.0], [1.0], [1.0], [1.0]], dtype=np.float32)[:, :, None]
    column = paint._sample_tiling(panel, (0, 0, 1, 10), (0, 0), 2.0)[:, 0, 0]
    assert np.allclose(column, [0.0, 0.5, 1, 1, 1, 1, 1, 0.5, 0.0, 0.5])


def test_a_tiling_scaled_by_a_fraction_keeps_its_period() -> None:
    panel = np.arange(4, dtype=np.float32)[None, :, None]
    row = paint._sample_tiling(panel, (0, 0, 40, 1), (0, 0), 2.5)[0, :, 0]
    assert np.allclose(row[:10], row[10:20]) and np.allclose(row[:10], row[30:])
    assert row[0] == 0.0 and row[5] == 2.0


@pytest.mark.parametrize(
    ("scale", "box"),
    # Fitted against Photoshop's renders at 89%, 50% (turned) and 7%.
    [(1.0, 1), (0.89, 1), (0.5, 2), (0.3, 2), (0.25, 4), (0.07, 8), (0.01, 64)],
)
def test_a_reduction_averages_the_deepest_halving_it_reaches(
    scale: float, box: int
) -> None:
    assert paint._reduction_box(scale) == box


def test_the_reduction_box_averages_forward_and_wraps() -> None:
    ramp = np.arange(4, dtype=np.float32)
    panel = (ramp[:, None] * 10 + ramp[None, :])[:, :, None]
    averaged = paint._box_average(panel, 2)[:, :, 0]
    forward = np.array([0.5, 1.5, 2.5, 1.5], dtype=np.float32)
    assert np.allclose(averaged, forward[:, None] * 10 + forward[None, :])


def test_a_half_scale_is_the_mean_of_each_pair() -> None:
    psd, _, desc = _overlay()
    desc[b"Scl "] = Double(50.0)
    panel = _panel(psd, desc)
    drawn, _ = paint.draw_pattern_fill((0, 0, 20, 20), psd, desc, origin=(0, 0))
    pairs = panel[:40, :40].reshape(20, 2, 20, 2, 3).mean(axis=(1, 3))
    assert drawn is not None
    assert np.allclose(drawn, pairs, atol=1e-6)
