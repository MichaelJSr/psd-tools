"""Where a pattern overlay's tiling starts, how it turns, and how it scales."""

import math

import numpy as np
import pytest
from PIL import Image

from psd_tools import PSDImage
from psd_tools.api import numpy_io
from psd_tools.api.layers import Layer
from psd_tools.composite import paint
from psd_tools.composite.composite import _draw_pattern_overlay, _style_bbox
from psd_tools.constants import Tag
from psd_tools.psd.descriptor import Bool, Descriptor, Double, String

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
    _psd, layer, desc = _overlay("layer_effects.psd", "Pattern Overlay")
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


def test_without_a_reference_point_the_tiling_starts_at_the_document_origin() -> None:
    psd, layer, desc = _overlay()
    del layer.tagged_blocks[Tag.REFERENCE_POINT]
    box = _style_bbox(layer)
    assert np.array_equal(_drawn(layer, desc), _tiling_from(psd, desc, (0, 0), box))


@pytest.mark.parametrize(
    ("scale", "angle", "origin"),
    [
        (100.0, 0.0, (3, 9)),
        (50.0, 0.0, (3, 9)),
        (12.5, 0.0, (3, 9)),
        (75.0, 90.0, (3, 9)),
        (75.0, 0.0, (3.4, 8.6)),
        (7.0, 0.0, (3.4, 8.6)),
        (200.0, 0.0, (3.4, 8.6)),
    ],
)
def test_pattern_origin_keeps_its_fraction_only_where_photoshop_does(
    scale: float, angle: float, origin: tuple[float, float]
) -> None:
    _, _, desc = _overlay()
    desc[b"Scl "] = Double(scale)
    desc[b"Angl"] = Double(angle)
    desc[b"phase"][b"Hrzn"] = Double(-0.25)
    desc[b"phase"][b"Vrtc"] = Double(0.75)
    drawn = paint.pattern_origin(desc, (0, 0, 1, 1), (3.65, 7.85))
    assert drawn == pytest.approx(origin)


# effects/pattern-origin-laws.psd: five 64 px grey squares, each with only a
# Normal pattern overlay of a 41 px noise tile, written by psd-tools; the
# descriptor is one Photoshop wrote. The cells: no reference point at 7%
# (phase -15.887, -13.963) and at 100%, a reference point at (+3.4, +5.6) at
# 75%, unlinked with a phase of (0.5, 0.5) at 7%, and (+3.4, +5.6) at 50%.
# The PNG is Photoshop 2023's PNG copy of it.
_ORIGINS = "effects/pattern-origin-laws.psd"


def _origin_errors() -> dict[str, int]:
    """Per cell, the most any channel is off Photoshop's, in levels."""
    psd = PSDImage.open(full_name(_ORIGINS))
    path = full_name(_ORIGINS.replace(".psd", ".png"))
    theirs = np.asarray(Image.open(path).convert("RGB"), dtype=int)
    ours = np.asarray(psd.composite(force=True).convert("RGB"), dtype=int)
    error = np.abs(ours - theirs).max(axis=2)
    return {
        cell.name: int(error[cell.top : cell.bottom, cell.left : cell.right].max())
        for cell in psd
    }


def test_pattern_origins_match_photoshop_2023() -> None:
    errors = _origin_errors()
    assert len(errors) == 5
    assert max(errors.values()) <= 1, errors


@pytest.mark.parametrize(
    ("sub_pixel", "missed"),
    [(False, ("no fxrp 7", "fraction 75", "half 7")), (True, ("fraction 50",))],
    ids=["always rounded", "never rounded"],
)
def test_a_rounded_or_unrounded_origin_misses(
    monkeypatch: pytest.MonkeyPatch, sub_pixel: bool, missed: tuple[str, ...]
) -> None:
    monkeypatch.setattr(paint, "_sub_pixel_origin", lambda desc: sub_pixel)
    errors = _origin_errors()
    for cell in errors:
        assert (errors[cell] > 4) == (cell in missed), (cell, errors)


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


@pytest.mark.parametrize(
    "angle", [Double(float("inf")), Double(float("nan")), String("left")]
)
def test_an_unreadable_pattern_angle_leaves_the_tiles_unturned(angle: object) -> None:
    """A fill layer's draw has no guard, so the angle cannot raise."""
    psd, _, desc = _overlay()
    expected, _ = paint.draw_pattern_fill(psd.viewbox, psd, desc)
    desc[b"Angl"] = angle
    drawn, _ = paint.draw_pattern_fill(psd.viewbox, psd, desc)
    assert drawn is not None and expected is not None
    assert np.array_equal(drawn, expected)


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
    # Fitted against Photoshop's renders at 89%, 50% (turned) and 7%; below
    # 1/8 Photoshop 2023 still samples the third halving (3.125-6.25%).
    [(1.0, 1), (0.89, 1), (0.5, 2), (0.3, 2), (0.25, 4), (0.07, 8), (0.01, 8)],
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


def _levels(psd: PSDImage, desc: Descriptor) -> np.ndarray:
    return np.rint(_panel(psd, desc) * 255).astype(np.int64)


def test_a_half_scale_rounds_each_quad_of_levels_half_up() -> None:
    """An 8-bit halving is ``(a + b + c + d + 2) >> 2``, in whole levels."""
    psd, _, desc = _overlay()
    desc[b"Scl "] = Double(50.0)
    quads = _levels(psd, desc)[:40, :40].reshape(20, 2, 20, 2, 3).sum(axis=(1, 3))
    drawn, _ = paint.draw_pattern_fill((0, 0, 20, 20), psd, desc, origin=(0, 0))
    assert drawn is not None
    assert np.array_equal(drawn * 255, np.rint(drawn * 255))
    assert np.array_equal(np.rint(drawn * 255), (quads + 2) >> 2)


def test_a_reduction_blends_level_samples_a_box_apart() -> None:
    """Photoshop 2023: at 30% (box 2) a pixel blends level samples 2k, 2k + 2."""
    psd, _, desc = _overlay()
    desc[b"Scl "] = Double(30.0)
    level = _levels(psd, desc)
    level = level + np.roll(level, -1, axis=0)
    level = (level + np.roll(level, -1, axis=1) + 2) >> 2
    position = np.arange(20) / 0.3 / 2
    t = np.floor((position - np.floor(position)) * 256).astype(np.int64)
    k = np.floor(position).astype(int) * 2
    rows = level[k] * (256 - t)[:, None, None] + level[k + 2] * t[:, None, None]
    blend = rows[:, k] * (256 - t)[None, :, None] + rows[:, k + 2] * t[None, :, None]
    drawn, _ = paint.draw_pattern_fill((0, 0, 20, 20), psd, desc, origin=(0, 0))
    assert drawn is not None
    assert np.array_equal(np.rint(drawn * 255), (blend + 32768) >> 16)


def test_an_8_bit_halving_rounds_as_photoshop() -> None:
    """Photoshop 2023's level for this 8x8 block at 12.5%: 132 (mean 131.47)."""
    block = np.array(
        [
            [150, 208, 145, 108, 252, 115, 54, 199],
            [255, 71, 120, 190, 186, 107, 77, 166],
            [188, 48, 177, 122, 111, 56, 40, 209],
            [120, 132, 34, 12, 11, 156, 69, 30],
            [73, 71, 107, 98, 60, 117, 246, 124],
            [130, 2, 191, 52, 197, 163, 177, 124],
            [129, 112, 147, 176, 255, 87, 160, 173],
            [204, 94, 237, 215, 216, 195, 35, 129],
        ],
        dtype=np.float32,
    )[:, :, None]
    assert paint._halve_levels(block, 8)[0, 0, 0] == 132


def test_an_8_bit_blend_weighs_in_256ths_as_photoshop() -> None:
    """Photoshop 2023's pixels for a 4x4 noise corner at 75% (float lerp: 3 off)."""
    panel = np.array(
        [
            [[53, 150, 108], [137, 208, 221], [93, 145, 1], [92, 108, 158]],
            [[152, 255, 151], [113, 71, 121], [146, 120, 227], [173, 190, 89]],
            [[61, 188, 115], [131, 48, 66], [132, 177, 244], [181, 122, 132]],
            [[63, 120, 20], [247, 132, 206], [73, 34, 225], [17, 12, 102]],
        ],
        dtype=np.float32,
    )
    photoshop = [
        [[53, 150, 108], [122, 187, 148], [92, 120, 105]],
        [[122, 233, 139], [126, 88, 146], [164, 158, 147]],
        [[62, 143, 52], [170, 97, 183], [79, 60, 152]],
    ]
    drawn = paint._sample_tiling(panel, (0, 0, 3, 3), (0, 0), 0.75, 1, levels=True)
    assert np.array_equal(drawn * 255, photoshop)
