"""Photoshop's layer-effect noise, one fixed field over the document.

``effects/effect-noise.psd``: over white, eight 120 x 120 cells, each a 60 x
60 black square at fill 0 with one noisy black effect, named for it: a drop
shadow of size 0 on the square at 50% (it paints the field itself), a soft
shadow at 60% opacity, a soft and a precise outer glow at range 50, inner
glows from the edge at range 50 and 100, an inner shadow, and a shadow
knocked out by a feathered square. ``effects/effect-noise-field.psd``: a
4200 x 3 layer at 50% and fill 0 whose drop shadow of size 0 and noise 50
paints the field across three runs of its table. ``effects/effect-noise-cone.psd``:
two such cells, a drop shadow of size 16 and an outer glow of size 16, spread
50 and range 100, both under the cone contour at noise 50, so each has a
full matte that the contour takes to nothing. Photoshop 2023's composites are
the reference.
"""

import numpy as np
import pytest

from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import composite, effect_noise, effects

from ..utils import full_name

pytest.importorskip("scipy")

_FIXTURE = "effects/effect-noise.psd"
_FIELD = "effects/effect-noise-field.psd"
_CONE = "effects/effect-noise-cone.psd"
_CELL = 120

# Levels a cell may stray from Photoshop's: Photoshop rounds each stage of
# an effect to 8 bits and the fork does not.
_BOUND = 2.0

# Levels the noise moves some pixel of every cell by, at the least.
_MOVED = 24.0


def _cell_errors(psd: PSDImage) -> dict[str, float]:
    """Per cell, the largest difference from Photoshop's composite, in 8-bit levels."""
    color, _, _ = composite(psd)
    ours = np.rint(color[..., :3] * 255.0)
    diff = np.abs(ours - np.rint(psd.numpy()[..., :3] * 255.0)).max(axis=2)
    errors = {}
    for layer in psd:
        if not layer.has_effects():
            continue
        left, top, right, bottom = layer.bbox
        x0 = (left + right) // 2 // _CELL * _CELL
        y0 = (top + bottom) // 2 // _CELL * _CELL
        errors[layer.name] = float(diff[y0 : y0 + _CELL, x0 : x0 + _CELL].max())
    return errors


def _levels(psd: PSDImage) -> np.ndarray:
    """A black effect's coverage over white, in 8-bit levels."""
    return np.rint((1.0 - psd.numpy()[..., :3].mean(axis=2)) * 255.0)


def test_the_field_is_photoshops_across_runs_of_its_table() -> None:
    psd = PSDImage.open(full_name(_FIELD))
    assert (psd.width, psd.height) == (4200, 3)
    field = effect_noise.field((0, 0, psd.width, psd.height))
    assert np.array_equal(_levels(psd), field)


def test_a_shadow_of_size_zero_at_half_coverage_paints_the_field() -> None:
    """At 50% coverage and noise 50 the coverage is the field's byte."""
    psd = PSDImage.open(full_name(_FIXTURE))
    layer = next(layer for layer in psd if layer.name == "ds-size0")
    box = layer.bbox
    left, top, right, bottom = box
    assert np.array_equal(_levels(psd)[top:bottom, left:right], effect_noise.field(box))


def test_noisy_effects_render_to_photoshops_composite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each cell matches Photoshop's, and strays from it without the noise."""
    errors = _cell_errors(PSDImage.open(full_name(_FIXTURE)))
    assert len(errors) == 8
    for name, error in errors.items():
        assert error <= _BOUND, (name, error)

    def noiseless(coverage: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        return coverage

    monkeypatch.setattr(effect_noise, "noisy", noiseless)
    errors = _cell_errors(PSDImage.open(full_name(_FIXTURE)))
    for name, error in errors.items():
        assert error > _MOVED, (name, error)


def _off_white(psd: PSDImage) -> dict[str, float]:
    """Per cell, how far ours darkens a pixel Photoshop leaves white, in levels."""
    color, _, _ = composite(psd)
    ours = np.rint(color[..., :3] * 255.0).min(axis=2)
    white = np.rint(psd.numpy()[..., :3] * 255.0).min(axis=2) == 255.0
    off = {}
    for layer in psd:
        if not layer.has_effects():
            continue
        x0 = layer.bbox[0] // _CELL * _CELL
        cell = np.s_[:, x0 : x0 + _CELL]
        off[layer.name] = float((255.0 - ours[cell])[white[cell]].max())
    return off


def test_no_noise_where_the_contour_takes_the_effect_to_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under the cone a full matte paints nothing, and Photoshop adds no noise there.

    The cone's own error is under the noise's reach.
    """
    psd = PSDImage.open(full_name(_CONE))
    off = _off_white(psd)
    assert sorted(off) == ["ds-cone", "og-cone"]
    for name, error in off.items():
        assert error < _MOVED, (name, error)

    monkeypatch.setattr(effects, "_contour_lit", lambda levels, table: None)
    for name, error in _off_white(PSDImage.open(full_name(_CONE))).items():
        assert error > _MOVED, (name, error)


@pytest.mark.parametrize(
    "noise, amount", [(0.0, 0), (6.0, 15), (50.0, 128), (70.0, 179), (100.0, 255)]
)
def test_the_noise_amount_rounds_half_up(noise: float, amount: int) -> None:
    assert effect_noise.noise_amount(noise) == amount


def test_no_noise_leaves_the_coverage_as_it_is() -> None:
    coverage = np.full((2, 3), 0.25, dtype=np.float32)
    assert effect_noise.noisy(coverage, 0.0, (0, 0, 3, 2), coverage) is coverage


def test_beneath_the_layer_the_noise_is_held_under_the_gain_on_the_matte() -> None:
    coverage = np.full((4, 4), 0.5, dtype=np.float32)
    matte = np.full((4, 4), 0.03, dtype=np.float32)
    moved = effect_noise.noisy(coverage, 100.0, (0, 0, 4, 4), matte)
    ceiling = 0.03 * effect_noise.BENEATH_GAIN
    assert moved.max() == pytest.approx(ceiling)
    assert moved.min() >= 0.0


def test_inside_the_layer_any_matte_lets_the_noise_through() -> None:
    """However faint the matte, the noise reaches full coverage; none without it."""
    coverage = np.zeros((2, 64), dtype=np.float32)
    matte = np.zeros((2, 64), dtype=np.float32)
    matte[0] = 1e-7
    moved = effect_noise.noisy(coverage, 100.0, (0, 0, 64, 2), matte, inside=True)
    assert moved[0].max() > 0.4
    assert not moved[1].any()


def test_no_noise_where_the_effect_is_not_lit() -> None:
    coverage = np.full((2, 64), 0.5, dtype=np.float32)
    lit = np.ones((2, 64), dtype=bool)
    lit[1] = False
    moved = effect_noise.noisy(
        coverage, 100.0, (0, 0, 64, 2), np.ones_like(coverage), lit=lit
    )
    assert moved[0].max() > 0.5
    assert not moved[1].any()


def test_the_coverage_sets_the_canvas_and_the_viewport_its_corner() -> None:
    coverage = np.full((2, 3), 0.5, dtype=np.float32)
    matte = np.ones_like(coverage)
    moved = effect_noise.noisy(coverage, 50.0, (5, 7, 100, 100), matte)
    assert np.array_equal(
        moved, effect_noise.noisy(coverage, 50.0, (5, 7, 8, 9), matte)
    )


def _field_at_once(viewport: tuple[int, int, int, int]) -> np.ndarray:
    """The field over ``viewport``, built in one piece as the module describes."""
    left, top, right, bottom = viewport
    table, rows, runs = effect_noise._tables()
    ys = np.arange(top, bottom) % len(rows)
    xs = np.arange(left, right) % (len(runs) * effect_noise._RUN)
    row, key = rows[ys, 0][:, None], rows[ys, 1][:, None]
    moves = runs[(xs // effect_noise._RUN)[None, :], row]
    column = (xs % effect_noise._RUN)[None, :] ^ key ^ moves[..., 1]
    return table[moves[..., 0], column]


@pytest.mark.parametrize(
    "viewport",
    [(2000, 100, 2100, 400), (-50, -300, 30, 20), (30700, 29900, 30760, 30100)],
)
def test_the_field_built_in_strips_is_the_field_built_at_once(
    viewport: tuple[int, int, int, int],
) -> None:
    """Strips of rows and runs of columns, across a run, the strip size and the wrap."""
    assert np.array_equal(effect_noise.field(viewport), _field_at_once(viewport))


def test_the_field_repeats_past_what_is_stored() -> None:
    near = effect_noise.field((0, 0, 8, 2))
    rows = len(effect_noise._tables()[1])
    assert np.array_equal(effect_noise.field((0, rows, 8, rows + 2)), near)
    assert effect_noise.field((-3, -2, 5, 1)).shape == (3, 8)
    assert effect_noise.field((4, 0, 4, 3)).shape == (3, 0)
