"""Layer effects in the blend modes that fade their source.

In Color Dodge, Linear Dodge, Color Burn, Linear Burn, Vivid Light, Linear
Light and Difference, Photoshop fades an effect's colour towards the mode's
neutral one by the effect's coverage and opacity together, rather than mixing
the full-strength blend with what lies beneath by them.
"""

import importlib
from typing import Any

import numpy as np
import pytest

from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import blend, composite
from psd_tools.constants import BlendMode

from ..utils import full_name

pytest.importorskip("scipy")

composite_module = importlib.import_module("psd_tools.composite.composite")

# effects/fading-mode-effects.psd: over a lilac backdrop, one layer per effect,
# named for it: inner shadows in Linear Burn and Color Burn on a feathered,
# spiked star (one over a colour overlay); satins in Difference and Linear
# Dodge; an inner glow in Linear Dodge; a gradient overlay in Linear Burn
# fading to clear; a bevel lit in Color Dodge and shaded in Linear Burn; an
# outer glow in Linear Dodge, also on a layer at 50% opacity; a drop shadow in
# Color Dodge; an inner shadow and a satin on layers at 50% fill; and
# Multiply, Screen and Normal controls. Photoshop 2023's composite is the
# reference.
_FIXTURE = "effects/fading-mode-effects.psd"

# Levels a cell may stray from Photoshop's: the bevel's shading is the
# approximation that strays furthest.
_BOUND = 10.0
_REACH = 12


def _errors(psd: PSDImage) -> dict[str, float]:
    """Per layer with an effect, its largest difference from Photoshop's, in levels."""
    color, _, _ = composite(psd)
    diff = np.abs(color[..., :3] - psd.numpy()[..., :3]).max(axis=2) * 255.0
    errors = {}
    for layer in psd:
        if not layer.has_effects():
            continue
        left, top, right, bottom = layer.bbox
        errors[layer.name] = float(
            diff[top - _REACH : bottom + _REACH, left - _REACH : right + _REACH].max()
        )
    return errors


def test_effects_in_the_fading_modes_render_to_photoshops_composite() -> None:
    errors = _errors(PSDImage.open(full_name(_FIXTURE)))
    assert len(errors) == 15
    for name, error in errors.items():
        assert error <= _BOUND, (name, error)


def test_mixing_the_full_blend_by_the_coverage_misses_photoshops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the coverage mixing the blend in instead, every fading cell strays.

    The controls in Multiply, Screen and Normal do not move.
    """
    fading = dict(_errors(PSDImage.open(full_name(_FIXTURE))))

    def no_fade(*args: Any, **kwargs: Any) -> None:
        return None

    def no_fade_source(blend_fn: Any, blend_mode: BlendMode, opacity: Any) -> Any:
        return blend_fn

    monkeypatch.setattr(composite_module, "faded_color", no_fade)
    monkeypatch.setattr(composite_module, "fade_source", no_fade_source)
    mixed = _errors(PSDImage.open(full_name(_FIXTURE)))
    controls = [
        name for name in mixed if any(m in name for m in ("Mltp", "Scrn", "Nrml"))
    ]
    assert len(controls) == 3
    for name, error in mixed.items():
        if name in controls:
            assert error == pytest.approx(fading[name], abs=1e-6), name
        else:
            assert error > 24.0, (name, error)


def test_a_layer_comp_bevel_lit_in_linear_dodge_matches_photoshops_preview() -> None:
    """``layer_comps.psb``'s bevels: Linear Dodge highlights, a Linear Burn shadow.

    The whole document lands within the bound.
    """
    psd = PSDImage.open(full_name("layer_comps.psb"))
    color, _, _ = composite(psd)
    error = float(((psd.numpy()[..., :3] - color[..., :3]) ** 2).mean())
    assert error <= 2.5e-4


def test_the_faded_colour_keeps_its_share_of_the_source() -> None:
    color = np.array([[0.8, 0.2, 0.6]], dtype=np.float32)
    share = np.array([[0.25]], dtype=np.float32)
    burned = blend.faded_color(BlendMode.LINEAR_BURN, color, share)
    assert burned is not None
    assert np.allclose(burned, 1.0 + 0.25 * (color - 1.0))
    dodged = blend.faded_color(BlendMode.LINEAR_DODGE, color, share)
    assert dodged is not None
    assert np.allclose(dodged, 0.25 * color)
    assert blend.faded_color(BlendMode.MULTIPLY, color, share) is None
    assert blend.faded_color(BlendMode.NORMAL, color, share) is None


def test_a_per_pixel_fade_lands_where_each_pixels_faded_source_blends() -> None:
    """Mixed in by its share, each pixel lands on the blend of its faded source.

    Where the share is zero the backdrop stays as it is.
    """
    cb = np.linspace(0.1, 0.9, 5, dtype=np.float32).reshape(5, 1, 1)
    cs = np.full((5, 1, 1), 0.3, dtype=np.float32)
    share = np.array([0.0, 0.1, 0.5, 0.9, 1.0], dtype=np.float32).reshape(5, 1, 1)
    blend_fn = blend.BLEND_FUNC[BlendMode.COLOR_BURN]
    faded = blend.fade_source(blend_fn, BlendMode.COLOR_BURN, share)
    result = (1.0 - share) * cb + share * faded(cb, cs)
    expected = blend_fn(cb, 1.0 + share * (cs - 1.0))
    assert np.all(np.isfinite(result))
    assert np.allclose(result, expected, atol=1e-5)
    assert np.allclose(result[0], cb[0])
