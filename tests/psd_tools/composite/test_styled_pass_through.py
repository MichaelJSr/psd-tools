"""A pass-through group carrying a layer style, over children that blend.

Photoshop renders such a group between its children blended through to the
backdrop and the group composited on its own in Normal, mixing the two by
``(1 - a) / (2 - a)`` where the group's alpha is ``a``; the group then goes on
as one layer, over its own shadow and glow.
"""

import importlib
from typing import Any

import numpy as np
import pytest

from psd_tools.api.layers import Layer
from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import composite

from ..utils import full_name

pytest.importorskip("scipy")

composite_module = importlib.import_module("psd_tools.composite.composite")

# effects/styled-pass-through.psd: over vertical stripes in six colours, one
# pass-through group per cell, named for its case. The squares are blue at 50%
# in Screen inside a group carrying a 0% outside stroke, unless the name says
# otherwise: Multiply, Pin Light, Overlay, Normal and Linear Dodge children;
# a Screen child at 50% fill, and one blurred; a group with no style ("plain")
# and a Normal one ("normal group"); groups whose only effect is a drop shadow
# or a colour overlay at 0%, a colour overlay at 1%, a Normal outer glow, a
# Multiply drop shadow; a styled group inside a styled group ("nested"); a
# Linear Dodge child at 50% fill; a Screen square over a Normal one at 50%
# ("overlap") and over an opaque one ("opaque under"); a bar with a Screen
# glow ("glowing"). Below them type layers: Screen text in a group with a 0%
# stroke, Pin Light text in the card template's 'Text' group style, the Screen
# text rasterized, and The_Radiance's Ability: a Normal type layer with a
# Screen glow and a stroke, under a group's Multiply shadow. Photoshop 2023's
# composite is the reference.
_FIXTURE = "effects/styled-pass-through.psd"

# The cells whose children blend with the backdrop, which the mix moves.
_MIXED = (
    "screen",
    "multiply",
    "pin light",
    "overlay",
    "screen fill",
    "screen soft",
    "color 1",
    "glow",
    "shadow",
    "nested",
    "glowing",
    "rasterized text",
    "glowing text",
)
# The cells Photoshop leaves as they pass through.
_KEPT = (
    "normal",
    "linear dodge",
    "plain",
    "normal group",
    "shadow 0",
    "color 0",
    "opaque under",
    "screen text",
    "pin light text",
)

# Levels a cell may stray from Photoshop's 8-bit composite.
_BOUND = 2.0
_BOUNDS = {
    # The type layer's stroke edge under its glow strays furthest.
    "glowing text": 16.0,
    # The template stroke's corners, whatever the group's children do.
    "pin light text": 10.0,
}


def _cell(layer: Layer) -> tuple[int, int, int, int]:
    """The grid cell a group sits in: 64 px squares above, 320 x 80 below."""
    left, top, right, bottom = layer.bbox
    cx, cy = (left + right) // 2, (top + bottom) // 2
    if cy < 138:
        x, y = 10 + (cx - 10) // 64 * 64, 10 + (cy - 10) // 64 * 64
        return x, y, x + 64, y + 64
    x, y = 10 + (cx - 10) // 320 * 320, 138 + (cy - 138) // 80 * 80
    return x, y, x + 320, y + 80


def _errors() -> dict[str, float]:
    """Per cell, its largest difference from Photoshop's, in levels."""
    psd = PSDImage.open(full_name(_FIXTURE))
    color, _, _ = composite(psd)
    diff = np.abs(color[..., :3] - psd.numpy()[..., :3]).max(axis=2) * 255.0
    errors = {}
    for layer in psd:
        if layer.is_group():
            left, top, right, bottom = _cell(layer)
            errors[layer.name] = float(diff[top:bottom, left:right].max())
    return errors


def test_styled_pass_through_groups_render_to_photoshops_composite() -> None:
    errors = _errors()
    assert set(_MIXED) | set(_KEPT) <= set(errors)
    for name in _MIXED + _KEPT:
        assert errors[name] <= _BOUNDS.get(name, _BOUND), (name, errors[name])


def test_without_the_mix_the_blending_children_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rendered as they pass through, the children Photoshop mixes stray.

    The cells it leaves alone do not move.
    """
    mixed = _errors()

    def never(*args: Any) -> bool:
        return False

    monkeypatch.setattr(composite_module, "_mixes_isolated", never)
    passed = _errors()
    for name in _MIXED:
        assert passed[name] > 20.0, (name, passed[name])
    for name in _KEPT:
        assert passed[name] == pytest.approx(mixed[name], abs=1e-6), name


def test_a_mixing_group_goes_on_over_its_whole_shadow_and_glow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cut from under the group by its children's modes, they show too little."""

    def by_children(self: Any, layer: Layer) -> bool:
        return composite_module._covers_beneath(layer, self._layer_filter)

    monkeypatch.setattr(composite_module.Compositor, "_covers_beneath", by_children)
    errors = _errors()
    for name in ("glow", "shadow", "glowing text"):
        assert errors[name] > 20.0, (name, errors[name])
    assert errors["screen"] <= _BOUND


@pytest.mark.parametrize(
    "name, mixes",
    [
        ("screen", True),
        ("color 1", True),
        ("glowing text", True),
        ("rasterized text", True),
        ("normal", False),
        ("plain", False),
        ("normal group", False),
        ("shadow 0", False),
        ("color 0", False),
        ("linear dodge", False),
        ("screen text", False),
        ("pin light text", False),
    ],
)
def test_which_groups_mix(name: str, mixes: bool) -> None:
    """What decides it: the group's style and its children's modes.

    A style at 0% does not count, nor do Normal children, nor a type layer's
    own paint or a mode faded by fill opacity; a stroke outside counts at 0%.
    """
    psd = PSDImage.open(full_name(_FIXTURE))
    group = next(layer for layer in psd if layer.name == name)
    assert composite_module._mixes_isolated(group, None) is mixes


@pytest.mark.xfail(
    strict=True,
    reason="Photoshop mixes a fill-faded child by its fill; such a group is "
    "left as it passes through",
)
def test_a_fill_faded_child_mixes_by_its_fill() -> None:
    assert _errors()["linear dodge fill"] <= _BOUND


@pytest.mark.xfail(
    strict=True,
    reason="where two children of a mixing group overlap, Photoshop lands "
    "elsewhere than one mix of the group's two renders",
)
def test_overlapping_children_mix_as_photoshop_mixes_them() -> None:
    assert _errors()["overlap"] <= _BOUND
