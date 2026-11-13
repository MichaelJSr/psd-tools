"""A pass-through group carrying a layer style, over children that blend.

Photoshop renders such a group between its children blended through to the
backdrop and the group composited on its own in Normal: one child by
``(1 - a) / (2 - a)`` where its alpha is ``a``, each blend-mode source after
it by the same law against what the group shows so far; the group then goes
on as one layer, over its own shadow and glow.
"""

import importlib
from typing import Any

import numpy as np
import pytest
from PIL import Image

from psd_tools.api.layers import Group, Layer, TypeLayer
from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import composite
from psd_tools.constants import Knockout, Tag

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
    "glowing text": 3.0,
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


def test_a_type_layer_drawn_as_its_pixels_mixes_as_they_do() -> None:
    """Composited from its stored raster, a type layer is pixels to the group.

    Photoshop 2023 does so with a type layer whose text it would lay out anew
    (``textWarningLevel`` 1), such as text a later Photoshop laid out; pinned
    straight, the fork takes it for one. "screen text" then lands where
    Photoshop puts "rasterized text", the same pixels over the same stripes
    80 px below, which it mixes; a substituted font is drawn the same way.
    """
    psd = PSDImage.open(full_name(_FIXTURE))
    cells = {layer.name: layer for layer in psd if isinstance(layer, Group)}
    text, raster = cells["screen text"][0], cells["rasterized text"][0]
    assert isinstance(text, TypeLayer)
    pixels, twin_pixels = text.numpy(), raster.numpy()
    assert pixels is not None and twin_pixels is not None
    assert np.array_equal(pixels, twin_pixels)
    left, top, right, bottom = _cell(cells["screen text"])
    shift = _cell(cells["rasterized text"])[1] - top
    twin = psd.numpy()[top + shift : bottom + shift, left:right, :3]
    for straight in ({"layer_text_gamma": {text.layer_id: None}}, {"text_gamma": None}):
        color, _, _ = composite(psd, **straight)
        assert np.abs(color[top:bottom, left:right, :3] - twin).max() * 255.0 <= _BOUND

    assert composite_module._mixes_isolated(cells["screen text"], None) is False
    text._data.text_data[b"substitutesUsed"] = True
    color, _, _ = composite(psd)
    assert np.abs(color[top:bottom, left:right, :3] - twin).max() * 255.0 <= _BOUND


def test_a_group_holding_a_knockout_does_not_mix() -> None:
    """No Photoshop render of one has been read: it keeps its pass-through one."""
    psd = PSDImage.open(full_name(_FIXTURE))
    group = next(layer for layer in psd if layer.name == "screen")
    assert isinstance(group, Group)
    assert composite_module._mixes_isolated(group, None) is True
    child = group[0]
    child.tagged_blocks.set_data(Tag.KNOCKOUT_SETTING, int(Knockout.SHALLOW))
    assert composite_module._mixes_isolated(group, None) is False


@pytest.mark.xfail(
    strict=True,
    reason="Photoshop mixes a fill-faded child by its fill; such a group is "
    "left as it passes through",
)
def test_a_fill_faded_child_mixes_by_its_fill() -> None:
    assert _errors()["linear dodge fill"] <= _BOUND


def test_overlapping_children_mix_as_photoshop_mixes_them() -> None:
    assert _errors()["overlap"] <= _BOUND


# effects/styled-pass-through-nested.psd: over the same stripes, a blue square
# at 50% in Screen per cell, under groups named for their nesting from the
# outside in: "pt" pass-through, "normal" Normal, "screen group" Screen,
# "styled" a pass-through group with a 0% outside stroke, "op99" a group at 99%
# opacity. Photoshop 2023's composite is the reference.
_NESTED = "effects/styled-pass-through-nested.psd"

_NESTED_MATCH = (
    "group pt>normal",
    "group normal>normal",
    "group styled>normal styled",
    "group normal",
    "group pt",
    "group styled",
    "group pt>normal op99",
    "group normal>pt",
)
# Left as it passes through; Photoshop lands near that, not on the mix.
_NESTED_KEPT = "group styled>screen group"
_NESTED_KEPT_BOUND = 10.0


def _nested_errors(psd: PSDImage | None = None) -> dict[str, float]:
    """Per group of the nested fixture, its largest difference, in levels."""
    psd = psd or PSDImage.open(full_name(_NESTED))
    color, _, _ = composite(psd)
    diff = np.abs(color[..., :3] - psd.numpy()[..., :3]).max(axis=2) * 255.0
    errors = {}
    for layer in psd:
        if layer.is_group():
            left, top, right, bottom = layer.bbox
            cell = diff[top - 20 : bottom + 20, left - 20 : right + 20]
            errors[layer.name] = float(cell.max())
    return errors


def test_nested_groups_render_to_photoshops_composite() -> None:
    errors = _nested_errors()
    for name in _NESTED_MATCH:
        assert errors[name] <= _BOUND, (name, errors[name])
    assert errors[_NESTED_KEPT] <= _NESTED_KEPT_BOUND


def test_a_group_child_in_a_mode_of_its_own_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A styled group over a Screen group does not mix; mixed, it strays."""
    psd = PSDImage.open(full_name(_NESTED))
    group = next(layer for layer in psd if layer.name == _NESTED_KEPT)
    assert composite_module._mixes_isolated(group, None) is False

    mixes = composite_module._mixes_isolated

    def forced(layer: Layer, *args: Any) -> bool:
        return layer.name == _NESTED_KEPT or mixes(layer, *args)

    monkeypatch.setattr(composite_module, "_mixes_isolated", forced)
    assert _nested_errors(PSDImage.open(full_name(_NESTED)))[_NESTED_KEPT] > 20.0


@pytest.mark.xfail(
    strict=True,
    reason="Photoshop passes a Normal group inside a styled pass-through group "
    "through, at 99% opacity too, and mixes its children with the group's",
)
@pytest.mark.parametrize(
    "name",
    ["group styled>normal", "group styled>normal op99", "group styled>pt>normal"],
)
def test_a_normal_group_child_mixes_as_a_layer_would(name: str) -> None:
    assert _nested_errors()[name] <= _BOUND


# effects/clipped-pass-through.psd: over vertical stripes in eight colours, one
# pass-through group per cell, named "<child>|<case>". The child is a blue
# square and a blue bar at 50% in Multiply or Screen, or a Normal square and
# bar with a Multiply drop shadow ("shadow"). The case is the group with
# nothing clipped to it ("pt"), with a 0% outside stroke ("styled"), or with a
# Curves layer clipped to it, hidden ("hadj") or shown ("vadj"), or a pixel
# layer, hidden and filled ("hpx") or shown and empty ("vempty"). Photoshop
# 2023's composite is the reference: every clipped case renders as "styled".
_CLIPPED = "effects/clipped-pass-through.psd"
_CLIPS = ("hadj", "vadj", "hpx", "vempty")


def _clipped_errors() -> dict[str, float]:
    """Per cell of the clipped fixture, its largest difference, in levels."""
    psd = PSDImage.open(full_name(_CLIPPED))
    color, _, _ = composite(psd)
    diff = np.abs(color[..., :3] - psd.numpy()[..., :3]).max(axis=2) * 255.0
    errors = {}
    for layer in psd:
        if layer.is_group():
            left, top = layer.bbox[:2]
            x, y = 10 + (left - 10) // 80 * 80, 10 + (top - 10) // 80 * 80
            errors[layer.name] = float(diff[y : y + 80, x : x + 80].max())
    return errors


def test_a_clipped_pass_through_group_renders_as_a_styled_one() -> None:
    """Any layer clipped to the group, hidden or empty, mixes it as a style does."""
    errors = _clipped_errors()
    assert len(errors) == 18
    for name, error in errors.items():
        assert error <= _BOUND, (name, error)


@pytest.mark.parametrize("case", ("pt", *_CLIPS))
def test_which_clipped_groups_mix(case: str) -> None:
    psd = PSDImage.open(full_name(_CLIPPED))
    group = next(layer for layer in psd if layer.name == f"multiply|{case}")
    assert composite_module._mixes_isolated(group, None) is (case != "pt")


# effects/mixing-law.psd: eight cells over stripes, opaque on the top row and at
# 50% on the bottom, each a pass-through group a hidden layer clips to (the
# last a 0% outside stroke instead): a Normal or Screen square fading in down
# the cell under a Screen or Multiply square fading out across it, or under a
# bar with a Screen outer glow. Photoshop 2023's PNG of it is the reference.
_LAW = "effects/mixing-law"


def _law_error(psd: PSDImage, name: str) -> int:
    """The largest difference from Photoshop's PNG ``name``, in levels."""
    theirs = np.asarray(Image.open(full_name(name + ".png")).convert("RGBA"), dtype=int)
    ours = np.asarray(psd.composite(force=True).convert("RGBA"), dtype=int)
    alpha = theirs[..., 3:] / 255.0
    color = np.abs(ours[..., :3] * ours[..., 3:] / 255.0 - theirs[..., :3] * alpha)
    return int(max(np.abs(ours[..., 3] - theirs[..., 3]).max(), color.max()))


def test_blend_sources_over_a_mixing_groups_content_follow_its_law(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Where a blend-mode source meets what the group already shows.

    Blended through, or mixed once as a whole, each cell strays by 14 to 20
    levels; by the law every pixel is Photoshop's within 3.
    """
    psd = PSDImage.open(full_name(_LAW + ".psd"))
    assert _law_error(psd, _LAW) <= 3

    def plain(self: Any, *args: Any) -> np.ndarray:
        return self._color

    monkeypatch.setattr(composite_module.Compositor, "_mixed_by_law", plain)
    assert _law_error(psd, _LAW) > 12


# effects/mixing-fixed-points.psd: three cells over stripes, each a pass-through
# group: a ramp fading in down the cell under a bar whose effect is in a colour
# its mode blends to itself, a white Screen outer glow over a dark ramp, then a
# black Multiply drop shadow over a light one. A hidden layer clips to those two
# groups; the third is the first without it. Photoshop 2023's PNG of it is the
# reference.
_FIXED = "effects/mixing-fixed-points"


def test_fixed_point_effects_follow_a_mixing_groups_law(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """White in Screen and black in Multiply blend to themselves, yet mix.

    Blended through, the two mixing cells stray by up to 29 levels where the
    effect meets the ramp; by the law every pixel is Photoshop's within 2.
    """
    psd = PSDImage.open(full_name(_FIXED + ".psd"))
    groups = [layer for layer in psd if layer.is_group()]
    assert [composite_module._mixes_isolated(g, None) for g in groups] == [
        True,
        True,
        False,
    ]
    assert _law_error(psd, _FIXED) <= 2

    def never(*args: Any) -> bool:
        return False

    monkeypatch.setattr(composite_module, "_mixes_isolated", never)
    assert _law_error(psd, _FIXED) > 20
