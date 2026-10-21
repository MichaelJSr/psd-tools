"""A viewport composite is the full composite cropped, effects included.

The compositor draws each effect on a box of its own and pastes it onto the
viewport. Since the box is cut to what can reach the viewport
(``_within_reach``), the cut must not move a pixel: every fixture that carries
an effect is rendered whole and through viewports that cut through its styled
layers, and the two have to agree byte for byte.
"""

import importlib
from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import pytest

from psd_tools.api.effects import DropShadow
from psd_tools.api.layers import Layer
from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import composite, effects
from psd_tools.composite.composite import Compositor, _within_reach
from psd_tools.psd.descriptor import Descriptor, Double, Enumerated
from psd_tools.terminology import Enum, Key, Type

from ..utils import full_name

pytest.importorskip("scipy")

# Every fixture with an enabled layer effect, so each kind the compositor
# draws -- strokes of the three positions, the overlays, drop and inner
# shadows, outer and inner glows, satins, bevels -- is cut through at least
# once, on an object and on a group.
_STYLED_FIXTURES = [
    "advanced-blending.psd",
    "blend-and-clipping.psd",
    "clipping-mask2.psd",
    "effect-stroke-gradient.psd",
    "effects/antialiased-stroke-edge.psd",
    "effects/blend-modes.psd",
    "effects/center-stroke-sizes.psd",
    "effects/double-stroke-effects.psd",
    "effects/effect-enums.psd",
    "effects/effects-enabled.psd",
    "effects/feathered-stroke.psd",
    "effects/group-clips-child-stroke.psd",
    "effects/group-stroke-off-canvas.psd",
    "effects/inset-stroke-sizes.psd",
    "effects/outside-stroke.psd",
    "effects/shape-fx2.psd",
    "effects/shape-stroke-raster-mask-density.psd",
    "effects/shape-stroke-raster-mask.psd",
    "effects/stroke-composite.psd",
    "effects/stroke-effect-transparent-shape.psd",
    "effects/stroke-effects.psd",
    "fill_adjustments.psd",
    "layer_comps.psb",
    "layer_effects.psd",
    "layer_params.psb",
    "masks.psb",
    "patterns.psd",
    "transparency/fill-opacity.psd",
]

# Fixtures whose redrawn vector paths meet the viewport: aggdraw's coverage
# of one path differs by an ulp from one viewport to another, and a stroke
# traced from it reads a pixel at ``1 - ulp`` as an edge the whole pixel is not.
# The stored-pixel render is unaffected.
_REDRAWN_PER_VIEWPORT = {
    "effects/shape-fx2.psd",
    "effects/shape-stroke-raster-mask-density.psd",
    "effects/shape-stroke-raster-mask.psd",
    "effects/stroke-composite.psd",
    "effects/stroke-effects.psd",
}

_Box = tuple[int, int, int, int]


def _case(filename: str, force: bool) -> Any:
    marks = []
    if force and filename in _REDRAWN_PER_VIEWPORT:
        marks.append(
            pytest.mark.xfail(
                strict=True, reason="aggdraw rasterizes a path differently per viewport"
            )
        )
    return pytest.param(
        filename,
        force,
        id=f"{filename}-{'redrawn' if force else 'stored'}",
        marks=marks,
    )


def _styled(psd: PSDImage) -> Iterator[Layer]:
    for layer in psd.descendants():
        try:
            if layer.has_effects():
                yield layer
        except Exception:
            continue


def _clamp(box: _Box, canvas: _Box) -> _Box | None:
    left = max(box[0], canvas[0])
    top = max(box[1], canvas[1])
    right = min(box[2], canvas[2])
    bottom = min(box[3], canvas[3])
    if left >= right or top >= bottom:
        return None
    return left, top, right, bottom


def _viewports(psd: PSDImage) -> list[_Box]:
    """Windows that cut through every styled layer, and the canvas's quadrants.

    One window strictly inside each styled layer, so every effect's box is
    cut on all four sides, and one a few pixels around it, so the cut falls
    through the effect's reach rather than through the layer.
    """
    canvas = psd.viewbox
    width, height = psd.width, psd.height
    boxes: list[_Box] = [
        (0, 0, width // 2, height // 2),
        (width // 2, height // 2, width, height),
        (width // 4, height // 4, 3 * width // 4, 3 * height // 4),
    ]
    for layer in _styled(psd):
        left, top, right, bottom = layer.bbox
        if left >= right or top >= bottom:
            continue
        dx, dy = max((right - left) // 3, 1), max((bottom - top) // 3, 1)
        for box in (
            (left + dx, top + dy, right - dx, bottom - dy),
            (left - 3, top - 3, right + 3, bottom + 3),
        ):
            clamped = _clamp(box, canvas)
            if clamped is not None and clamped not in boxes:
                boxes.append(clamped)
    return boxes


@pytest.mark.parametrize(
    "filename, force",
    [_case(name, force) for name in _STYLED_FIXTURES for force in (False, True)],
)
def test_a_viewport_render_is_the_full_render_cropped(
    filename: str, force: bool
) -> None:
    psd = PSDImage.open(full_name(filename))
    whole = composite(psd, force=force)
    for box in _viewports(psd):
        left, top, right, bottom = box
        part = composite(psd, viewport=box, force=force)
        for name, full, cut in zip(
            ("color", "shape", "alpha"), whole, part, strict=True
        ):
            assert np.array_equal(full[top:bottom, left:right], cut), (name, box)


def test_within_reach_cuts_a_box_to_the_viewport_grown_by_the_reach() -> None:
    assert _within_reach((0, 0, 100, 100), (40, 40, 60, 60), 5) == (35, 35, 65, 65)
    # A box already within reach is left alone.
    assert _within_reach((42, 42, 58, 58), (40, 40, 60, 60), 5) == (42, 42, 58, 58)
    # A box that misses the viewport draws nothing, whatever its reach.
    assert _within_reach((0, 0, 10, 10), (40, 40, 60, 60), 5) == (0, 0, 0, 0)
    assert _within_reach((0, 0, 10, 10), (40, 40, 60, 60), None) == (0, 0, 0, 0)
    # An unbounded reach keeps the whole box.
    assert _within_reach((0, 0, 100, 100), (40, 40, 60, 60), None) == (0, 0, 100, 100)


def _stroke(paint: bytes, style: bytes, size: float) -> Descriptor:
    desc = Descriptor()
    desc[Key.PaintType] = Enumerated(Type.FrameFill, paint)
    desc[Key.Style] = Enumerated(Type.FrameStyle, style)
    desc[Key.SizeKey] = Double(size)
    return desc


@pytest.mark.parametrize(
    "style, expected",
    [(Enum.OutsetFrame, 9), (Enum.CenteredFrame, 6), (Enum.InsetFrame, 9)],
    ids=["outset", "centered", "inset"],
)
def test_a_solid_strokes_reach_is_its_widest_limit_rounded_out(
    style: bytes, expected: int
) -> None:
    """Two pixels past the band's limit, whichever side of the edge it lies."""
    assert effects.stroke_reach(_stroke(Enum.SolidColor, style, 7.0)) == expected


@pytest.mark.parametrize("paint", [Enum.GradientFill, Enum.Pattern])
def test_a_painted_stroke_reads_its_whole_box(paint: bytes) -> None:
    assert effects.stroke_reach(_stroke(paint, Enum.OutsetFrame, 7.0)) is None


def test_a_cast_mattes_reach_is_its_blur_and_its_offset() -> None:
    assert effects.cast_reach(0.0, 0.0, 2.0, 0.0) == 5
    assert effects.cast_reach(4.0, 90.0, 2.0, 0.0) == 9
    assert effects.cast_reach(3.0, 45.0, 0.5, 0.0) == 7


def test_a_spread_matte_reads_its_whole_box() -> None:
    """Spread grows the silhouette at the levels it finds on the canvas."""
    assert effects.cast_reach(0.0, 0.0, 2.0, 10.0) is None
    assert effects.glow_reach(2.0, 10.0, precise=False) is None
    # A size of zero is not spread at all.
    assert effects.cast_reach(3.0, 90.0, 0.0, 69.0) == 6


def test_a_precise_glow_is_bounded_whatever_its_spread() -> None:
    assert effects.glow_reach(5.0, 40.0, precise=True) == 8
    assert effects.glow_reach(5.0, 0.0, precise=False) == 8


def test_a_small_viewport_draws_a_wide_layers_shadow_on_a_box_near_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cost of a viewport render follows the viewport, not the layer.

    ``masks.psb``'s 'Rectangle 2' runs past both sides of the canvas and
    casts a shadow a hundred pixels wide; asked for a window inside it, the
    shadow is drawn on a box no further from that window than its reach.
    """
    psd = PSDImage.open(full_name("masks.psb"))
    layer = next(sub for sub in psd.descendants() if sub.name == "Rectangle 2")
    shadow = cast(DropShadow, next(iter(layer.effects.find("dropshadow"))))
    geometry = effects.shadow_geometry(shadow.descriptor, shadow.angle)
    reach = effects.cast_reach(
        geometry.distance, geometry.angle, geometry.size, geometry.spread
    )
    assert reach is not None
    viewport = (300, 20, 340, 60)
    drawn: list[_Box] = []
    original = effects.draw_drop_shadow

    def recording(box: _Box, *args: Any, **kwargs: Any) -> np.ndarray:
        drawn.append(box)
        return original(box, *args, **kwargs)

    module = importlib.import_module("psd_tools.composite.composite")
    monkeypatch.setattr(module, "draw_drop_shadow", recording)
    composite(psd, viewport=viewport)
    assert drawn
    left, top, right, bottom = viewport
    for box in drawn:
        assert box[0] >= left - reach and box[1] >= top - reach, box
        assert box[2] <= right + reach and box[3] <= bottom + reach, box
        # Narrower than the layer it is cast from.
        assert box[2] - box[0] < layer.width


def test_one_group_composite_serves_a_stroke_and_a_shadow_on_one_box() -> None:
    """Tracing a group fills both the coverage and the alpha memo."""
    psd = PSDImage.open(full_name("effects/group-stroke-off-canvas.psd"))
    group = next(layer for layer in psd if layer.name == "Inside")
    compositor = Compositor(
        psd.viewbox,
        np.ones((psd.height, psd.width, 3), dtype=np.float32),
        np.zeros((psd.height, psd.width, 1), dtype=np.float32),
    )
    box = group.bbox
    shape = compositor._get_group_shape(group, box)
    key = (id(group), box)
    assert key in compositor._cache.group_shapes
    assert key in compositor._cache.group_alphas
    assert np.array_equal(compositor._get_group_shape(group, box, alpha=True), shape)
