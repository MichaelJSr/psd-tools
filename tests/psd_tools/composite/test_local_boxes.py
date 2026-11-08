"""A layer composited on the box it can change matches it composited on the whole canvas.

``Compositor.apply`` hands a layer whose change box is small to
``_apply_in_box``, which composites it on views of that box. Outside the box
nothing may change, and inside it the pixels are the ones the whole canvas
gets, up to float rounding. A layer whose change box misses the viewport is
skipped.
"""

import importlib

import numpy as np
import pytest

from psd_tools.api.layers import Layer
from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import utils, vector

from ..utils import full_name

pytest.importorskip("scipy")
pytest.importorskip("aggdraw")

composite_module = importlib.import_module("psd_tools.composite.composite")

# Rounding differs only in the last bits of a float32.
_ROUNDING = 1e-5


@pytest.mark.parametrize(
    "filename",
    [
        # Shapes whose antialiased corners lie past their rounded boxes.
        "adjustment-fillers.psd",
        "blend-modes/pass-through.psd",
        "clipping-mask.psd",
        "adjustments/curves_rgb.psd",
        "effects/center-stroke-sizes.psd",
        "effect-stroke-gradient.psd",
    ],
)
def test_a_layer_on_its_box_matches_the_whole_canvas(
    monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    psd = PSDImage.open(full_name(filename))
    boxes: list[str] = []
    real = composite_module.Compositor._apply_in_box

    def spy(self: object, layer: object, box: tuple[int, int, int, int]) -> None:
        boxes.append(getattr(layer, "name", ""))
        real(self, layer, box)  # type: ignore[arg-type]

    monkeypatch.setattr(composite_module.Compositor, "_apply_in_box", spy)
    boxed = composite_module.composite(psd, force=True)
    assert boxes, "no layer was composited on its own box"

    monkeypatch.setattr(composite_module, "_LOCAL_BOXES", False)
    whole = composite_module.composite(psd, force=True)
    for got, expected in zip(boxed, whole, strict=True):
        np.testing.assert_allclose(got, expected, rtol=0, atol=_ROUNDING)


@pytest.mark.parametrize(
    "filename, viewport",
    [
        ("effects/styled-pass-through-nested.psd", (20, 20, 90, 90)),
        # The knockout groups miss the box too but are composited whole.
        ("transparency/knockout-isolated-groups.psd", (140, 140, 245, 245)),
    ],
)
def test_a_group_whose_box_misses_the_viewport_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    viewport: tuple[int, int, int, int],
) -> None:
    psd = PSDImage.open(full_name(filename))
    cache = composite_module._PassCache()
    missed = {
        layer.name
        for layer in psd
        if layer.is_group()
        and composite_module._box_safe(layer, cache)
        and utils.intersect(viewport, composite_module._change_bbox(layer, cache))
        == (0, 0, 0, 0)
    }
    assert missed
    groups: list[str] = []
    real = composite_module.Compositor._get_group

    def spy(self: object, layer: Layer, knockout: object) -> object:
        groups.append(layer.name)
        return real(self, layer, knockout)  # type: ignore[arg-type]

    monkeypatch.setattr(composite_module.Compositor, "_get_group", spy)
    boxed = composite_module.composite(psd, viewport=viewport, force=True)
    assert not missed & set(groups)

    groups.clear()
    monkeypatch.setattr(composite_module, "_LOCAL_BOXES", False)
    whole = composite_module.composite(psd, viewport=viewport, force=True)
    assert missed <= set(groups)
    for got, expected in zip(boxed, whole, strict=True):
        np.testing.assert_allclose(got, expected, rtol=0, atol=_ROUNDING)


def test_a_canvas_written_in_place_is_never_the_seed() -> None:
    """Boxed layers reuse their own copy; what the canvases were seeded with stays."""
    psd = PSDImage.open(full_name("adjustment-fillers.psd"))
    color = np.ones((psd.height, psd.width, 3), dtype=np.float32)
    alpha = np.zeros((psd.height, psd.width, 1), dtype=np.float32)
    compositor = composite_module.Compositor(psd.viewbox, color, alpha, force=True)
    # A whole-canvas layer would replace the seed before a box could reach it.
    for layer in psd:
        if compositor._local_box(layer) is not None:
            compositor.apply(layer)
    assert compositor._owned, "no canvas was written in place"
    assert (color == 1.0).all() and (alpha == 0.0).all()


def test_the_drawn_box_holds_a_redrawn_shape_and_its_stroke() -> None:
    psd = PSDImage.open(full_name("adjustment-fillers.psd"))
    shapes = [layer for layer in psd.descendants() if layer.name.startswith("Polygon")]
    assert shapes
    viewport = (0, 0, psd.width, psd.height)
    for layer in shapes:
        box = vector.drawn_bbox(layer)
        assert box is not None
        coverage = np.maximum(
            vector.draw_vector_mask(layer, viewport),
            vector.draw_stroke(layer, viewport),
        )[..., 0]
        rows, cols = np.nonzero(coverage)
        # The path itself covers pixels past the layer's rounded box.
        bbox = layer.bbox
        assert cols.min() < bbox[0] or cols.max() >= bbox[2]
        assert box[0] <= cols.min() and cols.max() < box[2]
        assert box[1] <= rows.min() and rows.max() < box[3]


def test_a_layer_without_a_path_has_no_drawn_box() -> None:
    psd = PSDImage.open(full_name("adjustment-fillers.psd"))
    background = next(layer for layer in psd if layer.name == "Background")
    assert vector.drawn_bbox(background) is None
