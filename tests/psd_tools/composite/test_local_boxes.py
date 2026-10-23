"""A layer composited on the box it can change matches it composited on the whole canvas.

``Compositor.apply`` hands a layer whose change box is small to
``_apply_in_box``, which composites it on views of that box. Outside the box
nothing may change, and inside it the pixels are the ones the whole canvas
gets, up to float rounding.
"""

import importlib

import numpy as np
import pytest

from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import vector

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
