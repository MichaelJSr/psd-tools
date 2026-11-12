"""A vector stroke is drawn over the shape's clip layers, and past its stored box."""

import numpy as np

from psd_tools import PSDImage
from psd_tools.composite.composite import Compositor
from psd_tools.psd.descriptor import UnitFloat

from ..utils import full_name


def _plain(psd: PSDImage):
    return next(x for x in psd.descendants() if x.name == "Plain")


def test_an_unforced_render_draws_the_stroke_over_the_clip_layers() -> None:
    """``Plain`` has stored pixels, a 3 px inner red stroke and ``Clip`` above it.

    Photoshop's preview has the red stroke where ``Clip`` crosses it; the
    stored pixels under the clip layer gave ``Clip``'s blue there.
    """
    path = full_name("effects/stroke-composite.psd")
    psd = PSDImage.open(path)
    stored = psd.topil()
    assert stored is not None
    preview = np.asarray(stored.convert("RGBA"))
    drawn = np.asarray(psd.composite(ignore_preview=True).convert("RGBA"))
    for x, y in ((90, 20), (90, 22), (120, 50)):
        assert list(preview[y, x]) == [255, 0, 0, 255]
        assert list(drawn[y, x]) == [255, 0, 0, 255]
    forced = np.asarray(
        PSDImage.open(path).composite(force=True, ignore_preview=True).convert("RGBA")
    )
    assert np.abs(drawn.astype(int) - forced).max() <= 1


def test_only_a_stroke_with_a_width_and_a_shown_clip_layer_redraws() -> None:
    psd = PSDImage.open(full_name("effects/stroke-composite.psd"))
    layer = _plain(psd)
    compositor = Compositor(
        psd.viewbox,
        np.ones((psd.height, psd.width, 3), dtype=np.float32),
        np.zeros((psd.height, psd.width, 1), dtype=np.float32),
        layer_filter=lambda x: x.is_visible(),
    )
    assert compositor._redraws(layer)
    assert layer.stroke is not None
    width = layer.stroke._data[b"strokeStyleLineWidth"]
    layer.stroke._data[b"strokeStyleLineWidth"] = UnitFloat(value=0.0, unit=width.unit)
    assert not compositor._redraws(layer), "the templates' 'Clip ... to this'"
    layer.stroke._data[b"strokeStyleLineWidth"] = width
    layer.clip_layers[0].visible = False
    assert not compositor._redraws(layer), "no clip layer shown"


def test_a_stroke_past_its_layer_box_takes_its_colour() -> None:
    """A box that stops short of the path (a truncated origination box) left white."""
    psd = PSDImage.open(full_name("layers-minimal/shape-layer.psd"))
    layer = psd[0]
    assert not layer.has_pixels() and layer.bbox == (1, 1, 30, 30)
    layer._bbox = (1, 1, 30, 20)
    compositor = Compositor(
        psd.viewbox,
        np.ones((psd.height, psd.width, 3), dtype=np.float32),
        np.zeros((psd.height, psd.width, 1), dtype=np.float32),
    )
    color, shape, _ = compositor._get_stroke(layer)
    stroked = shape[..., 0] > 0
    assert stroked[25:].any()
    assert (color[stroked] == [1.0, 0.0, 1.0]).all()
