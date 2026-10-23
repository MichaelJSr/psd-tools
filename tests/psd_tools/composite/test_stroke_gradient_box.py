"""A gradient stroke's ramp is laid out on the band's own extent."""

import numpy as np
import pytest

from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import composite, paint
from psd_tools.composite.effects import draw_stroke_effect_split, stroke_gradient_box
from psd_tools.constants import ColorMode
from psd_tools.psd.descriptor import Descriptor, Double, Enumerated
from psd_tools.terminology import Enum, Key

from ..utils import full_name
from .test_gradient_fill import _gradient

# Where the silhouette's canvas sits in the document.
_VIEWPORT = (100, 200, 180, 270)
# The box of its pixels over half opaque, in the document.
_OPAQUE = (120, 220, 150, 240)


def _soft_tailed() -> np.ndarray:
    """A solid block with faint tails below it and to its left, as a soft edge leaves.

    The tails stop short of half opaque, so the record's box runs past the
    block on two sides and the block's own box does not.
    """
    width, height = _VIEWPORT[2] - _VIEWPORT[0], _VIEWPORT[3] - _VIEWPORT[1]
    shape = np.zeros((height, width, 1), dtype=np.float32)
    left, top = _OPAQUE[0] - _VIEWPORT[0], _OPAQUE[1] - _VIEWPORT[1]
    right, bottom = _OPAQUE[2] - _VIEWPORT[0], _OPAQUE[3] - _VIEWPORT[1]
    shape[bottom : bottom + 18, left:right] = 0.3
    shape[top:bottom, 4:left] = 0.2
    shape[top:bottom, left:right] = 1.0
    return shape


def _stroke(style: bytes, size: float, angle: float = 90.0) -> Descriptor:
    """A stroke effect painted with :py:func:`_gradient`'s ramp."""
    desc = _gradient(angle=angle)
    desc[Key.Style] = Enumerated(typeID=b"FStl", enum=style)
    desc[Key.SizeKey] = Double(size)
    desc[Key.PaintType] = Enumerated(typeID=b"FrFl", enum=Enum.GradientFill)
    return desc


def _grown(box: tuple[int, int, int, int], by: int) -> tuple[int, int, int, int]:
    return (box[0] - by, box[1] - by, box[2] + by, box[3] + by)


@pytest.mark.parametrize("size", [3.0, 13.0])
def test_a_soft_tail_does_not_stretch_a_gradient_strokes_ramp(size: float) -> None:
    desc = _stroke(Enum.OutsetFrame, size)
    shape = _soft_tailed()
    box = stroke_gradient_box(_VIEWPORT, shape, desc)
    assert box == _grown(_OPAQUE, int(size))
    color, _, _ = draw_stroke_effect_split(_VIEWPORT, shape, desc, _psd())
    expected, _ = paint.draw_gradient_fill(_VIEWPORT, ColorMode.RGB, desc, box)
    assert expected is not None
    assert np.array_equal(color, expected)


@pytest.mark.parametrize(
    ("style", "size", "grow"),
    [
        (Enum.OutsetFrame, 1.0, 1),
        (Enum.OutsetFrame, 3.0, 3),
        (Enum.OutsetFrame, 13.0, 13),
        (Enum.CenteredFrame, 1.0, 1),
        (Enum.CenteredFrame, 4.0, 2),
        (Enum.CenteredFrame, 7.0, 4),
        (Enum.InsetFrame, 3.0, 1),
        (Enum.InsetFrame, 13.0, 1),
    ],
)
def test_the_box_grows_by_the_whole_pixels_the_band_reaches_out(
    style: bytes, size: float, grow: int
) -> None:
    """Photoshop's renders give a centred band its half rounded up, an inset one a pixel."""
    box = stroke_gradient_box(_VIEWPORT, _soft_tailed(), _stroke(style, size))
    assert box == _grown(_OPAQUE, grow)


def test_an_extent_stands_in_for_the_silhouette() -> None:
    extent = (0, 0, 400, 300)
    box = stroke_gradient_box(
        _VIEWPORT, _soft_tailed(), _stroke(Enum.OutsetFrame, 7.0), extent
    )
    assert box == _grown(extent, 7)


def test_a_silhouette_never_half_opaque_keeps_the_viewport() -> None:
    shape = np.full((10, 10, 1), 0.5, dtype=np.float32)
    desc = _stroke(Enum.OutsetFrame, 3.0)
    assert stroke_gradient_box((0, 0, 10, 10), shape, desc) is None


def _psd() -> PSDImage:
    return PSDImage.open(full_name("effects/gradient-stroke-box.psd"))


# effects/gradient-stroke-box.psd: on white, one cell per layer, each green
# and carrying one red-to-blue stroke, linear and classic. A type layer reads
# "Hg"; the soft shape is a square with a faint skirt below and to its right,
# blurred, kept as pixels or a smart object or put in a pass-through or Normal
# group that carries the stroke. The names give the kind, the position, the
# size, the angle, the scale and the direction. Photoshop 2023's composite is
# the reference.
_FIXTURE_CELLS = 12

# Levels a pixel may stray: Photoshop rounds a scaled ramp's ends to whole
# pixels, which the ramp here does not.
_STRAY = 4


def _cell_errors() -> dict[str, int]:
    """Per cell, the most any pixel of it is off Photoshop's composite, in levels."""
    psd = _psd()
    color, _, _ = composite(psd)
    off = np.abs(color[..., :3] - psd.numpy()[..., :3]).max(axis=2) * 255
    errors = {}
    for layer in psd:
        stroke = next(iter(layer.effects.find("stroke")), None)
        if stroke is None:
            continue
        pad = int(float(stroke.descriptor[Key.SizeKey])) + 2
        left, top, right, bottom = layer.bbox
        cell = off[top - pad : bottom + pad, left - pad : right + pad]
        errors[layer.name] = int(np.ceil(cell.max() - 1e-6))
    return errors


def test_gradient_strokes_render_to_photoshops_composite() -> None:
    """Without the law a soft shape's cell strays 49-64 levels, a group's 98-169."""
    errors = _cell_errors()
    assert len(errors) == _FIXTURE_CELLS
    kinds = {name.split()[0] for name in errors}
    assert kinds == {"type", "smart", "pixel", "group-pass", "group-normal"}
    for name, error in errors.items():
        assert error <= _STRAY, name
