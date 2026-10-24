"""The compiled source arithmetic (``_blend.pyx``) is numpy's, bit for bit."""

import importlib
from collections.abc import Callable

import numpy as np
import pytest

from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import utils
from psd_tools.composite.composite import _over, _remix, _unmix

from ..utils import full_name

composite = importlib.import_module("psd_tools.composite.composite")

pytestmark = [
    pytest.mark.skipif(composite._cblend is None, reason="compiled loops not built"),
    pytest.mark.filterwarnings("ignore::RuntimeWarning"),
]

# What a canvas can hold besides [0, 1]: the signed zero and NaN np.clip passes
# through, subnormals, overshoot, and the zero alpha divide() fills under.
_SPECIAL = np.array(
    [0.0, -0.0, 1.0, 0.5, 1e-40, -1e-30, 1.5, -0.25, np.nan, np.inf],
    dtype=np.float32,
)


def _canvas(rng: np.random.Generator, channels: int) -> np.ndarray:
    """A strided 23x31 view into a larger canvas, as a local box hands one over."""
    whole = rng.random((29, 40, channels), dtype=np.float32)
    marked = rng.random(whole.shape[:2]) < 0.15
    whole[marked] = rng.choice(_SPECIAL, size=(int(marked.sum()), channels))
    return whole[3:26, 5:36]


def _numpy(
    monkeypatch: pytest.MonkeyPatch, fn: Callable[..., np.ndarray], *args: np.ndarray
) -> np.ndarray:
    with monkeypatch.context() as patch:
        patch.setattr(composite, "_cblend", None)
        return fn(*args)


def _same(compiled: np.ndarray, plain: np.ndarray) -> None:
    assert compiled.dtype == plain.dtype == np.float32
    assert compiled.shape == plain.shape
    assert compiled.tobytes() == plain.tobytes()


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_over_is_numpys(
    monkeypatch: pytest.MonkeyPatch, seed: int, channels: int
) -> None:
    rng = np.random.default_rng(seed)
    ones = [_canvas(rng, 1) for _ in range(5)]
    wide = [_canvas(rng, channels) for _ in range(4)]
    shape, alpha, alpha_b, alpha_previous, canvas_alpha = ones
    canvas, color_b, color, blended = wide
    args = [shape, alpha, alpha_b, color_b, color, blended, alpha_previous]
    args += [canvas, canvas_alpha]
    _same(_over(*args), _numpy(monkeypatch, _over, *args))


@pytest.mark.parametrize("seed", range(6))
def test_unmix_and_remix_are_numpys(monkeypatch: pytest.MonkeyPatch, seed: int) -> None:
    rng = np.random.default_rng(seed)
    color, backdrop_color = _canvas(rng, 3), _canvas(rng, 3)
    backdrop_alpha, alpha, canvas_alpha = (_canvas(rng, 1) for _ in range(3))
    unmixed = (color, backdrop_color, backdrop_alpha, alpha)
    _same(_unmix(*unmixed), _numpy(monkeypatch, _unmix, *unmixed))
    remixed = (color, backdrop_color, backdrop_alpha, alpha, canvas_alpha)
    _same(_remix(*remixed), _numpy(monkeypatch, _remix, *remixed))


@pytest.mark.parametrize("mask", [0.3, 1.0 / 3.0, 0.0, 1.0, "array"])
def test_passthrough_is_numpys(monkeypatch: pytest.MonkeyPatch, mask: object) -> None:
    """A Python float mask is rounded to float32 after ``1.0 - mask``, not before."""
    rng = np.random.default_rng(7)
    alpha, alpha_b, canvas_alpha = (_canvas(rng, 1) for _ in range(3))
    color_b, color = _canvas(rng, 3), _canvas(rng, 3)
    weight = _canvas(rng, 1) if mask == "array" else mask
    scalar = not isinstance(weight, np.ndarray)
    compiled = composite._cblend.passthrough(
        alpha if scalar else weight,
        weight if scalar else 0.0,
        scalar,
        alpha,
        alpha_b,
        color_b,
        color,
        canvas_alpha,
    )
    color_t = (1.0 - weight) * alpha_b * color_b + (  # type: ignore[operator]
        weight * alpha_b + alpha * (1.0 - alpha_b)  # type: ignore[operator]
    ) * color
    plain = np.clip(utils.divide(color_t, canvas_alpha, fill=color), 0, 1)
    _same(compiled, plain)


def test_unions_are_numpys() -> None:
    rng = np.random.default_rng(11)
    shape_g, shape, alpha_g, alpha, alpha_0 = (_canvas(rng, 1) for _ in range(5))
    union = utils.union
    plain_alpha_g = union(alpha_g, alpha)
    expected = (union(shape_g, shape), plain_alpha_g, union(alpha_0, plain_alpha_g))
    compiled = composite._cblend.unions(shape_g, shape, alpha_g, alpha, alpha_0)
    for got, want in zip(compiled, expected, strict=True):
        _same(got, want)


@pytest.mark.parametrize("operand", ["float64", "broadcast", "transposed"])
def test_numpy_keeps_what_the_loops_do_not_take(
    monkeypatch: pytest.MonkeyPatch, operand: str
) -> None:
    """A float64 operand, or one not in rows of pixels, stays with numpy."""
    rng = np.random.default_rng(3)
    ones = [_canvas(rng, 1) for _ in range(5)]
    wide = [_canvas(rng, 3) for _ in range(4)]
    if operand == "float64":
        wide[3] = wide[3].astype(np.float64)
    elif operand == "broadcast":
        ones[0] = np.broadcast_to(np.float32(0.5), ones[0].shape)
    else:
        wide[3] = np.ascontiguousarray(wide[3].transpose(2, 0, 1)).transpose(1, 2, 0)
    calls = []
    monkeypatch.setattr(composite._cblend, "over", lambda *a: calls.append(a))
    shape, alpha, alpha_b, alpha_previous, canvas_alpha = ones
    canvas, color_b, color, blended = wide
    args = (shape, alpha, alpha_b, color_b, color, blended, alpha_previous, canvas)
    out = _over(*args, canvas_alpha)
    assert not calls
    monkeypatch.undo()
    plain = _numpy(monkeypatch, _over, *args, canvas_alpha)
    _same(out.astype(np.float32), plain.astype(np.float32))
    if operand != "float64":
        with pytest.raises(ValueError, match="rows of pixels"):
            composite._cblend.over(*args, canvas_alpha)


@pytest.mark.parametrize(
    "filename",
    [
        "layer-edit.psd",
        "text.psd",
        "advanced-blending.psd",
        "passthrough_blendmodes1.psd",
        "passthrough_opacity.psd",
        "passthrough_fill_blendmode.psd",
        "gray1.psd",
        "blend-modes/cmyk-blend-modes.psd",
        "adjustments/adjustment_nested_composition_3.psd",
        "effects/styled-pass-through.psd",
        "effects/styled-pass-through-nested.psd",
        "effects/fading-mode-effects.psd",
        "effects/stroke-effects.psd",
    ],
)
@pytest.mark.parametrize("force", [True, False])
def test_a_render_is_numpys(
    monkeypatch: pytest.MonkeyPatch, filename: str, force: bool
) -> None:
    path = full_name(filename)
    psd = PSDImage.open(path)
    with np.errstate(all="ignore"):
        compiled = psd.composite(force=force, ignore_preview=True, apply_icc=False)
        with monkeypatch.context() as patch:
            patch.setattr(composite, "_cblend", None)
            plain = PSDImage.open(path).composite(
                force=force, ignore_preview=True, apply_icc=False
            )
    assert compiled.mode == plain.mode
    assert compiled.tobytes() == plain.tobytes()
