"""The compiled source arithmetic (``_blend.pyx``) is numpy's, bit for bit."""

import importlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from psd_tools.api import numpy_io
from psd_tools.api.psd_image import PSDImage
from psd_tools.composite import adjustments, utils
from psd_tools.composite.composite import _EffectCanvas, _over, _remix, _Source, _unmix
from psd_tools.constants import BlendMode, ColorMode, Knockout

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
    monkeypatch: pytest.MonkeyPatch, fn: Callable[..., Any], *args: object
) -> Any:
    with monkeypatch.context() as patch:
        patch.setattr(composite, "_cblend", None)
        patch.setattr(utils, "_cblend", None)
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


@pytest.mark.parametrize("channels", [1, 3])
def test_divide_and_clip_are_numpys(
    monkeypatch: pytest.MonkeyPatch, channels: int
) -> None:
    rng = np.random.default_rng(13)
    a, fill, b = _canvas(rng, channels), _canvas(rng, channels), _canvas(rng, 1)
    assert utils.in_rows((b,), (a, fill))
    for f in (0.0, 1.0, 0.1, fill):
        _same(utils.divide(a, b, f), _numpy(monkeypatch, utils.divide, a, b, f))
    _same(utils.clip(a), _numpy(monkeypatch, utils.clip, a))


class _Fit:
    _color_mode = ColorMode.RGB

    def _fit_source(self, color: np.ndarray) -> np.ndarray:
        return color


@pytest.mark.parametrize(
    "opacity, fill", [(1.0, 1.0), (0.6, 0.3), (0.0, 1.0), (1.0, 0)]
)
@pytest.mark.parametrize("channels", [1, 3])
def test_effect_canvas_is_numpys(
    monkeypatch: pytest.MonkeyPatch, opacity: float, fill: float, channels: int
) -> None:
    """Normal overs, one Multiply between them, and the source handed on."""
    rng = np.random.default_rng(17)
    shape = _canvas(rng, 1)
    source = _Source(
        color=_canvas(rng, channels),
        shape=shape,
        alpha=_canvas(rng, 1),
        mask=1.0,
        shape_mask=1.0,
        fill_opacity=fill,
        opacity=opacity,
        knockout=Knockout.NONE,
        adjustment_isolated=None,
        knockout_shape=shape,
        region=_canvas(rng, 1),
    )
    modes = (BlendMode.NORMAL, BlendMode.MULTIPLY, BlendMode.NORMAL)
    paints = [(_canvas(rng, 1), _canvas(rng, channels), m, 0.75) for m in modes]

    def paint() -> tuple[np.ndarray, ...]:
        canvas = _EffectCanvas(_Fit(), source)  # type: ignore[arg-type]
        for paint in paints:
            canvas.over(*paint)
        result = canvas.source()
        return result.color, result.shape, result.alpha

    for got, want in zip(paint(), _numpy(monkeypatch, paint), strict=True):
        _same(got, want)


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
            patch.setattr(utils, "_cblend", None)
            patch.setattr(adjustments, "_cblend", None)
            patch.setattr(numpy_io, "_cnumpy_io", None)
            plain = PSDImage.open(path).composite(
                force=force, ignore_preview=True, apply_icc=False
            )
    assert compiled.mode == plain.mode
    assert compiled.tobytes() == plain.tobytes()


@pytest.mark.parametrize("size", [2, 256, 4097, 65536])
def test_lut_levels_are_numpys(monkeypatch: pytest.MonkeyPatch, size: int) -> None:
    """Every level's edge and its float32 neighbours, the specials, strided views."""
    rng = np.random.default_rng(size)
    lut = rng.random(size, dtype=np.float32)
    edges = (np.arange(size, dtype=np.float32) + np.float32(0.5)) / np.float32(size - 1)
    values = np.concatenate(
        [
            np.linspace(0, 1, 300_001, dtype=np.float32),
            edges,
            np.nextafter(edges, np.float32(2)),
            np.nextafter(edges, np.float32(-1)),
            np.tile(_SPECIAL, 3),
            -_SPECIAL,
        ]
    )
    values = values[: values.size // 6 * 6].reshape(-1, 2, 3)
    for v in (values, values[:, :, 1], values[::2, :, 1:]):
        compiled = adjustments._apply_lut(v, lut)
        with monkeypatch.context() as patch:
            patch.setattr(adjustments, "_cblend", None)
            _same(compiled, adjustments._apply_lut(v, lut))


def test_bytes_are_numpys() -> None:
    """composite_pil's bytes: every level's edge and its neighbours, the specials."""
    edges = (np.arange(256, dtype=np.float32) + np.float32(0.5)) / np.float32(255)
    values = np.concatenate(
        [
            np.linspace(-0.01, 1.01, 100_001, dtype=np.float32),
            edges,
            np.nextafter(edges, np.float32(2)),
            np.nextafter(edges, np.float32(-1)),
            _SPECIAL,
            -_SPECIAL,
        ]
    )
    values = values[: values.size // 16 * 16].reshape(-1, 4, 4)
    rgb, alpha = values[..., :3], values[..., 3:]
    packed = np.ascontiguousarray(rgb), np.ascontiguousarray(alpha)
    for color, plane in (packed, (rgb, alpha), (rgb, None), (rgb[..., :1], alpha)):
        stacked = color if plane is None else np.concatenate((color, plane), 2)
        plain = 255 * stacked
        np.clip(plain, 0, 255, out=plain)
        np.rint(plain, out=plain)
        assert (
            composite._cblend.to_bytes(color, plane).tobytes()
            == plain.astype(np.uint8).tobytes()
        )
