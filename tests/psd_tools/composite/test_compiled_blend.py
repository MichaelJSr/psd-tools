"""The compiled source arithmetic (``_blend.pyx``) is numpy's, bit for bit."""

import importlib
import itertools
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from psd_tools.api import numpy_io
from psd_tools.api.psd_image import PSDImage
from psd_tools.color_convert import SRGB_CUTOFF
from psd_tools.composite import adjustments, blend, paint, utils
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


def _bits(a: np.ndarray) -> bytes:
    """The bytes, every NaN one NaN: a NaN's sign is the CPU's (x86 sets it)."""
    return np.where(np.isnan(a), np.nan, a).tobytes()


def _close32(compiled: np.ndarray, plain: np.ndarray) -> None:
    """Float32 maths within 2 float32 ULPs (or one ULP at 1.0 near zero), NaNs where numpy's.

    numpy's own float32 paths differ in the last bit by CPU (AVX-512's vector maths on
    the Linux runner), and so do the signs of a NaN and a zero.
    """
    a, b = np.asarray(compiled, np.float64), np.asarray(plain, np.float64)
    assert a.shape == b.shape
    assert np.array_equal(np.isnan(a), np.isnan(b))
    keep = ~np.isnan(a)
    np.testing.assert_allclose(
        a[keep], b[keep], rtol=2.4e-7, atol=1.2e-7
    )  # 1 ULP at 1.0


def _same(compiled: np.ndarray, plain: np.ndarray) -> None:
    assert compiled.dtype == plain.dtype == np.float32
    assert compiled.shape == plain.shape
    assert _bits(compiled) == _bits(plain)


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
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_unmix_and_remix_are_numpys(
    monkeypatch: pytest.MonkeyPatch, seed: int, channels: int
) -> None:
    rng = np.random.default_rng(seed)
    color, backdrop_color = _canvas(rng, channels), _canvas(rng, channels)
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


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("gamma", [1.45, 1.0, 3.0, 0.5])
@pytest.mark.parametrize("isolated", [False, True])
def test_text_mix_is_numpys(
    monkeypatch: pytest.MonkeyPatch, seed: int, gamma: float, isolated: bool
) -> None:
    """A type layer's mix in the text space; a gamma ``**`` shortcuts stays numpy's."""
    rng = np.random.default_rng(seed)
    ones = [_canvas(rng, 1) for _ in range(8)]
    shape, alpha, alpha_previous, group_previous, alpha_b, now, alpha_g, alpha_0 = ones
    color, canvas, color_0 = (_canvas(rng, 3) for _ in range(3))
    # Pixels the layer leaves alone, some with nothing under them; empty rows.
    alone = rng.random(shape.shape[:2]) < 0.4
    shape[alone] = alpha[alone] = 0.0
    now[alone & (rng.random(alone.shape) < 0.3)] = 0.0
    alpha_previous[alone] = now[alone]
    group_previous[alone] = rng.choice(
        np.array([0, 1], np.float32), (int(alone.sum()), 1)
    )
    shape[:2] = 0.0
    compositor = SimpleNamespace(
        _color=canvas,
        _alpha=now,
        _alpha_g=alpha_g,
        _color_0=color_0,
        _alpha_0=alpha_0,
        _channels=3,
    )
    args: tuple[Any, ...] = (color, shape, alpha, alpha_previous, group_previous)
    args += (canvas, alpha_b, blend.normal, gamma, isolated)
    kernel, calls = composite._cblend.text_mix, []

    def spy(*args: object) -> None:
        calls.append(args)
        kernel(*args)

    monkeypatch.setattr(composite._cblend, "text_mix", spy)
    mix = composite.Compositor._mix_in_text_space
    _same(mix(compositor, *args), _numpy(monkeypatch, mix, compositor, *args))
    assert len(calls) == (gamma != 0.5)


class _Fit:
    _color_mode = ColorMode.RGB
    _alpha_only = False

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


class _AlphaOnly(_Fit):
    _alpha_only = True


@pytest.mark.parametrize("opacity, fill", [(1.0, 1.0), (0.6, 0.3), (0.0, 1.0)])
def test_an_alpha_only_effect_canvas_keeps_shape_and_alpha(
    opacity: float, fill: float
) -> None:
    """Without colour, the shape and alpha handed on are the compiled ones, bit for bit."""
    rng = np.random.default_rng(23)
    shape = _canvas(rng, 1)
    source = _Source(
        color=_canvas(rng, 3),
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
    modes = (BlendMode.NORMAL, BlendMode.COLOR_DODGE, BlendMode.NORMAL)
    paints = [(_canvas(rng, 1), _canvas(rng, 3), m, 0.75) for m in modes]

    def paint(fit: _Fit) -> tuple[np.ndarray, ...]:
        canvas = _EffectCanvas(fit, source)  # type: ignore[arg-type]
        for paint in paints:
            canvas.over(*paint)
        canvas.replace(paints[0][0], paints[0][1], 0.5)
        result = canvas.source()
        return result.shape, result.alpha

    for got, want in zip(paint(_AlphaOnly()), paint(_Fit()), strict=True):
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
        "adjustments/curves_rgb.psd",
        "fill_adjustments.psd",
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


_NEWEST = (
    "divide",
    "clip",
    "effect_start",
    "effect_over",
    "effect_source",
    "vibrance_step",
    "text_mix",
    "distinct_pixels",
    "mix_toward",
    "gather_rows",
    "whole_levels",
)


class _Stale:
    """A ``_blend`` built before its newest loops."""

    def __init__(self, real: Any) -> None:
        self._real = real

    def __getattr__(self, name: str) -> Any:
        if name in _NEWEST:
            raise AttributeError(name)
        return getattr(self._real, name)


@pytest.mark.parametrize(
    "filename", ["effects/stroke-effects.psd", "effects/fading-mode-effects.psd"]
)
def test_a_stale_build_renders_as_before(
    monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    path = full_name(filename)
    kw: dict[str, Any] = {"force": True, "ignore_preview": True, "apply_icc": False}
    with np.errstate(all="ignore"):
        fresh = PSDImage.open(path).composite(**kw)
        for module in (composite, utils, adjustments):
            monkeypatch.setattr(module, "_cblend", _Stale(module._cblend))
        stale = PSDImage.open(path).composite(**kw)
    assert stale.tobytes() == fresh.tobytes()


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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("amount", [-1.0, -0.6, -0.25, 0.1, 0.25, 0.37, 0.5, 0.75, 1.0])
def test_vibrance_step_is_numpys(
    monkeypatch: pytest.MonkeyPatch, dtype: type[np.floating], amount: float
) -> None:
    """The 8-bit cube in linear light and every triple of the specials."""
    levels = paint._srgb_to_linear(np.arange(0, 256, 5, dtype=np.float32) / 255)
    cube = np.stack(np.meshgrid(levels, levels, levels, indexing="ij"), -1)
    cutoff = np.float32(SRGB_CUTOFF)
    specials = np.r_[_SPECIAL, -np.inf, np.nextafter(cutoff, [0, 1], dtype=np.float32)]
    triples = np.array(list(itertools.product(specials, repeat=3)))
    linear = np.concatenate([cube.reshape(-1, 3), triples]).astype(dtype)[:, None]
    kernel, calls = adjustments._cblend.vibrance_step, []

    def spy(*args: object) -> np.ndarray:
        calls.append(args)
        return kernel(*args)

    monkeypatch.setattr(adjustments._cblend, "vibrance_step", spy)
    for view in (linear, linear[::2]):
        compiled = adjustments._vibrance_step(view, amount)
        with monkeypatch.context() as patch:
            patch.setattr(adjustments, "_cblend", None)
            plain = adjustments._vibrance_step(view, amount)
        assert compiled.dtype == plain.dtype == np.float64
        if dtype == np.float32:
            _close32(compiled, plain)
        else:  # + 0.0: on x86 numpy's max and min give a zero either sign.
            assert _bits(compiled + 0.0) == _bits(plain + 0.0)
    assert len(calls) == 1, "a strided view stays with numpy"


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_mix_toward_is_numpys(monkeypatch: pytest.MonkeyPatch, channels: int) -> None:
    """An adjustment's strength, per pixel or one, over the specials and a box view."""
    kernel, calls = composite._cblend.mix_toward, []

    def spy(*args: object) -> np.ndarray:
        calls.append(args)
        return kernel(*args)

    monkeypatch.setattr(composite._cblend, "mix_toward", spy)
    rng = np.random.default_rng(channels + 7)
    content = rng.choice(_SPECIAL, size=(60, 70, channels)).astype(np.float32)
    blended = rng.choice(_SPECIAL, size=(60, 70, channels)).astype(np.float32)
    mask = rng.choice(_SPECIAL, size=(60, 70, 1)).astype(np.float32)
    for strength in (mask, mask[3:-2, 1:-4], 1.0, 0.3137254901960784, 0.0):
        box = (slice(None), slice(None))
        if isinstance(strength, np.ndarray) and strength.shape != mask.shape:
            box = (slice(3, -2), slice(1, -4))
        c, b = content[box], blended[box]
        got = composite._mix_toward(c, b, strength)
        expected = utils.clip(c + strength * (b - c))
        assert got.dtype == expected.dtype
        _close32(got, expected)
        # Whole levels where the strength is above 0.
        levels = np.rint(np.clip(expected, 0.0, 1.0) * np.float32(255)) / 255
        expected = np.where(np.asarray(strength) > 0, levels, expected)
        _close32(composite._mix_toward(c, b, strength, True), expected)
    assert len(calls) == 10


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_distinct_pixels_number_equal_bits(channels: int) -> None:
    """First seen first, -0.0 apart from 0.0, None past the limit."""
    rng = np.random.default_rng(channels)
    palette = rng.choice(_SPECIAL, size=(40, channels))
    pixels = palette[rng.integers(0, 40, 5000)]
    for rows in (pixels, np.repeat(pixels, rng.integers(1, 9, len(pixels)), axis=0)):
        bits = rows.view(np.uint32)
        first = np.unique(bits, axis=0, return_index=True)[1]
        index, found = composite._cblend.distinct_pixels(rows, len(first))
        assert np.array_equal(found, np.sort(first))
        assert np.array_equal(bits[found][index], bits)
        assert np.array_equal(
            composite._gather(rows[found], index).view(np.uint32), bits
        )
        assert composite._cblend.distinct_pixels(rows, len(first) - 1) is None


def test_a_pointwise_adjustment_runs_once_per_colour() -> None:
    """Hue ranges, a profile's space, black & white, brightness: bit for bit."""
    names = [
        "huesaturation",
        "vibrance_display",
        "vibrance_display_saturation",
        "brightnesscontrast",
        "brightnesscontrast_legacy",
    ]
    layers = [
        layer
        for name in names
        for layer in PSDImage.open(full_name(f"adjustments/{name}_rgb.psd"))
        if layer.kind in composite._POINTWISE
    ]
    layers.append(PSDImage.open(full_name("fill_adjustments.psd"))[11])
    rng = np.random.default_rng(0)
    palette = rng.random((50, 3), dtype=np.float32)
    palette[:2] = [[0.0, 0.5, 1.0], [-0.0, 0.5, 1.0]]
    image = palette[rng.integers(0, 50, (37, 41))]
    for layer in layers:
        fn, sizes = adjustments.ADJUSTMENT_FUNC[layer.kind], []

        def spy(img: np.ndarray, *args: Any) -> Any:
            sizes.append(img.shape)
            return fn(img, *args)

        out = composite._adjust(spy, layer, image, ColorMode.RGB)
        plain = fn(image, ColorMode.RGB, layer)
        assert sizes == [(1, 50, 3)], layer.name
        assert out is not None and plain is not None
        # 8-bit documents: whole levels.
        plain = composite._held_levels(layer, plain)
        assert out.dtype == plain.dtype
        assert out.tobytes() == plain.tobytes(), layer.name
