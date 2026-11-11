"""The same render on every CPU.

libm results (SVML under AVX-512, glibc, Apple's) and BLAS sums differ in
the last bits between CPUs. Non-integer powers go through ``paint._power``
(float32 worked in float64, rounded once), three-channel sums through
``paint._dot3``; ``tools/ulp_nudge.py`` checks the renders.
"""

import ast
import importlib.util
import math
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import psd_tools
from psd_tools.color_convert import SRGB_CUTOFF, SRGB_GAMMA, SRGB_OFFSET, SRGB_SLOPE
from psd_tools.composite import paint

from ..utils import full_name

_TOOL = Path(__file__).resolve().parents[3] / "tools" / "ulp_nudge.py"
_SRC = Path(psd_tools.__file__).parent

_POWERS = {"power", "float_power", "exp", "exp2", "expm1", "log", "log2", "log10"}
_POWERS |= {"log1p", "cbrt"}
_PRODUCTS = {"dot", "matmul", "tensordot", "einsum", "inner", "vdot"}

# Each site's reason: none is a float32 pixel array.
_ALLOWED = {
    # the helpers themselves
    ("composite/paint.py", "_powf", "np.power"): 1,
    ("composite/paint.py", "_power", "**"): 1,
    # scalars
    ("composite/adjustments.py", "_get_lut_size", "**"): 1,
    ("composite/adjustments.py", "_levels_gamma", "**"): 3,
    ("composite/adjustments.py", "_colorbalance_midtones", "**"): 3,
    # float64 tables
    ("composite/adjustments.py", "_brightness_curve", "**"): 1,
    ("composite/adjustments.py", "_brightness_curve", "np.exp"): 2,
    ("composite/paint.py", "_srgb_to_oklab", "np.cbrt"): 1,
    # 3 by 3 and 2 by 2 matrices, no pixels
    ("composite/adjustments.py", "_vibrance_space", "@"): 3,
    ("composite/text/raster.py", "_resample", "@"): 2,
    # float64 picture resample (BLAS; its sums are owed a fixed order)
    ("composite/text/svg.py", "picture_layer", "np.matmul"): 1,
    ("composite/text/svg.py", "picture_layer", "np.tensordot"): 1,
}


def _integer(node: ast.expr) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        node = node.operand
    return isinstance(node, ast.Constant) and type(node.value) is int


def _kind(node: ast.AST) -> str | None:
    """A CPU-dependent operation: a non-integer power, a libm ufunc, BLAS."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        return None if _integer(node.right) else "**"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
        return "@"
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return None
    name, owner = node.func.attr, node.func.value
    if not (isinstance(owner, ast.Name) and owner.id == "np"):
        return ".dot" if name == "dot" else None
    if name in _PRODUCTS:
        return f"np.{name}"
    exact = name == "power" and len(node.args) > 1 and _integer(node.args[1])
    return f"np.{name}" if name in _POWERS and not exact else None


def _sites() -> Counter[tuple[str, str, str]]:
    found: Counter[tuple[str, str, str]] = Counter()

    def visit(node: ast.AST, path: str, scope: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = f"{scope}.{node.name}" if scope else node.name
        kind = _kind(node)
        if kind:
            found[path, scope or "<module>", kind] += 1
        for child in ast.iter_child_nodes(node):
            visit(child, path, scope)

    for file in sorted((_SRC / "composite").rglob("*.py")):
        visit(ast.parse(file.read_text()), file.relative_to(_SRC).as_posix(), "")
    return found


def test_compositor_math_is_the_same_on_every_cpu() -> None:
    """No float32 libm power or BLAS sum on pixels: ``_power`` and ``_dot3``."""
    assert dict(_sites()) == _ALLOWED


def test_float32_powers_round_once() -> None:
    """float32 in, float32 out, the power correctly rounded; float64 as ``**``."""
    levels = np.arange(256, dtype=np.float32) / np.float32(255)
    linear = paint._srgb_to_linear(levels)
    assert linear.dtype == np.float32
    base = (levels + SRGB_OFFSET) / np.float32(1 + SRGB_OFFSET)
    gamma = float(np.float32(SRGB_GAMMA))
    expected = [np.float32(math.pow(float(b), gamma)) for b in base.tolist()]
    bright = levels > SRGB_SLOPE * SRGB_CUTOFF
    assert np.array_equal(linear[bright], np.array(expected, np.float32)[bright])
    wide = levels.astype(np.float64)
    assert np.array_equal(paint._power(wide, 1 / 2.4), wide ** (1 / 2.4))


@pytest.mark.skipif(sys.platform != "darwin", reason="DYLD_INSERT_LIBRARIES")
@pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler")
def test_renders_hold_under_a_nudged_libm() -> None:
    """Every fixed site reached: no float32 libm result, float64's last bit moot."""
    spec = importlib.util.spec_from_file_location("ulp_nudge", _TOOL)
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    names = [
        "adjustments/levels_gamma_rgb.psd",
        "adjustments/colorbalance_vibrance_rgb.psd",
        "adjustments/huesaturation_saturation_rgb.psd",
        "adjustments/vibrance_saturation_rgb.psd",
        "effects/effect-enums.psd",
    ]
    rows = list(tool.check([full_name(name) for name in names], f64=True))
    assert [(r.get("error"), r["f32"]) for r in rows] == [(None, 0)] * len(names)
    assert all(r["calls"] for r in rows)
    assert not any(tool.moved(r) for r in rows)
