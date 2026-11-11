"""Check that a render does not hang on the last bit of a libm result.

    PYTHONPATH=src python tools/ulp_nudge.py [--f64] [--sites] [--jobs N] <root> <card>...

CPUs differ in exactly those bits (SVML under AVX-512, glibc, Apple's libm).
An interposed libm moves every float32 result (``--f64``: float64) of the
powers, exponentials, logarithms, cube roots and trigonometry one last bit
up, then down; the 8-bit render must not change. It reaches numpy's ufuncs,
the ``**`` operator and the compiled modules alike, where numpy calls libm:
arm64, or x86 without AVX-512. One JSON line per card; exit 1 if one moved.
``--sites`` adds the Python functions whose own code made the calls (a
compiled kernel counts for its caller). macOS only (``DYLD_INSERT_LIBRARIES``),
with ``cc`` on PATH.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

_FLOAT = "powf expf exp2f expm1f logf log2f log10f log1pf cbrtf sinf cosf tanf atanf"
_DOUBLE = "pow exp exp2 expm1 log log2 log10 log1p cbrt sin cos tan atan"
_TWO = {"powf", "atan2f", "hypotf", "pow", "atan2", "hypot"}
_NAMES = (_FLOAT + " atan2f hypotf").split() + (_DOUBLE + " atan2 hypot").split()


def _source() -> str:
    """The interposer: each function calls libm's, then moves the result."""
    lines = [
        "#include <math.h>",
        "static int d32, d64;",
        "long n32, n64;",
        "void ulp_nudge_set(int f32, int f64) { d32 = f32; d64 = f64; }",
        "static float nf(float r) {",
        "  if (!isfinite(r) || r == 0.0f || r == 1.0f) return r;",
        "  __atomic_fetch_add(&n32, 1, __ATOMIC_RELAXED);",
        "  return d32 ? nextafterf(r, d32 > 0 ? INFINITY : -INFINITY) : r;",
        "}",
        "static double nd(double r) {",
        "  if (!isfinite(r) || r == 0.0 || r == 1.0) return r;",
        "  __atomic_fetch_add(&n64, 1, __ATOMIC_RELAXED);",
        "  return d64 ? nextafter(r, d64 > 0 ? INFINITY : -INFINITY) : r;",
        "}",
    ]
    for name in _NAMES:
        t, n = ("float", "nf") if name.endswith("f") else ("double", "nd")
        if name in _TWO:
            args, call = f"{t} x, {t} y", "x, y"
        else:
            args, call = f"{t} x", "x"
        lines.append(f"static {t} my_{name}({args}) {{ return {n}({name}({call})); }}")
    pairs = " ".join(f"{{(const void *)my_{n}, (const void *)&{n}}}," for n in _NAMES)
    lines += [
        "__attribute__((used)) static const struct { const void *with, *what; }",
        '  interpose[] __attribute__((section("__DATA,__interpose"))) = {',
        f"  {pairs}",
        "};",
    ]
    return "\n".join(lines) + "\n"


def build() -> Path:
    """The interposer, compiled once per source into the temp folder."""
    source = _source()
    key = hashlib.sha256(source.encode()).hexdigest()[:12]
    lib = Path(tempfile.gettempdir()) / f"ulp_nudge_{key}.dylib"
    if not lib.exists():
        with tempfile.TemporaryDirectory() as tmp:
            c = Path(tmp) / "nudge.c"
            c.write_text(source)
            out = Path(tmp) / lib.name
            cmd = ["cc", "-O2", "-fno-builtin", "-dynamiclib", "-o", str(out), str(c)]
            subprocess.run(cmd, check=True)
            os.replace(out, lib)
    return lib


def render(path: str | os.PathLike[str]) -> np.ndarray:
    """The card as AutoCroissant renders it, 8 bits."""
    from psd_tools import PSDImage  # noqa: PLC0415

    psd = PSDImage.open(path)
    return np.asarray(psd.composite(ignore_preview=True, apply_icc=False))


def _moved(a: np.ndarray, b: np.ndarray) -> dict[str, int]:
    if a.shape != b.shape:
        return {"px": -1, "max": -1}
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    diff = diff.max(axis=-1) if diff.ndim == 3 else diff
    return {"px": int((diff > 0).sum()), "max": int(diff.max(initial=0))}


def _profiled(count: ctypes.c_long, own: Counter[str]) -> Any:
    """A profile hook adding each call's delta to the function running it."""
    stack: list[str] = []
    last = [count.value]

    def hook(frame: Any, event: str, arg: Any) -> None:
        value = count.value
        if stack and value != last[0]:
            own[stack[-1]] += value - last[0]
        last[0] = value
        if event == "call":
            code = frame.f_code
            stack.append(f"{Path(code.co_filename).name}:{code.co_qualname}")
        elif event == "return" and stack:
            stack.pop()

    return hook


def _child(paths: Iterable[str], f64: bool, sites: bool) -> Iterator[dict[str, Any]]:
    """Runs under the interposer: base, nudged up, nudged down."""
    lib = ctypes.CDLL(os.environ["ULP_NUDGE_LIB"])
    n32, n64 = (ctypes.c_long.in_dll(lib, name) for name in ("n32", "n64"))
    count = n64 if f64 else n32
    for path in paths:
        row: dict[str, Any] = {"path": path}
        try:
            lib.ulp_nudge_set(0, 0)
            start, other, own = count.value, (n32 if f64 else n64).value, Counter[str]()
            if sites:
                sys.setprofile(_profiled(count, own))
            try:
                base = render(path)
            finally:
                sys.setprofile(None)
            row["calls"] = count.value - start
            row["f32" if f64 else "f64"] = (n32 if f64 else n64).value - other
            if sites:
                row["sites"] = dict(own.most_common(8))
            if row["calls"]:
                for name, step in (("up", 1), ("down", -1)):
                    lib.ulp_nudge_set(0 if f64 else step, step if f64 else 0)
                    row[name] = _moved(base, render(path))
                    lib.ulp_nudge_set(0, 0)
        except Exception as error:
            row["error"] = repr(error)[:300]
        yield row


def check(
    paths: Iterable[str | os.PathLike[str]],
    f64: bool = False,
    jobs: int = 1,
    sites: bool = False,
) -> Iterator[dict[str, Any]]:
    """Per path ``calls`` nudged, the other width's, ``up`` and ``down`` moves."""
    if sys.platform != "darwin":
        raise RuntimeError("the interposer is macOS's DYLD_INSERT_LIBRARIES")
    lib = build()
    env = dict(os.environ, DYLD_INSERT_LIBRARIES=str(lib), ULP_NUDGE_LIB=str(lib))
    items = [str(p) for p in paths]
    size = max(1, min(8, -(-len(items) // max(jobs, 1))))
    chunks = [items[i : i + size] for i in range(0, len(items), size)]

    def one(chunk: list[str]) -> list[dict[str, Any]]:
        flags = ["--f64"] * f64 + ["--sites"] * sites
        cmd = [sys.executable, __file__, "--child", *flags]
        done = subprocess.run(
            [*cmd, *chunk], env=env, capture_output=True, text=True, check=False
        )
        rows = [json.loads(line) for line in done.stdout.splitlines() if line]
        seen = {row["path"] for row in rows}
        tail = done.stderr[-300:]
        return rows + [{"path": p, "error": tail} for p in chunk if p not in seen]

    with ThreadPoolExecutor(max(jobs, 1)) as pool:
        for rows in pool.map(one, chunks):
            yield from rows


def moved(row: dict[str, Any]) -> bool:
    """Whether a row's render moved under the nudge, or failed."""
    return "error" in row or any(row.get(k, {}).get("px") for k in ("up", "down"))


def main() -> int:
    """The command line; see the module docstring."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--f64", action="store_true", help="nudge float64 results")
    parser.add_argument("--sites", action="store_true", help="name the callers")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("root")
    parser.add_argument("cards", nargs="*")
    args = parser.parse_args()
    if args.child:
        for row in _child([args.root, *args.cards], args.f64, args.sites):
            print(json.dumps(row), flush=True)
        return 0
    paths = [os.path.join(args.root, card) for card in args.cards] or [args.root]
    bad = 0
    for row in check(paths, args.f64, args.jobs, args.sites):
        bad += moved(row)
        print(json.dumps(row), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
