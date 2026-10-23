"""Fit one compositor constant to Photoshop's own renders.

    python tools/oracle/fit.py <module>:<NAME> <lo>:<hi>:<step> <case> [<case> ...]

A case is ``path.psd`` or ``path.psd@left,top,right,bottom``. Its truth is the
composite Photoshop stored in the file (saved with Maximize Compatibility), so
any Photoshop-saved PSD is a case: the oracle loop's outputs, the fixtures in
tests/psd_files, or a corpus card. Prints the mean squared error per value and
case, then the best value; the cases are re-read for every value.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np

from psd_tools import PSDImage
from psd_tools.composite import composite


def _case(spec: str) -> tuple[str, tuple[int, int, int, int] | None]:
    path, _, box = spec.partition("@")
    region = tuple(int(v) for v in box.split(",")) if box else None
    return path, region  # type: ignore[return-value]


def error(path: str, region: tuple[int, int, int, int] | None) -> float:
    psd = PSDImage.open(path)
    truth = psd.numpy()
    with np.errstate(all="ignore"):
        color, _, alpha = composite(psd)
    result = color
    if truth.shape[2] > color.shape[2]:
        result = np.concatenate((color, alpha), axis=2)
    if region is not None:
        left, top, right, bottom = region
        truth, result = truth[top:bottom, left:right], result[top:bottom, left:right]
    return float(np.mean((truth - result) ** 2))


def main(argv: list[str]) -> None:
    target, grid, *specs = argv
    module_name, _, name = target.partition(":")
    module = importlib.import_module(module_name)
    lo, hi, step = (float(v) for v in grid.split(":"))
    values = np.arange(lo, hi + step / 2, step)
    cases = [_case(s) for s in specs]
    original = getattr(module, name)
    table = []
    try:
        for value in values:
            setattr(module, name, type(original)(value))
            errors = [error(path, region) for path, region in cases]
            table.append((float(value), errors))
            print(
                f"{value:8.4f}  mean {np.mean(errors):.3e}  "
                + "  ".join(f"{e:.3e}" for e in errors)
            )
    finally:
        setattr(module, name, original)
    best = min(table, key=lambda row: np.mean(row[1]))
    print(
        f"best {name} = {best[0]:.4f} (now {original}); mean mse {np.mean(best[1]):.3e}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
