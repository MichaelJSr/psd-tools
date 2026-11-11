"""Rebuild ``src/psd_tools/composite/brightnesscontrast_tables.npz`` from Photoshop.

    python tools/brightnesscontrast_tables.py <probe dir> [out.npz]

Each probe is Photoshop's PNG of an 8-bit RGB ramp, 256 x 2 (row 0 gray,
row 1 ``(k, 7k, 101k + 50)`` mod 256), under one modern Brightness/Contrast
layer, named ``<brightness>_<contrast>.png``: every brightness with contrast
0, every contrast with brightness 0, and any pairs. Row 1 checks that each
channel goes through the same table. Without an output the stored tables are
checked against the probes instead, and a pair against contrast after
brightness.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

from psd_tools.composite import adjustments

LEVELS = np.arange(256)
RAMP = np.stack([LEVELS, (7 * LEVELS) % 256, (101 * LEVELS + 50) % 256], axis=1)


def read_probes(folder: str) -> dict[tuple[int, int], np.ndarray]:
    """Each probe's table by (brightness, contrast)."""
    probes = {}
    for name in sorted(os.listdir(folder)):
        stem, ext = os.path.splitext(name)
        if ext != ".png":
            continue
        brightness, contrast = map(int, stem.split("_"))
        pixels = np.asarray(Image.open(os.path.join(folder, name)).convert("RGB"))
        table = pixels[0, :, 0]
        assert (pixels[0] == table[:, None]).all(), name
        assert (pixels[1] == table[RAMP]).all(), name
        probes[(brightness, contrast)] = table.astype(np.uint8)
    return probes


def build(probes: dict[tuple[int, int], np.ndarray]) -> dict[str, np.ndarray]:
    """The stored arrays."""
    brightness = np.stack([probes[(b, 0)] for b in range(-150, 151)])
    contrast = np.stack([probes[(0, c)] for c in range(-50, 101)])
    pairs = sorted(key for key in probes if key[0] and key[1])
    return {
        "brightness": brightness,
        "contrast": contrast,
        "pairs": np.array(pairs, dtype=np.int16).reshape(-1, 2),
        "pair_tables": np.stack([probes[key] for key in pairs]),
    }


def check(probes: dict[tuple[int, int], np.ndarray]) -> None:
    """Fail on a probe the stored tables do not give."""
    bright, contrast, _ = adjustments._brightnesscontrast_tables()
    worst = 0
    for (b, c), table in probes.items():
        stored = adjustments._brightnesscontrast_levels(b, c)
        assert stored is not None and (stored == table).all(), (b, c)
        composed = contrast[c + 50][bright[b + 150]].astype(int)
        worst = max(worst, int(np.abs(composed - table).max()))
    print(f"{len(probes)} probes match; contrast after brightness is off by {worst}")


def main() -> None:
    """Write the tables, or check the stored ones."""
    found = read_probes(sys.argv[1])
    if len(sys.argv) > 2:
        tables = build(found)
        np.savez_compressed(
            sys.argv[2],
            brightness=tables["brightness"],
            contrast=tables["contrast"],
            pairs=tables["pairs"],
            pair_tables=tables["pair_tables"],
        )
    else:
        check(found)


if __name__ == "__main__":
    main()
