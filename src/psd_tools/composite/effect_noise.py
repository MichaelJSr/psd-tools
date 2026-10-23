"""
The noise of layer effects.

Photoshop's noise is not random from render to render: the drop shadow, outer
glow, inner shadow and inner glow of every document read one fixed field of
bytes at each document pixel, whatever the effect or the layer. The field is a
table of 64 rows of 2048 bytes. A document row reads one table row, a run of
2048 columns at a time, its column index XORed with a key; each row has its
own table row and key, and each further run moves both. ``noise_field.npz``
holds the table, every row's table row and key, and how each run moves them,
read off Photoshop 2023 renders over the largest canvas a PSD holds. Past it
the field repeats.

An effect's coverage moves by the field's distance from 128, scaled by the
noise amount, and is held within ``[0, ceiling]``. Beneath the layer the
ceiling is ``min(1, gain * matte)``, where ``matte`` is the coverage the
effect falls off along before its range and contour, so the noise fades out
with the effect's faint edge; inside the layer it is 1 wherever the matte is
not 0. Where the contour takes the effect to nothing, the ceiling is 0.
"""

import functools
import itertools
import math
from importlib import resources

import numpy as np

# The table's row length: a document row reads it this many columns at a time.
_RUN = 2048

# Rows of the field built at once: it bounds the memory of the indices.
_STRIP = 128

# The gain on the matte that bounds a noisy coverage beneath the layer.
BENEATH_GAIN = 25.0 / 3.0


@functools.cache
def _tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The table, each row's table row and key, and each run's moves."""
    source = resources.files(__package__).joinpath("noise_field.npz")
    with source.open("rb") as stream, np.load(stream) as data:
        return (
            data["table"].astype(np.uint8),
            data["rows"].astype(np.intp),
            data["tiles"].astype(np.intp),
        )


def field(viewport: tuple[int, int, int, int]) -> np.ndarray:
    """The noise bytes over ``viewport``, ``(height, width)`` uint8."""
    left, top, right, bottom = viewport
    table, rows, runs = _tables()
    height, width = max(bottom - top, 0), max(right - left, 0)
    out = np.empty((height, width), dtype=np.uint8)
    if out.size == 0:
        return out
    xs = np.arange(left, left + width) % (len(runs) * _RUN)
    run, column = xs // _RUN, xs % _RUN
    edges = [0, *(np.flatnonzero(np.diff(run)) + 1).tolist(), width]
    for start in range(0, height, _STRIP):
        stop = min(start + _STRIP, height)
        ys = np.arange(top + start, top + stop) % len(rows)
        row, key = rows[ys, 0], rows[ys, 1]
        for first, last in itertools.pairwise(edges):
            moves = runs[run[first], row]
            index = column[None, first:last] ^ (key ^ moves[:, 1])[:, None]
            out[start:stop, first:last] = table[moves[:, 0, None], index]
    return out


def noise_amount(noise: float) -> int:
    """A noise percentage as the byte Photoshop scales the field by."""
    return math.floor((min(max(noise, 0.0), 100.0) * 255.0 + 50.0) / 100.0)


def noisy(
    coverage: np.ndarray,
    noise: float,
    viewport: tuple[int, int, int, int],
    matte: np.ndarray,
    inside: bool = False,
    lit: np.ndarray | None = None,
) -> np.ndarray:
    """``coverage`` moved by ``noise`` percent of the field.

    ``coverage`` lies on the document from ``viewport``'s top left corner.
    ``matte``, and ``lit`` when given, lie on the same canvas: the coverage
    is held under the ceiling the module describes, for an effect ``inside``
    the layer or beneath it, and at 0 where ``lit`` is false. No noise
    leaves ``coverage`` as it is.
    """
    amount = noise_amount(noise) if math.isfinite(noise) else 0
    if amount <= 0:
        return coverage
    left, top = viewport[0], viewport[1]
    height, width = coverage.shape
    moved = field((left, top, left + width, top + height)).astype(np.float32)
    moved -= np.float32(128.0)
    moved *= np.float32(amount / (128.0 * 255.0))
    moved += coverage
    np.clip(moved, 0.0, 1.0, out=moved)
    if inside:
        moved[matte <= 0.0] = 0.0
    else:
        np.minimum(moved, matte * np.float32(BENEATH_GAIN), out=moved)
    if lit is not None:
        moved[~lit] = 0.0
    return moved
