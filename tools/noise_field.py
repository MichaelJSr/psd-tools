"""Rebuild ``src/psd_tools/composite/noise_field.npz`` from Photoshop renders.

    PYTHONPATH=src python tools/noise_field.py <block.psd> <tall.psd> <wide.psd> [out.npz]

Each probe is an 8-bit RGB document over white holding one layer filled black
at 50% opacity, at fill 0, with a black Normal drop shadow at 100% of size,
distance and spread 0, noise 50% and "layer knocks out drop shadow" off: its
composite is the noise field itself, one byte a pixel. ``block`` is a
rectangle from the origin whose rows, under their keys, read every byte of
the table (1920 x 2400 does); ``tall`` is 64 columns by as many rows as are
to be stored; ``wide`` is as many columns as are to be stored by enough rows
to show every table row in each run of 2048 columns (256 do).

The table rows and keys are found in ``block``: two document rows that read
the same table row hold the same bytes in each block of 16 columns, in an
order and at a place that their two keys set. Without an output the stored
field is checked against the probes instead.
"""

from __future__ import annotations

import collections
import sys

import numpy as np

from psd_tools import PSDImage
from psd_tools.composite import effect_noise

RUN = 2048
ROWS = 64
BLOCK = 16
PROBE = 8


def field_of(path: str) -> np.ndarray:
    """A probe's composite as the field's bytes."""
    rgb = PSDImage.open(path).numpy()[..., :3].astype(np.float64)
    return np.rint((1.0 - rgb.mean(axis=2)) * 255.0).astype(np.uint8)


def _xor_between(block: np.ndarray, a: int, b: int, high: int) -> int | None:
    """The key that maps row ``a`` onto row ``b``, given its bits above the block."""
    xs = np.arange(block.shape[1])
    for low in range(BLOCK):
        moved = xs ^ (high * BLOCK + low)
        inside = moved < block.shape[1]
        if inside.sum() > RUN // 2 and np.array_equal(
            block[b, moved[inside]], block[a, xs[inside]]
        ):
            return high * BLOCK + low
    return None


def table_of(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The table, and each block row's table row and key."""
    height, width = block.shape
    width -= width % BLOCK
    sorted_blocks = np.sort(block[:, :width].reshape(height, -1, BLOCK), axis=2)
    seen: dict[bytes, list[tuple[int, int]]] = collections.defaultdict(list)
    for y in range(height):
        for s in range(width // BLOCK):
            seen[sorted_blocks[y, s].tobytes()].append((y, s))
    votes: collections.Counter[tuple[int, int, int]] = collections.Counter()
    for places in seen.values():
        if len(places) > 4 * ROWS:
            continue
        for i, (ya, sa) in enumerate(places):
            for yb, sb in places[i + 1 :]:
                votes[(ya, yb, sa ^ sb)] += 1
    links: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for (ya, yb, high), count in votes.items():
        if count >= 20:
            links[ya].append((yb, high))
            links[yb].append((ya, high))
    row = np.full(height, -1)
    key = np.zeros(height, dtype=np.intp)
    roots = 0
    for start in range(height):
        if row[start] >= 0:
            continue
        row[start], queue = roots, [start]
        roots += 1
        while queue:
            y = queue.pop()
            for z, high in links[y]:
                if row[z] >= 0:
                    continue
                xor = _xor_between(block, y, z, high)
                if xor is not None:
                    row[z], key[z] = row[y], key[y] ^ xor
                    queue.append(z)
    if roots != ROWS:
        raise ValueError("found %d table rows, not %d" % (roots, ROWS))
    table = np.full((ROWS, RUN), -1, dtype=np.int16)
    xs = np.arange(block.shape[1])
    for y in range(height):
        table[row[y], xs ^ key[y]] = block[y]
    if (table < 0).any():
        raise ValueError("the block leaves %d table bytes unread" % (table < 0).sum())
    return table.astype(np.uint8), np.stack([row, key], axis=1)


def _reader(table: np.ndarray) -> dict[bytes, tuple[int, int]]:
    """Each table row and key by the first bytes it reads."""
    xs = np.arange(PROBE)
    return {table[b, xs ^ k].tobytes(): (b, k) for b in range(ROWS) for k in range(RUN)}


def _read(
    table: np.ndarray, reader: dict[bytes, tuple[int, int]], run: np.ndarray
) -> tuple[int, int]:
    """The table row and key one run of a document row reads."""
    found = reader.get(run[:PROBE].tobytes())
    if found is None:
        raise ValueError("a run matches no table row")
    b, k = found
    if not np.array_equal(table[b, np.arange(len(run)) ^ k], run):
        raise ValueError("a run matches a table row only in part")
    return found


def build(
    block: np.ndarray, tall: np.ndarray, wide: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The table, every row's table row and key, and how each run moves them."""
    table, _ = table_of(block)
    reader = _reader(table)
    rows = np.array([_read(table, reader, tall[y, :RUN]) for y in range(len(tall))])
    runs = np.zeros((-(-wide.shape[1] // RUN), ROWS, 2), dtype=np.intp)
    runs[0, :, 0] = np.arange(ROWS)
    known = np.zeros(runs.shape[:2], dtype=bool)
    known[0] = True
    for y in range(len(wide)):
        b0, k0 = rows[y]
        for t in range(1, len(runs)):
            b, k = _read(table, reader, wide[y, t * RUN : (t + 1) * RUN])
            if known[t, b0] and tuple(runs[t, b0]) != (b, k ^ k0):
                raise ValueError("run %d moves table row %d two ways" % (t, b0))
            runs[t, b0], known[t, b0] = (b, k ^ k0), True
    if not known.all():
        raise ValueError("the wide probe leaves %d moves unread" % (~known).sum())
    return table, rows.astype(np.uint16), runs.astype(np.uint16)


def main(argv: list[str]) -> None:
    """Write the field built from the probes, or check the stored one against them."""
    block, tall, wide = (field_of(path) for path in argv[:3])
    if len(argv) > 3:
        table, rows, runs = build(block, tall, wide)
        np.savez_compressed(argv[3], table=table, rows=rows, tiles=runs)
        return
    for name, probe in (("block", block), ("tall", tall), ("wide", wide)):
        height, width = probe.shape
        same = np.array_equal(effect_noise.field((0, 0, width, height)), probe)
        print("%s %dx%d: %s" % (name, width, height, "equal" if same else "DIFFERS"))


if __name__ == "__main__":
    main(sys.argv[1:])
