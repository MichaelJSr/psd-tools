"""Rebuild ``src/psd_tools/composite/vibrance_tables.npz`` from Photoshop renders.

    python tools/vibrance_tables.py <probe dir> [out.npz]

Each probe is a 16-bit RGB document without a profile, 4096 x 4096, holding
every 8-bit colour once (pixel ``i`` is ``(i >> 16, (i >> 8) & 255, i & 255)``)
under one vibrance layer at saturation 0, saved flattened as an uncompressed
little-endian TIFF named ``vibrance_<amount>.tif``, for the amounts +-25, +-50,
+-75 and +-100. Photoshop's 8-bit result is the 16-bit one rounded.

In linear light vibrance keeps each pixel's hue: its brightest channel becomes
``H'`` and every channel ``H' - k (H - c)``. Off the red hues ``H'`` and ``k``
depend on the brightest and darkest levels alone; a pixel whose red is the
brightest and whose other two are equal shows the spared skin tones. The
tables hold, at each encoded value and linear saturation, ``k - 1`` at each
amount, ``H' / H - 1`` at +-100 (it is linear in the amount), and for the
spared reds the amount whose unspared ``k`` they get, as a share of the
amount, and their share of the unspared ``H' - H``. Without an output the
stored tables are checked against the probes instead.
"""

from __future__ import annotations

import struct
import sys

import numpy as np

from psd_tools.composite import adjustments

AMOUNTS = (25, 50, 75, 100)
# Where the extrapolation past a saturation of 1 starts its fit.
REACH = 0.85
LEVELS = np.arange(256)


def _linear(value: np.ndarray) -> np.ndarray:
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def read_probe(path: str) -> np.ndarray:
    """A probe as linear light, ``(256, 256, 256, 3)`` by input red, green, blue."""
    with open(path, "rb") as stream:
        data = stream.read()
    assert data[:4] == b"II*\x00", "an uncompressed little-endian TIFF"
    (ifd,) = struct.unpack_from("<I", data, 4)
    (count,) = struct.unpack_from("<H", data, ifd)
    tags = {}
    for index in range(count):
        tag, kind, number, value = struct.unpack_from(
            "<HHII", data, ifd + 2 + 12 * index
        )
        tags[tag] = (kind, number, value)

    def values(tag: int) -> list[int]:
        kind, number, value = tags[tag]
        if number == 1:
            return [value]
        code = "I" if kind == 4 else "H"
        return list(struct.unpack_from(f"<{number}{code}", data, value))

    assert tags[259][2] == 1 and values(258) == [16, 16, 16]
    strips = zip(values(273), values(279), strict=True)
    raw = b"".join(data[start : start + size] for start, size in strips)
    pixels = np.frombuffer(raw, "<u2").astype(np.float64) / 65535.0
    return _linear(pixels.reshape(256, 256, 256, 3))


def lattice(probe: np.ndarray) -> dict[str, np.ndarray]:
    """``H'`` and the darkest channel by brightest and darkest input level."""
    high, low = np.meshgrid(LEVELS, LEVELS, indexing="ij")
    return {
        "value": probe[low, low, high, 2],
        "low": probe[low, low, high, 0],
        "red_value": probe[high, low, low, 0],
        "red_low": probe[high, low, low, 1],
    }


def nodes(field: np.ndarray, zero: float | None = None) -> np.ndarray:
    """``field[high, low]`` at the node values and saturations.

    Along each node value's row of levels the field is read linearly in
    saturation; a gray reads ``zero``, or the row's own trend when None. Past a
    saturation of 1 each row follows a quadratic through its nodes from
    ``REACH``.
    """
    linear = _linear(LEVELS / 255.0)
    sats = adjustments._VIBRANCE_SATURATIONS
    inside = sats <= 1
    out = np.zeros((len(adjustments._VIBRANCE_VALUES), len(sats)))
    for row, value in enumerate(adjustments._VIBRANCE_VALUES):
        high = int(value)
        if high == 0:
            continue
        low = np.arange(high)
        saturation = (linear[high] - linear[low]) / linear[high]
        order = np.argsort(saturation)
        saturation, values = saturation[order], field[high, low][order]
        if zero is not None:
            gray = zero
        elif len(saturation) > 1:
            slope = (values[1] - values[0]) / (saturation[1] - saturation[0])
            gray = values[0] - slope * saturation[0]
        else:
            gray = values[0]
        out[row, inside] = np.interp(
            sats[inside], np.r_[0.0, saturation], np.r_[gray, values]
        )
    out[0] = out[1]
    fit = inside & (sats >= REACH)
    curve = np.polyfit(sats[fit], out[:, fit].T, 2)
    out[:, ~inside] = (
        np.outer(sats[~inside] ** 2, curve[0])
        + np.outer(sats[~inside], curve[1])
        + curve[2]
    ).T
    return out


def chroma(table: dict[str, np.ndarray], red: bool = False) -> np.ndarray:
    """``k`` by brightest and darkest level, of the spared reds if ``red``."""
    linear = _linear(LEVELS / 255.0)
    span = np.maximum(linear[:, None] - linear[None, :], 1e-12)
    if red:
        return (table["red_value"] - table["red_low"]) / span
    return (table["value"] - table["low"]) / span


def build(directory: str) -> dict[str, np.ndarray]:
    """The tables, by name, from the probes in ``directory``."""
    linear = _linear(LEVELS / 255.0)[:, None]
    tables = {
        sign: {
            a: lattice(read_probe(f"{directory}/vibrance_{sign * a}.tif"))
            for a in AMOUNTS
        }
        for sign in (1, -1)
    }
    out: dict[str, np.ndarray] = {}
    for name, sign in (("raise", 1), ("lower", -1)):
        out[f"{name}_chroma"] = np.stack(
            [nodes(chroma(tables[sign][a]) - 1) for a in AMOUNTS]
        )
        top = tables[sign][100]
        out[f"{name}_value"] = nodes(
            (top["value"] - linear) / np.maximum(linear, 1e-12), zero=0.0
        )
    # The spared reds: the amount (as a share) whose unspared chroma theirs is,
    # found along the unspared tables' interpolation, and their share of H' - H.
    unspared = np.stack([np.ones((256, 256))] + [chroma(tables[1][a]) for a in AMOUNTS])
    shares, values = [], []
    for a in AMOUNTS:
        target = chroma(tables[1][a], red=True)
        lo, hi = np.zeros(target.shape), np.full(target.shape, float(a))
        for _ in range(40):
            mid = (lo + hi) / 2
            weights, base = adjustments._vibrance_amount_weights(mid / 100)
            reached = sum(
                weight * np.take_along_axis(unspared, (base + j)[None], 0)[0]
                for j, weight in enumerate(weights)
            )
            below = reached < target
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        share = (lo + hi) / 2 / a
        flat = np.abs(chroma(tables[1][a]) - 1) < 1e-4
        shares.append(np.clip(nodes(np.where(flat, 1.0, share)), 0, 1))
        if a < 100:
            rise = tables[1][a]["value"] - linear
            red_rise = tables[1][a]["red_value"] - linear
            moved = np.abs(rise) > 1e-5
            share = np.where(moved, red_rise / np.where(moved, rise, 1), 1.0)
            values.append(np.clip(nodes(np.clip(share, 0, 1)), 0, 1))
    out["skin_amount"] = np.stack(shares)
    out["skin_value"] = np.stack(values)
    return {key: value.astype(np.float16) for key, value in out.items()}


def check(directory: str) -> None:
    """The stored tables' worst miss, in 8-bit levels, on each probe."""
    for sign in (1, -1):
        for a in AMOUNTS:
            probe = read_probe(f"{directory}/vibrance_{sign * a}.tif").reshape(-1, 3)
            colours = np.stack(
                np.meshgrid(LEVELS, LEVELS, LEVELS, indexing="ij"), -1
            ).reshape(-1, 3)
            worst = 0.0
            for start in range(0, len(colours), 1 << 20):
                encoded = colours[start : start + (1 << 20)] / 255.0
                got = adjustments._vibrance_step(
                    _linear(encoded).astype(np.float32), sign * a / 100
                )
                expected = probe[start : start + (1 << 20)]
                miss = np.abs(_encode(got) - _encode(expected)) * 255
                worst = max(worst, float(miss.max()))
            print(f"vibrance {sign * a:+d}: worst {worst:.3f} levels")


def _encode(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 0, 1)
    return np.where(
        value <= 0.0031308, value * 12.92, 1.055 * value ** (1 / 2.4) - 0.055
    )


def main() -> None:
    """Write the tables, or check the stored ones."""
    if len(sys.argv) == 3:
        tables = build(sys.argv[1])
        np.savez_compressed(
            sys.argv[2],
            raise_chroma=tables["raise_chroma"],
            raise_value=tables["raise_value"],
            lower_chroma=tables["lower_chroma"],
            lower_value=tables["lower_value"],
            skin_amount=tables["skin_amount"],
            skin_value=tables["skin_value"],
        )
    elif len(sys.argv) == 2:
        check(sys.argv[1])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
