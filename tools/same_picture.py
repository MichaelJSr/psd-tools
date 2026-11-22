"""Edits that must not change a card's picture, checked without Photoshop.

    PYTHONPATH=src python tools/same_picture.py <cards> <out.jsonl> [--move]
        [--sample N] [--layers K] [list]

Each edit is made on a fresh copy of the card, rendered as AC renders it
(``composite(ignore_preview=True, apply_icc=False)``) and compared with its
reference; colour under alpha 0 is not compared, as it is never seen.

Relations (default):

- ``again``: a second render equals the first.
- ``wrap_all``: every top-level layer in one 100% pass-through group.
- On ``--layers`` layers a card (default 5, seeded by its path, kinds mixed):
  ``hide=delete``, ``hide=opacity0`` and, without effects, ``hide=fill0``;
  ``wrap`` (a pass-through group of one, from ``Group.new()``);
  ``empty_above`` (an empty layer above it); ``white_mask`` (an all-white mask
  on a layer without one); ``mask_off=mask_gone`` (a disabled mask equals a
  removed one); ``opacity=fill`` (50% each on a Normal layer without
  effects); ``swap`` (with its neighbour above when their boxes do not touch).

``--move`` (the ship check, every card): each pixel, smart-object or type
layer with effects, alone with its ancestors, moved by whole pixels; its
drawing, effects included, must move by the same pixels.

Skipped where a relation does not hold: clip bases and clipped layers
(reorders, wraps, inserted layers and moves; hiding skips only bases); for
moves, a mask of the layer's own, effects tied to the document (noise,
pattern, gradients not aligned with the layer, dither, jitter), a scaled
0 or 90 degree gradient overlay whose ramp starts before the canvas (its ends
are cut toward zero), and an ancestor with a mask or with any of those effects
or a gradient.

Rounding is reported, not failed: up to 4 pixels by up to 2 levels, and
any number by one level in the opacity and fill relations.

Cards: every PSD under <cards> with a PNG beside it (outside MDW/), or those
named in [list]; one process per card, ``SCORE_WORKERS`` (default 4, at most
4) at a time. Prints a tally per relation and exits 1 when any check fails
or errors.
"""

from __future__ import annotations

import collections
import glob
import importlib
import json
import math
import multiprocessing
import os
import random
import sys
import time
import traceback
from collections.abc import Callable
from typing import Any

import numpy as np
from PIL import Image

from psd_tools import PSDImage
from psd_tools.api.layers import Group
from psd_tools.constants import BlendMode

# The module, which the package's ``composite`` function shadows.
_composite = importlib.import_module("psd_tools.composite.composite")

Path = list[int]
# Any layer of the tree, groups and the document's children alike.
Layer = Any
Edit = Callable[[PSDImage], None]
LEAVES = ("pixel", "shape", "type", "smartobject")
SHIFT = (5, 3)
# Relations whose mix may round one level apart.
LOOSE = {"hide=opacity0", "hide=fill0", "opacity=fill"}
# Float32 rounding: a few pixels a level or two apart (reported, not failed).
NOISE_PX, NOISE_MAX = 4, 2


def render(psd: PSDImage, layer_filter: Callable | None = None) -> np.ndarray:
    """The card as AC renders it, as int16."""
    image = psd.composite(
        ignore_preview=True, apply_icc=False, layer_filter=layer_filter
    )
    return np.asarray(image).astype(np.int16)


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """Pixels that differ and by how much, colour under alpha 0 aside."""
    if a.shape != b.shape:
        return {"px": -1, "max": -1}
    d = np.abs(a - b).max(axis=2)
    if a.shape[2] == 4:
        d[(a[..., 3] == 0) & (b[..., 3] == 0)] = 0
    out: dict[str, Any] = {
        "px": int((d > 0).sum()),
        "max": int(d.max()) if d.size else 0,
    }
    if out["px"]:
        ys, xs = np.nonzero(d)
        out["box"] = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ]
    return out


def path_of(layer: Layer) -> Path:
    """Indices from the document down to ``layer``."""
    out = []
    while layer.parent is not None:
        out.append(layer.parent.index(layer))
        layer = layer.parent
    return out[::-1]


def at(psd: PSDImage, path: Path) -> Layer:
    """The layer at ``path``."""
    node: Any = psd
    for i in path:
        node = node[i]
    return node


def shown(layer: Layer, psd: PSDImage) -> bool:
    """Whether the layer and every group above it are visible."""
    node: Layer | None = layer
    while node is not None and node is not psd:
        if not node.visible:
            return False
        node = node.parent
    return True


def on_canvas(layer: Layer, psd: PSDImage) -> bool:
    """Whether the layer's box meets the canvas."""
    left, top, right, bottom = layer.bbox
    return left < psd.width and top < psd.height and right > 0 and bottom > 0


def clipped(layer: Layer) -> bool:
    """A clip base or a clipped layer."""
    return bool(layer.clipping or layer.clip_layers)


def _value(effect: Any, key: bytes) -> Any:
    try:
        value = effect.descriptor.get(key)
    except Exception:
        return None
    return getattr(value, "value", value)


def _ramp_cut(layer: Layer, effect: Any) -> bool:
    """Whether a scaled 0 or 90 degree ramp starts before the canvas.

    Its ends are cut to whole pixels toward zero, so there a move by whole
    pixels does not carry them (Photoshop's way there is not yet known).
    """
    scale = float(_value(effect, b"Scl ") or 100.0)
    angle = float(_value(effect, b"Angl") or 0.0) % 180.0
    if scale == 100.0 or angle not in (0.0, 90.0):
        return False
    box = _composite._opaque_bounds(layer, layer.bbox)
    lo, hi = (1, 3) if angle == 90.0 else (0, 2)
    half = math.floor((box[hi] - box[lo]) * scale / 200.0 + 1e-9)
    return (box[lo] + box[hi]) / 2.0 - half < 0


def _tied(layer: Layer) -> set[str]:
    """The layer's effects that are drawn from the document, not the layer."""
    out = set()
    for effect in layer.effects:
        if not effect.enabled:
            continue
        name = type(effect).__name__
        paint = getattr(getattr(effect, "fill_type", None), "value", b"")  # strokes
        if (_value(effect, b"Nose") or 0) > 0:
            out.add("noise")
        if name == "PatternOverlay" or paint == b"Ptrn":
            out.add("pattern")
        if name == "GradientOverlay" or paint == b"GrFl":
            out.add("gradient")
            if _value(effect, b"Algn") is False:
                out.add("gradient_unaligned")
            elif name == "GradientOverlay" and _ramp_cut(layer, effect):
                out.add("gradient_cut")
        if _value(effect, b"Dthr"):
            out.add("dither")
        if (_value(effect, b"ShdN") or 0) > 0:
            out.add("jitter")
    return out


def anchors(layer: Layer, psd: PSDImage) -> list[str]:
    """Why a whole-pixel move of ``layer`` need not move its drawing."""
    out = _tied(layer) - {"gradient"}
    node = layer.parent
    while node is not None and node is not psd:
        if node.mask is not None or node.has_vector_mask():
            out.add("ancestor_mask")
        if _tied(node):  # a group's ramp spans its other layers too
            out.add("ancestor_effects")
        node = node.parent
    return sorted(out)


def _solo(psd: PSDImage, layer: Layer) -> np.ndarray:
    """``layer`` rendered alone with its ancestors."""
    keep = {id(layer)}
    node = layer.parent
    while node is not None and node is not psd:
        keep.add(id(node))
        node = node.parent
    return render(psd, lambda x: id(x) in keep and x.is_visible())


def moves(path: str) -> list[dict]:
    """The whole-pixel move on every layer of the card it applies to."""
    psd = PSDImage.open(path)
    dx, dy = SHIFT
    rows = []
    skipped: collections.Counter = collections.Counter()
    for layer in psd.descendants():
        if not (
            layer.kind in ("pixel", "smartobject", "type")
            and layer.has_effects()
            and shown(layer, psd)
            and on_canvas(layer, psd)
        ):
            continue
        why = anchors(layer, psd)
        if layer.mask is not None or layer.has_vector_mask():
            why.append("mask")
        if clipped(layer):
            why.append("clip")
        if why:
            skipped.update(why)
            continue
        where = path_of(layer)
        row: dict = {"rel": "move", "layer": layer.name[:40], "kind": layer.kind}
        try:
            a = _solo(psd, layer)
            moved = PSDImage.open(path)
            m = at(moved, where)
            m.left, m.top = m.left + dx, m.top + dy
            b = _solo(moved, m)
            row.update(compare(a[: a.shape[0] - dy, : a.shape[1] - dx], b[dy:, dx:]))
        except Exception as error:
            row.update(px=-2, max=-2, error=repr(error)[:200])
        rows.append(row)
    rows.append({"rel": "move_skipped", **dict(skipped)})
    return rows


def _picks(psd: PSDImage, rel: str, k: int) -> list[Layer]:
    """``k`` shown layers on the canvas, one of each kind first; no artboards."""
    layers = [
        x
        for x in psd.descendants()
        if shown(x, psd) and on_canvas(x, psd) and x.kind != "artboard"
    ]
    random.Random(rel).shuffle(layers)
    seen, picks = set(), []
    for x in layers:
        key = (x.kind, bool(x.has_effects()), x.mask is not None)
        if key not in seen:
            seen.add(key)
            picks.append(x)
    return (picks + [x for x in layers if x not in picks])[:k]


def _wrap(where: Path) -> Edit:
    def edit(q: PSDImage) -> None:
        layer = at(q, where)
        parent, i = layer.parent, where[-1]
        group = Group.new(parent, "same_picture")
        group.move_to(parent, index=i)
        layer.move_to(group)
        assert parent.index(group) == i and len(group) == 1

    return edit


def _empty_above(where: Path) -> Edit:
    def edit(q: PSDImage) -> None:
        parent = at(q, where).parent
        empty = parent.create_pixel_layer(
            Image.new("RGBA", (1, 1), (0, 0, 0, 0)), name="same_picture"
        )
        empty.move_to(parent, index=where[-1] + 1)

    return edit


def _white_mask(where: Path) -> Edit:
    def edit(q: PSDImage) -> None:
        layer = at(q, where)
        layer.create_mask(Image.new("L", (1, 1), 255), top=0, left=0)
        layer._record.mask_data.background_color = 255  # white past its box too

    return edit


def _set(where: Path, **values: object) -> Edit:
    def edit(q: PSDImage) -> None:
        layer = at(q, where)
        for name, value in values.items():
            setattr(layer, name, value)

    return edit


def _call(where: Path, fn: Callable[[Layer], object]) -> Edit:
    def edit(q: PSDImage) -> None:
        fn(at(q, where))

    return edit


def _swappable(layer: Layer) -> bool:
    """Whether ``layer`` and its neighbour above may trade places."""
    parent, i = layer.parent, layer.parent.index(layer)
    if layer.kind not in LEAVES or layer.has_effects() or i + 1 >= len(parent):
        return False
    up = parent[i + 1]
    if (
        up.kind not in LEAVES
        or not up.visible
        or up.has_effects()
        or clipped(up)
        or (i + 2 < len(parent) and parent[i + 2].clipping)
    ):
        return False
    a, b = layer.bbox, up.bbox
    return a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def relations(path: str, rel: str, k: int) -> list[dict]:
    """The default relations on the card and ``k`` of its layers."""
    psd = PSDImage.open(path)
    base = render(psd)
    rows: list[dict] = []

    def check(
        name: str, edit: Edit, ref: np.ndarray | None, info: dict
    ) -> np.ndarray | None:
        try:
            q = PSDImage.open(path)
            edit(q)
            out = render(q)
        except Exception as error:
            rows.append(
                {"rel": name, **info, "px": -2, "max": -2, "error": repr(error)[:200]}
            )
            return None
        if ref is not None:
            rows.append({"rel": name, **info, **compare(out, ref)})
        return out

    check("again", lambda q: None, base, {})

    def wrap_all(q: PSDImage) -> None:
        q.create_group(list(q), name="same_picture")

    check("wrap_all", wrap_all, base, {})
    for layer in _picks(psd, rel, k):
        where = path_of(layer)
        effects, base_of = bool(layer.has_effects()), bool(layer.clip_layers)
        info = {"layer": layer.name[:40], "kind": layer.kind, "fx": effects}
        hidden = check("hide", _set(where, visible=False), None, info)
        if hidden is None:
            continue
        if not base_of:
            check(
                "hide=delete", _call(where, lambda x: x.parent.remove(x)), hidden, info
            )
            check("hide=opacity0", _set(where, opacity=0), hidden, info)
            if not effects and layer.kind != "group":
                check("hide=fill0", _set(where, fill_opacity=0), hidden, info)
        if not clipped(layer):
            check("wrap", _wrap(where), base, info)
            check("empty_above", _empty_above(where), base, info)
            if _swappable(layer):
                check(
                    "swap",
                    _call(where, lambda x: x.move_to(x.parent, index=where[-1] + 1)),
                    base,
                    info,
                )
        if layer.mask is None:
            check("white_mask", _white_mask(where), base, info)
        else:
            off = check(
                "mask_off",
                _call(where, lambda x: setattr(x.mask, "disabled", True)),
                None,
                info,
            )
            if off is not None:
                check(
                    "mask_off=mask_gone",
                    _call(where, lambda x: x.remove_mask()),
                    off,
                    info,
                )
        if (
            layer.kind != "group"
            and not base_of
            and not effects
            and layer.blend_mode == BlendMode.NORMAL
            and layer.opacity == 255
            and layer.fill_opacity == 255
        ):
            half = check("opacity50", _set(where, opacity=128), None, info)
            if half is not None:
                check("opacity=fill", _set(where, fill_opacity=128), half, info)
    return rows


def run(job: tuple[str, str, bool, int]) -> dict:
    """All checks on one card."""
    root, path, move, k = job
    rel = os.path.relpath(path, root)
    start = time.process_time()
    try:
        rows = moves(path) if move else relations(path, rel, k)
        out: dict = {"card": rel, "rows": rows}
    except Exception:
        out = {"card": rel, "rows": [], "error": traceback.format_exc()[-800:]}
    out["cpu_s"] = round(time.process_time() - start, 1)
    return out


def noise(row: dict) -> bool:
    """Whether a row differs only by rounding."""
    if row.get("px", 0) <= 0:
        return False
    loose = row["rel"] in LOOSE and row["max"] <= 1
    return loose or (row["px"] <= NOISE_PX and row["max"] <= NOISE_MAX)


def failed(row: dict) -> bool:
    """Whether a row breaks its relation."""
    if row["rel"] == "move_skipped" or "px" not in row:
        return False
    return row["px"] < 0 or (row["px"] > 0 and not noise(row))


def main() -> None:
    """Check the cards, ``SCORE_WORKERS`` processes at a time."""
    args = sys.argv[1:]
    move = "--move" in args
    args = [a for a in args if a != "--move"]
    options = {"--sample": 0, "--layers": 5}
    for name in options:
        if name in args:
            i = args.index(name)
            options[name] = int(args[i + 1])
            del args[i : i + 2]
    root, out = args[0].rstrip("/"), args[1]
    paths = sorted(
        p
        for p in glob.glob(root + "/**/*.psd", recursive=True)
        if "/MDW/" not in p and os.path.exists(p[:-4] + ".png")
    )
    if len(args) > 2:
        want = {line.strip() for line in open(args[2]) if line.strip()}
        paths = [p for p in paths if os.path.relpath(p, root) in want or p in want]
    if options["--sample"]:
        paths = sorted(
            random.Random(0).sample(paths, min(options["--sample"], len(paths)))
        )
    workers = min(4, int(os.environ.get("SCORE_WORKERS", "4")))
    tally: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    skipped: collections.Counter = collections.Counter()
    bad = errors = 0
    start = time.time()
    ctx = multiprocessing.get_context("spawn")
    jobs = [(root, p, move, options["--layers"]) for p in paths]
    with ctx.Pool(workers, maxtasksperchild=1) as pool, open(out, "w") as f:
        for n, card in enumerate(pool.imap_unordered(run, jobs), 1):
            f.write(json.dumps(card) + "\n")
            f.flush()
            errors += "error" in card
            for row in card["rows"]:
                if row["rel"] == "move_skipped":
                    skipped.update({k: v for k, v in row.items() if k != "rel"})
                elif "px" in row:
                    tally[row["rel"]][0] += 1
                    tally[row["rel"]][1] += failed(row)
                    tally[row["rel"]][2] += noise(row)
            fails = [r for r in card["rows"] if failed(r)]
            bad += len(fails)
            if fails or "error" in card:
                print(
                    f"[{n}/{len(paths)}] FAIL",
                    card["card"],
                    fails,
                    card.get("error", ""),
                )
    print(f"{len(paths)} cards in {time.time() - start:.0f} s, {errors} card errors")
    for name, (checked, wrong, near) in sorted(tally.items()):
        print(f"{name:20} {wrong:4} of {checked:5} differ, {near} by rounding")
    if skipped:
        print("skipped moves:", dict(sorted(skipped.items())))
    sys.exit(1 if bad or errors else 0)


if __name__ == "__main__":
    main()
