"""Check the render cache against uncached renders over a folder of cards.

    PYTHONPATH=src python tools/rcache_verify.py <cards> <fonts> <out.jsonl> [list]

Every PSD under <cards> with a PNG beside it (outside MDW/), or those named in
[list]. Per card: hide or show one seeded-random layer, change one word of
the Ability text and rasterize it with the fonts in <fonts>, save and reopen;
then render the baseline, the edit and the reopened file, on the Ability
region and on the whole card, under one cache with ``PSD_RCACHE_VERIFY=1``.
Each render runs with and without the cache and must come out equal. One
process per card, ``SCORE_WORKERS`` (default 4) at a time.
"""

from __future__ import annotations

import glob
import io
import json
import multiprocessing
import os
import random
import sys
import time
import traceback

import numpy as np

from psd_tools import PSDImage
from psd_tools.api.layers import TypeLayer
from psd_tools.composite import rcache
from psd_tools.composite.text import FontMap

ALIASES = {"Papyrus-Regular": "Papyrus", "Kung-Fu-Panda": "KungFuPanda"}


def ability_layer(psd: PSDImage) -> TypeLayer | None:
    """The card's visible Ability text layer."""
    for layer in psd.descendants():
        if (
            isinstance(layer, TypeLayer)
            and layer.name.strip().lower() == "ability"
            and layer.is_visible()
        ):
            return layer
    return None


def check(job: tuple[str, str, str]) -> dict:
    """Edit, save and render one card; the row for the report."""
    os.environ["PSD_RCACHE_VERIFY"] = "1"
    root, fonts, path = job
    rel = os.path.relpath(path, root)
    row: dict = {"card": rel}
    rng = random.Random(rel)
    start = time.process_time()
    try:
        old, new = PSDImage.open(path), PSDImage.open(path)
        row["cached"] = rcache.doc_key(old) is not None
        layers = list(new.descendants())
        toggled = layers[rng.randrange(len(layers))]
        toggled.visible = not toggled.visible
        row["toggled"] = toggled.name
        layer = ability_layer(new)
        region = None
        if layer is None:
            row["edit"] = "no ability"
        else:
            b = layer.bbox
            region = (
                max(0, b[0] - 40),
                max(0, b[1] - 40),
                min(new.width, b[2] + 40),
                min(new.height, b[3] + 40),
            )
            words = layer.text.split(" ")
            i = max(range(len(words)), key=lambda k: len(words[k]))
            words[i] = words[i][::-1]
            try:
                layer.set_text(" ".join(words), keep="runs")
                layer.rasterize(fonts=FontMap(dirs=[fonts], aliases=ALIASES))
                row["edit"] = "ability"
            except Exception as error:  # a card the bake refuses: toggle only
                row["edit"] = f"{type(error).__name__}: {error}"[:200]
        buffer = io.BytesIO()
        new.save(buffer)
        reopened = PSDImage.open(io.BytesIO(buffer.getvalue()))
        keys: dict[int, bytes] = {}
        row["same_keys"] = [rcache.layer_key(x, keys) for x in new] == [
            rcache.layer_key(x, keys) for x in reopened
        ]
        cache = rcache.RenderCache()
        renders = {}
        for viewport in ([region] if region else []) + [None]:
            for name, psd in (
                ("baseline", old),
                ("edited", new),
                ("reopened", reopened),
            ):
                with cache, np.errstate(all="ignore"):
                    renders[name] = np.asarray(
                        psd.composite(
                            viewport=viewport, ignore_preview=True, apply_icc=False
                        )
                    )
            row["reopened_equal" if viewport is None else "region_reopened_equal"] = (
                np.array_equal(renders["edited"], renders["reopened"])
            )
        row["hits"] = dict(cache.hits)
        row["misses"] = dict(cache.misses)
        row["result"] = "ok"
    except rcache.CacheMismatch as error:
        row["result"] = "mismatch"
        row["error"] = str(error)
    except Exception as error:
        row["result"] = "error"
        row["error"] = f"{type(error).__name__}: {error}"[:300]
        row["trace"] = traceback.format_exc()[-1500:]
    row["cpu_s"] = round(time.process_time() - start, 1)
    return row


def main() -> None:
    """Check every card, ``SCORE_WORKERS`` processes at a time."""
    root, fonts, out = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
    paths = sorted(
        p
        for p in glob.glob(root + "/**/*.psd", recursive=True)
        if "/MDW/" not in p and os.path.exists(p[:-4] + ".png")
    )
    if len(sys.argv) > 4:
        want = {line.strip() for line in open(sys.argv[4]) if line.strip()}
        paths = [p for p in paths if os.path.relpath(p, root) in want or p in want]
    workers = min(4, int(os.environ.get("SCORE_WORKERS", "4")))
    ctx = multiprocessing.get_context("spawn")
    counts: dict[str, int] = {}
    with ctx.Pool(workers, maxtasksperchild=1) as pool, open(out, "w") as f:
        jobs = [(root, fonts, p) for p in paths]
        for n, row in enumerate(pool.imap_unordered(check, jobs), 1):
            f.write(json.dumps(row) + "\n")
            f.flush()
            counts[row["result"]] = counts.get(row["result"], 0) + 1
            print(
                f"[{n}/{len(paths)}] {row['result']:8} {row['cpu_s']:6}s", row["card"]
            )
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
