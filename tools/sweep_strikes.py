"""Drop the listed type layers' stale strikes, nothing else.

    PYTHONPATH=src python tools/sweep_strikes.py <list.json> <cards root> <out dir>

``<list.json>`` maps a card's path under ``<cards root>`` to its layer ids:
``{"Field/3 Stars/Glitz_Pit.psd": [320]}``. Each card lands at the same path
under ``<out dir>``. Photoshop mixes a type layer's raster straight while its
story's strikes no longer match the story, and composes it in the text gamma,
as this fork draws it, once they are gone. Only Txt2 changes: the rasters, the
preview (none is written) and every other byte stay. A story set in a face of
``text_edit._KEEP_STRIKES`` keeps its strikes. Three cards at a time.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from typing import IO, Any

from psd_tools import PSDImage
from psd_tools.api.layers import TypeLayer
from psd_tools.api.text_edit import _story, drop_strikes
from psd_tools.composite.text import font_names


def _has_strikes(layer: TypeLayer) -> bool:
    try:
        story = _story(layer)
    except ValueError:
        return False
    return story is not None and story.k.strikes in story.story[story.k.view]


def with_strikes(psd: PSDImage) -> set[int]:
    """Ids of the type layers whose story has strikes."""
    return {
        layer.layer_id
        for layer in psd.descendants()
        if isinstance(layer, TypeLayer) and _has_strikes(layer)
    }


def sweep(src: str | IO[bytes], ids: Iterable[int]) -> tuple[bytes, dict[str, Any]]:
    """The card's bytes with the strikes of ``ids`` dropped, and what happened to each id."""
    psd = PSDImage.open(src)
    wanted = set(ids)
    report: dict[str, list[int]] = {"dropped": [], "kept": [], "none": []}
    for layer in psd.descendants():
        if not isinstance(layer, TypeLayer) or layer.layer_id not in wanted:
            continue
        wanted.discard(layer.layer_id)
        if not _has_strikes(layer):
            report["none"].append(layer.layer_id)
            continue
        drop_strikes(layer, font_names(layer))
        report["kept" if _has_strikes(layer) else "dropped"].append(layer.layer_id)
    report["missing"] = sorted(wanted)
    out = io.BytesIO()
    psd.save(out, update_preview=False)
    return out.getvalue(), report


def _one(job: tuple[str, str, str, list[int]]) -> tuple[str, dict[str, Any]]:
    rel, root, dest, ids = job
    logging.disable(logging.CRITICAL)
    data, report = sweep(os.path.join(root, rel), ids)
    target = os.path.join(dest, rel)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "wb") as f:
        f.write(data)
    return rel, report


def main(argv: list[str]) -> int:
    """Sweep every listed card; print the totals and each card a strike stayed on."""
    listing, root, dest = argv
    with open(listing, encoding="utf-8") as f:
        cards: dict[str, list[int]] = json.load(f)
    jobs = [(rel, root, dest, ids) for rel, ids in sorted(cards.items())]
    with ProcessPoolExecutor(3, max_tasks_per_child=10) as pool:
        done = dict(pool.map(_one, jobs, chunksize=1))
    total = {
        k: sum(len(r[k]) for r in done.values())
        for k in ("dropped", "kept", "none", "missing")
    }
    print(len(done), "cards", json.dumps(total))
    for rel, report in done.items():
        if report["kept"] or report["missing"]:
            print(
                rel,
                json.dumps({k: report[k] for k in ("kept", "missing") if report[k]}),
            )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
