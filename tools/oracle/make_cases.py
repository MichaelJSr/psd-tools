"""Write the PSDs one oracle family asks Photoshop to render.

    python tools/oracle/make_cases.py <spec.json> <dir>

The fork authors every case and Photoshop only opens and re-saves it
(render.jsx), so no feature needs ActionManager code: whatever the fork can
write, Photoshop renders. A spec names a seed and what to vary:

    {"family": "dropshadow-size",
     "seed": "tests/psd_files/layer_effects.psd",      # a whole document, or
     "excerpt": {"card": "<card>.psd", "layer": "Name", "pad": 48},
     "layer": "Drop Shadow", "effect": "dropshadow",
     "vary": {"blur": [0, 2, 8, 32], "Ckmt": [0, 50]}}

Every combination of ``vary`` becomes ``<dir>/in/<family>__k=v__....psd``;
an excerpt copies one card layer onto a transparent document of its own.
The stored preview is painted magenta, so a case Photoshop did not re-render
cannot pass for one it did.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from typing import Any

from PIL import Image

from psd_tools import PSDImage

_CANARY = (255, 0, 255)


def _seed(spec: dict[str, Any]) -> PSDImage:
    if "seed" in spec:
        return PSDImage.open(spec["seed"])
    ex = spec["excerpt"]
    card = PSDImage.open(ex["card"])
    layer = next(l for l in card.descendants() if l.name == ex["layer"])
    pad = int(ex.get("pad", 32))
    left, top, right, bottom = layer.bbox
    doc = PSDImage.new("RGB", (right - left + 2 * pad, bottom - top + 2 * pad), depth=8)
    copy = layer.copy_to(doc)
    dx, dy = pad - left, pad - top
    for moved in [copy, *copy.descendants()] if copy.is_group() else [copy]:
        if not moved.is_group():
            moved.left, moved.top = moved.left + dx, moved.top + dy
    return doc


def _target(doc: PSDImage, spec: dict[str, Any]) -> Any:
    name = spec.get("layer") or spec.get("excerpt", {}).get("layer")
    layer = next(l for l in doc.descendants() if l.name == name)
    return next(iter(layer.effects.find(spec["effect"])))


def _set(descriptor: Any, key: str, value: Any) -> None:
    old = descriptor[key.encode()]
    if hasattr(old, "value"):
        old.value = float(value)
    else:
        descriptor[key.encode()] = type(old)(value)


def main(spec_path: str, out_dir: str) -> None:
    spec = json.load(open(spec_path))
    os.makedirs(os.path.join(out_dir, "in"), exist_ok=True)
    keys = list(spec["vary"])
    names = []
    for values in itertools.product(*(spec["vary"][k] for k in keys)):
        doc = _seed(spec)
        effect = _target(doc, spec)
        for key, value in zip(keys, values):
            _set(effect.descriptor, key, value)
        name = (
            spec["family"]
            + "".join(f"__{k.strip()}={v}" for k, v in zip(keys, values))
            + ".psd"
        )
        canary = Image.new("RGB", (doc.width, doc.height), _CANARY)
        doc.save(os.path.join(out_dir, "in", name), preview=canary)
        names.append(name)
    json.dump(
        {"spec": spec, "cases": names},
        open(os.path.join(out_dir, "manifest.json"), "w"),
        indent=1,
    )
    jsx = open(os.path.join(os.path.dirname(__file__), "render.jsx")).read()
    jsx = jsx.replace("__DIR__", json.dumps(os.path.abspath(out_dir) + "/")).replace(
        "__FILES__", json.dumps(names)
    )
    open(os.path.join(out_dir, "render.jsx"), "w", encoding="ascii").write(jsx)
    print(
        f"{len(names)} cases in {out_dir}/in; render with: open -a 'Adobe Photoshop 2023' {out_dir}/render.jsx"
    )


if __name__ == "__main__":
    main(*sys.argv[1:])
