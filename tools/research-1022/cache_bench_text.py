"""cache_bench_text.py <mode cached|plain> <psd>: a bake-like sequence of three region renders.

old render, edited render (Ability hidden: stands in for a re-rasterized layer),
reopened render (a fresh open of the old file). Prints CPU times and saves arrays
for the A/B comparison."""
import sys, time, json, os
import numpy as np
from psd_tools import PSDImage
from psd_tools.composite.rcache import RenderCache

mode, path, out = sys.argv[1], sys.argv[2], sys.argv[3]

def ability(psd):
    for l in psd.descendants():
        if l.kind == "type" and l.name.strip().lower() == "ability" and l.is_visible():
            return l

FONTS = None
def edit(layer):
    """What a website ability edit does: new text, re-rasterized (psd_edit.set_text_layer's core)."""
    global FONTS
    from psd_tools.composite.text.fonts import FontMap
    if FONTS is None:
        FONTS = FontMap(dirs=["/Users/michaelsrouji/Desktop/TTSCardMaker/MDW/Fonts", os.path.expanduser("~/Library/Fonts")],
                        aliases={"Papyrus-Regular": "Papyrus", "Kung-Fu-Panda": "KungFuPanda"})
    text = layer.text
    words = text.split(" ")
    i = max(range(len(words)), key=lambda k: len(words[k]))
    words[i] = words[i][::-1]          # one word changed, same length
    layer.set_text(" ".join(words), keep="runs")
    layer.rasterize(fonts=FONTS)

def render(psd, region):
    with np.errstate(all="ignore"):
        im = psd.composite(viewport=region, ignore_preview=True, apply_icc=False)
    return np.asarray(im)

warm = PSDImage.open(path)
render(warm, (0, 0, 8, 8))
old = PSDImage.open(path)
lay = ability(old)
if lay is None:
    print(json.dumps({"card": os.path.basename(path), "skip": "no ability"})); sys.exit()
b = lay.bbox
region = (max(0, b[0] - 40), max(0, b[1] - 40), min(old.width, b[2] + 40), min(old.height, b[3] + 40))
rc = RenderCache() if mode == "cached" else None
res, arrs = {}, {}
def step(name, psd):
    c = time.process_time(); w = time.perf_counter()
    if rc is not None:
        with rc:
            arrs[name] = render(psd, region)
    else:
        arrs[name] = render(psd, region)
    res[name] = round(time.process_time() - c, 2); res[name + "_wall"] = round(time.perf_counter() - w, 2)
    if rc is not None:
        res[name + "_hm"] = (sum(rc.hits.values()), sum(rc.misses.values()))
step("old", old)
new = PSDImage.open(path); edit(ability(new))
step("edited", new)
again = PSDImage.open(path)
step("reopened", again)
res["card"] = os.path.basename(path); res["region"] = region; res["mode"] = mode
if rc is not None:
    res["hits"] = dict(rc.hits); res["misses"] = dict(rc.misses); res["cache_mb"] = round(rc.nbytes / 2**20, 1)
np.savez_compressed(out, **arrs)
print(json.dumps(res))
