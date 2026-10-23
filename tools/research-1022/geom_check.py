"""geom_check.py <n> [seed]: text_geometry() on the Name/Ability layers of n random corpus cards,
scored against each layer's stored raster (Photoshop's own render of the text).

recall: share of stored ink pixels (alpha > 0.5) inside the union of predicted glyph ink boxes (+1 px)
glyph_hit: share of predicted ink boxes that hold stored ink
"""
import json, os, random, sys, time
import numpy as np
from psd_tools import PSDImage
from psd_tools.composite.text.fonts import FontMap
from psd_tools.composite.text.geometry import text_geometry
from psd_tools.composite.text.verify import stored_raster

C = "/Users/michaelsrouji/Documents/Tools/psd-research/corpus-6e50c3ee"
FONTS = FontMap(dirs=["/Users/michaelsrouji/Desktop/TTSCardMaker/MDW/Fonts", os.path.expanduser("~/Library/Fonts")],
                aliases={"Papyrus-Regular": "Papyrus", "Kung-Fu-Panda": "KungFuPanda"}, policy="skip")
cards = []
for r, _, fs in os.walk(C):
    if "/MDW" in r: continue
    cards += [os.path.join(r, f) for f in fs if f.endswith(".psd")]
random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 3); random.shuffle(cards)
out = []
for p in cards[: int(sys.argv[1])]:
    psd = PSDImage.open(p)
    for l in psd.descendants():
        if l.kind != "type" or not l.is_visible() or l.name.strip().lower() not in ("name", "ability"):
            continue
        t = time.perf_counter()
        try:
            g = text_geometry(l, FONTS)
        except Exception as e:
            out.append({"card": os.path.relpath(p, C), "layer": l.name, "error": repr(e)[:120]}); continue
        ms = (time.perf_counter() - t) * 1000
        rgba, bb = stored_raster(l)
        if rgba is None:
            continue
        ink = rgba[..., 3] > 0.5
        mask = np.zeros_like(ink)
        hits = n = 0
        for gl in g["glyphs"]:
            if gl["ink"] is None: continue
            x0, y0, x1, y1 = gl["ink"]
            a, b_ = int(np.floor(x0)) - 1 - bb[0], int(np.floor(y0)) - 1 - bb[1]
            c, d = int(np.ceil(x1)) + 1 - bb[0], int(np.ceil(y1)) + 1 - bb[1]
            a, b_ = max(a, 0), max(b_, 0)
            if c <= a or d <= b_: continue
            mask[b_:d, a:c] = True
            n += 1; hits += bool(ink[b_:d, a:c].any())
        total = int(ink.sum())
        rec = float((ink & mask).sum() / total) if total else None
        pua = [gl for gl in g["glyphs"] if gl["ch"] and 0xE000 <= ord(gl["ch"]) <= 0xF8FF]
        out.append({"card": os.path.relpath(p, C), "layer": l.name, "ms": round(ms, 1), "recall": None if rec is None else round(rec, 4),
                    "glyph_hit": round(hits / n, 4) if n else None, "glyphs": n, "pua": len(pua), "warn": g["warnings"][:4]})
for o in out: print(json.dumps(o))
rs = [o["recall"] for o in out if o.get("recall") is not None]
gh = [o["glyph_hit"] for o in out if o.get("glyph_hit") is not None]
print("layers", len(out), "errors", sum(1 for o in out if "error" in o), "recall median %.4f p10 %.4f min %.4f" % (np.median(rs), np.percentile(rs, 10), min(rs)),
      "glyph_hit median %.4f p10 %.4f" % (np.median(gh), np.percentile(gh, 10)), ">=0.99 recall:", sum(1 for r in rs if r >= 0.99), "of", len(rs))
