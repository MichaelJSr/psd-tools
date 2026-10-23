"""fill_probe.py: template fill by hand on Creature_Template: name, ability, one type, stat digits; save and render."""
import os, sys, time, json
import numpy as np
from psd_tools import PSDImage
from psd_tools.composite.text.fonts import FontMap
T = "/Users/michaelsrouji/Desktop/TTSCardMaker/MDW/Templates/Creature_Template.psd"
OUT = "/Users/michaelsrouji/Documents/Tools/wt/feat-innovation-scratch/fill/"
FONTS = FontMap(dirs=["/Users/michaelsrouji/Desktop/TTSCardMaker/MDW/Fonts", os.path.expanduser("~/Library/Fonts")],
                aliases={"Papyrus-Regular": "Papyrus", "Kung-Fu-Panda": "KungFuPanda"})
req = {"name": "Croissant Knight", "ability": "(1CD) Deal 2 [fire] damage to an enemy creature.", "type": "Fire",
       "stats": {"Hp": 5, "Def": 2, "Atk": 4, "Spd": 3}}
t0 = time.perf_counter()
psd = PSDImage.open(T)
by = {}
for l in psd.descendants():
    by.setdefault(l.name.strip(), []).append(l)
report = {}
for field, layer in (("name", by["Name"][0]), ("ability", by["Ability"][0])):
    print(field, "fonts", [r.style.font for r in __import__("psd_tools.composite.text.model", fromlist=["TextModel"]).TextModel.from_layer(layer).runs][:3])
    layer.set_text(req[field], keep="runs")
    rep = layer.rasterize(fonts=FONTS)
    report[field] = {"warnings": list(rep.warnings), "fonts": {k: v[0] for k, v in rep.fonts.items()}, "bbox": list(layer.bbox)}
types = [l for l in psd.descendants() if l.parent is not None and getattr(l.parent, "name", "") == "Creature Types" and l.kind == "smartobject"]
for l in types:
    l.visible = l.name == req["type"]
for stat, v in req["stats"].items():
    grp = next(l for l in psd.descendants() if l.is_group() and l.name.strip().lower() == f"{stat.lower()} dark")
    for d in grp:
        d.visible = d.name.strip() == str(v)
psd.save(OUT + "Croissant_Knight.psd")
t1 = time.perf_counter()
im = PSDImage.open(OUT + "Croissant_Knight.psd").composite(ignore_preview=True)
im.save(OUT + "Croissant_Knight.png")
print(json.dumps(report, indent=1)); print("fill+save %.1f s, render %.1f s" % (t1 - t0, time.perf_counter() - t1))
