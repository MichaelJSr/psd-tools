import os, sys, json
from psd_tools import PSDImage
from psd_tools.composite.text.fonts import FontMap
from psd_tools.composite.text.geometry import text_geometry
FONTS = FontMap(dirs=["/Users/michaelsrouji/Desktop/TTSCardMaker/MDW/Fonts", os.path.expanduser("~/Library/Fonts")],
                aliases={"Papyrus-Regular": "Papyrus", "Kung-Fu-Panda": "KungFuPanda"}, policy="skip")
psd = PSDImage.open(sys.argv[1])
for l in psd.descendants():
    if l.kind == "type" and l.name == "Ability":
        g = text_geometry(l, FONTS)
        gl = g["glyphs"]
        text = g["text"]
        for k, x in enumerate(gl):
            if "stoog" in x["font"].lower():
                end = gl[k + 1]["i"] if k + 1 < len(gl) and gl[k + 1]["i"] > x["i"] else x["i"] + 1
                print(x["font"], repr(text[x["i"]:end]), "line", x["line"], "ink", x["ink"])
        print("lines:", [(ln["text"][:30], ln["box"]) for ln in g["lines"]])
