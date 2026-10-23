"""fx_lint.py <psd>...: each 'Ability FX: <word>' layer's box against the word's glyph boxes from text_geometry()."""
import os, sys, re
from psd_tools import PSDImage
from psd_tools.composite.text.fonts import FontMap
from psd_tools.composite.text.geometry import text_geometry
FONTS = FontMap(dirs=["/Users/michaelsrouji/Desktop/TTSCardMaker/MDW/Fonts", os.path.expanduser("~/Library/Fonts")],
                aliases={"Papyrus-Regular": "Papyrus", "Kung-Fu-Panda": "KungFuPanda"}, policy="skip")
for p in sys.argv[1:]:
    psd = PSDImage.open(p)
    ab = next((l for l in psd.descendants() if l.kind == "type" and l.name == "Ability" and l.is_visible()), None)
    fx = [l for l in psd.descendants() if l.name.startswith("Ability FX:") and l.is_visible()]
    if ab is None or not fx: continue
    g = text_geometry(ab, FONTS); text = g["text"]
    for l in fx:
        words = [w.strip() for w in l.name.split(":", 1)[1].split("|") if w.strip()]
        boxes = []
        for m in (m for w in words for m in re.finditer(re.escape(w), text, re.IGNORECASE)):
            ink = [x["ink"] for x in g["glyphs"] if m.start() <= x["i"] < m.end() and x["ink"]]
            if ink: boxes.append([min(b[0] for b in ink), min(b[1] for b in ink), max(b[2] for b in ink), max(b[3] for b in ink)])
        a = l.numpy("shape")
        if a is None:
            lb = l.bbox
        else:
            ys, xs = (a[..., 0] > 0.05).nonzero()
            lb = (l.left + int(xs.min()), l.top + int(ys.min()), l.left + int(xs.max()) + 1, l.top + int(ys.max()) + 1) if len(xs) else l.bbox
        def inside(b, pad=16): return lb[0] - pad <= b[0] and lb[1] - pad <= b[1] and b[2] <= lb[2] + pad and b[3] <= lb[3] + pad
        ok = bool(boxes) and all(inside(b) for b in boxes)
        print(("ok   " if ok else "STALE"), p.rsplit("/", 1)[1], repr(l.name), "layer", lb, "word", [[round(v) for v in b] for b in boxes])
