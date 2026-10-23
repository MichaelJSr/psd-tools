"""preview_floor.py <n> [seed]: each PSD's stored composite (Photoshop's render at save) against its committed PNG.
ICC: the better of raw and profile-converted (the harness's vote picks per card)."""
import io, os, random, sys
import numpy as np
from PIL import Image, ImageCms
from psd_tools import PSDImage
from psd_tools.constants import Resource
C = "/Users/michaelsrouji/Documents/Tools/psd-research/corpus-6e50c3ee"
def flat(im):
    a = np.asarray(im.convert("RGBA"), np.float32); k = a[..., 3:] / 255
    a[..., :3] = a[..., :3] * k + 255 * (1 - k); return np.rint(a[..., :3]).astype(np.int16)
cards = []
for r, _, fs in os.walk(C):
    if "/MDW" in r: continue
    cards += [os.path.join(r, f) for f in fs if f.endswith(".psd") and os.path.exists(os.path.join(r, f[:-4] + ".png"))]
random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 5); random.shuffle(cards)
rows = []
for p in cards[: int(sys.argv[1])]:
    psd = PSDImage.open(p)
    if not psd.has_preview():
        rows.append((os.path.relpath(p, C), "nopreview")); continue
    prev = psd.topil(); png = Image.open(p[:-4] + ".png")
    best = None
    for mode in ("raw", "conv"):
        im = prev
        if mode == "conv":
            if Resource.ICC_PROFILE not in psd.image_resources: continue
            src = ImageCms.ImageCmsProfile(io.BytesIO(psd.image_resources.get_data(Resource.ICC_PROFILE)))
            rgb = ImageCms.profileToProfile(im.convert("RGB"), src, ImageCms.createProfile("sRGB"))
            im = rgb if "A" not in prev.getbands() else Image.merge("RGBA", (*rgb.split(), prev.getchannel("A")))
        A, B = flat(im), flat(png)
        if A.shape != B.shape: best = ("size",); break
        d = np.abs(A - B); mx = d.max(axis=2)
        r = (round(float((mx > 24).mean() * 100), 3), round(float(d.mean()), 3), int(mx.max()), mode)
        if best is None or r[:2] < best[:2]: best = r
    rows.append((os.path.relpath(p, C),) + best)
num = [r for r in rows if isinstance(r[1], float)]
bp = sorted(r[1] for r in num); ma = [r[2] for r in num]
print("n", len(rows), "numeric", len(num), "nopreview", sum(1 for r in rows if r[1] == "nopreview"), "size", sum(1 for r in rows if r[1] == "size"))
print("bad%% median %.3f p90 %.3f p99 %.3f max %.3f; zero-bad %d; mae mean %.3f" % (np.median(bp), np.percentile(bp, 90), np.percentile(bp, 99), max(bp), sum(1 for b in bp if b == 0), np.mean(ma)))
for r in sorted(num, key=lambda r: -r[1])[:12]: print(r)
