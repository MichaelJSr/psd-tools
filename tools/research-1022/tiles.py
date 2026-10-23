"""tiles.py <psd> <bands>...: a full-card render as one pass and as horizontal bands; CPU time and bit-equality."""
import sys, time
import numpy as np
from psd_tools import PSDImage
p = sys.argv[1]
psd = PSDImage.open(p)
psd.composite(viewport=(0, 0, 8, 8), ignore_preview=True)      # warm imports
def cpu(f):
    t = time.process_time(); r = f(); return r, time.process_time() - t
full, tf = cpu(lambda: np.asarray(psd.composite(ignore_preview=True, apply_icc=False)))
print("full cpu %.1f" % tf, full.shape)
W, H = psd.width, psd.height
for n in map(int, sys.argv[2:]):
    edges = [round(H * k / n) for k in range(n + 1)]
    def run():
        return np.concatenate([np.asarray(psd.composite(viewport=(0, edges[k], W, edges[k + 1]), ignore_preview=True, apply_icc=False)) for k in range(n)], axis=0)
    tiled, tt = cpu(run)
    d = np.abs(tiled.astype(int) - full.astype(int))
    print("bands %d cpu %.1f  speedup %.2fx  max diff %d  px differ %d" % (n, tt, tf / tt, d.max(), int((d.max(axis=2) > 0).sum())))
