"""render_save.py <card rel> <out.npz> [viewport x0,y0,x1,y1]: harness-style render, saved as uint8 + timing."""
import io, json, sys, time, resource
import numpy as np
sys.path.insert(0, "/Users/michaelsrouji/Documents/Tools/wt/research-speed/_bench")
from bench_one import ROOT, icc_profile
from psd_tools import PSDImage
from psd_tools.api.pil_io import _apply_icc
rel, out = sys.argv[1], sys.argv[2]
vp = tuple(int(v) for v in sys.argv[3].split(",")) if len(sys.argv) > 3 else None
psd = PSDImage.open(ROOT + rel)
prof = icc_profile(psd)
t, c = time.perf_counter(), time.process_time()
with np.errstate(divide="ignore", invalid="ignore"):
    if vp is None:
        img = psd.composite(force=True, apply_icc=prof is None)
    else:
        img = psd.composite(viewport=vp, ignore_preview=True)
w, cpu = time.perf_counter() - t, time.process_time() - c
np.savez_compressed(out, img=np.asarray(img))
print(json.dumps({"card": rel, "wall": round(w, 2), "cpu": round(cpu, 2), "rss": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)}))
