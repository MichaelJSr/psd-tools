import sys, numpy as np
a, b = np.load(sys.argv[1])["img"].astype(np.int16), np.load(sys.argv[2])["img"].astype(np.int16)
if a.shape != b.shape:
    print("SHAPE", a.shape, b.shape); sys.exit()
d = np.abs(a - b)
px = d.max(axis=-1) if d.ndim == 3 else d
print(f"max {int(d.max())} px_diff {int((px > 0).sum())} px_gt1 {int((px > 1).sum())} px_gt24 {int((px > 24).sum())} of {px.size}")
