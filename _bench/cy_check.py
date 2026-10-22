import sys, time, numpy as np
sys.path.insert(0, "/Users/michaelsrouji/Documents/Tools/wt/research-speed/_bench/cy")
from psd_tools.composite import effects as E
import _chamfer
rng = np.random.default_rng(0)
cases = []
for h, w in [(1, 1), (1, 7), (7, 1), (2, 3), (5, 5), (40, 33), (300, 200)]:
    for dens in (0.0, 0.05, 0.5, 0.95, 1.0):
        cases.append(rng.random((h, w)) < dens)
yy, xx = np.mgrid[:600, :900]
cases.append(((yy - 300) ** 2 + (xx - 450) ** 2) < 200 ** 2)
for m in cases:
    a, b = E._chamfer_transform(m), _chamfer.chamfer_transform(m)
    assert a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True), (m.shape, np.nanmax(np.abs(a - b)))
print("bit-equal on", len(cases), "masks")
m = (rng.random((2400, 1200)) < 0.3) | (((yy[:1, :1]) == 0))
m = np.zeros((2400, 1200), bool); m[200:2200, 100:1100] = True
for f in (E._chamfer_transform, _chamfer.chamfer_transform):
    t = time.process_time(); [f(m) for _ in range(3)]; print(f.__module__, round((time.process_time() - t) / 3, 4), "s on 2400x1200")
m = m[1700:2300, :]
for f in (E._chamfer_transform, _chamfer.chamfer_transform):
    t = time.process_time(); [f(m) for _ in range(5)]; print(f.__module__, round((time.process_time() - t) / 5, 4), "s on 600x1200")
