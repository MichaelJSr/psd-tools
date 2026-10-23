import numpy as np, sys
sys.path.insert(0, sys.argv[1])
from psd_tools.composite import adjustments as A
def old(values, lut):
    lut_size = int(lut.shape[0]); depth = lut_size - 1
    values = np.floor(values * depth) / depth
    return np.interp(values, A._lut_domain(lut_size), lut).astype(np.float32, copy=False)
rng = np.random.default_rng(1)
for size in (256, 65536):
    for trial in range(20):
        lut = np.sort(rng.random(size)).astype(np.float32) if trial % 2 else rng.random(size).astype(np.float32)
        v = rng.random((300, 400, 3)).astype(np.float32) * 1.2 - 0.1
        v[0, :10, 0] = [0, 1, 0.5, 1/255, 254/255, 1-1e-7, 1e-7, -0.0, 2, -1]
        a, b = old(v, lut), A._apply_lut(v, lut)
        assert a.dtype == b.dtype and a.shape == b.shape, (a.shape, b.shape)
        assert np.array_equal(a, b), (size, trial, np.abs(a - b).max())
import time
v = np.random.default_rng(2).random((2400, 1200, 3)).astype(np.float32); lut = np.sort(np.random.default_rng(3).random(256)).astype(np.float32)
for f in (old, A._apply_lut):
    t = time.process_time(); [f(v, lut) for _ in range(5)]; print(f.__name__, round((time.process_time() - t) / 5, 4), "s per 1200x2400x3 call")
print("bit-equal on 40 random LUTs incl. out-of-range values")
