import sys, numpy as np
a = np.load(sys.argv[1]); b = np.load(sys.argv[2])
for k in a.files:
    x, y = a[k].astype(int), b[k].astype(int)
    print(k, x.shape, 'max diff', int(np.abs(x - y).max()) if x.shape == y.shape else 'shape!', 'px differ', int((np.abs(x - y).max(axis=-1) > 0).sum()) if x.shape == y.shape else '')
