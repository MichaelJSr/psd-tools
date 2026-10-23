"""sha_compare.py <log a> <log b>: composite sha logs (sitecustomize) compared by card and call."""
import sys


def load(p):
    out = {}
    for line in open(p):
        f = line.rstrip("\n").split("\t")
        out[(f[0].split("corpus-6e50c3ee/", 1)[-1], f[1])] = (f[2], float(f[3]))
    return out


a, b = load(sys.argv[1]), load(sys.argv[2])
common = sorted(set(a) & set(b))
diff = [k for k in common if a[k][0] != b[k][0]]
print(f"{len(common)} common composites, {len(common) - len(diff)} identical, {len(diff)} differ")
for k in diff[:20]:
    print("  differs:", k[0])
ca, cb = sum(a[k][1] for k in common), sum(b[k][1] for k in common)
print(f"composite CPU: {ca:.0f}s vs {cb:.0f}s (x{ca / cb if cb else 0:.2f})")
