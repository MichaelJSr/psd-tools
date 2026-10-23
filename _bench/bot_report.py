"""bot_report.py <tag.jsonl> <base tree> <proto tree>: open/parse/region renders, CPU, identity."""
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
base, proto = sys.argv[2], sys.argv[3]
by = {}
for r in rows:
    by.setdefault(r["card"], {})[r["tree"]] = r
tot = {}
def g(r, k):
    return r.get(k + "_cpu")
for c, d in by.items():
    b, p = d.get(base), d.get(proto)
    if not b or not p:
        continue
    s = f"{c.split('/')[-1][:24]:24s} open {g(b,'open'):.2f}/{g(p,'open'):.2f} parse {g(b,'parse'):.2f}/{g(p,'parse'):.2f}"
    for k in ("open", "parse"):
        tot.setdefault(k, [0, 0]); tot[k][0] += g(b, k); tot[k][1] += g(p, k)
    for w in ("name", "ability"):
        if g(b, w + "_render") is not None:
            same = b.get(w + "_sha") == p.get(w + "_sha")
            for k in (w + "_render", w + "_render2"):
                tot.setdefault(k, [0, 0]); tot[k][0] += g(b, k); tot[k][1] += g(p, k)
            s += f" {w} {g(b,w+'_render'):.2f}->{g(p,w+'_render'):.2f} {'=' if same else 'DIFF'}"
    print(s)
for k, (x, y) in tot.items():
    print(f"TOTAL {k:16s} {x:7.2f}s -> {y:7.2f}s  x{x / y if y else 0:.2f}")
