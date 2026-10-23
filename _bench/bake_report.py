"""bake_report.py <base.jsonl> <proto.jsonl>: per-card bake CPU by phase, totals, output identity."""
import json, sys
a = {d["card"]: d for d in map(json.loads, open(sys.argv[1]))}
b = {d["card"]: d for d in map(json.loads, open(sys.argv[2]))}
phases = ["open", "baseline", "render", "save", "reparse", "idempotence"]
tot = {k: [0.0, 0.0] for k in phases + ["cpu"]}
for c in a:
    x, y = a[c], b.get(c, {})
    if "timings" not in x or "timings" not in y:
        print(f"{c.split('/')[-1][:26]:26s} {x.get('error', '')[:60]} | {y.get('error', '')[:60]}")
        continue
    same = x.get("psd_sha") == y.get("psd_sha") and x.get("png_sha") == y.get("png_sha")
    for k in phases:
        tot[k][0] += x["timings"][k]; tot[k][1] += y["timings"][k]
    tot["cpu"][0] += x["cpu"]; tot["cpu"][1] += y["cpu"]
    print(f"{c.split('/')[-1][:26]:26s} cpu {x['cpu']:5.2f} -> {y['cpu']:5.2f}  " + " ".join(f"{k} {x['timings'][k]:.2f}/{y['timings'][k]:.2f}" for k in phases) + (" same-bytes" if same else " BYTES-DIFFER" if x.get("psd_sha") else ""))
print("TOTAL " + " ".join(f"{k} {u:.2f}->{v:.2f}" for k, (u, v) in tot.items()))
