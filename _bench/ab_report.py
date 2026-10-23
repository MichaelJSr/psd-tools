"""ab_report.py <tag.jsonl> [base tree name] [key]: per card CPU base vs others, identity of sha/float_sha."""
import json, sys, statistics as st
rows = [json.loads(l) for l in open(sys.argv[1])]
base = sys.argv[2] if len(sys.argv) > 2 else "feat-speed-base"
key = sys.argv[3] if len(sys.argv) > 3 else "composite"
by = {}
for r in rows:
    by.setdefault(r["card"], {}).setdefault(r["tree"], []).append(r)
trees = sorted({r["tree"] for r in rows} - {base})
tot = {t: [0.0, 0.0] for t in trees}
for card, d in by.items():
    if base not in d:
        continue
    b = min(d[base], key=lambda r: r.get(key + "_cpu", 1e9))
    line = f"{card.split('/')[-1][:28]:28s} base {b.get(key + '_cpu', float('nan')):7.2f}s rss {b.get('peak_rss_mb', 0):6.0f}"
    for t in trees:
        if t not in d:
            continue
        p = min(d[t], key=lambda r: r.get(key + "_cpu", 1e9))
        same = "=" if (p.get("sha") == b.get("sha") and p.get("float_sha") == b.get("float_sha")) else ("u8=" if p.get("sha") == b.get("sha") else "DIFF")
        bc, pc = b.get(key + "_cpu", 0), p.get(key + "_cpu", 0)
        tot[t][0] += bc; tot[t][1] += pc
        line += f" | {t} {pc:7.2f}s x{bc / pc if pc else 0:5.2f} rss {p.get('peak_rss_mb', 0):6.0f} {same}"
    print(line)
for t, (bc, pc) in tot.items():
    print(f"TOTAL {key}_cpu base {bc:.1f}s  {t} {pc:.1f}s  x{bc / pc if pc else 0:.2f}")
