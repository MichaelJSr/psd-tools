"""harness_report.py <proto run> <base run> <reference run>: identity and timing across harness runs (results/*.json + sha logs)."""
import json, statistics as st, sys
R = "/Users/michaelsrouji/Documents/Tools/psd-research/score/results/"
L = "/Users/michaelsrouji/Documents/Tools/wt/feat-speed-scratch/shalog/"
proto, base, ref = sys.argv[1:4]


def load(name):
    try:
        d = json.load(open(R + name + ".json"))
        return d["meta"], {c["card"]: c for c in d["cards"]}
    except FileNotFoundError:
        return None, {}


def shas(name):
    out = {}
    try:
        for line in open(L + name + ".log"):
            p, k, h, cpu, wall, ocpu, total, load = line.rstrip("\n").split("\t")
            rel = p.split("corpus-6e50c3ee/", 1)[-1]
            out[rel] = dict(sha=h, cpu=float(cpu), wall=float(wall), open_cpu=float(ocpu), proc_cpu=float(total), load=float(load))
    except FileNotFoundError:
        pass
    return out


def strip(c):
    return {k: v for k, v in c.items() if k not in ("render_s",)}


pm, pc = load(proto)
bm, bc = load(base)
rm, rc = load(ref)
ps, bs = shas(proto), shas(base)
print("runs:", proto, pm and pm.get("wall_s"), "s wall,", len(pc), "cards |", base, bm and bm.get("wall_s"), len(bc), "|", ref, rm and rm.get("fork"), len(rc))
common = [c for c in pc if c in rc]
same = [c for c in common if strip(pc[c]) == strip(rc[c])]
print(f"scores identical to {ref}: {len(same)}/{len(common)}")
for c in [c for c in common if c not in same][:10]:
    a, b = strip(pc[c]), strip(rc[c])
    print("  differs:", c, [k for k in a if a.get(k) != b.get(k)][:6])
both = [c for c in ps if c in bs]
eq = [c for c in both if ps[c]["sha"] == bs[c]["sha"]]
print(f"render sha identical to {base}: {len(eq)}/{len(both)}")
for c in [c for c in both if c not in eq]:
    print("  sha differs:", c)
if both:
    cb, cp = [bs[c]["cpu"] for c in both], [ps[c]["cpu"] for c in both]
    print(f"composite CPU on the {len(both)} common cards: base sum {sum(cb):.0f}s median {st.median(cb):.2f}s | proto sum {sum(cp):.0f}s median {st.median(cp):.2f}s | x{sum(cb) / sum(cp):.2f}")
    wb = [bc[c]["render_s"] for c in both if c in bc]; wp = [pc[c]["render_s"] for c in both if c in pc]
    if wb and wp:
        print(f"harness render_s (wall, load-affected) on them: base median {st.median(wb):.1f}s sum {sum(wb):.0f} | proto median {st.median(wp):.1f}s sum {sum(wp):.0f}")
if ps:
    allp = [v["cpu"] for v in ps.values()]
    allpc = [v["proc_cpu"] for v in ps.values()]
    print(f"proto, all {len(ps)} cards: composite CPU sum {sum(allp):.0f}s median {st.median(allp):.2f}s p90 {sorted(allp)[int(.9 * len(allp))]:.2f}s max {max(allp):.2f}s; process CPU at composite end median {st.median(allpc):.2f}s")
    if pc:
        w = sorted(c["render_s"] for c in pc.values())
        print(f"proto harness render_s (wall): median {st.median(w):.1f} p90 {w[int(.9 * len(w))]:.1f} max {w[-1]:.1f}; load1 median {st.median(v['load'] for v in ps.values()):.0f}")
    if rc:
        w = sorted(c["render_s"] for c in rc.values())
        print(f"{ref} render_s (wall, its own load): median {st.median(w):.1f} p90 {w[int(.9 * len(w))]:.1f} max {w[-1]:.1f}; wall {rm['wall_s']}s")
