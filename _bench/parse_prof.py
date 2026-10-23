"""parse_prof.py <out.prof> [cards file] [reps]: open + bot parse of each card in one process, cProfiled; prints per-card CPU."""
import cProfile, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_one
from psd_tools import PSDImage
cards = [l.strip() for l in open(sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "cards20.txt")) if l.strip()]
parser = bench_one._parser()
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 1
for c in cards[:2]:
    parser.parse(bench_one.ROOT + c, c)       # warm imports
rows = []
pr = cProfile.Profile()
for r in range(reps):
    for c in cards:
        t, w = time.process_time(), time.perf_counter()
        pr.enable()
        parser.parse(bench_one.ROOT + c, c)
        pr.disable()
        rows.append((c, round(time.process_time() - t, 3), round(time.perf_counter() - w, 3)))
pr.dump_stats(sys.argv[1])
tot = sum(r[1] for r in rows)
print(json.dumps({"cards": len(rows), "cpu_total": round(tot, 2), "per_card": rows}))
