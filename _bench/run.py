"""run.py <tag> <mode> <src,src2,...> [--cards F] [--workers N] [--prof] [--save] [--reps R]

Each card in its own process; with several src trees the runs interleave per card
(a, b, a, b ...) so load drifts hit both alike. JSON lines -> OUT/<tag>.jsonl.
"""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

B = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("SPEED_OUT", "/Users/michaelsrouji/Documents/Tools/wt/feat-speed-scratch/out")
PY = os.environ.get("SPEED_PY", "/Users/michaelsrouji/.venvs/ac-py314-fork/bin/python")


def main():
    a = sys.argv[1:]
    tag, mode, srcs = a[0], a[1], a[2].split(",")
    cards_f = a[a.index("--cards") + 1] if "--cards" in a else os.path.join(B, "cards20.txt")
    workers = int(a[a.index("--workers") + 1]) if "--workers" in a else 2
    reps = int(a[a.index("--reps") + 1]) if "--reps" in a else 1
    cards = [l.strip() for l in open(cards_f) if l.strip()]
    os.makedirs(os.path.join(OUT, tag), exist_ok=True)
    jobs = []
    for r in range(reps):
        for i, c in enumerate(cards):
            for s in srcs:
                jobs.append((r, i, c, s))

    def one(job):
        r, i, c, s = job
        name = os.path.basename(s.rstrip("/"))
        cmd = [PY, os.path.join(B, "bench_one.py"), mode, c]
        if "--prof" in a:
            cmd += ["--prof", os.path.join(OUT, tag, f"{name}_{i}_{r}.prof")]
        if "--save" in a:
            cmd += ["--save", os.path.join(OUT, tag, f"{name}_{i}.npz")]
        env = dict(os.environ, PYTHONPATH=os.path.join(s, "src"), PYTHONDONTWRITEBYTECODE="1")
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else json.dumps({"card": c, "error": p.stderr[-400:]})
        d = json.loads(line)
        d.update(tree=name, rep=r, idx=i)
        return d

    with open(os.path.join(OUT, tag + ".jsonl"), "w") as fh, ThreadPoolExecutor(workers) as pool:
        for d in pool.map(one, jobs):
            fh.write(json.dumps(d) + "\n")
            fh.flush()
            print(d.get("tree"), d.get("idx"), d.get("card"), d.get("composite", d.get("total")), d.get("error", "")[:200], flush=True)


if __name__ == "__main__":
    main()
