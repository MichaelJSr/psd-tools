"""open_bench.py <reps> [cards file]: PSDImage.open per card, best CPU of reps; sha1 of psd.save() bytes as the parse fingerprint."""
import hashlib, io, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_one import ROOT
from psd_tools import PSDImage
reps = int(sys.argv[1])
cards = [l.strip() for l in open(sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "cards20.txt")) if l.strip()]
out = {}
for c in cards:
    data = open(ROOT + c, "rb").read()
    best = 1e9
    for _ in range(reps):
        t = time.process_time()
        psd = PSDImage.open(io.BytesIO(data))
        best = min(best, time.process_time() - t)
    buf = io.BytesIO()
    psd.save(buf)
    texts = hashlib.sha1(repr([(l.name, l.kind, l.bbox, l.text if l.kind == "type" else None) for l in psd.descendants()]).encode()).hexdigest()[:12]
    out[c] = {"cpu": round(best, 4), "save_sha": hashlib.sha1(buf.getvalue()).hexdigest()[:12], "tree": texts}
print(json.dumps({"src": __import__("psd_tools").__file__, "total_cpu": round(sum(v["cpu"] for v in out.values()), 3), "cards": out}))
