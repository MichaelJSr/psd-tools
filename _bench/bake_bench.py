"""bake_bench.py <out dir> <field> <card>...: psd_edit.apply_jobs on a +1 stat (or a name/ability tweak) per card; JSON lines with the bake's timings."""
import json, os, sys, time, resource
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_one
import numpy as np
from psd_tools import PSDImage
parser = bench_one._parser()
from commands import psd_edit
psd_edit.perf_counter = time.process_time   # the bake's own timings in CPU seconds: the machine is shared
FONTS = os.environ.get("SPEED_FONTS", "/Users/michaelsrouji/Desktop/TTSCardMaker/MDW/Fonts")
out_dir, field = sys.argv[1], sys.argv[2]
chart_path = bench_one.ROOT + psd_edit.STATUS_CHART
chart = psd_edit.status_fills(PSDImage.open(chart_path)) if os.path.exists(chart_path) else None
cards = [l.strip() for l in open(sys.argv[3]) if l.strip()] if sys.argv[3].endswith(".txt") else sys.argv[3:]
for rel in cards:
    psd_path = bench_one.ROOT + rel
    png_path = psd_path[:-4] + ".png"
    row = {"card": rel, "field": field}
    t, c = time.perf_counter(), time.process_time()
    try:
        parsed = psd_edit.parse_card(parser, psd_path, rel)
        card = parsed["card"]
        if field in psd_edit.STAT_FIELDS:
            value = card.get(field)
            jobs = {field: (value if isinstance(value, int) and value >= 0 else 1) % 9 + 1}
        elif field == "name":
            jobs = {"name": (parsed["info"].display_name or parsed["info"].name) + " X"}
        else:
            jobs = {"ability": (card.get("ability") or "") + " Test."}
        os.makedirs(out_dir, exist_ok=True)
        with np.errstate(all="ignore"):
            baked = psd_edit.apply_jobs(psd_path, png_path, rel, jobs, parsed, parser, out_dir, FONTS, chart, update_preview=True)
        row["timings"] = {k: round(v, 3) for k, v in baked.timings.items()}
        row["region"] = list(baked.region)
        import hashlib
        for key, p in (("psd_sha", baked.psd_path), ("png_sha", baked.png_path)):
            row[key] = hashlib.sha1(open(p, "rb").read()).hexdigest()[:12]
            os.remove(p)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"[:160]
    row["wall"] = round(time.perf_counter() - t, 3)
    row["cpu"] = round(time.process_time() - c, 3)
    row["src"] = __import__("psd_tools").__file__.rsplit("/src/", 1)[0].rsplit("/", 1)[-1]
    print(json.dumps(row), flush=True)
