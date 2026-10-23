"""trace_card.py <src root> <psd> <out.json>: every line of psd_tools the harness render of one card executes.

sys.monitoring LINE events, each location disabled after its first hit, so the
trace costs about nothing. Started before psd_tools is imported, so import-time
lines count (a module-level change selects every card)."""
import json, sys, time
from sys import monitoring as M

src, psd_path, out = sys.argv[1].rstrip("/") + "/", sys.argv[2], sys.argv[3]
sys.path.insert(0, src)
seen = set()
TOOL = 4
M.use_tool_id(TOOL, "impact")

def on_line(code, line):
    f = code.co_filename
    if f.startswith(src):
        seen.add((f[len(src):], line))
    return M.DISABLE

M.register_callback(TOOL, M.events.LINE, on_line)
M.set_events(TOOL, M.events.LINE)
t = time.process_time()
import io
import numpy as np
from PIL import ImageCms
from psd_tools import PSDImage
from psd_tools.constants import ColorMode, Resource
psd = PSDImage.open(psd_path)
with np.errstate(all="ignore"):
    img = psd.composite(force=True)          # the harness renders with force=True
cpu = time.process_time() - t
M.set_events(TOOL, 0)
by_file = {}
for f, line in seen:
    by_file.setdefault(f, []).append(line)
json.dump({"psd": psd_path, "cpu": round(cpu, 2), "lines": {f: sorted(v) for f, v in sorted(by_file.items())}}, open(out, "w"))
print(psd_path.rsplit("/", 1)[1], "cpu", round(cpu, 2), "lines", len(seen))
