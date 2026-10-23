"""region_prof.py <out.prof> <which> <card>...: cProfile of render_region on the Name or Ability box, as psd_edit pads it."""
import cProfile, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_one
import numpy as np
from psd_tools import PSDImage
bench_one._parser()
from commands import psd_edit
from commands.psd_regions import fx_pad
out, which = sys.argv[1], sys.argv[2]
pr = cProfile.Profile()
for c in sys.argv[3:]:
    psd = PSDImage.open(bench_one.ROOT + c)
    layer = psd_edit.name_layer(psd) if which == "name" else psd_edit.ability_layer(psd)
    region = psd_edit._pad(tuple(layer.bbox), fx_pad(layer), psd)
    with np.errstate(all="ignore"):
        psd_edit.render_region(psd, region)       # warm imports
        t = time.process_time()
        pr.enable()
        psd_edit.render_region(psd, region)
        pr.disable()
    print(c, region, round(time.process_time() - t, 3))
pr.dump_stats(out)
