"""full_prof1.py <out.prof> <card>...: cProfile of composite(force=True) per card, in one process (imports warmed)."""
import cProfile, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_one import ROOT
import numpy as np
from psd_tools import PSDImage
PSDImage.open(ROOT + "Field/1 Stars/The_Freezer.psd").composite(force=True)  # warm imports
pr = cProfile.Profile()
for c in sys.argv[2:]:
    psd = PSDImage.open(ROOT + c)
    with np.errstate(all="ignore"):
        t = time.process_time()
        pr.enable()
        psd.composite(force=True)
        pr.disable()
    print(c, round(time.process_time() - t, 2))
pr.dump_stats(sys.argv[1])
