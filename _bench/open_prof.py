"""open_prof.py <out.prof> [cards]: cProfile of PSDImage.open over the cards (bytes preloaded)."""
import cProfile, io, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_one import ROOT
from psd_tools import PSDImage
cards = [l.strip() for l in open(sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "cards20.txt")) if l.strip()]
datas = [open(ROOT + c, "rb").read() for c in cards]
PSDImage.open(io.BytesIO(datas[0]))
pr = cProfile.Profile()
pr.enable()
for d in datas:
    PSDImage.open(io.BytesIO(d))
pr.disable()
pr.dump_stats(sys.argv[1])
