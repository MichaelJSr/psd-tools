"""truth.py <base.json> <run.json>: cards whose figures differ between two harness runs (any field but timing)."""
import json, sys
def load(p):
    return {c["card"]: c for c in json.load(open(p))["cards"]}
a, b = load(sys.argv[1]), load(sys.argv[2])
skip = {"render_s"}
ch = [k for k in a if k in b and {x: y for x, y in a[k].items() if x not in skip} != {x: y for x, y in b[k].items() if x not in skip}]
print(len(ch), "of", len(a))
for k in ch: print(" ", k)
