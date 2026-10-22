"""agg.py <prof files or dir> [--callers pat,...] [--top N]: aggregate cProfile dumps, short names."""
import glob, os, pstats, re, sys


def short(k):
    f, line, fn = k
    f = re.sub(r".*/src/psd_tools/", "", f)
    f = re.sub(r".*/site-packages/", "", f)
    f = re.sub(r".*/AutoCroissant/", "AC/", f)
    return f"{f}:{line}({fn})"


def main():
    args = sys.argv[1:]
    callers = []
    top = 40
    files = []
    i = 0
    while i < len(args):
        if args[i] == "--callers":
            callers = args[i + 1].split(","); i += 2
        elif args[i] == "--top":
            top = int(args[i + 1]); i += 2
        else:
            p = args[i]
            files += sorted(glob.glob(os.path.join(p, "*.prof"))) if os.path.isdir(p) else [p]
            i += 1
    s = pstats.Stats(files[0])
    for f in files[1:]:
        s.add(f)
    st = s.stats
    total = sum(v[2] for v in st.values())
    print(f"{len(files)} profiles, total tottime {total:.1f} s")
    print(f"{'tottime':>8} {'%':>5} {'cumtime':>8} {'ncalls':>8}  function")
    for k, v in sorted(st.items(), key=lambda kv: -kv[1][2])[:top]:
        print(f"{v[2]:8.2f} {100 * v[2] / total:5.1f} {v[3]:8.2f} {v[1]:8d}  {short(k)}")
    print("\nby cumtime (psd_tools functions)")
    for k, v in sorted(st.items(), key=lambda kv: -kv[1][3])[:top * 2]:
        if "psd_tools" in k[0] or "AutoCroissant" in k[0]:
            print(f"{v[3]:8.2f} {v[2]:8.2f} {v[1]:8d}  {short(k)}")
    for pat in callers:
        for k, v in st.items():
            if re.search(pat, short(k)):
                print(f"\ncallers of {short(k)}  tottime {v[2]:.2f} cum {v[3]:.2f}")
                for ck, cv in sorted(v[4].items(), key=lambda kv: -kv[1][2])[:12]:
                    print(f"   {cv[2]:8.2f} {cv[3]:8.2f} {cv[1]:6d}  {short(ck)}")


if __name__ == "__main__":
    main()
