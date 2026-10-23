"""bench_one.py <mode> <card rel> [--prof F] [--save F.npz]: one card, one process, one JSON line.

full:  what score_cards.render() times (open + composite(force=True) + ICC), split into parts;
       sha1 of the uint8 render and of the float colour/alpha the compositor returned.
bot:   open, the bot parser's parse, Name/Ability render_region as psd_edit pads them (twice each).
parse: open + parse only.
Paths: SPEED_ROOT (corpus), SPEED_AC (frozen bot code).
"""
import hashlib, io, json, os, resource, sys, time

ROOT = os.environ.get("SPEED_ROOT", "/Users/michaelsrouji/Documents/Tools/psd-research/corpus-6e50c3ee") + "/"
AC = os.environ.get("SPEED_AC", "/Users/michaelsrouji/Documents/Tools/wt/feat-speed-scratch/ac")


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20   # bytes on macOS


class T:
    def __init__(self, out, key):
        self.out, self.key = out, key

    def __enter__(self):
        self.w, self.c = time.perf_counter(), time.process_time()

    def __exit__(self, *a):
        self.out[self.key] = round(time.perf_counter() - self.w, 3)
        self.out[self.key + "_cpu"] = round(time.process_time() - self.c, 3)


def icc_profile(psd):
    from PIL import ImageCms
    from psd_tools.constants import ColorMode, Resource
    if psd.color_mode != ColorMode.RGB or Resource.ICC_PROFILE not in psd.image_resources:
        return None
    b = psd.image_resources.get_data(Resource.ICC_PROFILE)
    try:
        desc = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(b)))
    except Exception:
        return b
    return None if "srgb" in desc.lower() else b


def _sha(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    return hashlib.sha1(str((a.dtype.str, a.shape)).encode() + a.tobytes()).hexdigest()[:16]


def _capture_floats(out):
    """Wrap the compositor's entry so the float result is hashed too."""
    m = sys.modules["psd_tools.composite.composite"]
    inner = m.composite

    def wrapped(*a, **k):
        color, shape, alpha = inner(*a, **k)
        out.setdefault("float_sha", []).append(_sha(color) + "/" + _sha(alpha))
        return color, shape, alpha
    m.composite = wrapped


def full(rel, out, save=None):
    import numpy as np
    from psd_tools import PSDImage
    from psd_tools.api.pil_io import _apply_icc
    import psd_tools.composite.composite  # noqa: F401
    _capture_floats(out)
    p = ROOT + rel
    with T(out, "open"):
        psd = PSDImage.open(p)
    prof = icc_profile(psd)
    with T(out, "composite"):
        with np.errstate(divide="ignore", invalid="ignore"):
            img = psd.composite(force=True, apply_icc=prof is None)
    with T(out, "icc"):
        if prof is not None and img.mode in ("RGB", "RGBA"):
            img = _apply_icc(img, prof) or img
    arr = np.asarray(img)
    out["sha"] = _sha(arr)
    out["layers"] = sum(1 for _ in psd.descendants())
    out["size"] = [psd.width, psd.height]
    if save:
        np.savez_compressed(save, img=arr)


def _parser():
    sys.path.insert(0, AC)
    os.chdir(AC)
    from commands.card_data import CardBuild
    from commands.card_models import StatsDatabase
    db = StatsDatabase()
    b = CardBuild(db)
    b._populate_types_from_local(ROOT)
    from commands.psd_parser import PSDParser
    return PSDParser(db.all_types, db, b._icons_from_local(ROOT))


def parse(rel, out, save=None):
    from psd_tools import PSDImage   # noqa: F401
    with T(out, "import_bot"):
        parser = _parser()
    with T(out, "parse"):
        card = parser.parse(ROOT + rel, rel)
    out["ability_len"] = len(card.ability or "")


def bot(rel, out, save=None):
    import numpy as np
    from psd_tools import PSDImage
    import psd_tools.composite.composite  # noqa: F401
    with T(out, "import_bot"):
        parser = _parser()
        from commands import psd_edit
        from commands.psd_regions import fx_pad
    p = ROOT + rel
    with T(out, "open"):
        psd = PSDImage.open(p)
    with T(out, "parse"):
        parser.parse(p, rel)
    with np.errstate(divide="ignore", invalid="ignore"):
        for which in ("name", "ability"):
            try:
                layer = psd_edit.name_layer(psd) if which == "name" else psd_edit.ability_layer(psd)
            except Exception as e:
                out[which + "_err"] = str(e)[:80]
                continue
            region = psd_edit._pad(tuple(layer.bbox), fx_pad(layer), psd)
            out[which + "_region"] = list(region)
            with T(out, which + "_render"):
                img = psd_edit.render_region(psd, region)
            out[which + "_sha"] = _sha(np.asarray(img))
            with T(out, which + "_render2"):
                psd_edit.render_region(psd, region)


def main():
    args = sys.argv[1:]
    mode, rel = args[0], args[1]
    prof_out = args[args.index("--prof") + 1] if "--prof" in args else None
    save = args[args.index("--save") + 1] if "--save" in args else None
    out = {"card": rel, "mode": mode}
    import numpy, psd_tools   # noqa: F401  import cost outside the timers
    out["src"] = psd_tools.__file__.rsplit("/src/", 1)[0].rsplit("/", 1)[-1]
    fn = {"full": full, "bot": bot, "parse": parse}[mode]
    t, c = time.perf_counter(), time.process_time()
    if prof_out:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()
        fn(rel, out, save)
        pr.disable()
        pr.dump_stats(prof_out)
    else:
        fn(rel, out, save)
    out["total"] = round(time.perf_counter() - t, 3)
    out["total_cpu"] = round(time.process_time() - c, 3)
    out["peak_rss_mb"] = round(rss_mb(), 1)
    out["load1"] = round(os.getloadavg()[0], 1)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
