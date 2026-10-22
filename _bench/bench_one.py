"""bench_one.py <mode> <card rel path> [prof_out]: one card, one process, JSON line on stdout.

mode full: what score_cards.render() times (open + composite(force=True) + ICC), split into parts.
mode bot:  open, parser parse, Name/Ability render_region as psd_edit does, full no-force composite.
"""
import io, json, os, resource, sys, time

ROOT = "/Users/michaelsrouji/Desktop/TTSCardMaker/"
AC = "/Users/michaelsrouji/Desktop/AutoCroissant"


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


def full(rel, out):
    import numpy as np
    from psd_tools import PSDImage
    from psd_tools.api.pil_io import _apply_icc
    p = ROOT + rel
    with T(out, "open"):
        psd = PSDImage.open(p)
    prof = icc_profile(psd)
    with T(out, "composite"):
        with np.errstate(divide="ignore", invalid="ignore"):
            img = psd.composite(force=True, apply_icc=prof is None)
    with T(out, "icc"):
        if prof is not None and img.mode in ("RGB", "RGBA"):
            _apply_icc(img, prof)
    out["layers"] = sum(1 for _ in psd.descendants())
    out["size"] = [psd.width, psd.height]
    return img


def bot(rel, out):
    sys.path.insert(0, AC)
    os.chdir(AC)
    import numpy as np
    from psd_tools import PSDImage
    with T(out, "import_bot"):
        from commands import psd_edit
        from commands.psd_analyzer import PSDParser, RepositoryTraverser, StatsDatabase
        from commands.psd_regions import fx_pad
    with T(out, "parser_setup"):
        db = StatsDatabase()
        tr = RepositoryTraverser(db)
        tr._populate_types_from_local(ROOT)
        parser = PSDParser(db.all_types, db, tr._icons_from_local(ROOT))
    p = ROOT + rel
    with T(out, "open"):
        psd = PSDImage.open(p)
    with T(out, "open2"):          # warm: page cache, imports
        psd = PSDImage.open(p)
    with T(out, "parse"):
        parser.parse(p, rel)
    with T(out, "descendants"):
        n = sum(1 for _ in psd.descendants())
    out["layers"] = n
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
                psd_edit.render_region(psd, region)
            with T(out, which + "_render2"):     # same PSDImage again (any caching?)
                psd_edit.render_region(psd, region)
        with T(out, "full_noforce"):
            psd.composite(ignore_preview=True)


def main():
    mode, rel = sys.argv[1], sys.argv[2]
    prof_out = sys.argv[3] if len(sys.argv) > 3 else None
    out = {"card": rel, "mode": mode, "rss_start_mb": round(rss_mb(), 1)}
    import numpy, psd_tools   # noqa: F401  import cost outside the timers
    out["psd_tools"] = psd_tools.__file__
    fn = full if mode == "full" else bot
    t = time.perf_counter()
    if prof_out:
        import cProfile
        pr = cProfile.Profile()
        pr.enable()
        fn(rel, out)
        pr.disable()
        pr.dump_stats(prof_out)
    else:
        fn(rel, out)
    out["total"] = round(time.perf_counter() - t, 3)
    out["peak_rss_mb"] = round(rss_mb(), 1)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
