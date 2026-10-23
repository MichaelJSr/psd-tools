"""batch_bench.py <mode> <cards file> [workers]: the data build's per-card work (parse + psd_details.gather).

serial: as card_data does it (parse opens the PSD, gather opens it again)
shared: one PSDImage.open per card, handed to both
pool:   shared, in a process pool of <workers>
Prints one JSON line: wall, cpu (this process + children), per-card results digest.
"""
import hashlib, json, os, resource, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_one

_parser = None


def _setup():
    global _parser
    if _parser is None:
        _parser = bench_one._parser()
    return _parser


def one(rel, shared=True):
    parser = _setup()
    from psd_tools import PSDImage
    from commands import psd_parser, psd_details
    path = bench_one.ROOT + rel
    if shared:
        psd = PSDImage.open(path)
        real_open = psd_parser.PSDImage.open
        psd_parser.PSDImage.open = lambda p, *a, **k: psd if p == path else real_open(p, *a, **k)
        try:
            card = parser.parse(path, rel)
        finally:
            psd_parser.PSDImage.open = real_open
        record = psd_details.gather(psd, rel, card.name, None, tuple(parser.all_types))
    else:
        card = parser.parse(path, rel)
        record = psd_details.gather(path, rel, card.name, None, tuple(parser.all_types))
    record.pop("at", None)
    digest = hashlib.sha1(json.dumps([card.to_dict(), record], sort_keys=True, default=str).encode()).hexdigest()[:12]
    return rel, digest


def main():
    mode, cards_f = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    cards = [l.strip() for l in open(cards_f) if l.strip()]
    _setup()
    from commands import psd_details  # noqa: F401  imports outside the timer
    one(cards[0], True)              # warm
    t, c = time.perf_counter(), time.process_time()
    if mode == "pool":
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers, initializer=_setup) as pool:
            results = list(pool.map(one, cards, chunksize=4))
    else:
        results = [one(r, mode == "shared") for r in cards]
    wall = time.perf_counter() - t
    cpu = time.process_time() - c + resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
    print(json.dumps({"mode": mode, "src": __import__("psd_tools").__file__.rsplit("/src/", 1)[0].rsplit("/", 1)[-1], "n": len(cards), "wall": round(wall, 2), "cpu": round(cpu, 2), "digest": hashlib.sha1(json.dumps(sorted(results)).encode()).hexdigest()[:12], "load1": round(os.getloadavg()[0], 1)}))


if __name__ == "__main__":
    main()
