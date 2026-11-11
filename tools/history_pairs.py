"""Score the renderer on history pairs: a card PSD and the PNG a person pushed with it.

    python tools/history_pairs.py list <clone> <pairs.tsv> [--rev REV]
    PYTHONPATH=src python tools/history_pairs.py score <pairs.tsv> <out.jsonl>
        [--clone C] [--cache DIR] [--all] [--only LIST] [--batch 100] [--workers 2]
    python tools/history_pairs.py sum <out.jsonl> [<base.jsonl>] [--feature NAME]

A history pair is a PSD and its PNG changed in the same commit by a person, so
the PNG is that person's Photoshop export of that PSD version. Pairs are fixed
by blob sha: later sweeps cannot erode them.

``list`` reads only the log of a blobless clone (no blob is fetched) and writes
one row per pair: path, commit, date, author, psd and png blob, and ``live``
(the path is a card at REV) / ``newest`` (its newest pair) flags. Bot commits,
``Card-sweep:`` commits and the agents' commits are left out.

``score`` renders each PSD with ``composite(ignore_preview=True)`` and diffs it
with the PNG, both over white: bad = max RGB difference over 24 (bench's rule),
taking the better of the raw and the colour-converted render; ``mae`` is the
mean RGB difference. It scores each live card's newest pair unless told
otherwise. Blobs come from ``--cache`` (files hashed in place), then ``--clone``
(read only), then GitHub: ``--batch`` pairs per throwaway repo, one batched
fetch each, deleted after. One process per pair; rows are appended, so a rerun
resumes. ``tools/history_pairs.tsv`` is the pinned list: score it as it is.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from PIL import Image

from psd_tools import PSDImage
from psd_tools.api.pil_io import _apply_icc
from psd_tools.constants import ColorMode, Resource

THRESH = 24
TIMEOUT = 900
URL = "https://github.com/MichaelJSr/TTSCardMaker.git"
BOTS = {"AutoCroissant", "github-actions[bot]"}
# Agents commit as this author from this date on, often by script.
AGENT, AGENT_SINCE = "MichaelJSr", "2026-10-01"
COLS = ("path", "commit", "date", "author", "psd", "png", "live", "newest")
ENV = dict(os.environ, GIT_NO_LAZY_FETCH="1")
Row = dict[str, Any]


def git(gd: str, *args: str, data: bytes | None = None) -> bytes:
    """Run git on a git dir, never fetching a missing blob."""
    return subprocess.run(
        ["git", "--git-dir", gd, *args],
        input=data,
        env=ENV,
        capture_output=True,
        check=True,
    ).stdout


def git_dir(clone: str) -> str:
    """The clone's common git dir (a worktree's too)."""
    out = subprocess.run(
        ["git", "-C", clone, "rev-parse", "--path-format=absolute", "--git-common-dir"],
        env=ENV,
        capture_output=True,
        check=True,
        text=True,
    )
    return out.stdout.strip()


def is_hand(author: str, date: str, subject: str, body: str) -> bool:
    """Whether a commit's PNGs are a person's exports."""
    if author in BOTS or "Card-sweep:" in body or subject.startswith("Website"):
        return False
    return not (author == AGENT and date >= AGENT_SINCE)


# ------------------------------ list ------------------------------
def list_pairs(clone: str, rev: str) -> list[Row]:
    """Every hand pair in the log, newest first; trees only, no blob read."""
    gd = git_dir(clone)
    fmt = "--format=%x01%H%x02%an%x02%as%x02%s%x02%B%x03"
    raw = git(
        gd,
        "log",
        "--no-merges",
        "--raw",
        "--no-renames",
        "--no-abbrev",
        "-z",
        fmt,
        rev,
        "--",
        "*.psd",
        "*.png",
    )
    rows: list[Row] = []
    seen = set()
    for chunk in raw.split(b"\x01")[1:]:
        head, _, rest = chunk.partition(b"\x03")
        commit, author, date, subject, body = head.decode("utf-8", "replace").split(
            "\x02", 4
        )
        if not is_hand(author, date, subject, body):
            continue
        parts = rest.split(b"\0")
        files = {}
        for i, p in enumerate(parts[:-1]):
            meta = p.strip(b"\n")[1:].split() if p.strip(b"\n").startswith(b":") else []
            if len(meta) == 5 and meta[4] in (b"A", b"M"):
                files[parts[i + 1].decode()] = meta[3].decode()
        for path, sha in files.items():
            png = files.get(path[:-4] + ".png")
            if path.endswith(".psd") and png and (path, sha, png) not in seen:
                seen.add((path, sha, png))
                rows.append(
                    {
                        "path": path,
                        "commit": commit[:12],
                        "date": date,
                        "author": author,
                        "psd": sha,
                        "png": png,
                    }
                )
    names = set(git(gd, "ls-tree", "-r", "--name-only", "-z", rev).decode().split("\0"))
    live = {
        p
        for p in names
        if p.endswith(".psd") and not p.startswith("MDW/") and p[:-4] + ".png" in names
    }
    had = set()
    for r in rows:
        r["live"] = int(r["path"] in live)
        r["newest"] = int(r["live"] and r["path"] not in had)
        had.add(r["path"])
    return rows


def read_tsv(path: str) -> list[Row]:
    """Rows of a pairs file."""
    with open(path, encoding="utf-8") as f:
        head = f.readline().rstrip("\n").split("\t")
        return [
            dict(zip(head, line.rstrip("\n").split("\t"), strict=True))
            for line in f
            if line.strip()
        ]


def write_tsv(rows: list[Row], path: str) -> None:
    """Write a pairs file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(COLS) + "\n")
        f.writelines("\t".join(str(r[c]) for c in COLS) + "\n" for r in rows)


# ------------------------------ one pair ------------------------------
def load(ref: str) -> bytes:
    """A file's bytes, or a blob's as ``<git dir>::<sha>``."""
    gd, _, name = ref.rpartition("::")
    if gd:
        return git(gd, "cat-file", "blob", name)
    with open(name, "rb") as f:
        return f.read()


def over_white(im: Image.Image) -> np.ndarray:
    """RGB over white, int16 (H, W, 3)."""
    a = np.asarray(im.convert("RGBA"), np.float32)
    k = a[..., 3:] / 255
    return np.rint(a[..., :3] * k + 255 * (1 - k)).astype(np.int16)


def features(psd: PSDImage) -> list[str]:
    """Visible layer kinds and enabled effects, for picking the pairs a change touches."""
    f = set()
    for layer in psd.descendants():
        if layer.is_visible():
            f.add(layer.kind)
            if layer.has_effects() and layer.effects.enabled:
                f.update(type(e).__name__ for e in layer.effects if e.enabled)
    return sorted(f)


def score_one(psd_ref: str, png_ref: str) -> Row:
    """Bad px, mae, max and box of one pair, the better of raw and converted."""
    t = time.time()
    data = load(psd_ref)
    psd = PSDImage.open(io.BytesIO(data))
    off = Image.open(io.BytesIO(load(png_ref)))
    row: Row = {
        "psd_mb": round(len(data) / 1e6, 1),
        "depth": psd.depth,
        "png_mode": off.mode,
        "png_icc": bool(off.info.get("icc_profile")),
    }
    if off.size != psd.size:
        return row | {
            "skip": f"png {off.size[0]}x{off.size[1]}, psd {psd.width}x{psd.height}"
        }
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = psd.composite(ignore_preview=True, apply_icc=False)
        renders = [("raw", raw)]
        if Resource.ICC_PROFILE in psd.image_resources:
            icc = psd.image_resources.get_data(Resource.ICC_PROFILE)
            if psd.color_mode == ColorMode.RGB and raw.mode in ("RGB", "RGBA"):
                renders.append(("converted", _apply_icc(raw, icc)))  # = apply_icc=True
            else:
                renders.append(
                    ("converted", psd.composite(ignore_preview=True, apply_icc=True))
                )
    o = over_white(off)
    best = None
    for name, im in renders:
        e = np.abs(over_white(im) - o)
        d = e.max(axis=2)
        key = (int((d > THRESH).sum()), float(e.mean()))
        if best is None or key < best[0]:
            best = (key, name, d)
    assert best is not None
    (bad, mae), name, d = best
    row |= {
        "bad": bad,
        "mae": round(mae, 4),
        "max": int(d.max()),
        "render": name,
        "features": features(psd),
    }
    if bad:
        ys, xs = np.nonzero(d > THRESH)
        row["box"] = [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        ]
    row["seconds"] = round(time.time() - t, 1)
    return row


def spawn(psd_ref: str, png_ref: str) -> Row:
    """score_one() in its own process: a crash or a hang costs one pair."""
    try:
        p = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "_one", psd_ref, png_ref],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {TIMEOUT}s"}
    for line in p.stdout.splitlines():
        if line.startswith("PAIR "):
            return json.loads(line[5:])
    return {"error": (p.stderr or f"exit {p.returncode}")[-300:]}


# ------------------------------ score ------------------------------
def cached(rows: list[Row], cache: str | None) -> dict[str, str]:
    """Blob sha -> file, for the files at the pairs' paths under cache."""
    if not cache:
        return {}
    paths = sorted(
        {
            os.path.join(cache, r["path"][:-4] + e)
            for r in rows
            for e in (".psd", ".png")
        }
    )
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        return {}
    out = subprocess.run(
        ["git", "hash-object", "--no-filters", "--stdin-paths"],
        input="\n".join(paths),
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(zip(out.stdout.split(), paths, strict=True))


def present(gd: str | None, shas: set[str]) -> set[str]:
    """The blobs a clone already holds."""
    if not gd or not shas:
        return set()
    out = git(
        gd, "cat-file", "--batch-check", data="\n".join(sorted(shas)).encode() + b"\n"
    )
    return {w[0] for w in map(str.split, out.decode().splitlines()) if w[1] == "blob"}


def fetch(work: str, shas: list[str], url: str) -> str:
    """A throwaway repo holding just these blobs, in one batched fetch."""
    gd = os.path.join(work, "blobs.git")
    shutil.rmtree(gd, ignore_errors=True)
    subprocess.run(["git", "init", "-q", "--bare", gd], check=True, env=ENV)
    for k, v in (
        ("remote.origin.url", url),
        ("remote.origin.promisor", "true"),
        ("remote.origin.partialclonefilter", "blob:none"),
        ("extensions.partialClone", "origin"),
    ):
        git(gd, "config", k, v)
    git(
        gd,
        "-c",
        "fetch.negotiationAlgorithm=noop",
        "fetch",
        "--no-tags",
        "--no-write-fetch-head",
        "--filter=blob:none",
        "--stdin",
        "origin",
        data="\n".join(shas).encode() + b"\n",
    )
    return gd


def pick(a: argparse.Namespace) -> list[Row]:
    """The pairs to score, one per blob pair, less those already in the output."""
    rows = read_tsv(a.pairs)
    if not a.all:
        rows = [r for r in rows if r.get("newest") == "1"]
    if a.only:
        with open(a.only, encoding="utf-8") as f:
            want = {line.strip() for line in f if line.strip()}
        rows = [
            r for r in rows if r["path"] in want or f"{r['psd']}:{r['png']}" in want
        ]
    seen = set()
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            seen = {
                (j["psd"], j["png"]) for j in map(json.loads, f) if "error" not in j
            }
    todo = []
    for r in rows:
        if (r["psd"], r["png"]) not in seen:
            seen.add((r["psd"], r["png"]))
            todo.append(r)
    return todo


def score(a: argparse.Namespace) -> int:
    """Score the picked pairs, local ones first, then fetched batches."""
    todo = pick(a)
    files = cached(todo, a.cache)
    gd = git_dir(a.clone) if a.clone else None
    local = present(gd, {s for r in todo for s in (r["psd"], r["png"])} - set(files))
    near = [
        r
        for r in todo
        if r["psd"] in files.keys() | local and r["png"] in files.keys() | local
    ]
    far = [] if a.no_fetch else [r for r in todo if not any(r is x for x in near)]
    work = os.path.abspath(a.out) + ".work"
    print(f"{len(todo)} to score: {len(near)} here, {len(far)} to fetch", flush=True)
    batches = [(near, False)] + [
        (far[i : i + a.batch], True) for i in range(0, len(far), a.batch)
    ]
    with open(a.out, "a", encoding="utf-8") as out, ThreadPoolExecutor(a.workers) as ex:
        for n, (batch, fetched) in enumerate(batches):
            if not batch:
                continue
            store = None
            if fetched:
                if shutil.disk_usage(os.path.dirname(work)).free / 1e9 < a.min_free_gb:
                    print(f"stopped: under {a.min_free_gb} GB free", flush=True)
                    return 1
                os.makedirs(work, exist_ok=True)
                t = time.time()
                need = sorted(
                    {s for r in batch for s in (r["psd"], r["png"])}
                    - files.keys()
                    - local
                )
                store = fetch(work, need, a.url)
                mb = (
                    sum(
                        os.path.getsize(os.path.join(d, f))
                        for d, _, fs in os.walk(store)
                        for f in fs
                    )
                    / 1e6
                )
                print(
                    f"batch {n}: {len(need)} blobs, {mb:,.0f} MB in {time.time() - t:.0f} s",
                    flush=True,
                )

            def ref(sha: str, store: str | None = store) -> str:
                return (
                    files[sha]
                    if sha in files
                    else f"{gd if sha in local else store}::{sha}"
                )

            jobs = [(ref(r["psd"]), ref(r["png"])) for r in batch]
            for r, res in zip(batch, ex.map(lambda j: spawn(*j), jobs), strict=True):
                out.write(json.dumps({k: r[k] for k in COLS[:6]} | res) + "\n")
                out.flush()
                if "error" in res or res.get("bad"):
                    print(
                        r["path"],
                        res.get("bad", res.get("error", "")[-120:]),
                        flush=True,
                    )
            if store:
                shutil.rmtree(work)
    return 0


# ------------------------------ sum ------------------------------
def read_scores(path: str) -> dict[tuple[str, str], Row]:
    """Score rows by blob pair; a later row wins."""
    with open(path, encoding="utf-8") as f:
        return {(j["psd"], j["png"]): j for j in map(json.loads, f)}


def total(rows: list[Row]) -> str:
    """One line: bad px, cards off, mean mae."""
    ok = [r for r in rows if "bad" in r]
    if not ok:
        return f"{len(rows)} pairs, none scored"
    return (
        f"{len(ok)} scored ({len(rows) - len(ok)} skipped or failed): bad px {sum(r['bad'] for r in ok):,} "
        f"on {sum(1 for r in ok if r['bad'])} cards, mae mean {sum(r['mae'] for r in ok) / len(ok):.4f}"
    )


def summary(a: argparse.Namespace) -> int:
    """Totals of a run, and the pairs that moved against a base run."""
    new = read_scores(a.out)
    keys = [
        k for k, r in new.items() if not a.feature or a.feature in r.get("features", ())
    ]
    print(a.out + ":", total([new[k] for k in keys]))
    if a.base:
        old = read_scores(a.base)
        both = [k for k in keys if "bad" in new[k] and "bad" in old.get(k, {})]
        print(
            a.base + ":",
            total([old[k] for k in both]),
            f"(the {len(both)} pairs in both)",
        )
        moved = [
            (new[k]["path"], old[k]["bad"], new[k]["bad"], old[k]["mae"], new[k]["mae"])
            for k in both
            if (old[k]["bad"], old[k]["mae"]) != (new[k]["bad"], new[k]["mae"])
        ]
        print(
            f"moved {len(moved)}: worse {sum(1 for m in moved if (m[2], m[4]) > (m[1], m[3]))}"
        )
        for m in sorted(moved, key=lambda m: (m[1] - m[2], m[3] - m[4])):
            print(f"  {m[0]}: bad {m[1]} -> {m[2]}, mae {m[3]} -> {m[4]}")
    return 0


def main() -> int:
    """Command line."""
    if sys.argv[1:2] == ["_one"]:
        print("PAIR " + json.dumps(score_one(sys.argv[2], sys.argv[3])))
        return 0
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("clone")
    p.add_argument("pairs")
    p.add_argument("--rev", default="origin/main")
    p = sub.add_parser("score")
    p.add_argument("pairs")
    p.add_argument("out")
    p.add_argument("--clone")
    p.add_argument("--cache")
    p.add_argument(
        "--all", action="store_true", help="every pair, not each live card's newest"
    )
    p.add_argument("--only", help="file of card paths or psd:png blob pairs to keep")
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--url", default=URL)
    p.add_argument(
        "--no-fetch", action="store_true", help="only pairs already on this machine"
    )
    p.add_argument("--min-free-gb", type=float, default=80)
    p = sub.add_parser("sum")
    p.add_argument("out")
    p.add_argument("base", nargs="?")
    p.add_argument("--feature")
    a = ap.parse_args()
    if a.cmd == "list":
        rows = list_pairs(a.clone, a.rev)
        write_tsv(rows, a.pairs)
        live = {r["path"] for r in rows if r["live"]}
        print(
            f"{len(rows)} pairs on {len({r['path'] for r in rows})} paths; {len(live)} live cards have one"
        )
        return 0
    return score(a) if a.cmd == "score" else summary(a)


if __name__ == "__main__":
    sys.exit(main())
