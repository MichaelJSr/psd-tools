"""Tests for ``tools/history_pairs.py``: which commits count as pairs, and the score."""

import importlib.util
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from psd_tools import PSDImage

from .psd_tools.utils import full_name

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "history_pairs.py"
_spec = importlib.util.spec_from_file_location("history_pairs", _TOOL)
assert _spec is not None and _spec.loader is not None
history_pairs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(history_pairs)


def _commit(
    repo: Path, author: str, date: str, msg: str, files: dict[str, bytes]
) -> None:
    for name, data in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_bytes(data)
    env = dict(
        os.environ,
        GIT_AUTHOR_NAME=author,
        GIT_AUTHOR_EMAIL="a@b",
        GIT_AUTHOR_DATE=f"{date}T12:00:00",
        GIT_COMMITTER_NAME=author,
        GIT_COMMITTER_EMAIL="a@b",
        GIT_COMMITTER_DATE=f"{date}T12:00:00",
    )
    subprocess.run(["git", "-C", repo, "add", "-A"], check=True)
    subprocess.run(["git", "-C", repo, "commit", "-qm", msg], check=True, env=env)


def test_only_a_persons_psd_and_png_together_make_a_pair(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", tmp_path], check=True)
    _commit(
        tmp_path,
        "Ann",
        "2025-01-01",
        "a",
        {"A.psd": b"1", "A.png": b"1", "MDW/C.psd": b"1", "MDW/C.png": b"1"},
    )
    _commit(tmp_path, "MichaelJSr", "2025-02-01", "b", {"B.psd": b"1", "B.png": b"1"})
    _commit(tmp_path, "Ann", "2025-03-01", "a2", {"A.psd": b"2", "A.png": b"2"})
    _commit(
        tmp_path, "AutoCroissant", "2026-11-01", "bot", {"A.psd": b"3", "A.png": b"3"}
    )
    _commit(
        tmp_path,
        "Ann",
        "2026-11-02",
        "s\n\nCard-sweep: x",
        {"A.psd": b"4", "A.png": b"4"},
    )
    _commit(
        tmp_path, "MichaelJSr", "2026-10-05", "agent", {"B.psd": b"5", "B.png": b"5"}
    )
    _commit(tmp_path, "Ann", "2026-11-03", "psd only", {"B.psd": b"6"})
    rows = history_pairs.list_pairs(str(tmp_path), "HEAD")
    got = [(r["path"], r["date"], r["live"], r["newest"]) for r in rows]
    assert got == [
        ("A.psd", "2025-03-01", 1, 1),
        ("B.psd", "2025-02-01", 1, 1),
        ("A.psd", "2025-01-01", 1, 0),
        ("MDW/C.psd", "2025-01-01", 0, 0),
    ]


def test_score_counts_pixels_over_24_levels(tmp_path: Path) -> None:
    psd_path = full_name("clipping-mask3.psd")
    psd = PSDImage.open(psd_path)
    image = history_pairs.over_white(
        psd.composite(ignore_preview=True, apply_icc=False)
    )
    image = image.astype(np.uint8)
    Image.fromarray(image).save(tmp_path / "same.png")
    image[2:7, 3:8] = np.where(image[2:7, 3:8] < 128, 255, 0)
    Image.fromarray(image).save(tmp_path / "off.png")
    same = history_pairs.score_one(psd_path, str(tmp_path / "same.png"))
    assert (same["bad"], same["max"]) == (0, 0)
    off = history_pairs.score_one(psd_path, str(tmp_path / "off.png"))
    assert (off["bad"], off["box"]) == (25, [3, 2, 8, 7])
