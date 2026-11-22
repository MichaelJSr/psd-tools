"""Tests for ``tools/same_picture.py``: its relations hold, and catch a Normal wrap."""

import importlib.util
from pathlib import Path

import pytest

from psd_tools.api.layers import Group
from psd_tools.constants import BlendMode

from .psd_tools.utils import full_name

pytest.importorskip("aggdraw")
_TOOL = Path(__file__).resolve().parents[1] / "tools" / "same_picture.py"
_spec = importlib.util.spec_from_file_location("same_picture", _TOOL)
assert _spec is not None and _spec.loader is not None
same_picture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(same_picture)

FIXTURES = [
    "text.psb",
    "layer_effects.psd",
    "layer_mask_data.psd",
    pytest.param(
        "mask_parameters.psd",
        marks=pytest.mark.xfail(
            strict=True,
            reason="a pixel layer's vector mask is drawn only beside its pixel mask",
        ),
    ),
]


@pytest.mark.parametrize("name", FIXTURES)
def test_relations_hold(name: str) -> None:
    rows = same_picture.relations(full_name(name), name, 8)
    rows += same_picture.moves(full_name(name))
    checked = [r for r in rows if "px" in r]
    assert checked and not [r for r in checked if same_picture.failed(r)]


def test_a_normal_wrap_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    def normal(parent: Group, name: str = "Group") -> Group:
        group = Group.new(parent, name)
        group.blend_mode = BlendMode.NORMAL
        return group

    monkeypatch.setattr(same_picture, "Group", type("Normal", (), {"new": normal}))
    rows = same_picture.relations(full_name("text.psb"), "text.psb", 8)
    assert [r for r in rows if r["rel"] == "wrap" and same_picture.failed(r)]
