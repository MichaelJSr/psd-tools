import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("freetype")
pytest.importorskip("uharfbuzz")

from psd_tools import PSDImage
from psd_tools.api.layers import TypeLayer
from psd_tools.composite.text import (
    FontMap,
    Options,
    font_names,
    render_type_layer,
)
from psd_tools.composite.text import fonts as fonts_module

from ...utils import full_name
from .conftest import PS_TT


@pytest.fixture
def layer() -> TypeLayer:
    psd = PSDImage.open(full_name("text.psd"))
    found = [x for x in psd.descendants() if isinstance(x, TypeLayer)]
    return found[0]


@pytest.fixture(autouse=True)
def _no_platform_fonts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fonts_module, "platform_font_dirs", lambda: [])


def test_font_names(layer: TypeLayer) -> None:
    assert font_names(layer) == ["ArialMT"]


def test_missing_font_is_none(layer: TypeLayer, fixture_fonts_dir: str) -> None:
    assert render_type_layer(layer, FontMap(dirs=[fixture_fonts_dir])) is None
    assert render_type_layer(layer, FontMap(policy="skip")) is None


def test_resolved_font_renders(layer: TypeLayer, ttf_font_path: str) -> None:
    out = render_type_layer(layer, FontMap(aliases={"ArialMT": ttf_font_path}))
    assert out is not None
    rgba, bbox = out
    assert rgba.dtype == np.float32
    assert rgba.shape[2] == 4
    x, y = round(layer.transform[4]), round(layer.transform[5])
    assert rgba.shape[:2] == (bbox[3] - bbox[1], bbox[2] - bbox[0])
    assert (bbox[0], bbox[1]) == (x, y)


def test_substitute_policy_renders(layer: TypeLayer, fixture_fonts_dir: str) -> None:
    fm = FontMap(dirs=[fixture_fonts_dir], policy="substitute", fallback=PS_TT)
    assert render_type_layer(layer, fm, Options()) is not None


def test_composite_import_does_not_load_text() -> None:
    code = (
        "import sys, psd_tools.composite; "
        "assert 'psd_tools.composite.text' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
