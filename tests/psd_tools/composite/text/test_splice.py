from dataclasses import replace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("freetype")
pytest.importorskip("uharfbuzz")

from psd_tools import PSDImage
from psd_tools.api.layers import TypeLayer
from psd_tools.composite.text import FontMap, capture, render
from psd_tools.composite.text import fonts as fonts_module
from psd_tools.composite.text.splice import Unit
from psd_tools.composite.text.verify import stored_raster

from ...utils import full_name

RED = (1.0, 0.0, 0.0)


@pytest.fixture(autouse=True)
def _no_platform_fonts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fonts_module, "platform_font_dirs", lambda: [])


@pytest.fixture
def fonts(luckiest_guy_path: str) -> FontMap:
    return FontMap(aliases={"ArialMT": luckiest_guy_path})


def type_layer(psd: str, name: str | None = None) -> TypeLayer:
    return next(
        x
        for x in PSDImage.open(full_name(psd)).descendants()
        if isinstance(x, TypeLayer) and name in (None, x.name)
    )


def channels(layer: TypeLayer) -> list[Any]:
    return [(c.compression, c.data) for c in layer._channels]


def placed(layer: TypeLayer, canvas: tuple[int, int, int, int]) -> np.ndarray:
    """The layer's stored straight RGBA on ``canvas``, as 8-bit values."""
    rgba, bbox = stored_raster(layer)
    out = np.zeros((canvas[3] - canvas[1], canvas[2] - canvas[0], 4), np.float32)
    if rgba is not None and bbox is not None:
        x0, y0, x1, y1 = bbox
        out[y0 - canvas[1] : y1 - canvas[1], x0 - canvas[0] : x1 - canvas[0]] = rgba
    return np.rint(out * 255)


def rendered(
    layer: TypeLayer, fonts: FontMap, canvas: tuple[int, int, int, int]
) -> np.ndarray:
    """A full render of the layer on ``canvas``, as the stored 8-bit values."""
    result = render(layer, fonts=fonts)
    out = np.zeros((canvas[3] - canvas[1], canvas[2] - canvas[0], 4), np.float32)
    x0, y0, x1, y1 = result.bbox
    out[y0 - canvas[1] : y1 - canvas[1], x0 - canvas[0] : x1 - canvas[0]] = result.rgba
    out[out[..., 3] == 0] = 0
    return np.rint(np.clip(out, 0, 1) * 255)


def tint(layer: TypeLayer) -> None:
    """Paint the stored ink red, so stored pixels tell themselves from drawn ones."""
    rgba, bbox = stored_raster(layer)
    assert rgba is not None and bbox is not None
    rgba = rgba.copy()
    rgba[rgba[..., 3] > 0, :3] = RED
    layer._store_raster(rgba, bbox)


def canvas_of(*boxes: tuple[int, ...]) -> tuple[int, int, int, int]:
    pad = 4
    return (
        min(b[0] for b in boxes) - pad,
        min(b[1] for b in boxes) - pad,
        max(b[2] for b in boxes) + pad,
        max(b[3] for b in boxes) + pad,
    )


def test_unchanged_text_keeps_the_stored_channels(fonts: FontMap) -> None:
    # Photoshop's raster, which the fixture font does not reproduce.
    layer = type_layer("text.psd")
    before_channels, rect = channels(layer), tuple(layer.bbox)
    report = layer.rasterize(fonts=fonts, keep=capture(layer, fonts))
    assert report.splice is not None
    assert report.splice.zone is None and report.splice.fallback is None
    assert report.splice.kept == (0, 1, 2) and report.splice.redrawn == ()
    assert channels(layer) == before_channels
    assert tuple(layer.bbox) == rect


def test_unchanged_empty_layer_stays_empty(fonts: FontMap) -> None:
    layer = type_layer("text.psd")
    layer.set_text("")
    layer.rasterize(fonts=fonts)
    assert not layer.has_pixels()
    before_channels, rect = channels(layer), tuple(layer.bbox)
    report = layer.rasterize(fonts=fonts, keep=capture(layer, fonts))
    assert report.splice is not None
    assert report.splice.zone is None and report.splice.fallback is None
    assert channels(layer) == before_channels and tuple(layer.bbox) == rect


def test_one_line_edit_keeps_the_other_lines(fonts: FontMap) -> None:
    layer = type_layer("text.psd")
    layer.rasterize(fonts=fonts)
    tint(layer)
    before = capture(layer, fonts)
    old = tuple(layer.bbox)
    layer.set_text("Line 1\rLine 7\rLine 3 and text")
    report = layer.rasterize(fonts=fonts, keep=before)
    assert report.splice is not None
    assert report.splice.kept == (0, 2) and report.splice.redrawn == (1,)
    zone = report.splice.zone
    assert zone is not None
    canvas = canvas_of(old, tuple(layer.bbox))
    now = placed(layer, canvas)
    full = rendered(layer, fonts, canvas)
    inside = np.zeros(now.shape[:2], bool)
    inside[
        zone[1] - canvas[1] : zone[3] - canvas[1],
        zone[0] - canvas[0] : zone[2] - canvas[0],
    ] = True
    # Outside the zone the tinted raster stays; inside it, the engine's.
    ink = now[..., 3] > 0
    assert (ink & ~inside).any()
    assert (now[ink & ~inside, :3] == [255, 0, 0]).all()
    np.testing.assert_array_equal(now[inside & ink], full[inside & ink])
    assert not (now[inside, :3] == [255, 0, 0]).all(axis=-1)[ink[inside]].any()
    # The zone covers the edited line, not its neighbours' baselines.
    lines = report.lines
    assert zone[1] > layer.transform[5] + lines[0].y
    assert zone[3] < layer.transform[5] + lines[2].y


EDITS = [
    ("text.psd", "Line 1\rLine 7\rLine 3 and text"),
    ("text.psd", "Line 1\rLine 2\rLine 3 and more text"),
    ("text.psd", "Line 1\rLine 3 and text"),
    ("text.psd", "Line 0\rLine 1\rLine 2\rLine 3 and text"),
    ("text-story.psd", "(1CD) Attacks inflict 1T Frozen.\rLine 2."),
    ("text-story.psd", "(1CD) Attacks always inflict 1T Frozen.\rLine two."),
]


@pytest.mark.parametrize("unit", ["line", "glyph"])
@pytest.mark.parametrize(("psd", "text"), EDITS)
def test_splice_of_an_own_raster_is_the_full_render(
    fonts: FontMap, psd: str, text: str, unit: Unit
) -> None:
    # Over the engine's own raster a splice must give its full render: the
    # zone holds every pixel the edit changes.
    layer = type_layer(psd, "Ability" if psd == "text-story.psd" else None)
    layer.rasterize(fonts=fonts)
    old = tuple(layer.bbox)
    before = capture(layer, fonts)
    layer.set_text(text)
    report = layer.rasterize(fonts=fonts, keep=before, unit=unit)
    assert report.splice is not None and report.splice.fallback is None
    assert report.splice.redrawn
    canvas = canvas_of(old, tuple(layer.bbox), render(layer, fonts=fonts).bbox)
    np.testing.assert_array_equal(placed(layer, canvas), rendered(layer, fonts, canvas))


def test_reflow_redraws_every_line_it_moves(fonts: FontMap) -> None:
    layer = type_layer("text-story.psd", "Ability")
    layer.set_text("Gain 1 SP.\rAttacks inflict 1T Frozen.\rLine two.")
    assert len(layer.rasterize(fonts=fonts).lines) == 3
    before = capture(layer, fonts)
    # A longer second paragraph wraps and pushes the third down.
    layer.set_text(
        "Gain 1 SP.\rAttacks and blocks inflict 1T Frozen and more.\rLine two."
    )
    report = layer.rasterize(fonts=fonts, keep=before)
    assert report.splice is not None
    assert len(report.lines) == 4
    assert report.splice.kept == (0,)
    assert report.splice.redrawn == (1, 2, 3)
    # A same-width edit changes its own line alone.
    before = capture(layer, fonts)
    layer.set_text(
        "Gain 1 SP.\rAttacks and blocks inflict 1T Frozen and more.\rLine twa."
    )
    report = layer.rasterize(fonts=fonts, keep=before)
    assert report.splice is not None
    assert report.splice.redrawn == (3,)


def test_glyph_unit_redraws_only_the_changed_glyphs(ttf_font_path: str) -> None:
    fonts = FontMap(aliases={"ArialMT": ttf_font_path})
    layer = type_layer("text.psd")
    layer.set_text("IAI IAI")
    layer.rasterize(fonts=fonts)
    zones = {}
    for unit in ("line", "glyph"):
        layer.set_text("IAI IAI")
        layer.rasterize(fonts=fonts)
        before = capture(layer, fonts)
        # A and V share an advance: nothing else moves.
        layer.set_text("IAI IVI")
        report = layer.rasterize(fonts=fonts, keep=before, unit=unit)
        assert report.splice is not None and report.splice.fallback is None
        zones[unit] = report.splice.zone
    line, glyph = zones["line"], zones["glyph"]
    assert line is not None and glyph is not None
    size = layer.engine_dict["StyleRun"]["RunArray"][0]["StyleSheet"]["StyleSheetData"][
        "FontSize"
    ]
    assert glyph[2] - glyph[0] < 0.7 * float(size) + 6 < line[2] - line[0]


def test_stray_stored_ink_renders_the_whole_layer(fonts: FontMap) -> None:
    layer = type_layer("text.psd")
    layer.rasterize(fonts=fonts)
    rgba, bbox = stored_raster(layer)
    assert rgba is not None and bbox is not None
    rgba = rgba.copy()
    rgba[0, rgba.shape[1] // 2] = (0.0, 0.0, 0.0, 1.0)  # between the glyphs' boxes
    layer._store_raster(rgba, bbox)
    before = capture(layer, fonts)
    layer.set_text("Line 1\rLine 7\rLine 3 and text")
    report = layer.rasterize(fonts=fonts, keep=before)
    assert report.splice is not None
    assert report.splice.fallback is not None
    assert report.splice.fallback.startswith("stored ink outside the layout")
    assert report.splice.kept == ()
    canvas = canvas_of(bbox, tuple(layer.bbox))
    np.testing.assert_array_equal(placed(layer, canvas), rendered(layer, fonts, canvas))


def test_a_moved_layer_renders_the_whole_layer(fonts: FontMap) -> None:
    layer = type_layer("text.psd")
    before = capture(layer, fonts)
    moved = replace(before.model, transform=before.model.transform[:4] + (0.0, 0.0))
    layer.set_text("Line 1\rLine 7\rLine 3 and text")
    report = layer.rasterize(fonts=fonts, keep=replace(before, model=moved))
    assert report.splice is not None
    assert report.splice.fallback == "the transform changed"
