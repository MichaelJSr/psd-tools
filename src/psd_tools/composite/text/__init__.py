"""Type layer rendering from EngineData.

Requires the ``text`` extra (``uharfbuzz``, ``freetype-py``). Importing
:mod:`psd_tools.composite` does not import this package.
"""

from __future__ import annotations

import time

import numpy as np

from psd_tools.api.layers import TypeLayer
from psd_tools.composite.text import layout as _layout
from psd_tools.composite.text.fonts import (
    FontMap,
    MissingFontError,
    Policy,
    TextRenderError,
)
from psd_tools.composite.text.geometry import text_geometry
from psd_tools.composite.text.layout import Layout
from psd_tools.composite.text.model import TextModel, Transform
from psd_tools.composite.text.raster import (
    Bbox,
    GlyphCache,
    Options,
    raster_warnings,
    rasterize,
)
from psd_tools.composite.text.report import Fidelity, RenderReport, RenderResult
from psd_tools.composite.text.verify import fidelity, stored_raster

__all__ = [
    "Fidelity",
    "FontMap",
    "MissingFontError",
    "Options",
    "RenderReport",
    "RenderResult",
    "TextRenderError",
    "default_font_map",
    "font_names",
    "render",
    "render_type_layer",
    "text_geometry",
    "verify",
]

_DEFAULT_FONTS: dict[Policy, FontMap] = {}
#: Rendered glyph bitmaps, shared by every render in the process.
_GLYPH_CACHE = GlyphCache()


def default_font_map(policy: Policy = "raise") -> FontMap:
    """The process-wide :class:`FontMap` over the environment's directories."""
    fonts = _DEFAULT_FONTS.get(policy)
    if fonts is None:
        fonts = _DEFAULT_FONTS[policy] = FontMap(policy=policy)
    return fonts


def font_names(layer: TypeLayer) -> list[str]:
    """PostScript names the layer's style runs use, in order of first use.

    ``FontSet`` also lists faces no run uses (Photoshop's invisible font and
    its default); those are not required.
    """
    return list(
        dict.fromkeys(
            r.style.font for r in TextModel.from_layer(layer).runs if r.end > r.start
        )
    )


def _report(
    lay: Layout, transform: Transform, options: Options, t0: float
) -> RenderReport:
    warnings = list(lay.warnings)
    for w in raster_warnings(lay, transform, options):
        if w not in warnings:
            warnings.append(w)
    return RenderReport(
        fonts={name: (r.how, r.path) for name, r in lay.fonts.items()},
        lines=lay.lines,
        warnings=tuple(warnings),
        ms=(time.perf_counter() - t0) * 1000,
    )


def render(
    layer: TypeLayer,
    *,
    fonts: FontMap | None = None,
    options: Options | None = None,
    breaks: list[int] | None = None,
) -> RenderResult:
    """Render ``layer`` and report on the render.

    ``breaks`` lists ``str`` indices that start a line and replaces the
    composer's line breaking. Raises :class:`MissingFontError` under the
    ``raise`` font policy; under ``skip`` the runs of a missing font draw
    nothing and the report lists the font as ``"missing"``.
    """
    t0 = time.perf_counter()
    options = options or Options()
    fonts = fonts or default_font_map(options.missing_font)
    model = TextModel.from_layer(layer)
    lay = _layout.layout(model, fonts, breaks)
    # FontSize and the transform are in document pixels at any ppi.
    transform = model.transform
    rgba, bbox = rasterize(
        lay,
        transform,
        model.antialias,
        options,
        _GLYPH_CACHE,
        max_alloc_bytes=layer._psd._max_alloc_bytes,
    )
    return RenderResult(rgba, bbox, _report(lay, transform, options, t0))


def render_type_layer(
    layer: TypeLayer,
    fonts: FontMap | None = None,
    options: Options | None = None,
) -> tuple[np.ndarray, Bbox] | None:
    """Render ``layer``: ``(rgba, bbox)``, or ``None`` when a font is missing.

    ``rgba`` is straight-alpha ``float32`` HxWx4 and ``bbox`` its
    ``(x0, y0, x1, y1)`` in document pixels. The scoring entry: a layer with
    a missing font is reported, never rendered partially.
    """
    try:
        result = render(layer, fonts=fonts, options=options)
    except MissingFontError:
        return None
    if any(how == "missing" for how, _ in result.report.fonts.values()):
        return None
    return result.rgba, result.bbox


def verify(
    layer: TypeLayer, *, fonts: FontMap | None = None, options: Options | None = None
) -> Fidelity:
    """Render ``layer`` and compare with its stored raster.

    Every type layer Photoshop saved carries its own render, so this checks
    the engine (and the installed fonts) against Photoshop before an edit
    replaces that raster.
    """
    stored = stored_raster(layer)
    budget = layer._psd._max_alloc_bytes
    try:
        result = render(layer, fonts=fonts, options=options)
    except MissingFontError:
        return fidelity(*stored, None, None, "missing", max_alloc_bytes=budget)
    if any(how == "missing" for how, _ in result.report.fonts.values()):
        return fidelity(*stored, None, None, "missing", max_alloc_bytes=budget)
    return fidelity(*stored, result.rgba, result.bbox, max_alloc_bytes=budget)
