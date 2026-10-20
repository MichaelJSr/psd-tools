"""Type layer rendering from EngineData.

Requires the ``text`` extra (``uharfbuzz``, ``freetype-py``). Importing
:mod:`psd_tools.composite` does not import this package.
"""

from __future__ import annotations

import numpy as np

from psd_tools.api.layers import TypeLayer
from psd_tools.composite.text.fonts import (
    FontMap,
    MissingFontError,
    Policy,
    TextRenderError,
)
from psd_tools.composite.text.raster import Bbox, Options
from psd_tools.composite.text.report import Fidelity, RenderReport, RenderResult

__all__ = [
    "Fidelity",
    "FontMap",
    "MissingFontError",
    "Options",
    "RenderReport",
    "RenderResult",
    "TextRenderError",
    "render",
    "render_type_layer",
    "verify",
]

_DEFAULT_FONTS: dict[Policy, FontMap] = {}


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
    font_set = [str(f["Name"].value) for f in layer.resource_dict["FontSet"]]
    sheets = layer.resource_dict.get("StyleSheetSet")
    normal_index = int(layer.resource_dict.get("TheNormalStyleSheet", 0))
    default = 0
    if sheets:
        default = int(sheets[normal_index]["StyleSheetData"].get("Font", 0))
    names: dict[str, None] = {}
    runs = layer.engine_dict["StyleRun"]["RunArray"]
    lengths = layer.engine_dict["StyleRun"]["RunLengthArray"]
    for run, length in zip(runs, lengths, strict=False):
        if int(length) <= 0:
            continue
        index = int(run["StyleSheet"]["StyleSheetData"].get("Font", default))
        names[font_set[index]] = None
    return list(names)


def render(
    layer: TypeLayer,
    *,
    fonts: FontMap | None = None,
    options: Options | None = None,
    breaks: list[int] | None = None,
) -> RenderResult:
    """Render ``layer`` and report on the render."""
    raise NotImplementedError


def verify(
    layer: TypeLayer, *, fonts: FontMap | None = None, options: Options | None = None
) -> Fidelity:
    """Render ``layer`` and compare with its stored raster."""
    raise NotImplementedError


def render_type_layer(
    layer: TypeLayer,
    fonts: FontMap | None = None,
    options: Options | None = None,
) -> tuple[np.ndarray, Bbox] | None:
    """Render ``layer``: ``(rgba, bbox)``, or ``None`` when a font is missing.

    The glyph pipeline is not wired yet: a layer whose fonts all resolve
    renders empty.
    """
    options = options or Options()
    fonts = fonts or default_font_map(options.missing_font)
    try:
        resolved = fonts.resolve_all(font_names(layer))
    except MissingFontError:
        return None
    if any(r.face is None for r in resolved.values()):
        return None
    x, y = round(layer.transform[4]), round(layer.transform[5])
    return np.zeros((0, 0, 4), dtype=np.float32), (x, y, x, y)
