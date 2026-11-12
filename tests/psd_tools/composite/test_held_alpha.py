"""An 8-bit layer's alpha times its opacity goes on at whole levels."""

import numpy as np
from PIL import Image

from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer

# Backdrop and source greys per row; alpha 0-255 along each row.
_PAIRS = [(d, s) for d in (0, 37, 200, 255) for s in (0, 90, 163, 255)]


def _probe(opacity: int) -> tuple[np.ndarray, ...]:
    d = np.array([p[0] for p in _PAIRS], dtype=np.uint8)[:, None].repeat(256, 1)
    s = np.array([p[1] for p in _PAIRS], dtype=np.uint8)[:, None].repeat(256, 1)
    a = np.arange(256, dtype=np.uint8)[None, :].repeat(len(_PAIRS), 0)
    psd = PSDImage.new(mode="RGB", size=(256, len(_PAIRS)))
    backdrop = np.dstack([d, d, d, np.full_like(d, 255)])
    psd.create_pixel_layer(image=Image.fromarray(backdrop), name="Backdrop")
    layer = PixelLayer.frompil(Image.fromarray(np.dstack([s, s, s, a])), psd)
    layer.opacity = opacity
    psd.append(layer)
    out = np.asarray(psd.composite(ignore_preview=True).convert("RGB"))[..., 0]
    return out.astype(int), d.astype(int), s.astype(int), a.astype(int)


def test_alpha_times_opacity_is_rounded_before_the_blend() -> None:
    """Photoshop 2023 blends with round(a * opacity / 255), not the float product.

    Its probe of every (backdrop, source, alpha) at 50% matched this on all
    16.7M triples and the float product on 91.8%.
    """
    out, d, s, a = _probe(128)
    held = np.floor(a * 128 / 255 + 0.5)
    assert np.array_equal(out, np.floor(d + (s - d) * held / 255 + 0.5))
    product = np.floor(d + (s - d) * (a * 128 / 255) / 255 + 0.5)
    assert (out != product).sum() > 0
