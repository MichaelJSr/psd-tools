"""The space a type layer's anti-aliased edge is blended in.

Photoshop composites a type layer under its "Blend Text Colors Using Gamma"
preference: the coverage of a glyph edge mixes the text colour with what lies
beneath in a space of that gamma relative to linear light, rather than in the
document's own encoding. The channel values are linearised with the sRGB
transfer curve and re-encoded with the preference gamma, which is what the
preference's default fits on Photoshop's own output.
"""

import numpy as np

from psd_tools.composite.paint import _linear_to_srgb, _srgb_to_linear
from psd_tools.constants import TEXT_BLEND_GAMMA

__all__ = ["TEXT_BLEND_GAMMA", "from_text_space", "to_text_space"]


def to_text_space(color: np.ndarray, gamma: float = TEXT_BLEND_GAMMA) -> np.ndarray:
    """Channel values in the document's encoding to the text blend space."""
    return _srgb_to_linear(np.maximum(color, 0.0)) ** (1.0 / gamma)


def from_text_space(value: np.ndarray, gamma: float = TEXT_BLEND_GAMMA) -> np.ndarray:
    """The inverse of :py:func:`to_text_space`."""
    return _linear_to_srgb(np.maximum(value, 0.0) ** gamma)
