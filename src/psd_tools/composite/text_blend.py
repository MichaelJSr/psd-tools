"""The space a type layer's anti-aliased edge is blended in.

Photoshop composites a type layer under its "Blend Text Colors Using Gamma"
preference: the coverage of a glyph edge mixes the text colour with what lies
beneath in a space of that gamma relative to linear light, rather than in the
document's own encoding. The channel values are linearised with the sRGB
transfer curve and re-encoded with the preference gamma, which is what the
preference's default fits on Photoshop's own output.
"""

import numpy as np

# Photoshop's "Blend Text Colors Using Gamma" preference, at its default.
TEXT_BLEND_GAMMA = 1.45

_SRGB_LINEAR_CUTOFF = 0.04045
_SRGB_LINEAR_SLOPE = 12.92
_SRGB_OFFSET = 0.055
_SRGB_EXPONENT = 2.4


def _linearize(value: np.ndarray) -> np.ndarray:
    value = np.maximum(value, 0.0)
    return np.where(
        value > _SRGB_LINEAR_CUTOFF,
        ((value + _SRGB_OFFSET) / (1.0 + _SRGB_OFFSET)) ** _SRGB_EXPONENT,
        value / _SRGB_LINEAR_SLOPE,
    )


def _encode(linear: np.ndarray) -> np.ndarray:
    linear = np.maximum(linear, 0.0)
    return np.where(
        linear > _SRGB_LINEAR_CUTOFF / _SRGB_LINEAR_SLOPE,
        (1.0 + _SRGB_OFFSET) * linear ** (1.0 / _SRGB_EXPONENT) - _SRGB_OFFSET,
        linear * _SRGB_LINEAR_SLOPE,
    )


def to_text_space(color: np.ndarray, gamma: float = TEXT_BLEND_GAMMA) -> np.ndarray:
    """Channel values in the document's encoding to the text blend space."""
    return _linearize(color) ** (1.0 / gamma)


def from_text_space(value: np.ndarray, gamma: float = TEXT_BLEND_GAMMA) -> np.ndarray:
    """The inverse of :py:func:`to_text_space`."""
    return _encode(np.maximum(value, 0.0) ** gamma)
