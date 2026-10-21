import pytest

pytest.importorskip("freetype")
pytest.importorskip("uharfbuzz")

from psd_tools.composite.text import constants as c


def test_values() -> None:
    assert c.SUB_SIZE == 0.583
    assert c.SUB_POSITION == 0.333
    assert c.AUTO_LEADING == 1.2
    assert c.ASCENT_GLYPHS == ("d", "h", "l", "b")
    assert c.BOX_ABSOLUTE_FRAME_L == 1.5
    assert c.WORD_SPACING_MIN == 0.8
    assert c.FAUX_BOLD_EM == 0.027
    assert c.FAUX_ITALIC_SHEAR == 0.2126
    assert c.STRONG_EMBOLDEN_26_6 == 19
    assert pytest.approx(0.5 ** (1 / 1.25), abs=5e-4) == c.NONE_THRESHOLD
    assert c.CRISP_SUBPIXEL_X == ((24.0, 4), (32.0, 2))
    assert c.GENERAL_TRANSFORM_EPS == 0.01
    assert c.MAX_CANVAS_PIXELS == 64_000_000
    assert c.GLYPH_CACHE_SIZE == 8192
    assert c.GLYPH_CACHE_BYTES == 64 * 1024 * 1024
    assert c.FACE_CACHE_SIZE == 64
    assert c.IOU_COVER == 0.5
    assert c.BAD_DIFF == 24 / 255
    assert c.ANY_ALPHA == 0.1
