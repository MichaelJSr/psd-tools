"""Numbers the text engine relies on, each named once.

Tests assert these values; a change here is a behaviour change.
"""

# Photoshop's ResourceDict defaults for the subscript/superscript size and
# position ratios and the auto-leading ratio; a document's own values win.
SUB_SIZE = 0.583
SUB_POSITION = 0.333
AUTO_LEADING = 1.2

# The first baseline of a box sits one 'd' ascent below the top; a face
# without 'd' falls back through the others, then to ``face.ascender``.
ASCENT_GLYPHS = ("d", "h", "l", "b")

# A box whose left edge is at least this far from zero stores an absolute
# frame; its top is then taken as zero.
BOX_ABSOLUTE_FRAME_L = 1.5

# Justified text pulls a word onto the line only while the compressed
# word-space factor stays at or above this.
WORD_SPACING_MIN = 0.8

# Faux bold widens each advance by this fraction of the em.
FAUX_BOLD_EM = 1 / 30

# Faux italic shears the outline by tan 12 degrees.
FAUX_ITALIC_SHEAR = 0.2126

# AntA Strong and Smooth embolden the outline by this 26.6 amount in x and y.
STRONG_EMBOLDEN_26_6 = 19

# AntA None keeps a pixel whose coverage exceeds 0.5 ** (1 / 1.25).
NONE_THRESHOLD = 0.574

# An off-diagonal transform coefficient above this makes the transform
# "general": a shear or rotation rather than a scale or flip.
GENERAL_TRANSFORM_EPS = 0.01

# A canvas larger than this raises before allocation.
MAX_CANVAS_PIXELS = 64_000_000

# Bounded caches: rendered glyph bitmaps and open font faces.
GLYPH_CACHE_SIZE = 8192
FACE_CACHE_SIZE = 64

# Fidelity metrics, shared with the scoring harness: alpha above IOU_COVER
# counts as ink for the IoU; a pixel is bad when its premultiplied channel
# difference exceeds BAD_DIFF where either alpha exceeds ANY_ALPHA.
IOU_COVER = 0.5
BAD_DIFF = 24 / 255
ANY_ALPHA = 0.1
