"""Numbers the text engine relies on, each named once.

Tests assert these values; a change here is a behaviour change.
"""

# Photoshop's ResourceDict defaults for the subscript/superscript size and
# position ratios and the auto-leading ratio; a document's own values win.
SUB_SIZE = 0.583
SUB_POSITION = 0.333
SUPER_SIZE = 0.583
SUPER_POSITION = 0.333
AUTO_LEADING = 1.2

# Photoshop's default tab stops are this many pt apart, from the line's start
# (its paragraph sheets' DefaultTabWidth); its type tool sets no other stops.
TAB_WIDTH = 36.0

# The first baseline of a box sits one 'd' ascent below the top; a face
# without 'd' falls back through the others, then to ``face.ascender``.
ASCENT_GLYPHS = ("d", "h", "l", "b")

# Photoshop takes a TrueType face's ascent from its ascent glyph run through
# the font's instructions at this many pixels per em.
ASCENT_PPEM = 1000

# A box whose left edge is at least this far from zero stores an absolute
# frame; its top is then taken as zero.
BOX_ABSOLUTE_FRAME_L = 1.5

# Photoshop 2023 keeps a word on a box line that runs past the box by up to
# this fraction of the box width (fixtures at 128-483 pt: 1.43e-5..1.48e-5).
FIT_SLACK = 1.4e-5

# Justified text pulls a word onto the line only while the compressed
# word-space factor stays at or above this.
WORD_SPACING_MIN = 0.8

# Faux bold strokes the outline by this fraction of the rendered em and
# widens each advance by the same amount; a faux-bold run's ascent grows by
# half of it at the nominal size.
FAUX_BOLD_EM = 0.027

# Faux italic shears the outline by tan 12 degrees.
FAUX_ITALIC_SHEAR = 0.2126

# AntA Strong, and Smooth from SMOOTH_SMALL_PX up, embolden colour glyphs
# and the exact-area coverage by this 26.6 amount in x and y, fitted to
# Photoshop 2023's relayouts; scaled to the sample render, rounded.
STRONG_EMBOLDEN_26_6 = 17.5

# Their other glyphs are touch-sampled, as Photoshop 2023: a sample counts
# where the outline covers any of its cell, so a straight edge grows 1/8 px.
# Origins go on the sample render's 26.6 grid and a second render, one unit
# towards the exact origin, joins the first: right (down) when the origin is
# more than this many units past the grid, else left (up). An edge on a
# cell border so counts on the left and top only.
TOUCH_TIE = 0.05

# Smooth with both ems (x and y) under this many px is not emboldened and puts
# each glyph origin on 1/SMOOTH_GRID px, rotated too (Photoshop 2023).
SMOOTH_SMALL_PX = 100.0
SMOOTH_GRID = 4

# AntA None keeps a pixel whose coverage exceeds 0.5 ** (1 / 1.25).
NONE_THRESHOLD = 0.574

# AntA Crisp, Strong and Smooth take coverage from this many point samples
# a side per pixel: Photoshop stores their edges in sixteenths. A colour
# glyph's colours keep the exact coverage, except where it is emboldened.
COVERAGE_SAMPLES = 4

# AntA Crisp snaps glyph origins to whole pixels in y and, in x, to a grid
# that gets finer for smaller text: ``(pixel-size bound, positions per
# pixel)`` for sizes below each bound, one position from the last bound up.
CRISP_SUBPIXEL_X = ((24.0, 4), (32.0, 2))

# An off-diagonal transform coefficient above this makes the transform
# "general": a shear or rotation rather than a scale or flip.
GENERAL_TRANSFORM_EPS = 0.01

# A canvas larger than this raises before allocation; the document's
# allocation budget applies as well.
MAX_CANVAS_PIXELS = 64_000_000

# Bounded caches: rendered glyph bitmaps, by count and by their total bytes,
# open font faces, and the font indexes FontMaps share, by folder list.
GLYPH_CACHE_SIZE = 8192
GLYPH_CACHE_BYTES = 64 * 1024 * 1024
FACE_CACHE_SIZE = 64
INDEX_CACHE_SIZE = 8

# A splice treats the pixels this far around a glyph's outline box as the
# glyph's own: anti-aliasing, the Strong embolden and Photoshop's placement.
SPLICE_PAD = 2

# Fidelity metrics, shared with the scoring harness: alpha above IOU_COVER
# counts as ink for the IoU; a pixel is bad when its premultiplied channel
# difference exceeds BAD_DIFF where either alpha exceeds ANY_ALPHA.
IOU_COVER = 0.5
BAD_DIFF = 24 / 255
ANY_ALPHA = 0.1
