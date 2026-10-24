Type layers
===========

A type layer stores two things: the text with its styles (``EngineData``)
and the pixels Photoshop rendered from them. psd-tools reads both, edits the
first, and can render the second itself.

Installation
------------

Rendering needs the ``text`` extra (HarfBuzz for shaping, FreeType for
rasterization)::

    pip install 'psd-tools[text]'

Editing needs nothing beyond psd-tools. The ``resample`` setting of
:py:class:`~psd_tools.composite.text.Options` also needs ``scipy``.

Reading
-------

:py:attr:`~psd_tools.api.layers.TypeLayer.text` is the text;
:py:attr:`~psd_tools.api.layers.TypeLayer.typesetting` exposes the fonts,
paragraphs and styled runs::

    from psd_tools import PSDImage

    psd = PSDImage.open("card.psd")
    for layer in psd.descendants():
        if layer.kind == "type":
            print(layer.text)
            for paragraph in layer.typesetting:
                for run in paragraph.runs:
                    print(run.text, run.style.font_name, run.style.font_size)

Editing
-------

Assigning :py:attr:`~psd_tools.api.layers.TypeLayer.text` or calling
:py:meth:`~psd_tools.api.layers.TypeLayer.set_text` replaces the text and
re-fits the style runs: the unchanged prefix and suffix keep their styles,
replaced text takes the style of the first character it replaces, inserted
text continues the run before it, and paragraph runs are rebuilt to end at
each ``'\r'`` and each forced line break ``'\x03'``.
:py:meth:`~psd_tools.api.layers.TypeLayer.set_span_style` and
:py:meth:`~psd_tools.api.layers.TypeLayer.set_run_style` write
``StyleSheetData`` keys on a range of characters::

    layer.text = "Spend all Void counters.\rStats start at 0."
    layer.set_span_style(10, 14, fill=(0.8, 0.1, 0.1), FauxBold=True)
    layer.set_run_style(0, Tracking=20)

A run's ``Font`` names a face in the layer's ``FontSet``.
:py:meth:`~psd_tools.api.layers.TypeLayer.add_font` appends one, to the
document's ``Txt2`` font set too, and returns its index; ``font_type`` is
``1`` for TrueType outlines and ``0`` for PostScript (CFF or Type 1) ones::

    layer.add_font("stoogedings-Regular")
    layer.set_span_style(0, 6, Font="stoogedings-Regular")

Edits never render. Until :py:meth:`~psd_tools.api.layers.TypeLayer.rasterize`
runs, the layer's pixels, ``composite()`` and a saved file still show
Photoshop's raster of the old text.

A layer's text is also stored in its story in the document's ``Txt2``
block (the layer's ``TextIndex`` is the story's index), and that story is
what Photoshop reads: it rebuilds the layer's own EngineData from it on
save. Every edit above writes the story too, its glyph layout left for
Photoshop to redo. A style key with no story feature, a style edit on a
stale story and any edit of a story another layer shares raise
``ValueError`` and leave the layer as it was.
:py:attr:`~psd_tools.api.layers.TypeLayer.story_text`
reads the story and :py:meth:`~psd_tools.api.layers.TypeLayer.sync_story`
brings a stale one (a file edited by an older version) back to the
layer's text. A type layer that
:py:meth:`~psd_tools.api.layers.Layer.duplicate` or
:py:meth:`~psd_tools.api.layers.Layer.copy_to` copies, or that moves into
another document, gets a copy of its story and text frame under a new
``TextIndex``; across documents the fonts and style sheets it names are
matched by name or added. A story cannot go between a Photoshop CC file
and an older one, whose ``Txt2`` keys are spelled differently.

Fitting a line
--------------

:py:meth:`~psd_tools.api.layers.TypeLayer.fit_text` shrinks point text
until its widest line fits a width in document pixels, as the layout
measures it (advance widths, not ink). Every run's ``FontSize`` is scaled
by one factor, rounded down to a hundredth of a point, in the layer and
its story. It returns the factor, ``1.0`` when the text fits already, and
refuses with ``ValueError``, changing nothing, below ``min_scale``::

    layer.text = "Croissant Knight"
    scale = layer.fit_text(660, min_scale=0.6, fonts=fonts)
    layer.rasterize(fonts=fonts)

Centred text keeps its centre, so the width for a name between two icons
is twice the smaller distance from the text's anchor to either icon.

Rendering
---------

:py:meth:`~psd_tools.api.layers.TypeLayer.rasterize` renders the current
text into the layer's channels, so ``topil()``, ``numpy()``, ``composite()``
and ``save()`` show it::

    from psd_tools.composite.text import FontMap

    fonts = FontMap(dirs=["/path/to/project/fonts"])
    report = layer.rasterize(fonts=fonts)
    print(report.fonts)      # PostScript name -> (how it resolved, path)
    print(report.warnings)   # ("font_loose", "overflow", ...)
    psd.save("card-edited.psd")

``report.lines`` lists every composed line with its position and its
``start`` and ``end`` in the text; a line an overflowing box did not draw
has ``drawn`` false.

An edit can keep Photoshop's raster where the text did not change.
:py:func:`~psd_tools.composite.text.capture` the layer before editing it and
pass that to ``rasterize``; only the lines whose glyphs, styles or positions
changed are drawn, and a reflow redraws every line it moves::

    from psd_tools.composite.text import capture

    before = capture(layer, fonts)
    layer.set_text("Deal 7 damage.\rDraw a card.")
    report = layer.rasterize(fonts=fonts, keep=before)
    print(report.splice.kept, report.splice.redrawn, report.splice.zone)

``unit="glyph"`` draws only the changed glyphs of a changed line. When no
inked glyph changed (a trailing space, say) the stored channels stay byte
for byte and the TySh bounds as Photoshop wrote them. The whole layer is
drawn instead, with ``report.splice.fallback`` saying why, when the stored
ink falls outside the layout's glyph boxes, the transform or anti-aliasing
changed, the document is not RGB, or the layer had no stored pixels.

The render is stored at the document's depth for RGB documents and
converted through PIL, at 8 bits, for the other colour modes.

Fonts are looked up by PostScript name in the ``FontMap``'s directories,
then ``PSD_TOOLS_FONT_DIRS`` and the platform font directories, each with
its subdirectories; ``aliases`` map a name to a file. A name no file
matches, or whose file cannot be opened, raises
:py:class:`~psd_tools.composite.text.MissingFontError`;
``FontMap(policy="substitute")`` draws it with a fallback face and
``policy="skip"`` leaves its runs blank, both reported in ``report.fonts``.

:py:func:`~psd_tools.composite.text.render` returns the pixels without
touching the layer, and :py:func:`~psd_tools.composite.text.verify`
compares a render with the raster Photoshop stored, which every type layer
carries. Checking an unedited layer this way shows how closely the engine
and the installed fonts reproduce Photoshop before an edit replaces that
raster::

    from psd_tools.composite.text import verify

    fidelity = verify(layer, fonts=fonts)
    print(fidelity.iou, fidelity.bad_pct)

Geometry
--------

:py:func:`~psd_tools.composite.text.text_geometry` lays a layer out without
rendering it and returns where its lines and glyphs sit, in document pixels,
as JSON-ready data. Every index points into the result's ``text``, so a
word finds its boxes::

    from psd_tools.composite.text import text_geometry

    geometry = text_geometry(layer, fonts)
    start = geometry["text"].index("Frozen")
    end = start + len("Frozen")
    boxes = [g["ink"] for g in geometry["glyphs"] if start <= g["i"] < end]
    for line in geometry["lines"]:
        print(line["text"], line["start"], line["end"], line["box"])

A glyph draws ``text[i:end]``, several characters for a ligature; its
``cell`` is its pen advance by the face's ascent and descent and its ``ink``
the box of its outline. Warped text is laid out unwarped, as it is rendered.
Under ``FontMap(policy="skip")`` the runs of a font listed in
``missing_fonts`` have no glyphs.

What is rendered
----------------

Point and paragraph (box) text; FontSize, Leading and AutoLeading, Tracking,
metrics kerning and ligatures, super- and subscript, BaselineShift,
Horizontal and Vertical scale, all caps, faux bold and italic, underline
and strikethrough, fill colour, the five anti-aliasing modes, and any
affine transform, at any document resolution. Paragraph text is broken
with Photoshop's single-line composer, with justification, and its first
line sits one ``d`` below the box top. A TrueType glyph is drawn as the
font's instructions leave it at one pixel per font unit, and its ``d``
measured as they place it at 1000 pixels per em, as Photoshop does. Warp is
rendered unwarped, colour emoji fonts as monochrome silhouettes and the
platform anti-aliasing modes (Mac LCD, Mac and Windows Gray) as Strong,
each with a warning in the report.
