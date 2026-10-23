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

Edits never render. Until :py:meth:`~psd_tools.api.layers.TypeLayer.rasterize`
runs, the layer's pixels, ``composite()`` and a saved file still show
Photoshop's raster of the old text.

A layer's text is also stored in its story in the document's ``Txt2``
block (the layer's ``TextIndex`` is the story's index), and that story is
what Photoshop reads: it rebuilds the layer's own EngineData from it on
save. Every edit above writes the story too, its glyph layout left for
Photoshop to redo; a style key with no story feature raises
``ValueError``. :py:attr:`~psd_tools.api.layers.TypeLayer.story_text`
reads the story and :py:meth:`~psd_tools.api.layers.TypeLayer.sync_story`
brings a stale one (a file edited by an older version) back to the
layer's text.

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
has ``drawn`` false. The render is stored at the document's depth for RGB
documents and converted through PIL, at 8 bits, for the other colour modes.

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
with Photoshop's single-line composer, with justification. Warp is
rendered unwarped, colour emoji fonts as monochrome silhouettes and the
platform anti-aliasing modes (Mac LCD, Mac and Windows Gray) as Strong,
each with a warning in the report.
