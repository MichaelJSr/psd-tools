Migration Guide
===============

v1.12
-----

psd-tools 1.12 moves the composite dependencies into the ``composite`` extra.

Breaking Change: Composite Extra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The main breaking change in version 1.12 is that compositing needs the ``composite``
extra, which must be explicitly installed; there is no fallback without it.

**What changed:**

- Dependencies ``aggdraw``, ``scipy``, and ``scikit-image`` moved to the ``composite`` extra
- Compositing requires the composite extra

**Migration steps:**

If you composite, install with the composite extra::

    pip install 'psd-tools[composite]'

Or install the dependencies separately::

    pip install psd-tools aggdraw scipy

**What works without composite dependencies:**

- Reading and writing PSD files
- Accessing layer information (names, dimensions, etc.)
- Extracting raw pixel data with NumPy

**Error handling:**

A render that needs a missing package raises ``ImportError`` naming it::

    ModuleNotFoundError: No module named 'scipy'

Module Structure Changes
~~~~~~~~~~~~~~~~~~~~~~~~

Version 1.12 includes some internal refactoring that generally doesn't affect public APIs:

- The ``PSD`` class moved from ``psd_tools.psd`` to ``psd_tools.psd.document`` (still importable from ``psd_tools.psd``)
- Utils module split into ``registry`` and ``bin_utils`` (internal change)
- Composite module reorganized for better type safety (internal change)

These changes maintain backward compatibility for public imports.

Type Annotations
~~~~~~~~~~~~~~~~

Version 1.12 adds comprehensive type annotations throughout the codebase. If you use
type checkers like mypy, you may discover type errors in your code that were previously
undetected. This is a good thing - the annotations help catch bugs earlier!

v1.11
-----

psd-tools 1.11 introduces stronger type-safety via annotation and new public APIs for layer creation.
Now the following approach is possible to create a new layered document::

    from PIL import Image
    from psd_tools import PSDImage

    image = Image.new("RGBA", (width, height))
    psdimage = PSDImage.new(mode='RGB', size=(640, 480), depth=8)
    layer = psdimage.create_pixel_layer(image, name="Layer 1", top=0, left=0, opacity=255)
    psdimage.save('new_image.psd')

Version 1.11 introduces some breaking changes.

Layer creation now disables orphaned layers. They must be given a valid PSDImage object.

version 1.11.x::

    image = Image.new("RGBA", (width, height))
    psdimage.create_pixel_layer(psdimage, image)

version 1.10.x::

    image = Image.new("RGBA", (width, height))
    PixelLayer.frompil(None, image, parent=None)

The same layer cannot be shared between multiple container objects.

version 1.11.x::

    layer = psdimage.create_pixel_layer(group, image)
    psdimage.append(layer)  # This won't duplicate the layer.

v1.10
-----

psd-tools 1.10 has a few breaking changes.

Basic layer structure editing is supported in 1.10. You can add or remove a pixel layer, or change the grouping of layers.

psd-tools 1.10 drops `compose` module. Use `composite` instead.

version 1.10.x::

    image = psd.composite()
    layer_image = layer.composite()

v1.9
----

psd-tools 1.9 switches to NumPy based compositing.

version 1.8.x::

    psd = PSDImage.open(filename)
    image = psd.compose()
    layer = psd[0]
    layer_image = layer.compose()

version 1.9.x::

    psd = PSDImage.open(filename)
    image = psd.composite()
    layer = psd[0]
    layer_image = layer.composite()

NumPy array API is introduced::

    image = psd.numpy()
    layer_image = layer.numpy()

v1.8
----

There are major API changes in version 1.8.x.

.. note:: In version 1.8.0 - 1.8.7, the package name was `psd_tools2`.

PSDImage
~~~~~~~~

File open method is changed from `load` to
:py:meth:`~psd_tools.PSDImage.open`.

version 1.7.x::

    psd = PSDImage.load(filename)
    with open(filename, 'rb') as f:
        psd = PSDImage.from_stream(f)

version 1.8.x::

    psd = PSDImage.open(filename)
    with open(filename, 'rb') as f:
        psd = PSDImage.open(f)

Layers
~~~~~~

Children of PSDImage or Group is directly accessible by iterator or indexing.

version 1.7.x::

    for layer in group.layers:
        print(layer)

    first_child = group.layers[0]

version 1.8.x::

    for layer in group:
        print(layer)

    first_child = group[0]


In version 1.8.x, the order of layers is reversed to reflect that the index
should not change when a new layer is added on top.

PIL export
~~~~~~~~~~

Primary PIL export method is :py:func:`~psd_tools.compose`.

version 1.7.x::

    image = psd.as_PIL()

    layer_image = compose(layer)
    raw_layer_image = layer.as_PIL()

version 1.8.x::

    image = psd.compose()

    layer_image = layer.compose()
    raw_layer_image = layer.topil()


Low-level data structure
~~~~~~~~~~~~~~~~~~~~~~~~

Data structures are completely rewritten to support writing functionality.
See :py:mod:`psd_tools.psd` subpackage.

version 1.7.x::

    psd.decoded_data

version 1.8.x::

    psd._record

Drop pymaging support
~~~~~~~~~~~~~~~~~~~~~

Pymaging support is dropped.
