Contributing
============

Development happens at github: `bug tracker <https://github.com/psd-tools/psd-tools/issues>`__.
Feel free to submit `bug reports <https://github.com/psd-tools/psd-tools/issues/new>`_
or pull requests. Attaching an erroneous PSD file makes the debugging process
faster. Such PSD file might be added to the test suite. See `Files in reports
and pull requests`_ first.

The license is MIT. To report a security vulnerability, follow the
`security policy <https://github.com/psd-tools/psd-tools/security/policy>`__
instead of opening an issue.

Files in reports and pull requests
----------------------------------

Treat every PSD from outside the project as untrusted, including one attached
to a report or a pull request.

- **Reports.** Attach a minimal or forged file, not a real document. A real
  document can carry private or copyrighted content in layer names, embedded
  smart objects, thumbnails, XMP and file paths, and an attachment on a public
  issue is public.
- **Fixtures.** A file added under ``tests/psd_files/`` must be one you have the
  right to license under the MIT license. Prefer a file authored or forged for
  the test. A file crafted with Photoshop or a third-party tool is accepted too.
  State in the pull request how each new file was created, including the tool
  and its version. Keep the file as small as the test allows.
- **Hostile files.** A file that declares far more data than it holds, to
  exercise a limit, should be forged inside the test instead of committed as a
  binary, so that it cannot exhaust the memory of a contributor or of CI.
- **Opening files.** Parse files from outside the project with the limits in
  :doc:`untrusted`.

Package design
--------------

See :doc:`architecture` for the package layout and the contracts a change
must preserve.

Testing
-------

In order to run tests, make sure PIL/Pillow is built with LittleCMS
or LittleCMS2 support. For example, on Ubuntu, install the following packages::

    apt-get install liblcms2-dev libjpeg-dev libfreetype6-dev zlib1g-dev

Then install `psd-tools` with development dependencies::

    uv sync --group dev --extra composite

Finally, run tests::

    uv run pytest

The Cython extensions build in place with ``python setup.py build_ext
--inplace`` (Cython and setuptools on the path). Without
``psd_tools.composite._distance``, ``psd_tools.composite._blend`` or
``psd_tools.api._numpy_io`` the NumPy code runs instead;
``test_the_compiled_loops_match_numpy``, ``test_compiled_blend.py`` and
``test_the_compiled_8bit_read_is_parse_arrays`` check the two agree bit for bit.

Things to know when writing compositor tests:

- ``check_composite_quality()`` defaults to ``force=False``. A regression in
  the drawing path shows only under ``force=True``, so run both.
- Assert pixel values, not only the PIL mode of the result.
- ``_mse`` in ``tests/psd_tools/composite/test_composite.py`` scores colour
  where alpha is 0, so split alpha from RGB before reading a change as a
  regression.
- A fixture whose layer runs past the canvas measures the viewport clip, not
  the feature under test. Compare ``layer.bbox`` with ``psd.size`` first.
- CI runs Python 3.14 on Linux and macOS.

Acknowledgments
---------------

Great thanks to `all the contributors <https://github.com/psd-tools/psd-tools/graphs/contributors>`_.
