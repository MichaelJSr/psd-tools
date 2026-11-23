#!/usr/bin/env python
"""Build script for the Cython extension."""

import sys

from Cython.Build import cythonize
from setuptools import Extension, setup

use_limited_api = sys.version_info >= (3, 13)

extensions = [
    Extension(
        "psd_tools.compression._rle",
        ["src/psd_tools/compression/_rle.pyx"],
        define_macros=[("Py_LIMITED_API", 0x030D0000)] if use_limited_api else [],
        py_limited_api=use_limited_api,
    ),
    Extension(
        "psd_tools.api._numpy_io",
        ["src/psd_tools/api/_numpy_io.pyx"],
        define_macros=[("Py_LIMITED_API", 0x030D0000)] if use_limited_api else [],
        py_limited_api=use_limited_api,
    ),
    Extension(
        "psd_tools.psd._descriptor",
        ["src/psd_tools/psd/_descriptor.pyx"],
        define_macros=[("Py_LIMITED_API", 0x030D0000)] if use_limited_api else [],
        py_limited_api=use_limited_api,
    ),
    Extension(
        "psd_tools.psd._engine",
        ["src/psd_tools/psd/_engine.pyx"],
        define_macros=[("Py_LIMITED_API", 0x030D0000)] if use_limited_api else [],
        py_limited_api=use_limited_api,
    ),
    # Bit-identical to the numpy code only without FMA contraction.
    Extension(
        "psd_tools.composite._distance",
        ["src/psd_tools/composite/_distance.pyx"],
        extra_compile_args=["-ffp-contract=off"],
        define_macros=[("Py_LIMITED_API", 0x030D0000)] if use_limited_api else [],
        py_limited_api=use_limited_api,
    ),
    Extension(
        "psd_tools.composite._blend",
        ["src/psd_tools/composite/_blend.pyx"],
        extra_compile_args=["-ffp-contract=off"],
        define_macros=[("Py_LIMITED_API", 0x030D0000)] if use_limited_api else [],
        py_limited_api=use_limited_api,
    ),
]


setup(ext_modules=cythonize(extensions))
