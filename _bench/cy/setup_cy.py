from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
setup(ext_modules=cythonize([Extension("_chamfer", ["_chamfer.pyx"], include_dirs=[np.get_include()], extra_compile_args=["-O3"])]), script_args=["build_ext", "--inplace"])
