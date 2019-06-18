from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

modules = [Extension("calculation",["calculation.pyx"],
                     include_dirs=[numpy.get_include()],
                     extra_compile_args=["-O2"])]
setup(
    cmdclass = {'build_ext' : build_ext},
    ext_modules = modules
    )
