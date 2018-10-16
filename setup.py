from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension('collapse_cython', ['collapse_cython.pyx']),
]

setup(
  name = 'cython stuff',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
