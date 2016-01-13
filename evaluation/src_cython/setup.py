from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("mainDefs",
                         ["mainDefs.pyx", "main2.cpp"],
                         library_dirs = ["."],
                         include_dirs = ["."],
                         language='c++',
                         # std= 'c++11',
                         extra_link_args=["-std=c++11"],
                         extra_compile_args=["-std=c++11", "-w"])]

setup(cmdclass = {'build_ext': build_ext}, ext_modules = ext_modules)

'''
ext_modules = [
        Extension("m2",
                  ["m2.pyx", "main2.cpp"],
                  library_dirs = ["./zi/*,./zi/disjoint_sets/*"],
                  include_dirs = ["./zi/*,./zi/disjoint_sets/*"],
                  language='c++',
                  std= 'c++11')
            ]

setup(cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules,
      extra_compile_args=["-std=c++11"],extra_link_args=["-std=c++11"])
'''