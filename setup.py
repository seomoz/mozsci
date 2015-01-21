#!/usr/bin/env python

# Copyright (c) 2012 SEOmoz
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [Extension('mozsci.spearmanr_by_fast',
    sources=["mozsci/spearmanr_by_fast.pyx", "mozsci/cspearmanr_by_fast.cc"],
    include_dirs = [np.get_include()],
    language="c++"),
    Extension('mozsci._c_utils',
        sources=["mozsci/_c_utils.pyx"],
        include_dirs = [np.get_include()],
        language="c++"),
    ]

with open('requirements.txt', 'r') as fin:
    requires = fin.read().strip().split('\n')

setup(
    name             = 'mozsci',
    version          = '0.9.0',
    description      = 'Data science tools from SEOmoz',
    author           = 'Matt Peters',
    author_email     = 'matt@moz.com',
    url              = 'http://github.com/seomoz/mozsci',
    packages         = ['mozsci', 'mozsci.models', 'mozsci.glm'],
    license          = 'MIT',
    platforms        = 'Posix; MacOS X',
    cmdclass         = {'build_ext': build_ext},
    ext_modules      = ext_modules,
    install_requires = requires,
    classifiers      = [
        'License :: OSI Approved :: MIT License',
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        ],
)
