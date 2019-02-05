from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

setup(
        name='funlib.evaluate',
        version='0.1',
        description='Popular metrics and reporting tools for volume comparison.',
        url='https://github.com/funkelab/funlib.evaluate',
        author='Jan Funke',
        author_email='funkej@janelia.hhmi.org',
        license='MIT',
        packages=[
            'funlib.evaluate',
            'funlib.evaluate.impl'
        ],
        ext_modules=cythonize([
            Extension(
                'funlib.evaluate.rand_voi',
                sources=[
                    'funlib/evaluate/rand_voi.pyx'
                ],
                extra_compile_args=['-O3'],
                include_dirs=[np.get_include()],
                language='c++')
        ])
)
