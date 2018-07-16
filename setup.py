"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""
# Always prefer setuptools over distutils
from setuptools import setup
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='udkm1Dsimpy',
    version='0.1',
    packages=['udkm1Dsimpy'],
    url='https://github.com/dschick/udkm1Dsimpy',  # Optional
    install_requires=['numpy',
                      'scipy',
                      'numericalunits',
                      'sympy'],
    license='GPL3',
    author='Daniel Schick, et. al.',
    author_email='schick.daniel@gmail.com',
    description='A Simulation Toolkit for ' \
                '1D Ultrafast Dynamics in Condensed Matter',  # Required
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',  # Optional (see note above)
)
