from setuptools import setup, find_packages

setup(
    name='udkm1Dsim',
    version='0.1.4',
    packages=find_packages(),
    package_data={
        '': ['*.txt', '*.dat', '*.nff', '*.mf'],
    },
    url='https://github.com/dschick/udkm1Dsim',
    install_requires=['tqdm',
                      'numpy',
                      'pint',
                      'pytest',
                      'scipy',
                      'sympy',
                      'tabulate'],
    extras_require={
        'parallel':  ['dask'],
    },
    license='MIT',
    author='Daniel Schick, et. al.',
    author_email='schick.daniel@gmail.com',
    description='A Python Simulation Toolkit for 1D Ultrafast Dynamics '
                + 'in Condensed Matter',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    python_requires='>=3.5',
    keywords='ultrafast dynamics condensed matter 1D',
)
