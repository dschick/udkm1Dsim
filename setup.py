from setuptools import setup

setup(
    name='udkm1Dsim',
    version='0.1.4',
    packages=['udkm1Dsim'],
    package_data={
        'udkm1Dsim': ['parameters/atomic_form_factors/chantler/*.cf',
                      'parameters/atomic_form_factors/chantler/*.md',
                      'parameters/atomic_form_factors/henke/*.nff',
                      'parameters/atomic_form_factors/henke/*.md',
                      'parameters/atomic_form_factors/cromermann.txt',
                      'parameters/magnetic_form_factors/*.mf',
                      'parameters/elements.dat',
                      'matlab/*.m',
                      ],
    },
    url='https://github.com/dschick/udkm1Dsim',
    install_requires=['tqdm>=4.43.0',
                      'numpy>=1.18.2',
                      'pint>=0.9',
                      'scipy>=1.4.1',
                      'sympy>=1.5.1',
                      'tabulate'],
    extras_require={
        'parallel':  ['dask>=2.6.0'],
        'testing': ['flake8', 'pytest'],
        'documentation': ['sphinx', 'nbsphinx', 'sphinxcontrib-napoleon'],
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
