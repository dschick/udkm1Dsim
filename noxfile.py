import nox


@nox.session(python='3.13',
             venv_backend='mamba', reuse_venv=True)
def lint(session):
    session.install('flake8')
    session.run('flake8', '.')


@nox.parametrize(
    'python,numpy',
    [
        ('3.10', '1.26.4'),
        ('3.10', '2.0.0'),

        ('3.11', '1.26.4'),
        ('3.11', '2.0.0'),
        ('3.11', '2.3.0'),
        ('3.11', '2.4.1'),

        ('3.12', '1.26.4'),
        ('3.12', '2.0.0'),
        ('3.12', '2.3.0'),
        ('3.12', '2.4.1'),

        ('3.13', '2.3.0'),
        ('3.13', '2.4.1'),

        ('3.14', '2.3.0'),
        ('3.14', '2.4.1'),
    ],
)
@nox.session(venv_backend='mamba', reuse_venv=True)
def tests(session, numpy):
    session.install(f'numpy=={numpy}')
    session.install('pytest-cov')
    session.install('.')
    session.run('pytest', '--cov=udkm1Dsim', 'test/')


@nox.session(python='3.13', reuse_venv=False)
def docs(session):
    session.install('.')
    session.install('-r', './docs/requirements.txt')
    session.run('make', '--directory', './docs', 'html', external=True)
