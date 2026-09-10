import nox

reuse_venv = False


@nox.session(python='3.13',
             venv_backend='uv', reuse_venv=reuse_venv)
def lint(session):
    session.install('.', '--group', 'lint')
    session.run('ruff', 'check', 'udkm1Dsim/', 'tests/', 'examples/')


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
@nox.session(venv_backend='uv', reuse_venv=reuse_venv)
def tests(session, numpy):
    session.install(f'numpy=={numpy}')
    session.install('.', '--group', 'test')
    session.run('pytest', '--cov=udkm1Dsim', '--ignore=tests/benchmarks', 'tests/')


@nox.session(venv_backend='uv', python='3.13', reuse_venv=reuse_venv)
def docs(session):
    session.install('.[docs]')
    session.run('make', '--directory', './docs', 'html', external=True)


@nox.session(venv_backend='uv', python='3.13', reuse_venv=reuse_venv)
def benchmarks(session):
    session.install('.', '--group', 'test')
    session.run('pytest', '--codspeed', 'tests/benchmarks/')
