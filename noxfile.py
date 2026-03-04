import nox


python_versions = ['3.10', '3.11', '3.12', '3.13']


@nox.session(python=python_versions,
             venv_backend='mamba')
def lint(session):
    session.install('flake8')
    session.run('flake8', '.')


@nox.session(python=python_versions,
             venv_backend='mamba')
def tests(session):
    session.install('pytest-cov')
    session.install('.')
    session.run('pytest', '--cov=udkm1Dsim', 'test/')


@nox.session(python='3.13')
def docs(session):
    session.install('.')
    session.install('-r', './docs/requirements.txt')
    session.run('make', '--directory', './docs', 'html', external=True)
