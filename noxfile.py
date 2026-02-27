import nox


python_versions = ["3.10", "3.11", "3.12", "3.13"]


@nox.session(python=python_versions,
             venv_backend="mamba")
def lint(session):
    session.install("flake8")
    session.run("flake8", ".")


@nox.session(python=python_versions,
             venv_backend="mamba")
def tests(session):
    session.install("pytest-cov")
    session.install(".")
    session.run("pytest", "--cov=udkm1Dsim", "test/")
