import pytest


# Define the function we want to benchmark
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    else:
        return fibonacci(n - 2) + fibonacci(n - 1)


@pytest.mark.benchmark
def test_fib_parametrized():
    result = fibonacci(10)
    assert result > 0
