# rprob

`rprob` is a Python package that provides a `R` style collection of functions for generating random numbers, computing densities, distribution functions, quantile functions, and more for various statistical distributions.

[pypi](https://pypi.org/project/rprob/)
[github](https://github.com/jinwook-chang/rprob)

## Installation

To install `rprob`, you can use pip:

```bash
pip install rprob
```

## Usage

Here's an example of how to use `rprob` to generate random numbers from a normal distribution:

```python
from rprob import rnorm

random_numbers = rnorm(100)
print(random_numbers)
```


## Notice

Since `df` is reserved keywords, `df` function in R is replaced as `df_`.
Please be careful to use it.

