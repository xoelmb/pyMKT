"""
Cython Fisher's exact test:
Fisher's exact test is a statistical significance test used in the
analysis of contingency tables where sample sizes are small.
Function lngamma(), lncombination(), hypergeometric_probability(),
were originally written by Oyvind Langsrud:
Oyvind Langsrud
Copyright (C) : All right reserved.
Contact Oyvind Langsrud for permission.
Adapted to Cython version by:
Haibao Tang, Brent Pedersen
"""

from numba import jit, prange
from numba.types import int64, float64
import numpy as np
from math import log, exp, lgamma


@jit(float64(int64), nopython=True, parallel=True, cache=True)
def _naive_lnfactorial(n):
    acc = 0.0
    for i in prange(2, n + 1):
        acc += log(i)
    return acc

# Tabulated ln n! for n \in [0, 1023]
_lnfactorials1 = np.zeros(1024)
for i in range(1024):
    _lnfactorials1[i] = _naive_lnfactorial(i)


# Logarithm of n! with algorithmic approximation
@jit(float64(int64), nopython=True, cache=True)
def lnfactorial(n):
    return _lnfactorials1[n] if n < 1024 else lgamma(n + 1)


# Logarithm of the number of combinations of 'n' objects taken 'p' at a time
@jit(float64(int64, int64), nopython=True, cache=True)
def lncombination(n, p):
    return lnfactorial(n) - lnfactorial(p) - lnfactorial(n - p)


# Compute the hypergeometric probability, or probability that a list of
# 'n' objects should contain 'x' ones with a particular property when the
# list has been selected randomly without replacement from a set of 'N'
# objects in which 'K' exhibit the same property
@jit(float64(int64,int64,int64,int64), nopython=True, cache=True)
def hypergeometric_probability(x, n, K, N):
    return exp(lncombination(K, x)
               + lncombination(N - K, n - x)
               - lncombination(N, n))


@jit(float64(int64, int64, int64, int64), nopython=True, cache=True)
def pval(a_true, a_false, b_true, b_false):
    # Convert the a/b groups to study vs population.
    k = a_true
    n = a_false + a_true  # total in study.
    K = a_true + b_true
    N = K + a_false + b_false

    lm = max(0, n - (N - K))
    um = min(n, K)
    if lm == um:
        return 1.0

    epsilon = 1e-6
    cutoff = hypergeometric_probability(k, n, K, N)
    left_tail = 0
    right_tail = 0
    two_tail = 0

    for x in range(lm, um + 1):
        p = hypergeometric_probability(x, n, K, N)
        if x <= k:
            left_tail += p
        if x >= k:
            right_tail += p

        if p <= cutoff + epsilon:
            two_tail += p

    return min(two_tail, 1.0)