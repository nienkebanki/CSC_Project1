# src/theory.py
import numpy as np
from scipy.stats import binom

def _max_iid_binom_cdf(N: int, q: float, k: int) -> np.ndarray:
    # If X1..Xk are i.i.d. Bin(N,q), then: P(max <= t) = P(X <= t)^k
    t = np.arange(N + 1)
    F = binom.cdf(t, N, q)
    return F ** k

def pid_theory(N: int, m: int, p: float) -> float:
    # Eq 8: sum_j P(Nt=j) * P(Nmax < j)
    e = (1 - p) / (m - 1)  # Eq 1
    j = np.arange(N + 1)
    pmf_t = binom.pmf( j, N, p )  # Eq 2

    # P(Nmax_s <= t) where Nmax_s = max of (m-1) Bin(N,e)
    Fmax = _max_iid_binom_cdf(N, e, m - 1) 

    # P(Nmax_s < j) = P(Nmax_s <= j-1)
    Pmax_lt = np.concatenate(([0.0], Fmax[:-1]))

    return float(np.sum(pmf_t * Pmax_lt))

def pdir_theory(N: int, m: int, p: float, T: int) -> float:
    # Eq 15: Pdir = sum_{j=T..N} P(Nt=j) * P(Nmax_s < j)
    e = (1 - p) / (m - 1)
    j = np.arange(N + 1)
    pmf_t = binom.pmf(j, N, p)

    Fmax = _max_iid_binom_cdf(N, e, m - 1)
    Pmax_lt = np.concatenate(([0.0], Fmax[:-1]))

    mask = j >= T
    return float(np.sum(pmf_t[mask] * Pmax_lt[mask]))

def pfar_theory(N: int, m: int, T: int) -> float:
    # Eq 16-20: outsider votes uniformly => q = 1/m, Pf ar = P(max >= T) = 1 - P(max <= T-1)
    if T <= 0:
        return 1.0
    if T - 1 >= N:
        return 0.0

    q = 1.0 / m  # Eq 16: e' = 1/m
    Fmax = _max_iid_binom_cdf(N, q, m)
    return float(1.0 - Fmax[T - 1]) # Pfar = 1 - P(Nmax <= T-1)
