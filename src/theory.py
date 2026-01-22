# src/theory.py
# Consists of all necessary formulas from the paper!

from __future__ import annotations
import math

# Eq1: e = (1-p)/(m-1)
def eq1_e(p, m):
    return (1.0 - p) / (m - 1)

# THE IDENTIFICATION TASK

# Eq2: P(Nt=j) = (N choose j) * p^j * (1-p)^(N-j)
def eq2_P_Nt_eq_j(N, p, j):
    return math.comb(N, j) * (p ** j) * ((1.0 - p) ** (N - j))

# Eq3: P(Ns=j) = (N choose j) * e^j * (1-e)^(N-j)
def eq3_P_Ns(N, e, j):
    return math.comb(N, j) * (e ** j) * ((1.0 - e) ** (N - j))

# Eq5: P(Ns_max=j) = sum_{k=1..m-1} C(m-1,k) [P(Ns=j)]^k [P(Ns<j)]^(m-1-k)
def eq5_P_Nsmax(N, m, p, j):
    e_prob = eq1_e(p, m)
    P_eq = eq3_P_Ns(N, e_prob, j)

    # P(Ns < j) = sum_{t=0..j-1} P(Ns=t) with Eq. (3)
    P_lt = 0.0
    for t in range(0, j):
        P_lt += eq3_P_Ns(N, e_prob, t)

    # sum_{k=1..m-1} C(m-1,k) [P(Ns=j)]^k [P(Ns<j)]^(m-1-k)
    total = 0.0
    for k in range(1, m):
        total += math.comb(m - 1, k) * (P_eq ** k) * (P_lt ** (m - 1 - k))

    return total

# Eq8: Pid = sum_{j=1..N} P(Nt=j) * sum_{k=0..j-1} P(Ns_max=k)
def pid_theory(N, m, p):
    Nsmax_pmf = [eq5_P_Nsmax(N, m, p, k) for k in range(0, N + 1)]

    total = 0.0
    for j in range(1, N + 1):
        outer = eq2_P_Nt_eq_j(N, p, j)

        inner = 0.0
        for k in range(0, j):
            inner += Nsmax_pmf[k]

        total += outer * inner
    return total

# THE WATCHLIST TEST

# Eq15 Pdir = sum_{j=T..N} P(Nt=j) * sum_{k=0..j-1} P(Ns_max=k)
def pdir_theory(N, m, p, T):
    Nsmax_pmf = [eq5_P_Nsmax(N, m, p, k) for k in range(0, N + 1)]
    total = 0.0
    for j in range(T, N + 1):
        outer = eq2_P_Nt_eq_j(N, p, j)
        inner = 0.0
        for k in range(0, j):
            inner += Nsmax_pmf[k]
        total += outer * inner
    return total

# Eq16: e' = 1/m
def eq16_eprime(m):
    return 1.0 / m

# Eq17: P(Ns'=j) = (N choose j) * (e')^j * (1-e')^(N-j)
def eq17_P_Nsprime(N, eprime, j):
    return math.comb(N, j) * (eprime ** j) * ((1.0 - eprime) ** (N - j))

# Eq18: P(Ns'_max=j)
def eq18_P_Nsprime_max(N, m, j):
    eprime = eq16_eprime(m)
    P_eq = eq17_P_Nsprime(N, eprime, j)

    # P(Ns' < j) = sum_{t=0..j-1} P(Ns'=t)
    P_lt = 0.0
    for t in range(0, j):
        P_lt += eq17_P_Nsprime(N, eprime, t)

    # sum_{k=1..m} C(m,k) [P(Ns'=j)]^k [P(Ns'<j)]^(m-k)
    total = 0.0
    for k in range(1, m + 1):
        total += math.comb(m, k) * (P_eq ** k) * (P_lt ** (m - k))

    return total

# Eq20: Pfar = sum_{j=T..N} P(Ns'_max=j)
def pfar_theory(N, m, T):
    total = 0.0
    for j in range(T, N + 1):
        total += eq18_P_Nsprime_max(N, m, j)
    return total