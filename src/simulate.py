# src/simulate.py
from __future__ import annotations
import numpy as np

def single_run_identification(N: int, m: int, p: float, rng: np.random.Generator) -> tuple[int, int]:
    # if random number is less than p, classifier i votes for target class
    correct = rng.random(N) < p
    nt = int(correct.sum())
    wrong = N - nt

    if wrong > 0:
        # wrong votes go uniformly to classes 1..m-1, equation 1 from paper
        non_votes = rng.integers(1, m, size=wrong)  # 1..m-1 inclusive
        counts = np.bincount(non_votes, minlength=m)
        max_non = int(counts[1:].max())
    else:
        max_non = 0

    return nt, max_non

# Identification Rate, Pid = P(Nt > Ns max), Eq 6
def pid(N: int, m: int, p: float, trials: int = 5000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    successes = 0

    for _ in range(trials):
        nt, max_non = single_run_identification(N, m, p, rng) # Simulates one vote profile
        if nt > max_non:
            successes += 1 # Counts successes if Nt > Ns max

    return successes / trials # Ratio of number of successes over the total trials

# Detection and Identification Rate, Pdir = P(Nt > Ns max and Nt >= T), Eq 14
def pdir(N: int, m: int, p: float, T: int, trials: int = 5000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    successes = 0

    for _ in range(trials):
        nt, max_non = single_run_identification(N, m, p, rng)
        if (nt > max_non) and (nt >= T):
            successes += 1

    return successes / trials

# False Acceptance Rate, Pfar = P(Ns' max >= T), Eq 19
def pfar(N: int, m: int, T: int, trials: int = 5000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    false_accepts = 0

    for _ in range(trials):
        votes = rng.integers(0, m, size=N)  # Uniform among all m classes
        counts = np.bincount(votes, minlength=m)
        if int(counts.max()) >= T:
            false_accepts += 1

    return false_accepts / trials

# EXTENSION Heterogeneous accuracies (pi) + weighted plurality
def pid_hetero_unweighted(N: int, m: int, pi: np.ndarray, trials: int = 5000, seed: int = 0) -> float:
    # Extension identification with heterogeneous accuracies pi, unweighted.
    rng = np.random.default_rng(seed)
    successes = 0

    for _ in range(trials):
        votes = np.empty(N, dtype=int)

        # Simulate one vote from each classifier i
        for i in range(N):
            if rng.random() < pi[i]:
                votes[i] = 0
            else:
                votes[i] = rng.integers(1, m)

        counts = np.bincount(votes, minlength=m)
        Nt = int(counts[0])
        Nmax_s = int(counts[1:].max())

        if Nt > Nmax_s:
            successes += 1

    return successes / trials


def pid_hetero_weighted(N: int, m: int, pi: np.ndarray, trials: int = 5000, seed: int = 0) -> float:
    # Extension identification with heterogeneous accuracies pi, weighted.
    rng = np.random.default_rng(seed)

    # Compute weights w_i from pi
    eps = 1e-12
    e_i = (1.0 - pi) / (m - 1)
    w = np.log((pi + eps) / (e_i + eps))
    w = np.maximum(w, 0.0)

    # Normalize weights so sum(w)=N
    s = float(w.sum())
    if s <= 0:
        w = np.ones(N)
    else:
        w = w * (N / s)

    successes = 0

    for _ in range(trials):
        votes = np.empty(N, dtype=int)

        # Simulate votes
        for i in range(N):
            if rng.random() < pi[i]:
                votes[i] = 0
            else:
                votes[i] = rng.integers(1, m)

        # Weighted plurality scores per class
        scores = np.zeros(m, dtype=float)
        for i in range(N):
            scores[votes[i]] += w[i]

        # Strict win for target
        if scores[0] > scores[1:].max():
            successes += 1

    return successes / trials


def pdir_hetero_unweighted(N: int, m: int, pi: np.ndarray, T: int, trials: int = 5000, seed: int = 0) -> float:
    # Extension watchlist with heterogeneous accuracies pi, unweighted.
    rng = np.random.default_rng(seed)
    successes = 0

    for _ in range(trials):
        votes = np.empty(N, dtype=int)

        for i in range(N):
            if rng.random() < pi[i]:
                votes[i] = 0
            else:
                votes[i] = rng.integers(1, m)

        counts = np.bincount(votes, minlength=m)
        Nt = int(counts[0])
        Nmax_s = int(counts[1:].max())

        if (Nt > Nmax_s) and (Nt >= T):
            successes += 1

    return successes / trials


def pdir_hetero_weighted(N: int, m: int, pi: np.ndarray, T: int, trials: int = 5000, seed: int = 0) -> float:
    # Extension watchlist with heterogeneous accuracies pi, weighted.
    rng = np.random.default_rng(seed)

    eps = 1e-12
    e_i = (1.0 - pi) / (m - 1)
    w = np.log((pi + eps) / (e_i + eps))
    w = np.maximum(w, 0.0)

    s = float(w.sum())
    if s <= 0:
        w = np.ones(N)
    else:
        w = w * (N / s)

    successes = 0

    for _ in range(trials):
        votes = np.empty(N, dtype=int)

        for i in range(N):
            if rng.random() < pi[i]:
                votes[i] = 0
            else:
                votes[i] = rng.integers(1, m)

        scores = np.zeros(m, dtype=float)
        for i in range(N):
            scores[votes[i]] += w[i]

        if (scores[0] > scores[1:].max()) and (scores[0] >= T):
            successes += 1

    return successes / trials


def pfar_weighted_outsider(N: int, m: int, pi: np.ndarray, T: int, trials: int = 5000, seed: int = 0) -> float:
    # Extension outsider false acceptance for weighted plurality.
    rng = np.random.default_rng(seed)

    eps = 1e-12
    e_i = (1.0 - pi) / (m - 1)
    w = np.log((pi + eps) / (e_i + eps))
    w = np.maximum(w, 0.0)

    s = float(w.sum())
    if s <= 0:
        w = np.ones(N)
    else:
        w = w * (N / s)

    false_accepts = 0

    for _ in range(trials):
        votes = rng.integers(0, m, size=N)  # Uniform among m classes
        scores = np.zeros(m, dtype=float)

        for i in range(N):
            scores[votes[i]] += w[i]

        if float(scores.max()) >= T:
            false_accepts += 1

    return false_accepts / trials