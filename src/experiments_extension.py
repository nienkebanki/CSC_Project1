# src/experiments_extension.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulate import (
    pid_hetero_unweighted,
    pid_hetero_weighted,
    pdir_hetero_unweighted,
    pdir_hetero_weighted,
    pfar,
    pfar_weighted_outsider,
)

# Sample heterogeneous accuracies p_i from a Beta distribution
def sample_pi_beta(N, mean_p, strength, seed):
    rng = np.random.default_rng(seed)
    alpha = mean_p * strength
    beta = (1.0 - mean_p) * strength
    return rng.beta(alpha, beta, size=N)

# Extension Fig 3 plot:
# Compare Pid for unweighted vs weighted plurality under heterogeneous accuracies
def Pid_vs_p_weighted(N=50, m=100, mean_p=0.10, strengths=(5, 10, 30, 100, 300), trials=5000):
    pid_unw = []
    pid_w = []

    for strength in strengths:
        pi = sample_pi_beta(N, mean_p, strength, seed=123)

        pid_u_val = pid_hetero_unweighted(N, m, pi, trials=trials, seed=1)
        pid_w_val = pid_hetero_weighted(N, m, pi, trials=trials, seed=1)

        pid_unw.append(pid_u_val)
        pid_w.append(pid_w_val)

    plt.figure()
    plt.plot(strengths, pid_unw, marker="o", label="Unweighted plurality")
    plt.plot(strengths, pid_w, marker="o", label="Weighted plurality")
    plt.xscale("log")
    plt.xlabel("strength (higher = less heterogeneity)")
    plt.ylabel("Pid")
    plt.title(f"Pid vs Strength of Heterogenity with N={N}, m={m}, mean_p={mean_p}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Extension Fig 5 ROC plot:
# Compare ROC curves (Pdir vs Pfar) for unweighted vs weighted plurality under heterogeneity
def Pdir_vs_Pfar_weighted(N=50, m=100, mean_p=0.10, strength=10, trials=5000):
    pi = sample_pi_beta(N, mean_p, strength, seed=456)
    Ts = np.arange(1, N + 1)

    # Unweighted ROC
    pfar_unw = np.array([pfar(N, m, int(T), trials=trials, seed=7) for T in Ts])
    pdir_unw = np.array([pdir_hetero_unweighted(N, m, pi, int(T), trials=trials, seed=8) for T in Ts])

    # Weighted ROC
    pfar_w = np.array([pfar_weighted_outsider(N, m, pi, int(T), trials=trials, seed=7) for T in Ts])
    pdir_w = np.array([pdir_hetero_weighted(N, m, pi, int(T), trials=trials, seed=8) for T in Ts])

    # Sort by Pf ar so the ROC curves look clean
    ord_unw = np.argsort(pfar_unw)
    ord_w = np.argsort(pfar_w)

    plt.figure()
    plt.plot(pfar_unw[ord_unw], pdir_unw[ord_unw], label="Unweighted plurality")
    plt.plot(pfar_w[ord_w], pdir_w[ord_w], label="Weighted plurality")
    plt.xlabel("Pfar")
    plt.ylabel("Pdir")
    plt.title(f"Pdir vs Pfar with N={N}, m={m}, mean_p={mean_p}, strength={strength}")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    Pid_vs_p_weighted()
    Pdir_vs_Pfar_weighted()

if __name__ == "__main__":
    main()
