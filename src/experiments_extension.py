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

def sample_pi_beta(N: int, mean_p: float, kappa: float, seed: int) -> np.ndarray:
    # Heterogeneous accuracies p_i sampled from Beta distribution
    rng = np.random.default_rng(seed)
    alpha = mean_p * kappa
    beta = (1.0 - mean_p) * kappa
    return rng.beta(alpha, beta, size=N)


def extension_pid_print(
    N: int = 50,
    m: int = 100,
    mean_p: float = 0.10,
    kappas: list[float] = [5, 10, 30, 100, 300],
    trials: int = 5000,
    show_plot: bool = True,
):
    # Extension: Pid heterogeneous unweighted plurality vs. weighted plurality
    rows = []
    for kappa in kappas:
        pi = sample_pi_beta(N, mean_p, kappa, seed=123)

        pid_u = pid_hetero_unweighted(N, m, pi, trials=trials, seed=1)
        pid_w = pid_hetero_weighted(N, m, pi, trials=trials, seed=1)

        rows.append({
            "kappa": kappa,
            "pi_mean": float(pi.mean()),
            "pi_std": float(pi.std()),
            "pid_unweighted": pid_u,
            "pid_weighted": pid_w,
        })

    df = pd.DataFrame(rows)

    print("\n Extension: Pid with heterogeneous accuracies ")
    print(f"N={N}, m={m}, mean_p_target={mean_p}, trials={trials}")
    print(df.to_string(index=False))

    if show_plot:
        plt.figure()
        plt.plot(df["kappa"], df["pid_unweighted"], marker="o", label="Unweighted plurality")
        plt.plot(df["kappa"], df["pid_weighted"], marker="o", label="Weighted plurality")
        plt.xscale("log")
        plt.xlabel("kappa")
        plt.ylabel("Pid")
        plt.title("Extension: Unweighted vs Weighted (Pid)")
        plt.legend()
        plt.tight_layout()
        plt.show()


def extension_roc_print(
    N: int = 50,
    m: int = 100,
    mean_p: float = 0.10,
    kappa: float = 10,
    trials: int = 5000,
    show_plot: bool = True,
):
    # Extension: ROC curves of unweighted vs weighted
    pi = sample_pi_beta(N, mean_p, kappa, seed=456)
    Ts = np.arange(1, N + 1)

    # Unweighted ROC
    pfar_u = np.array([pfar(N, m, int(T), trials=trials, seed=7) for T in Ts])
    pdir_u = np.array([pdir_hetero_unweighted(N, m, pi, int(T), trials=trials, seed=8) for T in Ts])

    # Weighted ROC
    pfar_w = np.array([pfar_weighted_outsider(N, m, pi, int(T), trials=trials, seed=7) for T in Ts])
    pdir_w = np.array([pdir_hetero_weighted(N, m, pi, int(T), trials=trials, seed=8) for T in Ts])

    df = pd.DataFrame({
        "T": Ts,
        "pfar_unweighted": pfar_u,
        "pdir_unweighted": pdir_u,
        "pfar_weighted": pfar_w,
        "pdir_weighted": pdir_w,
    })

    print("\n Extension: ROC unweighted vs weighted ")
    print(f"N={N}, m={m}, mean_p~{pi.mean():.4f}, std_p~{pi.std():.4f}, kappa={kappa}, trials={trials}")
    print(df.to_string(index=False))

    if show_plot:
        ord_u = np.argsort(pfar_u)
        ord_w = np.argsort(pfar_w)

        plt.figure()
        plt.plot(pfar_u[ord_u], pdir_u[ord_u], label="Unweighted plurality")
        plt.plot(pfar_w[ord_w], pdir_w[ord_w], label="Weighted plurality")
        plt.xlabel("Pfar")
        plt.ylabel("Pdir")
        plt.title("Extension ROC: Unweighted vs Weighted")
        plt.legend()
        plt.tight_layout()
        plt.show()


def main():
    extension_pid_print(show_plot=True)
    extension_roc_print(show_plot=True)


if __name__ == "__main__":
    main()
