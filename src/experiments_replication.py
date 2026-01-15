# src/experiments_replication.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simulate import pid, pdir, pfar
from theory import pid_theory, pdir_theory, pfar_theory


def replicate_fig3_print(N: int = 50, m: int = 100, trials: int = 5000, show_plot: bool = True):
    # Replication of Figure 3
    ps = np.linspace(0.01, 0.30, 25)

    rows = []
    for p_val in ps:
        rows.append({
            "p": float(p_val),
            "pid_theory": pid_theory(N, m, float(p_val)),
            "pid_sim": pid(N, m, float(p_val), trials=trials, seed=1),
        })

    df = pd.DataFrame(rows)

    print("\n Replication of Figure 3")
    print(f"N={N}, m={m}, trials={trials}")
    print(df.to_string(index=False))

    if show_plot:
        plt.figure()
        plt.plot(df["p"], df["pid_theory"])
        plt.scatter(df["p"], df["pid_sim"], s=20, label=f"Simulation ({trials} trials)")
        plt.xlabel("p (individual classifier recognition rate)")
        plt.ylabel("Pid")
        plt.title(f"Replication of Figure 3 (N={N}, m={m})")
        plt.legend()
        plt.tight_layout()
        plt.show()


def replicate_fig5_print(N: int = 50, m: int = 100, trials: int = 5000, show_plot: bool = True):
    # Paper Fig. 5 replication: ROC curves (Pdir vs Pfar) for N=50, m=100,
    ps = [0.05, 0.10, 0.15, 0.20]
    Ts = np.arange(1, N + 1)

    all_rows = []

    if show_plot:
        plt.figure()

    for p_val in ps:
        # Theory arrays
        pfar_th = np.array([pfar_theory(N, m, int(T)) for T in Ts])
        pdir_th = np.array([pdir_theory(N, m, p_val, int(T)) for T in Ts])

        # Simulation arrays
        Ts_sim = Ts[::2]
        pfar_sim = np.array([pfar(N, m, int(T), trials=trials, seed=3) for T in Ts_sim])
        pdir_sim = np.array([pdir(N, m, p_val, int(T), trials=trials, seed=2) for T in Ts_sim])

        # Store printable table rows
        for T, a, b in zip(Ts, pfar_th, pdir_th):
            all_rows.append({
                "p": p_val,
                "T": int(T),
                "pfar_theory": float(a),
                "pdir_theory": float(b),
            })

        if show_plot:
            # Sort by Pf ar for a clean curve
            order = np.argsort(pfar_th)
            plt.plot(pfar_th[order], pdir_th[order], label=f"Theory p={p_val}")
            plt.scatter(pfar_sim, pdir_sim, s=18, label=f"Sim p={p_val}")

    df = pd.DataFrame(all_rows)

    print("\n Replication of Figure 5: ")
    print(f"N={N}, m={m}, trials={trials}")
    print(df.to_string(index=False))

    if show_plot:
        plt.xlabel("Pfar")
        plt.ylabel("Pdir")
        plt.title(f"Replication of Figure 5 ROC (N={N}, m={m})")
        plt.legend()
        plt.tight_layout()
        plt.show()


def main():
    replicate_fig3_print(show_plot=True)
    replicate_fig5_print(show_plot=True)


if __name__ == "__main__":
    main()
