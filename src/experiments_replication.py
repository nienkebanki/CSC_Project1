# src/experiments_replication.py
import numpy as np
import matplotlib.pyplot as plt

from simulate import pid, pdir, pfar
from theory import pid_theory, pdir_theory, pfar_theory

# Replication of Figure 3: Pid vs p for N = 50 and m = 100
def Pid_vs_p(N=50, m=100, trials=5000):
    ps = np.linspace(0.01, 0.30, 25)
    
    pid_th = []
    pid_sim = []
    
    for p_val in ps:
        p = float(p_val)
        pid_t = pid_theory(N, m, p)
        pid_th.append(pid_t)
        pid_s = pid(N, m, p, trials=trials, seed=1)
        pid_sim.append(pid_s)

    plt.figure()
    plt.plot(ps, pid_th)
    plt.scatter(ps, pid_sim, s=20)
    plt.xlabel("p")
    plt.ylabel("Pid")
    plt.title(f"Replication of Figure 3 (N={N}, m={m})")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Paper Fig. 5 replication: ROC curves showing how Pdir and Pfar vary for N=50 and m=100,
def Pdir_vs_Pfar(N=50, m=100, trials=5000):
    ps = [0.05, 0.10, 0.15, 0.20]
    Ts = np.arange(1, N + 1)

    plt.figure()

    pfar_th = np.array([pfar_theory(N, m, int(T)) for T in Ts])

    Ts_sim = Ts[::2]
    pfar_sim = np.array([pfar(N, m, int(T), trials=trials, seed=3) for T in Ts_sim])

    for p_val in ps:
        pdir_th = np.array([pdir_theory(N, m, p_val, int(T)) for T in Ts])
        pdir_sim= np.array([pdir(N, m, p_val, int(T), trials=trials, seed=2) for T in Ts_sim])
        order = np.argsort(pfar_th)
        plt.plot(pfar_th[order], pdir_th[order], label=f"Theory p={p_val}")
        plt.scatter(pfar_sim, pdir_sim, s=18, label=f"Sim p={p_val}")


    plt.xlabel("Pfar")
    plt.ylabel("Pdir")
    plt.title(f"Replication Fig. 5 ROC (N={N}, m={m})")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    Pid_vs_p()
    Pdir_vs_Pfar()

if __name__ == "__main__":
    main()
