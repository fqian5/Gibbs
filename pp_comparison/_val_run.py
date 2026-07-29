import json, sys, numpy as np
from lindblad_exact import run_exact_lindblad, ness_energy_direct
from mps_lindblad import run_mcwf

L = int(sys.argv[1]); gamma = float(sys.argv[2]); T = float(sys.argv[3])
n_traj = int(sys.argv[4]); chi = int(sys.argv[5])
tau = 0.1
ex = run_exact_lindblad(L, gamma, T, tau, experiment="B")
e_ness = ness_energy_direct(L, gamma, experiment="B")
mc = run_mcwf(L, gamma, T, tau, dt=0.05, max_bond=chi, n_traj=n_traj, seed=1)
d = np.abs(mc["energy"] - ex["energy"]) / L
print(f"L={L} gamma={gamma} T={T} n_traj={n_traj} chi={chi}")
print(f"  exact H(0)={ex['energy'][0]:.4f} H(T)={ex['energy'][-1]:.4f}  NESS(kernel)={e_ness:.4f}  E_GS={ex['E_GS']:.4f}")
print(f"  mcwf  H(0)={mc['energy'][0]:.4f} H(T)={mc['energy'][-1]:.4f}+/-{mc['stderr'][-1]:.4f}")
print(f"  max|dH|/L={d.max():.3e} mean|dH|/L={d.mean():.3e}  mcwf_wall={mc['wall_time']:.1f}s")
out = dict(L=L, gamma=gamma, T=T, tau=tau, n_traj=n_traj, chi=chi,
           t=ex["t"].tolist(), exact=ex["energy"].tolist(), E_GS=ex["E_GS"],
           E_NESS_kernel=e_ness, neg_pzL=-1.5*L,
           mcwf=mc["energy"].tolist(), mcwf_stderr=mc["stderr"].tolist(),
           mcwf_wall=mc["wall_time"], mcwf_maxdH_L=float(d.max()),
           mcwf_meandH_L=float(d.mean()))
with open(f"phase2_val_L{L}.json", "w") as f:
    json.dump(out, f)
print(f"  wrote phase2_val_L{L}.json")
