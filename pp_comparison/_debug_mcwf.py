import numpy as np, time
from lindblad_exact import run_exact_lindblad
from mps_lindblad import run_mcwf

L, gamma, T, tau = 6, 0.1, 20.0, 0.1
ex = run_exact_lindblad(L, gamma, T, tau, experiment="B")
t0 = time.time()
mc = run_mcwf(L, gamma, T, tau, dt=0.05, max_bond=32, n_traj=40, seed=1)
print("40-traj wall=%.1fs" % (time.time() - t0))
# compare at a few time indices
idxs = [0, 20, 50, 100, len(ex["t"]) - 1]
print("  t      exact      mcwf(40)   stderr")
for i in idxs:
    print("  %5.1f  %9.5f  %9.5f  %.4f" % (ex["t"][i], ex["energy"][i], mc["energy"][i], mc["stderr"][i]))
d = np.abs(mc["energy"] - ex["energy"]) / L
print("max|dH|/L=%.3e  mean=%.3e" % (d.max(), d.mean()))
