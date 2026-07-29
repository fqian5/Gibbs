"""
Plot fidelity |<GS|ψ(t)>|² vs iteration for the efficient Lindblad simulator.

Usage:
    conda activate lindbladian
    cd /path/to/Gibbs/efficient
    python plot_fidelity.py
"""

import sys, os

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works in all terminals
import matplotlib.pyplot as plt
import numpy as np
import torch
import quimb as qu
import quimb.tensor as qtn
from simulator import EfficientLindbladSimulator


# ── parameters ──────────────────────────────────────────────────────────────
L         = 4
pxx       = -1.0
pz        = 1.5
sigma     = 4.0
alpha     = 0.25      # tuned for dt=0.05
dt        = 0.05
Ss_factor = 2.0
max_bond  = 32
n_steps   = 60
n_seeds   = 5         # independent trajectories
# ────────────────────────────────────────────────────────────────────────────

print(f"System: L={L}, pxx={pxx}, pz={pz}")
print(f"Params: dt={dt}, Ss_factor={Ss_factor}, alpha={alpha}, max_bond={max_bond}")


# ── ground state via numpy DMRG (avoids torch/numpy backend conflict) ────────
def get_ground_state_numpy(L, pxx, pz):
    """Build ground state MPS using pure-numpy DMRG via SpinHam1D."""
    builder = qtn.SpinHam1D(S=1/2)
    for i in range(L - 1):
        builder[i, i + 1] += pxx, qu.pauli("X"), qu.pauli("X")
    for i in range(L):
        builder[i] += pz, qu.pauli("Z")
    ham_np = builder.build_mpo(L)
    dmrg = qtn.DMRG2(ham_np)
    dmrg.solve(max_sweeps=80, verbosity=0)
    psi_gs = dmrg.state
    psi_gs.normalize()
    E = float(np.real(qtn.expec_TN_1D(psi_gs.H, ham_np, psi_gs)))
    # Convert to torch so it's compatible with the simulator's psi
    for t in psi_gs.tensors:
        t.modify(data=torch.tensor(np.array(t.data), dtype=torch.complex128))
    return psi_gs, E


print("Computing ground state via DMRG …")
psi_gs, E_exact = get_ground_state_numpy(L, pxx, pz)
print(f"Ground energy: {E_exact:.6f}\n")

print(f"Running {n_seeds} trajectories × {n_steps} steps …\n")

all_fidelities = []
all_energies   = []

for seed in range(n_seeds):
    print(f"Seed {seed + 1}/{n_seeds} …", end=" ", flush=True)
    sim = EfficientLindbladSimulator(
        L=L, pxx=pxx, pz=pz, dt=dt, Ss_factor=Ss_factor, sigma=sigma,
        max_bond=max_bond, seed=seed, preseed=True,
    )
    out = sim.run(n_steps=n_steps, alpha=alpha, psi_gs=psi_gs, verbose=False)
    all_fidelities.append(out["fidelities"])
    all_energies.append(out["energies"])
    print(f"final fidelity={out['fidelities'][-1]:.4f}  E={out['energies'][-1]:.4f}")

fid  = np.array(all_fidelities)   # (n_seeds, n_steps)
eng  = np.array(all_energies)
steps = np.arange(1, n_steps + 1)

mean_fid = fid.mean(axis=0)
std_fid  = fid.std(axis=0)
mean_eng = eng.mean(axis=0)
std_eng  = eng.std(axis=0)


# ── plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

# --- fidelity panel ---
ax = axes[0]
for f in fid:
    ax.plot(steps, f, alpha=0.25, linewidth=0.8, color="steelblue")
ax.plot(steps, mean_fid, color="steelblue", linewidth=2, label="mean fidelity")
ax.fill_between(steps, mean_fid - std_fid, mean_fid + std_fid,
                alpha=0.25, color="steelblue", label="±1σ")
ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, label="perfect overlap")
ax.set_ylabel(r"Fidelity $|\langle\mathrm{GS}|\psi\rangle|^2$", fontsize=12)
ax.set_ylim(-0.05, 1.15)
ax.legend(fontsize=9)
ax.set_title(
    f"Randomized Lindblad cooling  (L={L}, pxx={pxx}, pz={pz}, α={alpha}, dt={dt})\n"
    f"{n_seeds} independent trajectories",
    fontsize=11,
)

# --- energy panel ---
ax = axes[1]
for e in eng:
    ax.plot(steps, e, alpha=0.25, linewidth=0.8, color="tomato")
ax.plot(steps, mean_eng, color="tomato", linewidth=2, label="mean energy")
ax.fill_between(steps, mean_eng - std_eng, mean_eng + std_eng,
                alpha=0.25, color="tomato", label="±1σ")
ax.axhline(E_exact, color="k", linestyle="--", linewidth=0.8,
           label=f"$E_{{exact}}$ = {E_exact:.4f}")
ax.set_xlabel("Iteration", fontsize=12)
ax.set_ylabel("Energy $\\langle H \\rangle$", fontsize=12)
ax.legend(fontsize=9)

plt.tight_layout()
out_path = "fidelity_plot.png"
plt.savefig(out_path, dpi=150)
print(f"\nSaved → {out_path}")
plt.show()
