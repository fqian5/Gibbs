"""
Performance test for zhiyan/sim.py
Plots fidelity |<GS|ψ(t)>|² and energy vs iteration over multiple parallel trajectories.

Usage:
    conda activate lindbladian
    cd /path/to/Gibbs/zhiyan
    python test_performance.py
"""

import sys, os
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim import LindbladSimulator, run_parallel

if __name__ == "__main__":  # required: macOS spawn re-imports __main__ in workers

    # ── parameters ──────────────────────────────────────────────────────────
    L         = 4
    pxx       = -1.0
    pz        = 1.5
    alpha     = 0.25
    dt        = 0.05
    Ss_factor = 2.0
    sigma     = 4.0
    max_bond  = 32
    n_steps   = 80
    n_traj    = 8
    n_workers = min(n_traj, os.cpu_count() or 4)
    # ────────────────────────────────────────────────────────────────────────

    print(f"System:  L={L}, pxx={pxx}, pz={pz}")
    print(f"Params:  dt={dt}, Ss={Ss_factor}σ, alpha={alpha}, max_bond={max_bond}")
    print(f"Run:     {n_traj} trajectories × {n_steps} steps  ({n_workers} workers)")
    print()

    # ── ground state (numpy DMRG, then convert to torch) ────────────────────
    print("Computing ground state via DMRG …")
    ref = LindbladSimulator(L=L, pxx=pxx, pz=pz, dt=dt, Ss_factor=Ss_factor,
                            sigma=sigma, max_bond=max_bond, preseed=False, seed=0)
    psi_gs, E_exact = ref.get_ground_state()
    print(f"E_exact = {E_exact:.6f}\n")

    # ── parallel run ─────────────────────────────────────────────────────────
    import time
    print(f"Launching {n_workers} parallel workers …")
    t0 = time.perf_counter()

    results = run_parallel(
        n_traj    = n_traj,
        n_steps   = n_steps,
        alpha     = alpha,
        sim_params= dict(L=L, pxx=pxx, pz=pz, dt=dt, Ss_factor=Ss_factor,
                         sigma=sigma, max_bond=max_bond, preseed=True),
        psi_gs    = psi_gs,
        n_workers = n_workers,
    )

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s  ({elapsed/(n_traj*n_steps):.2f}s / traj-step)")

    fid = results["fidelities"]   # (n_traj, n_steps)
    eng = results["energies"]     # (n_traj, n_steps)

    # ── print summary ────────────────────────────────────────────────────────
    steps = np.arange(1, n_steps + 1)
    for traj_id in range(n_traj):
        print(f"  traj {traj_id+1}: final fidelity={fid[traj_id,-1]:.4f}  E={eng[traj_id,-1]:.4f}")

    mean_fid = fid.mean(0); std_fid = fid.std(0)
    mean_eng = eng.mean(0); std_eng = eng.std(0)
    print(f"\nMean final fidelity: {mean_fid[-1]:.4f} ± {std_fid[-1]:.4f}")
    print(f"Mean final energy:   {mean_eng[-1]:.4f} ± {std_eng[-1]:.4f}  (exact: {E_exact:.4f})")

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax = axes[0]
    for f in fid:
        ax.plot(steps, f, alpha=0.25, linewidth=0.8, color="steelblue")
    ax.plot(steps, mean_fid, color="steelblue", linewidth=2, label="mean fidelity")
    ax.fill_between(steps, mean_fid - std_fid, mean_fid + std_fid,
                    alpha=0.25, color="steelblue", label="±1σ")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, label="perfect")
    ax.set_ylabel(r"Fidelity $|\langle\mathrm{GS}|\psi\rangle|^2$", fontsize=12)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=9)
    ax.set_title(
        f"Randomized Lindblad cooling  (L={L}, pxx={pxx}, pz={pz}, α={alpha}, dt={dt})\n"
        f"{n_traj} trajectories via run_parallel  [zhiyan/sim.py]",
        fontsize=11,
    )

    ax = axes[1]
    for e in eng:
        ax.plot(steps, e, alpha=0.25, linewidth=0.8, color="tomato")
    ax.plot(steps, mean_eng, color="tomato", linewidth=2, label="mean energy")
    ax.fill_between(steps, mean_eng - std_eng, mean_eng + std_eng,
                    alpha=0.25, color="tomato", label="±1σ")
    ax.axhline(E_exact, color="k", linestyle="--", linewidth=0.8,
               label=f"$E_{{exact}}$ = {E_exact:.4f}")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(r"Energy $\langle H\rangle$", fontsize=12)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig("fidelity_plot.png", dpi=150)
    print("\nSaved → fidelity_plot.png")
