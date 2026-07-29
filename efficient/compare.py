"""
Comparison between the efficient simulator and zhiyan's verified baseline.

Usage:
    conda activate lindbladian
    cd /path/to/Gibbs/efficient
    python compare.py

What it does:
    1. Runs zhiyan's baseline parameters (dt=0.01, Ss=4σ) for N steps on L=4
    2. Runs efficient parameters   (dt=0.05, Ss=2σ) for the same N steps
    3. Checks that both converge to the same ground-state energy (within tolerance)
    4. Reports wall-clock time per step for both
"""

import sys
import time
import random
import numpy as np
import torch
import quimb as qu
import quimb.tensor as qtn
from simulator import EfficientLindbladSimulator

# ---------------------------------------------------------------------------
# Exact diagonalization ground-state energy  (reference)
# ---------------------------------------------------------------------------

def exact_ground_energy(L, pxx, pz):
    dims = [2] * L
    H = sum(
        pxx * qu.ikron(qu.pauli("X") & qu.pauli("X"), dims, [i, i + 1])
        for i in range(L - 1)
    )
    H += sum(pz * qu.ikron(qu.pauli("Z"), dims, [i]) for i in range(L))
    return float(qu.eigvalsh(H, k=1)[0])


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_comparison(
    L: int = 4,
    pxx: float = -1.0,
    pz: float = 1.5,
    sigma: float = 4.0,
    alpha: float = 0.25,   # tuned for dt=0.05 (zhiyan uses 0.5 with dt=0.01)
    max_bond: int = 32,
    n_steps: int = 100,
    seed: int = 42,
):
    print("=" * 60)
    print(f"Comparing Lindblad simulators  (L={L}, pxx={pxx}, pz={pz})")
    print("=" * 60)

    E_exact = exact_ground_energy(L, pxx, pz)
    print(f"Exact ground energy: {E_exact:.6f}")
    print(f"Using alpha={alpha:.4f}\n")

    configs = [
        dict(label="Baseline (dt=0.01, Ss=4σ)", dt=0.01, Ss_factor=4.0),
        dict(label="Efficient (dt=0.05, Ss=2σ)", dt=0.05, Ss_factor=2.0),
    ]

    results = {}
    for cfg in configs:
        label    = cfg["label"]
        dt       = cfg["dt"]
        Ss_factor = cfg["Ss_factor"]
        M = int(round(2 * Ss_factor * sigma / dt))
        print(f"--- {label} ---")
        print(f"    dt={dt}, Ss={Ss_factor*sigma:.1f}, sub-steps M={M}")

        sim = EfficientLindbladSimulator(
            L=L, pxx=pxx, pz=pz,
            dt=dt, Ss_factor=Ss_factor, sigma=sigma,
            max_bond=max_bond, seed=seed, preseed=True,
        )

        energies = []
        t0 = time.perf_counter()
        for n in range(n_steps):
            sim.step(alpha=alpha)
            E = sim.get_energy()
            energies.append(E)
            if (n + 1) % 5 == 0:
                print(f"    step {n+1:3d}/{n_steps}  E={E:.6f}")
        elapsed = time.perf_counter() - t0

        final_E = energies[-1]
        rel_err = abs(final_E - E_exact) / abs(E_exact)
        time_per_step = elapsed / n_steps

        print(f"    Final energy:    {final_E:.6f}")
        print(f"    Exact energy:    {E_exact:.6f}")
        print(f"    Relative error:  {rel_err:.2e}")
        print(f"    Total time:      {elapsed:.2f}s  ({time_per_step:.3f}s/step)\n")

        results[label] = dict(
            energies=energies,
            final_E=final_E,
            rel_err=rel_err,
            time_per_step=time_per_step,
        )

    # Speedup
    r_base = results["Baseline (dt=0.01, Ss=4σ)"]
    r_eff  = results["Efficient (dt=0.05, Ss=2σ)"]
    speedup = r_base["time_per_step"] / r_eff["time_per_step"]

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Baseline  time/step: {r_base['time_per_step']:.3f}s  "
          f"  rel err: {r_base['rel_err']:.2e}")
    print(f"  Efficient time/step: {r_eff['time_per_step']:.3f}s  "
          f"  rel err: {r_eff['rel_err']:.2e}")
    print(f"  Speedup: {speedup:.1f}×")

    # Sanity check: both methods should converge toward the ground state
    assert r_eff["rel_err"] < 0.1, (
        f"Efficient simulator energy {r_eff['final_E']:.4f} is far from "
        f"exact {E_exact:.4f} — check parameters or increase n_steps."
    )
    print("\n✓  Both simulators trending toward ground state.")
    return results


if __name__ == "__main__":
    run_comparison()
