"""
Phase-2 validation: at small L, prove exact-DM, MPS-MCWF, and PP-adjoint all
agree on the surrogate <H>(t) relaxation curve and the NESS plateau (-pz*L).

Run order:
  1. exact-DM reference (lindblad_exact.run_exact_lindblad, experiment B).
  2. MPS-MCWF (mps_lindblad.run_mcwf).
  3. PP (pp_lindblad.jl via subprocess) -- requires the `julia` command.

Writes phase2_validation.json (so plotting/report can use measured numbers even
if one engine is rerun later). Prints agreement metrics: max & mean |dH|/L over t.

Usage: python validate_phase2.py [L] [gamma] [T] [n_traj] [chi]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JULIA = "julia"
JPROJ = HERE
PXX, PZ, DT, TAU = -1.0, 1.5, 0.05, 0.1


def run_pp(L, gamma, T, tau=TAU, dt=DT, min_abs_coeff=1e-4, max_weight="Inf"):
    cfg = dict(L=L, gamma=gamma, T=T, tau=tau, dt=dt, pxx=PXX, pz=PZ,
               min_abs_coeff=min_abs_coeff, max_weight=max_weight)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f); cfg_path = f.name
    out_path = cfg_path.replace(".json", "_out.json")
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "JULIA_NUM_THREADS"):
        env[v] = "4"
    cmd = [JULIA, f"--project={JPROJ}", os.path.join(HERE, "pp_lindblad.jl"), cfg_path, out_path]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"PP failed:\n{proc.stdout}\n{proc.stderr}")
    with open(out_path) as fh:
        out = json.load(fh)
    os.remove(cfg_path); os.remove(out_path)
    return out


def main():
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    gamma = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    T = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
    n_traj = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    chi = int(sys.argv[5]) if len(sys.argv) > 5 else 32

    from lindblad_exact import run_exact_lindblad, ness_energy_direct
    from mps_lindblad import run_mcwf

    print(f"=== PHASE-2 VALIDATION  L={L} gamma={gamma} T={T} n_traj={n_traj} chi={chi} ===")

    ex = run_exact_lindblad(L, gamma, T, TAU, PXX, PZ, experiment="B")
    e_ness = ness_energy_direct(L, gamma, PXX, PZ, experiment="B")
    print(f"  exact-DM: <H>(0)={ex['energy'][0]:.5f}  <H>(T)={ex['energy'][-1]:.5f}")
    print(f"  exact NESS(kernel)={e_ness:.5f}  E_GS={ex['E_GS']:.5f}  (-pz*L={-PZ*L:.3f})")

    mc = run_mcwf(L, gamma, T, TAU, dt=DT, pxx=PXX, pz=PZ,
                  max_bond=chi, n_traj=n_traj, seed=1)
    ref = ex["energy"]
    d_mc = np.abs(mc["energy"] - ref) / L
    print(f"  MPS-MCWF: <H>(T)={mc['energy'][-1]:.5f}+/-{mc['stderr'][-1]:.4f}  "
          f"max|dH|/L={d_mc.max():.3e} mean={d_mc.mean():.3e}  wall={mc['wall_time']:.1f}s")

    pp_ok = True
    try:
        pp = run_pp(L, gamma, T, min_abs_coeff=1e-4)
        d_pp = np.abs(np.array(pp["energy"]) - ref) / L
        print(f"  PP-adjoint: <H>(T)={pp['energy'][-1]:.5f}  "
              f"max|dH|/L={d_pp.max():.3e} mean={d_pp.mean():.3e}  wall={pp['wall_time']:.3f}s")
    except Exception as e:
        pp_ok = False
        pp = None
        print(f"  PP-adjoint: SKIPPED/FAILED ({type(e).__name__}: {str(e)[:200]})")

    out = {
        "L": L, "gamma": gamma, "T": T, "tau": TAU, "n_traj": n_traj, "chi": chi,
        "t": ex["t"].tolist(), "exact": ref.tolist(), "E_GS": ex["E_GS"],
        "E_NESS_kernel": e_ness, "neg_pzL": -PZ * L,
        "mcwf": mc["energy"].tolist(), "mcwf_stderr": mc["stderr"].tolist(),
        "mcwf_wall": mc["wall_time"], "mcwf_maxdH_L": float(d_mc.max()),
        "pp": (pp["energy"] if pp_ok else None),
        "pp_wall": (pp["wall_time"] if pp_ok else None),
        "pp_maxdH_L": (float(np.abs(np.array(pp["energy"]) - ref).max() / L) if pp_ok else None),
    }
    with open(os.path.join(HERE, "phase2_validation.json"), "w") as f:
        json.dump(out, f)
    print("  wrote phase2_validation.json")


if __name__ == "__main__":
    main()
