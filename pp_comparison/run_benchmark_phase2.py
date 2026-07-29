"""
Phase-2 orchestration: dissipative cooling benchmark (surrogate sigma^+ Lindbladian).

Phases:
  1. VALIDATE (L=6,8): exact-DM vs MPS-MCWF (and PP if julia available) agree on
     <H>(t) and the NESS plateau (-pz*L).  [validate_phase2.py does the 3-way]
  2. MPS scaling (Exp B): wall-clock & NESS vs L for the MCWF, chi sweep.
  3. PP scaling (Exp B): emits ready-to-run configs and, if julia is available,
     runs them (wall-clock & NESS vs L, min_abs_coeff sweep).

All MPS results are computed here (python). PP results are loaded from JSON files
produced by pp_lindblad.jl (run separately if the julia CLI is unavailable in this
environment). Writes benchmark_phase2_results.json.

Time params (locked): tau=0.1, dt=0.05, gamma=0.1, pxx=-1.0, pz=1.5.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JULIA = "julia"
JPROJ = HERE
PXX, PZ, DT, TAU, GAMMA = -1.0, 1.5, 0.05, 0.1, 0.1


def julia_available():
    try:
        r = subprocess.run([JULIA, "--version"], capture_output=True, text=True)
        return r.returncode == 0
    except Exception:
        return False


def run_pp(L, gamma, T, min_abs_coeff=1e-4, max_weight="Inf", tau=TAU, dt=DT):
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


def write_pp_configs(Ls, gamma, T, macs, outdir):
    """Emit JSON configs so pp_lindblad.jl can be batch-run even without python."""
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for L in Ls:
        for mac in macs:
            cfg = dict(L=L, gamma=gamma, T=T, tau=TAU, dt=DT, pxx=PXX, pz=PZ,
                       min_abs_coeff=mac, max_weight="Inf")
            p = os.path.join(outdir, f"pp_cfg_L{L}_mac{mac:.0e}.json")
            with open(p, "w") as f:
                json.dump(cfg, f)
            paths.append(p)
    return paths


def phase_mps_scaling(Ls, gamma=GAMMA, T=30.0, chi=32, n_traj=100):
    from mps_lindblad import run_mcwf
    print(f"\n=== MPS-MCWF scaling (gamma={gamma}, T={T}, chi={chi}, n_traj={n_traj}) ===")
    rows = []
    for L in Ls:
        mc = run_mcwf(L, gamma, T, TAU, dt=DT, pxx=PXX, pz=PZ,
                      max_bond=chi, n_traj=n_traj, seed=1)
        row = dict(L=L, chi=chi, n_traj=n_traj, wall=mc["wall_time"],
                   H_T=float(mc["energy"][-1]), stderr_T=float(mc["stderr"][-1]),
                   maxbond=mc["max_bond_reached"], neg_pzL=-PZ * L,
                   t=mc["t"].tolist(), energy=mc["energy"].tolist(),
                   energy_stderr=mc["stderr"].tolist())
        rows.append(row)
        print(f"  L={L:4d}: wall={mc['wall_time']:7.1f}s  H(T)={row['H_T']:.4f}+/-{row['stderr_T']:.4f}  "
              f"(NESS -pz*L={-PZ*L:.1f})  bond={row['maxbond']}")
    return rows


def phase_pp_scaling(Ls, gamma=GAMMA, T=30.0, mac=1e-4):
    if not julia_available():
        print("\n=== PP scaling SKIPPED (julia CLI unavailable in this environment) ===")
        return None
    print(f"\n=== PP-adjoint scaling (gamma={gamma}, T={T}, mac={mac:.0e}) ===")
    rows = []
    for L in Ls:
        pp = run_pp(L, gamma, T, min_abs_coeff=mac)
        row = dict(L=L, mac=mac, wall=pp["wall_time"], H_T=float(pp["energy"][-1]),
                   neg_pzL=-PZ * L, t=pp["t"], energy=pp["energy"])
        rows.append(row)
        print(f"  L={L:4d}: wall={pp['wall_time']:7.3f}s  H(T)={row['H_T']:.4f}  (NESS -pz*L={-PZ*L:.1f})")
    return rows


def main():
    results = {"params": dict(pxx=PXX, pz=PZ, dt=DT, tau=TAU, gamma=GAMMA)}
    t0 = time.time()

    # MPS scaling (capped for time): modest traj count + shorter T to keep the
    # MCWF tractable. T=20 still shows clear relaxation; the wall-vs-L trend and
    # the trajectory tax are the point, not a fully-plateaued NESS here.
    results["mps_scaling"] = phase_mps_scaling(Ls=(10, 16, 20), chi=32, n_traj=40, T=20.0)

    # PP scaling (runs only if julia available); always emit configs for manual run.
    cfgdir = os.path.join(HERE, "pp_configs_phase2")
    paths = write_pp_configs(Ls=(10, 16, 24, 50, 100), gamma=GAMMA, T=30.0,
                             macs=(1e-3, 1e-4, 1e-5), outdir=cfgdir)
    print(f"\nWrote {len(paths)} PP configs to {cfgdir} (run via pp_lindblad.jl).")
    results["pp_scaling"] = phase_pp_scaling(Ls=(10, 16, 24, 50, 100), mac=1e-4, T=20.0)

    with open(os.path.join(HERE, "benchmark_phase2_results.json"), "w") as f:
        json.dump(results, f)
    print(f"\nTotal phase-2 wall: {time.time()-t0:.1f}s -> benchmark_phase2_results.json")


if __name__ == "__main__":
    main()
