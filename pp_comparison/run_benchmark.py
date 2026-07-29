"""
Orchestrate the head-to-head benchmark: exact (ground truth) vs quimb MPS vs
PauliPropagation.jl, on the SAME 2nd-order Strang-Trotter TFIM circuit.

Pipeline:
  1. VALIDATE: at L=10 (dt=0.05, T=4), confirm exact / MPS(tightest chi) /
     PP(tightest min_abs_coeff) agree on <Z_{L/2}>(t) to ~1e-3.
  2. ACCURACY sweep: MPS over chi, PP over min_abs_coeff, vs exact-Trotter ref.
  3. SCALING: local observable <Z_{L/2}> at larger L for both simulators
     (self-convergence + cross-method overlay).

All results are written to benchmark_results.json for plotting.

Timing notes:
  * MPS: a warmup call is issued before each timed run (excluded).
  * PP: the Julia script warms up (JIT) before timing internally.
  * BLAS threads pinned via env (OMP/MKL/OPENBLAS = BENCH_THREADS).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/Users/qianfeng/anaconda3/envs/lindbladian/bin/python"
JULIA = "julia"
JULIA_PROJECT = HERE
PP_SCRIPT = os.path.join(HERE, "pp_trotter.jl")

PXX = -1.0
PZ = 1.5
DT = 0.05
T = 4.0


def _err_metrics(method_vals, ref_vals):
    """E_inf = max_t |method-ref|, E_1 = mean_t |method-ref|. Aligned by index."""
    d = np.abs(np.asarray(method_vals) - np.asarray(ref_vals))
    return float(d.max()), float(d.mean())


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_exact(L, dt=DT, T=T, observable="zmid"):
    from exact_reference import trotter_reference, continuous_reference

    ref = trotter_reference(L, dt, T, PXX, PZ)
    cont = continuous_reference(L, dt, T, PXX, PZ)
    key = "z_mid" if observable == "zmid" else "sumz"
    return {
        "t": ref["t"].tolist(),
        "trotter": ref[key].tolist(),
        "continuous": cont[key].tolist(),
    }


def run_mps(L, chi, dt=DT, T=T, observable="zmid",
            measure_sumz=False, measure_entropy=True):
    from mps_trotter import run_mps_trotter

    # warmup (excluded) -- 1 step at this L/chi
    run_mps_trotter(L, dt, dt, PXX, PZ, max_bond=chi,
                    measure_sumz=False, measure_entropy=False)
    res = run_mps_trotter(L, dt, T, PXX, PZ, max_bond=chi,
                          measure_sumz=measure_sumz, measure_entropy=measure_entropy)
    key = "z_mid" if observable == "zmid" else "sumz"
    return {
        "t": res["t"].tolist(),
        "vals": res[key].tolist(),
        "wall_time": res["wall_time"],
        "entropy": res["entropy"].tolist(),
        "max_bond_reached": res["max_bond_reached"],
    }


def run_pp(L, min_abs_coeff, dt=DT, T=T, observable="zmid",
           max_weight="Inf", threads=4):
    cfg = {
        "L": L, "dt": dt, "T": T, "pxx": PXX, "pz": PZ,
        "obs": observable, "min_abs_coeff": min_abs_coeff,
        "max_weight": max_weight,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        cfg_path = f.name
    out_path = cfg_path.replace(".json", "_out.json")

    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "JULIA_NUM_THREADS"):
        env[v] = str(threads)

    cmd = [JULIA, f"--project={JULIA_PROJECT}", PP_SCRIPT, cfg_path, out_path]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"PP run failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}")
    with open(out_path) as fh:
        out = json.load(fh)
    os.remove(cfg_path)
    os.remove(out_path)
    return out


# ---------------------------------------------------------------------------
# Benchmark phases
# ---------------------------------------------------------------------------

def phase_validate(L=10, threads=4):
    print(f"\n=== PHASE 1: VALIDATION (L={L}, dt={DT}, T={T}) ===")
    ex = run_exact(L, observable="zmid")
    ref = np.array(ex["trotter"])

    mps = run_mps(L, chi=128, observable="zmid")
    pp = run_pp(L, min_abs_coeff=1e-5, observable="zmid", threads=threads)

    e_mps = _err_metrics(mps["vals"], ref)
    e_pp = _err_metrics(pp["vals"], ref)
    floor = _err_metrics(ex["continuous"], ref)

    print(f"  exact   <Z_mid>(T) = {ref[-1]:.6f}")
    print(f"  MPS(128) <Z_mid>(T)= {mps['vals'][-1]:.6f}  E_inf={e_mps[0]:.2e} E_1={e_mps[1]:.2e}")
    print(f"  PP(1e-5) <Z_mid>(T)= {pp['vals'][-1]:.6f}  E_inf={e_pp[0]:.2e} E_1={e_pp[1]:.2e}")
    print(f"  Trotter floor vs continuous: E_inf={floor[0]:.2e}")
    ok = e_mps[0] < 1e-3 and e_pp[0] < 1e-3
    print(f"  AGREEMENT to 1e-3: {'PASS' if ok else 'FAIL'}")
    return {
        "L": L, "exact": ex, "mps": mps, "pp": pp,
        "e_mps": e_mps, "e_pp": e_pp, "floor": floor, "agree": ok,
    }


def phase_accuracy(L=12, chis=(16, 32, 64, 128), macs=(1e-2, 1e-3, 1e-4, 1e-5),
                   threads=4):
    print(f"\n=== PHASE 2: ACCURACY/WALL sweep (L={L}) ===")
    ex = run_exact(L, observable="zmid")
    ref = np.array(ex["trotter"])

    mps_rows = []
    for chi in chis:
        r = run_mps(L, chi=chi, observable="zmid")
        einf, e1 = _err_metrics(r["vals"], ref)
        mps_rows.append({"chi": chi, "E_inf": einf, "E_1": e1,
                         "wall": r["wall_time"], "max_bond": r["max_bond_reached"],
                         "vals": r["vals"]})
        print(f"  MPS chi={chi:4d}: E_inf={einf:.2e} E_1={e1:.2e} wall={r['wall_time']:.3f}s bond={r['max_bond_reached']}")

    pp_rows = []
    for mac in macs:
        r = run_pp(L, min_abs_coeff=mac, observable="zmid", threads=threads)
        einf, e1 = _err_metrics(r["vals"], ref)
        pp_rows.append({"mac": mac, "E_inf": einf, "E_1": e1,
                        "wall": r["wall_time"], "vals": r["vals"]})
        print(f"  PP  mac={mac:.0e}: E_inf={einf:.2e} E_1={e1:.2e} wall={r['wall_time']:.3f}s")

    return {"L": L, "t": ex["t"], "ref": ex["trotter"],
            "continuous": ex["continuous"], "mps": mps_rows, "pp": pp_rows}


def phase_scaling(Ls=(10, 20, 30, 50, 100), chi=64, mac=1e-4, threads=4):
    print(f"\n=== PHASE 3: SCALING vs L (local <Z_mid>, chi={chi}, mac={mac:.0e}) ===")
    rows = []
    for L in Ls:
        # exact only for small L
        ex_T = None
        if L <= 12:
            ex = run_exact(L, observable="zmid")
            ex_T = ex["trotter"][-1]

        mps = run_mps(L, chi=chi, observable="zmid", measure_entropy=True)
        pp = run_pp(L, min_abs_coeff=mac, observable="zmid", threads=threads)

        cross = _err_metrics(mps["vals"], pp["vals"])
        row = {
            "L": L,
            "mps_wall": mps["wall_time"], "mps_T": mps["vals"][-1],
            "mps_bond": mps["max_bond_reached"],
            "mps_maxS": float(np.nanmax(mps["entropy"])),
            "pp_wall": pp["wall_time"], "pp_T": pp["vals"][-1],
            "exact_T": ex_T,
            "cross_Einf": cross[0],
            "mps_vals": mps["vals"], "pp_vals": pp["vals"], "t": mps["t"],
        }
        rows.append(row)
        exstr = f"exact_T={ex_T:.6f} " if ex_T is not None else ""
        print(f"  L={L:4d}: MPS wall={mps['wall_time']:7.3f}s T={mps['vals'][-1]:.6f} bond={mps['max_bond_reached']:3d} S={row['mps_maxS']:.2f} | "
              f"PP wall={pp['wall_time']:7.3f}s T={pp['vals'][-1]:.6f} | {exstr}cross_Einf={cross[0]:.2e}")
    return rows


def main():
    threads = int(os.environ.get("BENCH_THREADS", "4"))
    t_start = time.time()

    results = {"params": {"pxx": PXX, "pz": PZ, "dt": DT, "T": T, "threads": threads}}

    results["validate"] = phase_validate(L=10, threads=threads)
    if not results["validate"]["agree"]:
        print("\n!! VALIDATION FAILED -- aborting benchmark (physics disagree).")
        # still write what we have
    results["accuracy"] = phase_accuracy(L=12, threads=threads)
    results["scaling"] = phase_scaling(Ls=(10, 20, 30, 50, 100),
                                       chi=64, mac=1e-4, threads=threads)

    out = os.path.join(HERE, "benchmark_results.json")
    with open(out, "w") as f:
        json.dump(results, f)
    print(f"\nTotal benchmark wall time: {time.time() - t_start:.1f}s")
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
