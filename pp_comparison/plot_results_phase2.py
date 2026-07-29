"""
Phase-2 plots (dissipative cooling). Produces:
  (a) <H>(t) relaxation overlay at L=6/8: exact-DM + MPS-MCWF (+/- stderr) (+ PP if present).
  (b) MCWF error-vs-time at small L (|<H>_mcwf - <H>_exact|/L vs MC stderr band).
  (c) wall-clock vs L (MPS-MCWF; + PP if present).
  (d) steady-state energy vs L (H(T)) with the analytic NESS line -pz*L and E_GS.
Saves PNGs to the working directory. Loads phase2_val_L*.json and
benchmark_phase2_results.json (whichever exist).
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PZ = 1.5


def save(fig, name):
    p = os.path.join(HERE, name)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    print("wrote", p)
    plt.close(fig)


# ---- (a)+(b) validation overlays from phase2_val_L*.json ----
val_files = sorted(glob.glob(os.path.join(HERE, "phase2_val_L*.json")))
for vf in val_files:
    with open(vf) as f:
        V = json.load(f)
    L = V["L"]
    t = np.array(V["t"]); ex = np.array(V["exact"])
    mc = np.array(V["mcwf"]); se = np.array(V["mcwf_stderr"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, ex, "k-", lw=2.2, label="exact density-matrix Lindblad")
    ax.plot(t, mc, "o", ms=3, color="tab:blue", markevery=10,
            label=f"MPS-MCWF (N_traj={V['n_traj']}, $\\chi$={V['chi']})")
    ax.fill_between(t, mc - se, mc + se, color="tab:blue", alpha=0.25, label="MC stderr")
    if V.get("pp") is not None:
        ax.plot(t, np.array(V["pp"]), "s", ms=3, color="tab:red", markevery=10,
                label="PP adjoint")
    ax.axhline(V["neg_pzL"], color="green", ls="--", lw=1, label=f"NESS $=-p_zL={V['neg_pzL']:.0f}$")
    ax.axhline(V["E_GS"], color="purple", ls=":", lw=1, label=f"true $E_{{GS}}={V['E_GS']:.2f}$")
    ax.set_xlabel("time $t$"); ax.set_ylabel(r"$\langle H\rangle(t)$")
    ax.set_title(f"(a) Dissipative cooling, L={L}, $\\gamma$={V['gamma']}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    save(fig, f"plot2_a_relax_L{L}.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(t, np.abs(mc - ex) / L + 1e-12, "-", color="tab:blue",
                label="MPS-MCWF |dH|/L")
    ax.semilogy(t, se / L + 1e-12, ":", color="gray", label="MC stderr / L")
    ax.set_xlabel("time $t$"); ax.set_ylabel(r"$|\langle H\rangle_{\rm method}-\langle H\rangle_{\rm exact}|/L$")
    ax.set_title(f"(b) MCWF error vs time, L={L} (MC-noise dominated)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    save(fig, f"plot2_b_error_L{L}.png")


# ---- (c)+(d) scaling from benchmark_phase2_results.json ----
bp = os.path.join(HERE, "benchmark_phase2_results.json")
if os.path.exists(bp):
    with open(bp) as f:
        R = json.load(f)
    mps = R.get("mps_scaling") or []
    pp = R.get("pp_scaling") or []

    if mps:
        Ls = [r["L"] for r in mps]; w = [r["wall"] for r in mps]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(Ls, w, "o-", color="tab:blue", label=f"MPS-MCWF (N_traj={mps[0]['n_traj']}, $\\chi$={mps[0]['chi']})")
        if pp:
            Lp = [r["L"] for r in pp]; wp = [r["wall"] for r in pp]
            ax.plot(Lp, wp, "s-", color="tab:red", label=f"PP (mac={pp[0]['mac']:.0e})")
        ax.set_xlabel("system size $L$"); ax.set_ylabel("wall-clock (s)")
        ax.set_title("(c) Dissipative wall-clock vs L")
        ax.legend(); ax.grid(alpha=0.3)
        save(fig, "plot2_c_wall_vs_L.png")

        fig, ax = plt.subplots(figsize=(7, 4.5))
        Ls = np.array([r["L"] for r in mps])
        hT = np.array([r["H_T"] for r in mps])
        seT = np.array([r["stderr_T"] for r in mps])
        ax.errorbar(Ls, hT / Ls, yerr=seT / Ls, fmt="o-", color="tab:blue",
                    label=r"MPS-MCWF $\langle H\rangle(T)/L$")
        if pp:
            Lp = np.array([r["L"] for r in pp]); hp = np.array([r["H_T"] for r in pp])
            ax.plot(Lp, hp / Lp, "s-", color="tab:red", label=r"PP $\langle H\rangle(T)/L$")
        ax.axhline(-PZ, color="green", ls="--", label=r"NESS/L $=-p_z=-1.5$")
        ax.set_xlabel("system size $L$"); ax.set_ylabel(r"$\langle H\rangle(T)/L$")
        ax.set_title("(d) Steady-state energy density vs L (T=30, not fully plateaued)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        save(fig, "plot2_d_ness_vs_L.png")

print("phase-2 plots done.")
