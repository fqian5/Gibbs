"""
Plot benchmark_results.json:
  (a) <Z_{L/2}>(t) overlay at L=10 (MPS + PP + exact-Trotter + continuous)
  (b) error-vs-time curves at L=10
  (c) wall-clock vs L at fixed accuracy (local observable)
  (d) wall-clock vs accuracy Pareto at L=12
Plus a self-convergence panel at L=30.

Saves PNGs to the working directory.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "benchmark_results.json")) as f:
    R = json.load(f)


def save(fig, name):
    p = os.path.join(HERE, name)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    print("wrote", p)
    plt.close(fig)


# (a) + (b) overlay & error at L=10 -----------------------------------------
v = R["validate"]
t = np.array(v["exact"]["t"])
exact = np.array(v["exact"]["trotter"])
cont = np.array(v["exact"]["continuous"])
mps = np.array(v["mps"]["vals"])
pp = np.array(v["pp"]["vals"])

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(t, exact, "k-", lw=2.4, label="exact Trotter (ground truth)")
ax.plot(t, cont, color="gray", ls=":", lw=1.6, label="continuous $e^{-iHt}$ (Trotter floor)")
ax.plot(t, mps, "o", ms=4, color="tab:blue", label="MPS $\\chi=128$", markevery=4)
ax.plot(t, pp, "s", ms=4, color="tab:red", label="PP min_abs=1e-5", markevery=4)
ax.set_xlabel("time $t$"); ax.set_ylabel(r"$\langle Z_{L/2}\rangle(t)$")
ax.set_title(f"(a) Local observable overlay, L={v['L']}")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
save(fig, "plot_a_overlay_L10.png")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogy(t, np.abs(mps - exact) + 1e-16, "-", color="tab:blue", label="MPS $\\chi=128$")
ax.semilogy(t, np.abs(pp - exact) + 1e-16, "-", color="tab:red", label="PP 1e-5")
ax.semilogy(t, np.abs(cont - exact) + 1e-16, ":", color="gray", label="Trotter floor")
ax.axhline(1e-3, color="k", ls="--", lw=0.8, label="1e-3 target")
ax.set_xlabel("time $t$"); ax.set_ylabel(r"$|\langle Z\rangle_{\rm method}-\langle Z\rangle_{\rm exact}|$")
ax.set_title(f"(b) Error vs time, L={v['L']}")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
save(fig, "plot_b_error_vs_time_L10.png")

# (c) wall vs L --------------------------------------------------------------
sc = R["scaling"]
Ls = [row["L"] for row in sc]
mps_wall = [row["mps_wall"] for row in sc]
pp_wall = [row["pp_wall"] for row in sc]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(Ls, mps_wall, "o-", color="tab:blue", label="MPS $\\chi=64$")
ax.plot(Ls, pp_wall, "s-", color="tab:red", label="PP min_abs=1e-4")
ax.set_xlabel("system size $L$"); ax.set_ylabel("wall-clock (s)")
ax.set_title(r"(c) Wall-clock vs $L$, local $\langle Z_{L/2}\rangle$, T=4")
ax.legend(); ax.grid(alpha=0.3)
save(fig, "plot_c_wall_vs_L.png")

# (d) Pareto: wall vs accuracy at L=12 --------------------------------------
ac = R["accuracy"]
fig, ax = plt.subplots(figsize=(7, 4.5))
mps_e = [r["E_inf"] for r in ac["mps"]]
mps_w = [r["wall"] for r in ac["mps"]]
pp_e = [r["E_inf"] for r in ac["pp"]]
pp_w = [r["wall"] for r in ac["pp"]]
ax.loglog(mps_e, mps_w, "o-", color="tab:blue", label="MPS (sweep $\\chi$)")
for r in ac["mps"]:
    ax.annotate(f"$\\chi$={r['chi']}", (r["E_inf"], r["wall"]), fontsize=7,
                textcoords="offset points", xytext=(4, 4))
ax.loglog(pp_e, pp_w, "s-", color="tab:red", label="PP (sweep min_abs)")
for r in ac["pp"]:
    ax.annotate(f"{r['mac']:.0e}", (r["E_inf"], r["wall"]), fontsize=7,
                textcoords="offset points", xytext=(4, -8))
ax.axvline(1e-3, color="k", ls="--", lw=0.8, label="1e-3 target")
ax.set_xlabel(r"accuracy $E_\infty$ (lower better)"); ax.set_ylabel("wall-clock (s)")
ax.set_title(f"(d) Wall-vs-accuracy Pareto, L={ac['L']}")
ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
save(fig, "plot_d_pareto_L12.png")

# (e) self-convergence at L=30 ----------------------------------------------
if "selfconv_L30" in R:
    sc30 = R["selfconv_L30"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    mw = [r["wall"] for r in sc30["mps"]]; mt = [r["T"] for r in sc30["mps"]]
    pw = [r["wall"] for r in sc30["pp"]]; pt = [r["T"] for r in sc30["pp"]]
    ax.plot(mw, mt, "o-", color="tab:blue", label="MPS ($\\chi$=64,128,192)")
    ax.plot(pw, pt, "s-", color="tab:red", label="PP (mac=1e-4,1e-5,1e-6)")
    for r in sc30["mps"]:
        ax.annotate(f"$\\chi$={r['chi']}", (r["wall"], r["T"]), fontsize=7,
                    textcoords="offset points", xytext=(4, 4))
    for r in sc30["pp"]:
        ax.annotate(f"{r['mac']:.0e}", (r["wall"], r["T"]), fontsize=7,
                    textcoords="offset points", xytext=(4, -10))
    ax.set_xlabel("wall-clock (s)")
    ax.set_ylabel(r"$\langle Z_{L/2}\rangle(T)$ (converging value)")
    ax.set_title("(e) Self-convergence at L=30 (both -> ~0.775)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    save(fig, "plot_e_selfconv_L30.png")

print("All plots written.")
