"""
MPS quantum-jump (MCWF) unraveling of the site-local sigma^+ surrogate Lindbladian
(Experiment B).

Master equation:
    drho/dt = -i[H,rho] + sum_i gamma ( K_i rho K_i^dag - 1/2 {K_i^dag K_i, rho} ),
    K_i = sigma_i^+ = |1><0|_i      (damps each site toward |1>; pz>0 => MF GS).

MCWF unraveling (Dalibard-Castin-Molmer):
  * Non-Hermitian effective Hamiltonian
        H_eff = H - (i/2) gamma sum_i K_i^dag K_i = H - (i/2) gamma sum_i |0><0|_i .
    (K_i = sigma^+ = |1><0|  =>  K_i^dag K_i = |0><0|, the |0> projector.)
    Between jumps the (unnormalized) state evolves under exp(-i H_eff dt).
    We Strang-split exp(-i H_eff tau) over an OUTER step tau as
        [coherent Strang step]^{n_sub}  followed by  per-site no-jump damping gate
        exp(-(gamma/2) |1><1|_i tau)
    matching the PP Strang split  e^{(tau/2)L_coh} e^{tau L_diss} e^{(tau/2)L_coh}
    only at first order in the no-jump part; we therefore apply the damping as a
    half/half sandwich around the coherent block to keep the same 2nd-order Strang
    structure as PP (see _outer_step).
  * The norm-loss over the step gives the total jump probability
        dp = 1 - ||psi||^2 ;  with per-site weights  dp_i ~ gamma <psi|1><1|_i|psi> tau.
    If a uniform random draw < dp, a jump occurs on site i chosen with prob dp_i/dp,
    applying K_i = sigma_i^+ and renormalizing. Otherwise renormalize the no-jump state.

  * <H>(t) is averaged over N_traj trajectories; MC stderr ~ std/sqrt(N_traj).

The coherent Strang step reuses the phase-1 gate construction verbatim:
    Z half-layer exp(-i pz Z dt/2) -> XX even -> XX odd -> Z half-layer.

Convention / ordering identical to lindblad_exact.py and phase-1 modules.
Backend: numpy (complex128) for robust MPO energy expectation; bond knob `max_bond`.
"""

from __future__ import annotations

import time

import numpy as np
import quimb as qu
import quimb.tensor as qtn

# Robust-SVD patch (numpy) reused from phase-1 convention.
import autoray
import scipy.linalg


def _robust_svd_numpy(a, full_matrices=False, **kwargs):
    kwargs.pop("backend", None)
    kwargs["lapack_driver"] = "gesvd"
    try:
        return scipy.linalg.svd(a, full_matrices=full_matrices, **kwargs)
    except scipy.linalg.LinAlgError:
        kwargs["check_finite"] = False
        return scipy.linalg.svd(a, full_matrices=full_matrices, **kwargs)


autoray.register_function("numpy", "linalg.svd", _robust_svd_numpy)

_X = np.asarray(qu.pauli("X"))
_Z = np.asarray(qu.pauli("Z"))
_SP = np.array([[0, 0], [1, 0]], dtype=complex)   # sigma^+ = |1><0|  (|0>->|1>)
# K = sigma^+  =>  K^dag K = |0><0| = P0.  The no-jump generator and the per-site
# jump weight are therefore controlled by P0 (population in |0>), NOT |1><1|.
_P0 = np.array([[1, 0], [0, 0]], dtype=complex)    # |0><0|


def _coh_gates(pxx, pz, dt):
    Uz = np.asarray(qu.expm(-1j * (pz * qu.pauli("Z")) * dt / 2.0))
    Uxx = np.asarray(qu.expm(-1j * (pxx * (qu.pauli("X") & qu.pauli("X"))) * dt)).reshape(2, 2, 2, 2)
    return Uz, Uxx


def _build_ham_mpo(L, pxx, pz):
    builder = qtn.SpinHam1D(S=1 / 2)
    for i in range(L - 1):
        builder[i, i + 1] += pxx, qu.pauli("X"), qu.pauli("X")
    for i in range(L):
        builder[i] += pz, qu.pauli("Z")
    return builder.build_mpo(L)


def _coherent_strang(mps, Uz, Uxx, L, split_opts):
    """One 2nd-order Strang step of the unitary part (phase-1 layer)."""
    for i in range(L):
        mps.gate_(Uz, i, contract=True)
    for i in range(0, L - 1, 2):
        mps.gate_split_(Uxx, (i, i + 1), **split_opts)
    for i in range(1, L - 1, 2):
        mps.gate_split_(Uxx, (i, i + 1), **split_opts)
    for i in range(L):
        mps.gate_(Uz, i, contract=True)


def _apply_nojump_damping(mps, L, gamma, t):
    """Apply per-site no-jump factor exp(-(gamma/2) K^dag K t) with K=sigma^+.

    K^dag K = |0><0|, so the factor is diag(e^{-(gamma/2)t}, 1): it shrinks the
    |0> amplitude (no-jump bias toward having jumped to |1>). In-place.
    """
    d = np.exp(-0.5 * gamma * t)
    G = np.array([[d, 0.0], [0.0, 1.0]], dtype=complex)  # exp(-(g/2) P0 t)
    for i in range(L):
        mps.gate_(G, i, contract=True)


def _site_pop0(mps, i):
    """<psi| |0><0|_i |psi> / <psi|psi> for a (possibly unnormalized) MPS.

    This is the per-site jump weight (rate ~ gamma <K^dag K> = gamma <P0>)."""
    nrm = mps.H @ mps
    val = mps.H @ mps.gate(_P0, i)
    return float(np.real(complex(val) / complex(nrm)))


def run_mcwf(L, gamma, T, tau, dt=0.05, pxx=-1.0, pz=1.5,
             max_bond=64, n_traj=200, seed=0, cutoff=1e-12,
             measure_every=1):
    """Run N_traj MCWF trajectories; return averaged <H>(t) with MC stderr.

    Outer step tau, coherent sub-Trotterized with n_sub = round(tau/dt) steps.
    Strang structure per outer step:  half damping -> coherent(tau) -> half damping
    (matches PP's e^{(tau/2)diss?}). NOTE: we put the FULL coherent block in the
    middle and split the dissipative no-jump factor into two halves of tau/2, so
    the unraveling has the same 2nd-order Strang ordering as pp_lindblad.jl.

    Returns dict: 't', 'energy' (mean over traj), 'stderr', 'wall_time',
                  'max_bond_reached', 'n_traj'.
    """
    rng = np.random.default_rng(seed)
    N = int(round(T / tau))
    n_sub = max(1, int(round(tau / dt)))
    dt_eff = tau / n_sub

    Uz, Uxx = _coh_gates(pxx, pz, dt_eff)
    split_opts = dict(max_bond=max_bond, cutoff=cutoff, cutoff_mode="rel",
                      renorm=True, absorb="both")
    H_mpo = _build_ham_mpo(L, pxx, pz)

    ts = np.array([k * tau for k in range(N + 1)])
    # accumulate energy per time index across trajectories
    e_sum = np.zeros(N + 1)
    e_sq = np.zeros(N + 1)
    max_bond_reached = 1

    t0 = time.perf_counter()
    for traj in range(n_traj):
        psi = qtn.MPS_computational_state("0" * L)
        # energy at t=0
        e_sum[0] += pz * L  # |0..0> energy exactly pz*L
        e_sq[0] += (pz * L) ** 2

        for k in range(N):
            # --- 2nd-order Strang outer step on the no-jump (effective) dynamics ---
            # half dissipative no-jump (tau/2)
            _apply_nojump_damping(psi, L, gamma, tau / 2.0)
            # coherent block (full tau, sub-Trotterized)
            for _ in range(n_sub):
                _coherent_strang(psi, Uz, Uxx, L, split_opts)
            # half dissipative no-jump (tau/2)
            _apply_nojump_damping(psi, L, gamma, tau / 2.0)

            # --- jump decision from norm loss ---
            nrm2 = float(np.real(complex(psi.H @ psi)))
            dp = 1.0 - nrm2
            if dp < 0:
                dp = 0.0
            if rng.random() < dp:
                # choose site by per-site jump weight: w_i ~ gamma*tau*<P0_i>
                # (K=sigma^+ => K^dag K = |0><0|; jumps fire where |0> population is)
                weights = np.array([_site_pop0(psi, i) for i in range(L)])
                wsum = weights.sum()
                if wsum <= 0:
                    site = rng.integers(L)
                else:
                    site = rng.choice(L, p=weights / wsum)
                psi.gate_(_SP, site, contract=True)
            # renormalize (either no-jump branch or post-jump)
            psi.normalize()

            max_bond_reached = max(max_bond_reached, psi.max_bond())

            if (k + 1) % measure_every == 0 or (k + 1) == N:
                e = float(np.real(qtn.expec_TN_1D(psi.H, H_mpo, psi)))
                e_sum[k + 1] += e
                e_sq[k + 1] += e * e
    wall = time.perf_counter() - t0

    e_mean = e_sum / n_traj
    e_var = e_sq / n_traj - e_mean ** 2
    e_var = np.clip(e_var, 0, None)
    stderr = np.sqrt(e_var / n_traj)

    return {
        "t": ts,
        "energy": e_mean,
        "stderr": stderr,
        "wall_time": wall,
        "max_bond_reached": int(max_bond_reached),
        "n_traj": n_traj,
    }


def _sanity(L=6, gamma=0.1, T=40.0, tau=0.1, n_traj=200, chi=32):
    print(f"=== mps_lindblad MCWF sanity (L={L}, gamma={gamma}, n_traj={n_traj}) ===")
    res = run_mcwf(L, gamma, T, tau, max_bond=chi, n_traj=n_traj, seed=1)
    print(f"  <H>(0) = {res['energy'][0]:.5f} (expect {1.5*L:.1f})")
    print(f"  <H>(T) = {res['energy'][-1]:.5f} +/- {res['stderr'][-1]:.4f} "
          f"(target NESS -pz*L = {-1.5*L:.1f})")
    print(f"  wall={res['wall_time']:.2f}s  max_bond={res['max_bond_reached']}")


if __name__ == "__main__":
    _sanity()
