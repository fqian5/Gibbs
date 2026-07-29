"""
quimb MPS implementation of the 2nd-order Strang-Trotter TFIM evolution.

Same circuit as exact_reference.py:
    H = pxx * sum X_i X_{i+1} + pz * sum Z_i   (open boundary, Pauli +/-1)
    One Strang step U(dt):
        Z half-layer (exp(-i pz Z dt/2)) -> XX even bonds (exp(-i pxx XX dt))
        -> XX odd bonds -> Z half-layer.

Uses quimb's `gate_split_` (the same idiom as efficient/simulator.py) which
applies a 2-site gate and SVD-truncates the bond to `max_bond` in one call.
Single-site Z gates are exact (no truncation).

Backend: torch (complex128) to reuse the robust-SVD patch from the repo, with a
numpy fallback path. Qubit ordering matches quimb's native site_ind (site 0 =
leftmost), consistent with exact_reference.py.

Returns per-time <Z_{L/2}>(t), <sum Z>(t), wall-clock, and half-chain
entanglement entropy S(t).

Angles in radians; times t = n*dt.
"""

from __future__ import annotations

import time

import numpy as np
import quimb as qu
import quimb.tensor as qtn

# ---------------------------------------------------------------------------
# Robust SVD patch (verbatim convention from efficient/simulator.py).
# Guards against LAPACK gesdd convergence failures on near-degenerate spectra.
# ---------------------------------------------------------------------------
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


try:
    import torch

    _HAVE_TORCH = True

    def _robust_svd_torch(a, full_matrices=False, **kwargs):
        try:
            return torch.linalg.svd(a, full_matrices=full_matrices)
        except RuntimeError:
            a_cpu = a.detach().cpu().numpy()
            U_c, S_c, Vh_c = _robust_svd_numpy(a_cpu, full_matrices=full_matrices)
            U = torch.from_numpy(U_c).to(device=a.device, dtype=a.dtype)
            S = torch.from_numpy(S_c).to(device=a.device)
            Vh = torch.from_numpy(Vh_c).to(device=a.device, dtype=a.dtype)
            return U, S, Vh

    autoray.register_function("torch", "linalg.svd", _robust_svd_torch)
except ImportError:  # pragma: no cover
    _HAVE_TORCH = False

autoray.register_function("numpy", "linalg.svd", _robust_svd_numpy)

# Pauli matrices (eigenvalues +/-1)
_X = qu.pauli("X")
_Z = qu.pauli("Z")
_I = qu.pauli("I")


def _gates(pxx: float, pz: float, dt: float):
    """Build the three gate matrices for one Strang step.

    Returns (Uz, Uxx) as numpy complex arrays. Uz is 2x2, Uxx is 4x4
    (reshaped to (2,2,2,2) by quimb internally; we pass 4x4 -> reshape here).
    """
    Uz = qu.expm(-1j * (pz * _Z) * dt / 2.0)
    Uxx = qu.expm(-1j * (pxx * (_X & _X)) * dt)
    return np.asarray(Uz), np.asarray(Uxx).reshape(2, 2, 2, 2)


def _to_backend(mps, backend, dtype):
    if backend == "torch" and _HAVE_TORCH:
        dt_map = {"complex128": torch.complex128, "complex64": torch.complex64}
        tdt = dt_map[dtype]
        mps.apply_to_arrays(
            lambda x: torch.tensor(np.array(x), dtype=tdt)
            if not isinstance(x, torch.Tensor)
            else x.to(tdt)
        )
    # numpy backend: leave as-is (complex128)


def _z_expect(mps, site: int) -> float:
    """<Z_site>. Uses a fresh copy so the simulation MPS is untouched."""
    val = mps.H @ mps.gate(np.asarray(_Z), site)
    return float(np.real(complex(val)))


def run_mps_trotter(
    L: int,
    dt: float,
    T: float,
    pxx: float = -1.0,
    pz: float = 1.5,
    max_bond: int = 64,
    backend: str = "torch",
    dtype: str = "complex128",
    cutoff: float = 1e-12,
    measure_sumz: bool = True,
    measure_entropy: bool = True,
):
    """Run the Strang-Trotter evolution with an MPS of max bond `max_bond`.

    Returns dict:
        't', 'z_mid', 'sumz', 'entropy', 'wall_time', 'max_bond_reached'
    """
    N = int(round(T / dt))
    mid = L // 2

    split_opts = dict(
        max_bond=max_bond,
        cutoff=cutoff,
        cutoff_mode="rel",
        renorm=True,
        absorb="both",
    )

    Uz, Uxx = _gates(pxx, pz, dt)
    if backend == "torch" and _HAVE_TORCH:
        tdt = torch.complex128 if dtype == "complex128" else torch.complex64
        Uz_b = torch.tensor(Uz, dtype=tdt)
        Uxx_b = torch.tensor(Uxx, dtype=tdt)
        Z_b = torch.tensor(np.asarray(_Z), dtype=tdt)
    else:
        Uz_b, Uxx_b, Z_b = Uz, Uxx, np.asarray(_Z)

    mps = qtn.MPS_computational_state("0" * L)
    _to_backend(mps, backend, dtype)

    def z_mid_val():
        return float(np.real(complex(mps.H @ mps.gate(Z_b, mid))))

    def sumz_val():
        if not measure_sumz:
            return np.nan
        s = 0.0
        for i in range(L):
            s += float(np.real(complex(mps.H @ mps.gate(Z_b, i))))
        return s

    def entropy_val():
        if not measure_entropy:
            return np.nan
        try:
            return float(mps.entropy(mid))
        except Exception:
            return np.nan

    ts = [0.0]
    z_mid = [z_mid_val()]
    sumz = [sumz_val()]
    ent = [entropy_val()]

    max_bond_reached = mps.max_bond()
    t0 = time.perf_counter()
    for n in range(N):
        # 1. Z half-layer (single-site, exact)
        for i in range(L):
            mps.gate_(Uz_b, i, contract=True)
        # 2. XX even bonds
        for i in range(0, L - 1, 2):
            mps.gate_split_(Uxx_b, (i, i + 1), **split_opts)
        # 3. XX odd bonds
        for i in range(1, L - 1, 2):
            mps.gate_split_(Uxx_b, (i, i + 1), **split_opts)
        # 4. Z half-layer
        for i in range(L):
            mps.gate_(Uz_b, i, contract=True)

        mps.normalize()
        max_bond_reached = max(max_bond_reached, mps.max_bond())

        ts.append((n + 1) * dt)
        z_mid.append(z_mid_val())
        sumz.append(sumz_val())
        ent.append(entropy_val())
    wall = time.perf_counter() - t0

    return {
        "t": np.array(ts),
        "z_mid": np.array(z_mid),
        "sumz": np.array(sumz),
        "entropy": np.array(ent),
        "wall_time": wall,
        "max_bond_reached": int(max_bond_reached),
    }


def _sanity(L=8, dt=0.05, T=4.0, pxx=-1.0, pz=1.5, chi=64):
    print(f"=== mps_trotter sanity (L={L}, chi={chi}) ===")
    # warmup (excluded from any timing the caller does)
    run_mps_trotter(L, dt, dt, pxx, pz, max_bond=chi)
    res = run_mps_trotter(L, dt, T, pxx, pz, max_bond=chi)
    print(f"  <Z_mid>(0)={res['z_mid'][0]:.6f} (expect 1.0)")
    print(f"  <sumZ>(0)={res['sumz'][0]:.6f} (expect {L})")
    print(f"  <Z_mid>(T)={res['z_mid'][-1]:.6f}")
    print(f"  max half-chain S(t)={np.nanmax(res['entropy']):.4f}")
    print(f"  wall={res['wall_time']:.3f}s  max_bond_reached={res['max_bond_reached']}")


if __name__ == "__main__":
    _sanity()
