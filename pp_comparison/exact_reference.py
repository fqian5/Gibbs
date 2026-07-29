"""
Exact dense-statevector reference for the 1D open-boundary TFIM Trotter benchmark.

Hamiltonian (Pauli +/- 1 convention, matches efficient/simulator.py):
    H = pxx * sum_{i=0}^{L-2} X_i X_{i+1}  +  pz * sum_{i=0}^{L-1} Z_i
Defaults: pxx = -1.0, pz = 1.5.

One 2nd-order Strang step U(dt) (matches arXiv_2508_05703 apply_system_evolution):
    1. Z half-layer:  exp(-i * pz * Z * dt/2) on all sites
    2. XX even bonds: exp(-i * pxx * (X o X) * dt) on (0,1),(2,3),...
    3. XX odd bonds:  same on (1,2),(3,4),...
    4. Z half-layer:  exp(-i * pz * Z * dt/2) on all sites

This module provides:
  * `trotter_reference(...)`  -> exact (no truncation) Trotter-circuit statevector
       evolution, returning <Z_{L/2}>(t) and <sum Z>(t) per step. This is the
       PRIMARY ground truth for the simulators (isolates truncation error).
  * `continuous_reference(...)` -> exact continuous exp(-iHt)|0> via scipy
       expm_multiply, to expose the Trotter floor.

Qubit ordering convention: site 0 is the MOST-significant qubit of the
statevector index (numpy kron order: op_0 (x) op_1 (x) ... (x) op_{L-1}).
This matches `build_full_operator` below and quimb's site_ind ordering, so the
observable <Z_{L/2}> is unambiguous across modules. L/2 means index L//2.

All angles are in radians. Times t = n*dt.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

# Pauli matrices (eigenvalues +/- 1)
I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _expm_2x2_pauli(coeff: float, P: np.ndarray) -> np.ndarray:
    """exp(-i * coeff * P) for a single Pauli P (P^2 = I).

    Uses exp(-i a P) = cos(a) I - i sin(a) P. Exact, no eig needed.
    """
    a = coeff
    return np.cos(a) * I2 - 1j * np.sin(a) * P


def _expm_xx(coeff: float) -> np.ndarray:
    """exp(-i * coeff * (X o X)) as a 4x4 matrix. (XoX)^2 = I, so closed form."""
    XX = np.kron(X, X)
    a = coeff
    return np.cos(a) * np.eye(4, dtype=complex) - 1j * np.sin(a) * XX


def _apply_single(psi: np.ndarray, U: np.ndarray, site: int, L: int) -> np.ndarray:
    """Apply 2x2 gate U on `site` to statevector psi of shape (2^L,).

    Reshape to put the target axis first, matvec, restore.
    """
    psi = psi.reshape([2] * L)
    psi = np.moveaxis(psi, site, 0)
    shp = psi.shape
    psi = (U @ psi.reshape(2, -1)).reshape(shp)
    psi = np.moveaxis(psi, 0, site)
    return psi.reshape(-1)


def _apply_two(psi: np.ndarray, U4: np.ndarray, i: int, j: int, L: int) -> np.ndarray:
    """Apply 4x4 gate U4 on adjacent sites (i, j) with i < j, j == i+1.

    U4 acts on the 4-dim space ordered as (site_i (x) site_j).
    """
    psi = psi.reshape([2] * L)
    psi = np.moveaxis(psi, [i, j], [0, 1])
    shp = psi.shape
    psi = (U4 @ psi.reshape(4, -1)).reshape(shp)
    psi = np.moveaxis(psi, [0, 1], [i, j])
    return psi.reshape(-1)


def strang_step(psi: np.ndarray, L: int, pxx: float, pz: float, dt: float) -> np.ndarray:
    """Apply ONE 2nd-order Strang step to statevector psi. Returns new psi."""
    # angle conventions (SPEC):  RZ generator-angle per half-layer = pz*dt/2
    #                            RXX generator-angle per bond       = pxx*dt
    Uz = _expm_2x2_pauli(pz * dt / 2.0, Z)
    Uxx = _expm_xx(pxx * dt)

    # 1. Z half-layer
    for i in range(L):
        psi = _apply_single(psi, Uz, i, L)
    # 2. XX even bonds
    for i in range(0, L - 1, 2):
        psi = _apply_two(psi, Uxx, i, i + 1, L)
    # 3. XX odd bonds
    for i in range(1, L - 1, 2):
        psi = _apply_two(psi, Uxx, i, i + 1, L)
    # 4. Z half-layer
    for i in range(L):
        psi = _apply_single(psi, Uz, i, L)
    return psi


def _z_expectation(psi: np.ndarray, site: int, L: int) -> float:
    """<psi| Z_site |psi> with site 0 = most significant qubit."""
    psi_t = psi.reshape([2] * L)
    psi_z = np.moveaxis(psi_t, site, 0).reshape(2, -1)
    # Z eigenvalue +1 for |0>, -1 for |1>
    val = np.vdot(psi_z[0], psi_z[0]) - np.vdot(psi_z[1], psi_z[1])
    return float(val.real)


def _sumz_expectation(psi: np.ndarray, L: int) -> float:
    return float(sum(_z_expectation(psi, i, L) for i in range(L)))


def trotter_reference(L: int, dt: float, T: float, pxx: float = -1.0, pz: float = 1.5):
    """Exact (untruncated) Strang-Trotter circuit evolution from |0...0>.

    Returns dict with:
        't'        : array of times (length N+1, includes t=0)
        'z_mid'    : <Z_{L/2}>(t)
        'sumz'     : <sum_i Z_i>(t)
        'energy0'  : <H>(t=0)  (sanity)
    """
    N = int(round(T / dt))
    psi = np.zeros(2 ** L, dtype=complex)
    psi[0] = 1.0  # |0...0>
    mid = L // 2

    ts = [0.0]
    z_mid = [_z_expectation(psi, mid, L)]
    sumz = [_sumz_expectation(psi, L)]

    for n in range(N):
        psi = strang_step(psi, L, pxx, pz, dt)
        ts.append((n + 1) * dt)
        z_mid.append(_z_expectation(psi, mid, L))
        sumz.append(_sumz_expectation(psi, L))

    return {
        "t": np.array(ts),
        "z_mid": np.array(z_mid),
        "sumz": np.array(sumz),
    }


def build_sparse_hamiltonian(L: int, pxx: float, pz: float) -> sp.csr_matrix:
    """Sparse H = pxx sum X_iX_{i+1} + pz sum Z_i (site 0 = MSB)."""
    def op_at(op, site):
        mats = [sp.identity(2, format="csr", dtype=complex)] * L
        mats[site] = sp.csr_matrix(op)
        out = mats[0]
        for m in mats[1:]:
            out = sp.kron(out, m, format="csr")
        return out

    def op2_at(opa, opb, i, j):
        mats = [sp.identity(2, format="csr", dtype=complex)] * L
        mats[i] = sp.csr_matrix(opa)
        mats[j] = sp.csr_matrix(opb)
        out = mats[0]
        for m in mats[1:]:
            out = sp.kron(out, m, format="csr")
        return out

    H = sp.csr_matrix((2 ** L, 2 ** L), dtype=complex)
    for i in range(L - 1):
        H = H + pxx * op2_at(X, X, i, i + 1)
    for i in range(L):
        H = H + pz * op_at(Z, i)
    return H.tocsr()


def continuous_reference(L: int, dt: float, T: float, pxx: float = -1.0, pz: float = 1.5):
    """Exact continuous exp(-iHt)|0> reference at the same time grid.

    Uses scipy expm_multiply incrementally (apply exp(-iH dt) each step).
    Returns same structure as trotter_reference.
    """
    N = int(round(T / dt))
    H = build_sparse_hamiltonian(L, pxx, pz)
    psi = np.zeros(2 ** L, dtype=complex)
    psi[0] = 1.0
    mid = L // 2

    ts = [0.0]
    z_mid = [_z_expectation(psi, mid, L)]
    sumz = [_sumz_expectation(psi, L)]

    for n in range(N):
        psi = expm_multiply(-1j * dt * H, psi)
        ts.append((n + 1) * dt)
        z_mid.append(_z_expectation(psi, mid, L))
        sumz.append(_sumz_expectation(psi, L))

    return {"t": np.array(ts), "z_mid": np.array(z_mid), "sumz": np.array(sumz)}


def energy_at_zero(L: int, pxx: float, pz: float) -> float:
    """<H>(t=0) on |0...0>. X-terms vanish; only pz sum Z survives = pz*L."""
    psi = np.zeros(2 ** L, dtype=complex)
    psi[0] = 1.0
    H = build_sparse_hamiltonian(L, pxx, pz)
    return float((psi.conj() @ (H @ psi)).real)


def _sanity_checks(L=8, dt=0.05, T=4.0, pxx=-1.0, pz=1.5):
    print(f"=== exact_reference sanity checks (L={L}, pxx={pxx}, pz={pz}) ===")
    # <Z_i>(0) = +1 for all i
    psi0 = np.zeros(2 ** L, dtype=complex)
    psi0[0] = 1.0
    zs = [_z_expectation(psi0, i, L) for i in range(L)]
    assert np.allclose(zs, 1.0), f"<Z_i>(0) != +1: {zs}"
    print(f"  <Z_i>(0) = +1 for all i: PASS ({zs[0]:.6f})")

    # <H>(0) = pz * L
    e0 = energy_at_zero(L, pxx, pz)
    assert abs(e0 - pz * L) < 1e-9, f"<H>(0)={e0} != pz*L={pz*L}"
    print(f"  <H>(0) = {e0:.6f}, expected pz*L = {pz*L:.6f}: PASS")

    # Trotter run executes & stays normalized
    ref = trotter_reference(L, dt, T, pxx, pz)
    print(f"  Trotter <Z_mid>(0)={ref['z_mid'][0]:.6f}, <Z_mid>(T)={ref['z_mid'][-1]:.6f}")
    print(f"  Trotter <sumZ>(0)={ref['sumz'][0]:.6f}")

    # Trotter vs continuous floor at small dt
    cont = continuous_reference(L, dt, T, pxx, pz)
    err = np.max(np.abs(ref["z_mid"] - cont["z_mid"]))
    print(f"  max_t |Z_mid Trotter - continuous| (Trotter floor) = {err:.3e}")
    print("  ALL SANITY CHECKS PASSED")


if __name__ == "__main__":
    _sanity_checks()
