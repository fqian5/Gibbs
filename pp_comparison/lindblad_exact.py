"""
Exact density-matrix Lindblad reference (gold standard, L <= 8).

Vectorized Liouvillian on 4^L: vec(rho)(t) = exp(L t) vec(rho0),
<H>(t) = Tr[H rho(t)]. Column-stacking vec convention:
    vec(A X B) = (B^T (x) A) vec(X).
So for  drho/dt = -i[H,rho] + sum_k ( K rho K^dag - 1/2 {K^dag K, rho} ):
    Lcoh = -i ( I (x) H  -  H^T (x) I )
    Ldiss(K) = conj(K) (x) K  - 1/2 ( I (x) K^dag K + (K^dag K)^T (x) I )

Two dissipators:
  * Experiment B (PRIMARY): site-local surrogate K_i = sigma_i^+ = |1><0|_i.
    Damps each site toward |1> (the mean-field single-site ground state of pz*Z
    with pz>0). NESS energy E_NESS(gamma) lies between true E_GS (gamma->0+) and
    E_MF = -pz*L (gamma->inf, frozen |1...1>).
  * Experiment A (validation only): the exact eigenbasis-filtered jump K from
    PRR_2024/lindblad.py (construct_jump_exact) -> cools toward true E_GS.

Hamiltonian convention (locked, Pauli +/-1):
    H = pxx * sum X_i X_{i+1} + pz * sum Z_i,  pxx=-1.0, pz=+1.5.

Qubit ordering: site 0 = most-significant qubit (kron order op_0 (x) ... (x) op_{L-1}),
matching exact_reference.py / mps_trotter.py.

Sanity: <H>(0) on |0...0> = pz*L (asserted = 12.0 at L=8).
Damping-direction guard: E_NESS(gamma->inf) -> -pz*L = -1.5L (asserted).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from scipy.sparse.linalg import expm_multiply

I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
SP = np.array([[0, 0], [1, 0]], dtype=complex)   # sigma^+ = |1><0|  (maps |0>->|1>)


def _op_at(op, site, L):
    mats = [I2] * L
    mats[site] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def build_hamiltonian(L, pxx=-1.0, pz=1.5):
    """Dense H = pxx sum X_iX_{i+1} + pz sum Z_i (site 0 = MSB)."""
    H = np.zeros((2 ** L, 2 ** L), dtype=complex)
    for i in range(L - 1):
        H += pxx * (_op_at(X, i, L) @ _op_at(X, i + 1, L))
    for i in range(L):
        H += pz * _op_at(Z, i, L)
    return H


def _vec(rho):
    """Column-stacking vectorization (Fortran order)."""
    return rho.reshape(-1, order="F")


def _unvec(v, n):
    return v.reshape((n, n), order="F")


def build_liouvillian(H, Ks):
    """Dense Liouvillian super-operator L (n^2 x n^2), column-stacking convention.

    drho/dt = -i[H,rho] + sum_k ( K rho K^dag - 1/2 {K^dag K, rho} ).
    """
    n = H.shape[0]
    Id = np.eye(n, dtype=complex)
    Lcoh = -1j * (np.kron(Id, H) - np.kron(H.T, Id))
    Ldiss = np.zeros((n * n, n * n), dtype=complex)
    for K in Ks:
        KdK = K.conj().T @ K
        Ldiss += np.kron(K.conj(), K)
        Ldiss -= 0.5 * (np.kron(Id, KdK) + np.kron(KdK.T, Id))
    return Lcoh + Ldiss


def build_liouvillian_sparse(H, Ks):
    """Sparse Liouvillian (csr, n^2 x n^2), same column-stacking convention.

    Enables L=7,8 surrogate evolution via expm_multiply (dense would be ~GBs).
    """
    n = H.shape[0]
    Hs = sp.csr_matrix(H)
    Id = sp.identity(n, format="csr", dtype=complex)
    Lcoh = -1j * (sp.kron(Id, Hs, format="csr") - sp.kron(Hs.T, Id, format="csr"))
    Ldiss = sp.csr_matrix((n * n, n * n), dtype=complex)
    for K in Ks:
        Ks_ = sp.csr_matrix(K)
        KdK = (Ks_.conj().T @ Ks_).tocsr()
        Ldiss = Ldiss + sp.kron(Ks_.conj(), Ks_, format="csr")
        Ldiss = Ldiss - 0.5 * (sp.kron(Id, KdK, format="csr")
                               + sp.kron(KdK.T, Id, format="csr"))
    return (Lcoh + Ldiss).tocsr()


def surrogate_jumps(L, gamma):
    """Per-site K_i = sqrt(gamma) * sigma_i^+ (damp toward |1>)."""
    return [np.sqrt(gamma) * _op_at(SP, i, L) for i in range(L)]


def exact_filtered_jump(L, pxx=-1.0, pz=1.5, A_site=None,
                        filter_params=None, gamma=1.0):
    """Build the exact eigenbasis-filtered jump (Experiment A) from a coupling A.

    Reuses the PRR_2024 filter: K = sum_ij fhat(E_i - E_j) <i|A|j> |i><j|.
    A defaults to sum_i X_i (a simple global coupling). filter_params default to a
    broad low-pass that selects de-exciting (E_i<E_j) transitions toward the GS.
    Returns [sqrt(gamma)*K].
    """
    from scipy.special import erf

    H = build_hamiltonian(L, pxx, pz)
    if A_site is None:
        A = sum(_op_at(X, i, L) for i in range(L))
    else:
        A = _op_at(X, A_site, L)

    E, V = la.eigh(H)
    A_ovlp = V.conj().T @ A @ V

    if filter_params is None:
        # low-pass that keeps energy-lowering transitions (x = E_i - E_j < 0)
        a, b, da, db = 0.0, (E.max() - E.min()) * 1.5, 1.0, 1.0
    else:
        a, b, da, db = (filter_params[k] for k in ("a", "b", "da", "db"))

    def fhat(x):
        return 0.5 * (erf((x + a) / da) - erf((x + b) / db))

    n = 2 ** L
    K = np.zeros((n, n), dtype=complex)
    Emat = E[:, None] - E[None, :]
    F = fhat(Emat)
    K = V @ (F * A_ovlp) @ V.conj().T
    return [np.sqrt(gamma) * K]


def run_exact_lindblad(L, gamma, T, tau, pxx=-1.0, pz=1.5,
                       experiment="B", filter_params=None):
    """Evolve the exact Lindblad and record <H>(t) on a tau grid.

    experiment="B": surrogate K_i = sqrt(gamma) sigma^+ (toward |1>).
    experiment="A": exact filtered jump (cools toward true GS).

    Returns dict: 't', 'energy', 'E_GS' (true GS energy), 'E_NESS' (=energy[-1]).

    For the surrogate (experiment B) the jumps are sparse, so for L>=7 we build a
    SPARSE Liouvillian and use expm_multiply (the dense n^2 x n^2 propagator would
    be ~34 GB at L=8). For L<=6 the dense one-step propagator is used (faster).
    Experiment A's filtered jump is dense -> dense path only (L<=6).
    """
    H = build_hamiltonian(L, pxx, pz)
    n = 2 ** L
    if experiment == "B":
        Ks = surrogate_jumps(L, gamma)
    elif experiment == "A":
        Ks = exact_filtered_jump(L, pxx, pz, filter_params=filter_params, gamma=gamma)
    else:
        raise ValueError(experiment)

    N = int(round(T / tau))
    rho = np.zeros((n, n), dtype=complex)
    rho[0, 0] = 1.0                        # |0...0><0...0|
    v = _vec(rho)
    ts = [0.0]
    energies = [float(np.real(np.trace(H @ rho)))]

    use_sparse = (experiment == "B") and (L >= 7)
    if use_sparse:
        Lsup = build_liouvillian_sparse(H, Ks)   # sparse n^2 x n^2
        for k in range(N):
            v = expm_multiply(Lsup * tau, v)     # action of exp(L*tau) on vec(rho)
            rho = _unvec(v, n)
            ts.append((k + 1) * tau)
            energies.append(float(np.real(np.trace(H @ rho))))
    else:
        Lsup = build_liouvillian(H, Ks)
        step = la.expm(Lsup * tau)               # dense one-step propagator
        for k in range(N):
            v = step @ v
            rho = _unvec(v, n)
            ts.append((k + 1) * tau)
            energies.append(float(np.real(np.trace(H @ rho))))

    E_GS = float(np.min(la.eigvalsh(H)))
    return {
        "t": np.array(ts),
        "energy": np.array(energies),
        "E_GS": E_GS,
        "E_NESS": energies[-1],
    }


def ness_energy_direct(L, gamma, pxx=-1.0, pz=1.5, experiment="B"):
    """Steady-state <H> from the null space of L (kernel), independent of T.

    Solves L vec(rho_ss) = 0 with Tr[rho_ss]=1 via SVD, returns Tr[H rho_ss].
    Used as a T-independent NESS cross-check.
    """
    H = build_hamiltonian(L, pxx, pz)
    n = 2 ** L
    Ks = surrogate_jumps(L, gamma) if experiment == "B" else \
        exact_filtered_jump(L, pxx, pz, gamma=gamma)
    Lsup = build_liouvillian(H, Ks)
    # smallest singular vector -> kernel of L
    u, s, vh = la.svd(Lsup)
    rho_ss = _unvec(vh[-1].conj(), n)
    rho_ss = rho_ss / np.trace(rho_ss)
    rho_ss = 0.5 * (rho_ss + rho_ss.conj().T)   # hermitize
    return float(np.real(np.trace(H @ rho_ss)))


def _sanity():
    pxx, pz = -1.0, 1.5
    print("=== lindblad_exact sanity checks ===")

    # <H>(0) = pz*L at L=8
    L = 8
    H = build_hamiltonian(L, pxx, pz)
    rho0 = np.zeros((2 ** L, 2 ** L), dtype=complex)
    rho0[0, 0] = 1.0
    e0 = float(np.real(np.trace(H @ rho0)))
    assert abs(e0 - pz * L) < 1e-9, f"<H>(0)={e0} != {pz*L}"
    print(f"  L=8 <H>(0) = {e0:.6f} == pz*L = {pz*L:.1f}  PASS")

    # Damping-direction guard: gamma->inf NESS -> -pz*L
    L = 6
    e_big = ness_energy_direct(L, gamma=50.0, pxx=pxx, pz=pz, experiment="B")
    target = -pz * L
    print(f"  L=6 E_NESS(gamma=50) = {e_big:.5f}, target -pz*L = {target:.5f}, "
          f"diff = {abs(e_big - target):.2e}")
    assert abs(e_big - target) < 1e-2, "DAMPING DIRECTION WRONG (should -> -pz*L)"
    print("  damping direction toward |1>: PASS")

    # True (T-independent) NESS energy from the kernel of L.
    # KEY PHYSICS: for the single-site sigma^+ surrogate, E_NESS = -pz*L EXACTLY
    # for every gamma (the dissipator pins <Z_i>=-1 and kills <X_iX_{i+1}> in the
    # steady state). So the gamma-scan does NOT tune E_NESS for this surrogate;
    # it is pinned at the mean-field value E_MF = -pz*L. The discriminating
    # quantity for the benchmark is the TRANSIENT relaxation curve <H>(t).
    e_ss = ness_energy_direct(L, gamma=0.1, experiment="B")
    e_gs = float(np.min(la.eigvalsh(build_hamiltonian(L, pxx, pz))))
    print(f"  L=6 gamma=0.1: E_GS={e_gs:.5f}  E_NESS(kernel)={e_ss:.5f} "
          f"(target -pz*L={-pz*L:.5f}, diff {abs(e_ss + pz*L):.2e})")
    assert abs(e_ss + pz * L) < 1e-6, "surrogate NESS should pin to -pz*L"
    print("  ALL SANITY CHECKS PASSED")


if __name__ == "__main__":
    _sanity()
