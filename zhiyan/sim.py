"""
Efficient Randomized Lindblad Simulator (zhiyan/ version)
==========================================================
Standalone, production-quality re-implementation of zhiyan's
TNN_TFIM_state.ipynb with the following optimizations:

  1. dt = 0.05  (vs notebook 0.01)  ─┐
  2. Ss = 2σ    (vs notebook 4σ)    ─┘  → 10× fewer TEBD sub-steps

  3. Pre-seeded gate cache  ─  system bonds are cache-hits from step 1

  4. Vectorized energy / fidelity  ─  qtn.expec_TN_1D (no MPO.apply + rename)

  5. Parallel trajectories  ─  run_parallel() uses multiprocessing.Pool;
                               N processes give ≈ N× wall-clock speedup

Physics follows zhiyan's notebook exactly:
  • Random site_i  ∈ {0 … L-1}
  • Random pauli   ∈ {X, Y, Z}
  • Random sign    ∈ {+1, -1}
  • Random ω       ∈ [0, BB]
  • 4th-order Suzuki–Yoshida TEBD, projective measurement

Hamiltonian convention (same as zhiyan):
  H = pxx · Σ X_i X_{i+1}  +  pz · Σ Z_i
  TFIM ground truth (L=4, pxx=-1, pz=1.5):  E₀ ≈ -6.5039

Usage
-----
    from sim import LindbladSimulator, run_parallel

    # Single trajectory
    sim = LindbladSimulator(L=12, pxx=-1.0, pz=1.5)
    for _ in range(200):
        sim.step(alpha=0.25)
    print(sim.get_energy())

    # Parallel (N trajectories, N × speedup)
    results = run_parallel(n_traj=8, n_steps=200, alpha=0.25,
                           sim_params=dict(L=12, pxx=-1.0, pz=1.5))
    import numpy as np
    print("mean energy:", np.mean(results["energies"][:, -1]))
"""

from __future__ import annotations

import os
import random
from typing import Optional, Dict, List

import autoray
import numpy as np
import quimb as qu
import quimb.tensor as qtn
import scipy.linalg
import torch


# ── numerical stability: robust SVD (identical to zhiyan's patch) ─────────────

def _svd_numpy(a, full_matrices=False, **kwargs):
    kwargs.pop("backend", None)
    kwargs["lapack_driver"] = "gesvd"
    try:
        return scipy.linalg.svd(a, full_matrices=full_matrices, **kwargs)
    except scipy.linalg.LinAlgError:
        kwargs["check_finite"] = False
        return scipy.linalg.svd(a, full_matrices=full_matrices, **kwargs)


def _svd_torch(a, full_matrices=False, **kwargs):
    try:
        return torch.linalg.svd(a, full_matrices=full_matrices)
    except RuntimeError:
        U_c, S_c, Vh_c = _svd_numpy(a.detach().cpu().numpy(), full_matrices=full_matrices)
        U  = torch.from_numpy(U_c ).to(device=a.device, dtype=a.dtype)
        S  = torch.from_numpy(S_c ).to(device=a.device)
        Vh = torch.from_numpy(Vh_c).to(device=a.device, dtype=a.dtype)
        return U, S, Vh


autoray.register_function("numpy", "linalg.svd", _svd_numpy)
autoray.register_function("torch", "linalg.svd", _svd_torch)


# ── Pauli matrices ─────────────────────────────────────────────────────────────

_X = qu.pauli("X")
_Y = qu.pauli("Y")
_Z = qu.pauli("Z")
_PAULIS = {1: _X, 2: _Y, 3: _Z}


# ── 4th-order Yoshida coefficients ─────────────────────────────────────────────

_CBRT2 = 2.0 ** (1.0 / 3.0)
_W1    =  1.0 / (2.0 - _CBRT2)          #  ≈ +1.3512
_W0    = -_CBRT2 / (2.0 - _CBRT2)       #  ≈ -1.7025  (negative time step)
_COEFFS = (_W1, _W0, _W1)               #  S4 = S2(W1·dt)·S2(W0·dt)·S2(W1·dt)


# ── SWAP gate ──────────────────────────────────────────────────────────────────

def _make_swap(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    S = torch.zeros((4, 4), dtype=dtype, device=device)
    S[0, 0] = S[1, 2] = S[2, 1] = S[3, 3] = 1.0
    return S.reshape(2, 2, 2, 2).contiguous()


# ── MPS helpers (verbatim from zhiyan's notebook) ─────────────────────────────

def _insert_site(psi: qtn.MatrixProductState,
                 site_i: int,
                 device: torch.device,
                 dtype: torch.dtype) -> qtn.MatrixProductState:
    """Insert a |0⟩ bath qubit AFTER site_i into the MPS."""
    def _to(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.tensor(np.array(x), dtype=dtype, device=device)

    clean = []
    for i in range(psi.L):
        t = psi[i]
        d = _to(t.data)
        p = t.inds.index(psi.site_ind(i))
        if d.ndim == 2:
            d = d.transpose(0, 1).reshape(1, d.shape[1], 2) if p == 0 \
                else d.reshape(d.shape[0], 1, 2)
        else:
            axes = [a for a in range(3) if a != p] + [p]
            d = d.permute(*axes).contiguous()
        clean.append(d)

    pos = site_i + 1
    if pos <= 0:
        l, r = 1, clean[0].shape[0]
    elif pos >= psi.L:
        l, r = clean[-1].shape[1], 1
    else:
        l, r = clean[pos - 1].shape[1], clean[pos].shape[0]

    bath = torch.zeros((l, r, 2), dtype=dtype, device=device)
    for b in range(min(l, r)):
        bath[b, b, 0] = 1.0
    clean.insert(pos, bath)

    new_psi = qtn.MatrixProductState(clean, shape="lrp")
    new_psi.cyclic = False
    t0, tL = new_psi[0], new_psi[new_psi.L - 1]
    for ind in set(t0.inds) & set(tL.inds):
        if not ind.startswith("k"):
            tL.reindex_({ind: qtn.rand_uuid()}, inplace=True)
    return new_psi


def _measure_remove(mps: qtn.MatrixProductState,
                    i: int,
                    device: torch.device,
                    dtype: torch.dtype):
    """Projective measurement on site i, remove from MPS.  Returns (new_mps, outcome)."""
    mps.canonize(i)
    T   = mps[i]
    def _to(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.tensor(np.array(x), dtype=dtype, device=device)

    data = _to(T.data)
    p_ind = mps.site_ind(i)
    p_ax  = T.inds.index(p_ind)

    total_w = float(torch.norm(data).item() ** 2)
    # slice along physical axis
    slices = [slice(None)] * data.ndim
    slices[p_ax] = 0
    T0 = data[tuple(slices)]
    prob0 = max(0.0, min(1.0, float(torch.norm(T0).item() ** 2 / total_w)))

    outcome = 0 if random.random() < prob0 else 1
    norm_f  = 1.0 / np.sqrt(prob0 if outcome == 0 else 1.0 - prob0)
    proj    = torch.tensor([1.0, 0.0] if outcome == 0 else [0.0, 1.0],
                           dtype=dtype, device=device)

    def _lrp(idx):
        t = mps[idx]; d = _to(t.data); pi = mps.site_ind(idx)
        if d.ndim == 2:
            pa = t.inds.index(pi)
            d  = d.transpose(0, 1).reshape(1, -1, 2) if pa == 0 \
                 else d.reshape(-1, 1, 2)
        else:
            if idx > 0:
                common = set(t.inds) & set(mps[idx - 1].inds)
                if common:
                    la = t.inds.index(list(common)[0])
                    pa = t.inds.index(pi)
                    ra = [x for x in range(3) if x != la and x != pa][0]
                    d  = d.permute(la, ra, pa)
                else:
                    bds = [x for x in t.inds if x != pi]
                    im  = {ix: n for n, ix in enumerate(t.inds)}
                    d   = d.permute(im[bds[0]], im[bds[1]], im[pi])
            else:
                pa   = t.inds.index(pi)
                axes = [a for a in range(d.ndim) if a != pa] + [pa]
                d    = d.permute(*axes)
                if d.ndim == 2:
                    d = d.reshape(1, -1, 2)
        return d.contiguous()

    A    = _lrp(i)
    M    = torch.tensordot(A, proj, dims=([2], [0])) * norm_f
    L    = mps.L
    out  = []
    if i == 0:
        An = _lrp(1)
        out.append(torch.tensordot(M, An, dims=([1], [0])))
        for s in range(2, L): out.append(_lrp(s))
    else:
        Ap  = _lrp(i - 1)
        An  = torch.tensordot(Ap, M, dims=([1], [0])).permute(0, 2, 1).contiguous()
        for s in range(L):
            if   s == i - 1: out.append(An)
            elif s == i:     continue
            else:            out.append(_lrp(s))

    if out[0].ndim == 2:  out[0]  = out[0].reshape(1, *out[0].shape)
    if out[-1].ndim == 2: out[-1] = out[-1].reshape(*out[-1].shape[:-1], 1, out[-1].shape[-1])

    new_mps = qtn.MatrixProductState(out, shape="lrp",
                                     site_ind_id="k{}", site_tag_id="I{}")
    new_mps.normalize()
    return new_mps, outcome


# ── Hamiltonian constructor ────────────────────────────────────────────────────

def _build_local_ham(L, pxx, pz, omega, alpha, site_i, pauli_op):
    """Build LocalHam on the (L+1)-site extended chain (system + bath).

    Same construction as zhiyan's build_hamiltonian_part_vector.
    The 'jumper' XX bond across the bath is excluded (handled via SWAP gates).
    """
    def get_idx(j):
        return j if j <= site_i else j + 1

    bath = site_i + 1
    b    = qtn.SpinHam1D(S=1 / 2, cyclic=False)

    for j in range(site_i):          b[j, j + 1]         += pxx, _X, _X
    for j in range(site_i + 1, L - 1):
        u, v = j + 1, j + 2;        b[u, v]              += pxx, _X, _X
    for j in range(L):               b[get_idx(j)]        += pz,  _Z
    b[bath]                                                += (-omega / 2.0), _Z
    b[site_i, bath]                                        += alpha, _PAULIS[pauli_op], _X
    if bath < L:                     b[bath, bath + 1]    += 0.0, _X, _X

    return b.build_local_ham(L + 1)


# ── Gate caching ───────────────────────────────────────────────────────────────

def _c_dtype(dtype):
    return torch.complex128 if dtype in (torch.float64, torch.complex128) \
           else torch.complex64


def _get_gates(H_local, dt, expm_cache, dtype, device):
    """Compute exp(-i H_bond dt) for every bond; cache by (dt, matrix_bytes)."""
    def _to(x):
        if isinstance(x, torch.Tensor): return x.to(device=device, dtype=dtype)
        return torch.tensor(np.array(x), dtype=dtype, device=device)

    cd    = _c_dtype(dtype)
    gates = {}
    for where, term in H_local.terms.items():
        dim = int(np.sqrt(term.size))
        mat = term.reshape(dim, dim)
        if np.allclose(mat, 0.0, atol=1e-15):
            gates[where] = torch.eye(4, dtype=dtype, device=device).reshape(2,2,2,2)
            continue
        key = (round(float(dt), 15), mat.tobytes())
        U   = expm_cache.get(key)
        if U is None:
            m = _to(mat)
            w, V = torch.linalg.eigh(m)
            V_c  = V.to(cd)
            phase = torch.exp(-1j * w * dt).to(cd)
            U    = (V_c @ torch.diag(phase) @ V_c.mH).to(dtype)
            expm_cache[key] = U
        gates[where] = U.reshape(2, 2, 2, 2).contiguous()
    return gates, expm_cache


def _get_jumper(dt, pxx, expm_cache, dtype, device):
    """exp(-i pxx X⊗X dt) — handles the XX bond that straddles the bath site."""
    cd   = _c_dtype(dtype)
    sX   = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype, device=device)
    H2   = pxx * torch.kron(sX, sX)
    key  = (round(float(dt), 15), H2.cpu().numpy().tobytes())
    U2   = expm_cache.get(key)
    if U2 is None:
        if abs(dt) < 1e-15:
            U2 = torch.eye(4, dtype=dtype, device=device)
        else:
            w, V = torch.linalg.eigh(H2)
            V_c  = V.to(cd)
            phase = torch.exp(-1j * w * dt).to(cd)
            U2   = (V_c @ torch.diag(phase) @ V_c.mH).to(dtype)
        expm_cache[key] = U2
    return U2.reshape(2, 2, 2, 2).contiguous(), expm_cache


# ── Core TEBD step ─────────────────────────────────────────────────────────────

def _run_step(psi, duration, dt, L, pxx, pz,
              omega, alpha, site_i, pauli_op,
              max_bond, expm_cache, device, dtype, swap):
    """Insert bath → 4th-order TEBD for `duration` → measure bath.

    Returns (new_psi, outcome, expm_cache, max_bond_reached).
    """
    mps = _insert_site(psi, site_i, device, dtype)
    mps.cyclic = False

    H_local = _build_local_ham(L, pxx, pz, omega, alpha, site_i, pauli_op)

    opts = dict(max_bond=max_bond, cutoff=1e-12,
                cutoff_mode="rel", renorm=True, absorb="both")

    n_sub = int(round(duration / dt))
    if n_sub <= 0:
        new_psi, out = _measure_remove(mps, site_i + 1, device, dtype)
        return new_psi, out, expm_cache, mps.max_bond()

    iL, iM, iR = site_i, site_i + 1, site_i + 2
    has_jumper  = (iR < mps.L) and (iL >= 0)
    max_bd      = mps.max_bond()

    # Cache gate sets keyed by rounded sub-step size
    _gsets: Dict = {}

    def _gset(dt_sub):
        nonlocal expm_cache
        k = round(float(dt_sub), 15)
        if k in _gsets:
            return _gsets[k]
        full, expm_cache = _get_gates(H_local, dt_sub,       expm_cache, dtype, device)
        half, expm_cache = _get_gates(H_local, dt_sub / 2.0, expm_cache, dtype, device)
        ev = sorted(b for b in full if b[0] % 2 == 0)
        od = sorted(b for b in full if b[0] % 2 == 1)
        jh = None
        if has_jumper:
            jh, expm_cache = _get_jumper(dt_sub / 2.0, pxx, expm_cache, dtype, device)
        _gsets[k] = (full, half, ev, od, jh)
        return _gsets[k]

    def _strang(dt_sub):
        """One 2nd-order Strang step."""
        _, half, ev, od, jh = _gset(dt_sub)

        # jumper half (handles X_{i-1} X_{i+1} across bath via SWAP trick)
        if has_jumper:
            mps.gate_split_(swap, (iM, iR), **opts)
            mps.gate_split_(jh,   (iL, iM), **opts)
            mps.gate_split_(swap, (iM, iR), **opts)

        # Strang: odd_half → even_full → odd_half
        for b in od: mps.gate_split_(half[b], b, **opts)
        for b in ev: mps.gate_split_(_gset(dt_sub)[0][b], b, **opts)
        for b in od: mps.gate_split_(half[b], b, **opts)

        # jumper half (symmetric)
        if has_jumper:
            mps.gate_split_(swap, (iM, iR), **opts)
            mps.gate_split_(jh,   (iL, iM), **opts)
            mps.gate_split_(swap, (iM, iR), **opts)

    # 4th-order loop: S4(dt) = S2(W1·dt) · S2(W0·dt) · S2(W1·dt)
    for _ in range(n_sub):
        for w in _COEFFS:
            _strang(w * dt)
        mps.normalize()
        max_bd = max(max_bd, mps.max_bond())

    new_psi, outcome = _measure_remove(mps, site_i + 1, device, dtype)
    return new_psi, outcome, expm_cache, max_bd


# ── Cache pre-seeding ─────────────────────────────────────────────────────────

def _preseed(L, pxx, pz, dt, expm_cache, device, dtype):
    """Warm up the gate cache before the main loop.

    Runs a 3-sub-step dummy simulation (alpha=0) to populate the cache
    with all system-bond gates for every Yoshida sub-step size.
    After preseeding, all system bonds are cache-hits for every future step.
    """
    swap  = _make_swap(dtype, device)
    dummy = qtn.MPS_computational_state("0" * L)
    dummy.apply_to_arrays(
        lambda x: (x.to(dtype=dtype, device=device) if isinstance(x, torch.Tensor)
                   else torch.tensor(np.array(x), dtype=dtype, device=device))
    )
    _, _, cache, _ = _run_step(
        dummy, duration=dt * 3, dt=dt, L=L,
        pxx=pxx, pz=pz, omega=1.0, alpha=0.0,
        site_i=0, pauli_op=1,
        max_bond=4, expm_cache=expm_cache,
        device=device, dtype=dtype, swap=swap,
    )
    return cache


# ── LindbladSimulator ─────────────────────────────────────────────────────────

class LindbladSimulator:
    """Pure-state MPS Lindblad cooling simulator.

    Optimized implementation of zhiyan's TNN_TFIM_state.ipynb.

    Parameters
    ----------
    L          : number of system qubits
    pxx        : XX coupling (e.g. pxx=-1.0 for TFIM)
    pz         : transverse field
    dt         : TEBD sub-step. Default 0.05 (10× fewer steps than zhiyan's 0.01)
    Ss_factor  : interaction window = Ss_factor × sigma. Default 2.0 (vs 4.0)
    sigma      : filter width. Default 4.0
    BB         : max bath frequency. Default 5.0
    max_bond   : MPS bond dimension cap
    seed       : RNG seed
    preseed    : warm up cache before first step (recommended)
    device     : 'cpu' or 'cuda'
    dtype      : torch.complex128 (default)
    """

    def __init__(
        self, L: int, pxx: float, pz: float, *,
        dt: float = 0.05,
        Ss_factor: float = 2.0,
        sigma: float = 4.0,
        BB: float = 5.0,
        max_bond: int = 64,
        seed: Optional[int] = None,
        preseed: bool = True,
        device: str = "cpu",
        dtype: torch.dtype = torch.complex128,
    ):
        self.L        = L
        self.pxx      = pxx
        self.pz       = pz
        self.dt       = dt
        self.Ss       = Ss_factor * sigma
        self.BB       = BB
        self.max_bond = max_bond
        self.device   = torch.device(device)
        self.dtype    = dtype

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        self._swap  = _make_swap(dtype, self.device)
        self._cache: dict = {}

        # Initial state |0…0⟩
        self.psi = qtn.MPS_computational_state("0" * L)
        self._to_dev(self.psi)

        if preseed:
            self._cache = _preseed(L, pxx, pz, dt, self._cache, self.device, self.dtype)

        self._ham_mpo = None

    # ── internals ──────────────────────────────────────────────────────────────

    def _to_dev(self, obj):
        obj.apply_to_arrays(
            lambda x: (x.to(dtype=self.dtype, device=self.device)
                       if isinstance(x, torch.Tensor)
                       else torch.tensor(np.array(x), dtype=self.dtype, device=self.device))
        )

    def _ham(self):
        if self._ham_mpo is None:
            b = qtn.SpinHam1D(S=1 / 2)
            for i in range(self.L - 1): b[i, i + 1] += self.pxx, _X, _X
            for i in range(self.L):     b[i]         += self.pz,  _Z
            self._ham_mpo = b.build_mpo(self.L)
            self._to_dev(self._ham_mpo)
        return self._ham_mpo

    # ── public API ─────────────────────────────────────────────────────────────

    def step(self, alpha: float) -> int:
        """Run one cooling step.  Returns bath measurement outcome (0 or 1).

        Randomly draws: site_i, pauli ∈ {X,Y,Z}, sign ∈ {±1}, ω ∈ [0, BB].
        Applies 4th-order TEBD for total time 2·Ss, then projects the bath.
        """
        site_i   = random.randint(0, self.L - 1)
        pauli_op = random.randint(1, 3)           # 1=X, 2=Y, 3=Z
        omega    = random.uniform(0.0, self.BB)
        sign     = random.choice([-1, 1])

        new_psi, outcome, self._cache, _ = _run_step(
            self.psi, duration=2.0 * self.Ss, dt=self.dt,
            L=self.L, pxx=self.pxx, pz=self.pz,
            omega=omega, alpha=sign * alpha,
            site_i=site_i, pauli_op=pauli_op,
            max_bond=self.max_bond, expm_cache=self._cache,
            device=self.device, dtype=self.dtype, swap=self._swap,
        )
        self.psi = new_psi
        return outcome

    def get_energy(self) -> float:
        """⟨ψ|H|ψ⟩ using qtn.expec_TN_1D (fast, no index renaming needed)."""
        return float(np.real(qtn.expec_TN_1D(self.psi.H, self._ham(), self.psi)))

    def get_fidelity(self, psi_gs: qtn.MatrixProductState) -> float:
        """Return |⟨GS|ψ⟩|²."""
        return float(abs(complex(psi_gs.H @ self.psi)) ** 2)

    def get_ground_state(self):
        """Run DMRG to find the ground state. Returns (psi_gs, E0)."""
        # Build numpy-based MPO so DMRG stays on numpy backend
        b = qtn.SpinHam1D(S=1 / 2)
        for i in range(self.L - 1): b[i, i + 1] += self.pxx, _X, _X
        for i in range(self.L):     b[i]         += self.pz,  _Z
        ham_np = b.build_mpo(self.L)
        dmrg   = qtn.DMRG2(ham_np)
        dmrg.solve(max_sweeps=80, verbosity=0)
        gs = dmrg.state
        gs.normalize()
        E0 = float(np.real(qtn.expec_TN_1D(gs.H, ham_np, gs)))
        # Convert to torch for fidelity computation
        gs.apply_to_arrays(
            lambda x: torch.tensor(np.array(x), dtype=self.dtype, device=self.device)
        )
        return gs, E0

    def run(self, n_steps: int, alpha: float,
            psi_gs=None, verbose: bool = True,
            log_every: int = 10) -> dict:
        """Run n_steps and return energies (and optionally fidelities).

        Parameters
        ----------
        n_steps   : number of cooling steps
        alpha     : coupling strength
        psi_gs    : if provided, compute fidelity at each step
        verbose   : print progress
        log_every : print every this many steps

        Returns
        -------
        dict with keys 'energies' and 'fidelities' (None if psi_gs not given)
        """
        energies   = np.zeros(n_steps)
        fidelities = np.zeros(n_steps) if psi_gs is not None else None

        for n in range(n_steps):
            self.step(alpha)
            energies[n] = self.get_energy()
            if fidelities is not None:
                fidelities[n] = self.get_fidelity(psi_gs)
            if verbose and (n % log_every == 0 or n == n_steps - 1):
                msg = f"  step {n+1:4d}/{n_steps}  E={energies[n]:.6f}"
                if fidelities is not None:
                    msg += f"  F={fidelities[n]:.4f}"
                print(msg, flush=True)

        return {"energies": energies, "fidelities": fidelities}


# ── Parallel trajectory execution ────────────────────────────────────────────
# The worker function must live at module level for multiprocessing "spawn"
# (default on macOS/Windows).

def _mps_to_lrp_arrays(mps) -> List[np.ndarray]:
    """Serialize an MPS to a list of (left, right, phys) numpy arrays.

    quimb stores tensors in arbitrary index order depending on how the MPS
    was built.  This function reads the physical-index position from the
    tensor metadata and permutes to a canonical (L, R, P) layout that can
    be safely reconstructed with shape='lrp'.
    """
    out = []
    for i in range(mps.L):
        t    = mps[i]
        d    = t.data
        if isinstance(d, torch.Tensor):
            d = d.detach().cpu().numpy()
        else:
            d = np.array(d)

        p_ind = mps.site_ind(i)

        if d.ndim == 2:
            # Boundary tensor: determine which axis is physical
            p_ax = t.inds.index(p_ind)
            if p_ax == 0:
                # (phys, bond) → (1, bond, phys)
                d = np.transpose(d, (1, 0)).reshape(1, d.shape[1], d.shape[0])
            else:
                # (bond, phys) → (bond, 1, phys)
                d = d.reshape(d.shape[0], 1, d.shape[1])
        else:
            # Bulk tensor: permute to (left, right, phys)
            p_ax = t.inds.index(p_ind)
            axes = [a for a in range(d.ndim) if a != p_ax] + [p_ax]
            d = np.transpose(d, axes)

        out.append(d)
    return out


def _lrp_arrays_to_mps(arrays, dtype, device):
    """Reconstruct an MPS from (left, right, phys) numpy arrays."""
    tensors = [torch.tensor(a, dtype=dtype, device=device) for a in arrays]
    return qtn.MatrixProductState(tensors, shape="lrp")


def _worker(args):
    """Worker: create a fresh simulator, run it, return numpy arrays."""
    seed, n_steps, alpha, sim_params, gs_arrays = args

    sim = LindbladSimulator(**sim_params, seed=seed)
    psi_gs = None

    if gs_arrays is not None:
        psi_gs = _lrp_arrays_to_mps(gs_arrays, sim.dtype, sim.device)

    out = sim.run(n_steps, alpha, psi_gs=psi_gs, verbose=False)
    fid = out["fidelities"]
    return out["energies"], fid


def run_parallel(
    n_traj: int,
    n_steps: int,
    alpha: float,
    sim_params: dict,
    psi_gs=None,
    n_workers: Optional[int] = None,
) -> dict:
    """Run multiple independent cooling trajectories in parallel.

    Each trajectory starts from the |0…0⟩ product state with a unique RNG seed,
    so all trajectories are statistically independent.

    Parameters
    ----------
    n_traj      : number of independent trajectories to run
    n_steps     : cooling steps per trajectory
    alpha       : coupling strength
    sim_params  : keyword arguments for LindbladSimulator (L, pxx, pz, …)
                  Do NOT include 'seed' — it is set automatically per trajectory.
    psi_gs      : optional ground-state MPS for fidelity tracking.
                  Pass the result of LindbladSimulator.get_ground_state()[0].
    n_workers   : number of parallel processes. Defaults to os.cpu_count().

    Returns
    -------
    dict with:
      'energies'   : np.ndarray of shape (n_traj, n_steps)
      'fidelities' : np.ndarray of shape (n_traj, n_steps), or None

    Example
    -------
        results = run_parallel(
            n_traj=8, n_steps=200, alpha=0.25,
            sim_params=dict(L=12, pxx=-1.0, pz=1.5, dt=0.05, max_bond=32),
        )
        import numpy as np
        print("mean final energy:", np.mean(results["energies"][:, -1]))
    """
    from multiprocessing import Pool, cpu_count

    if n_workers is None:
        n_workers = min(n_traj, cpu_count())

    # Serialize psi_gs in canonical (left, right, phys) order so workers can
    # reconstruct it correctly regardless of DMRG's internal index layout.
    gs_arrays = None
    if psi_gs is not None:
        gs_arrays = _mps_to_lrp_arrays(psi_gs)

    tasks = [
        (seed, n_steps, alpha, sim_params, gs_arrays)
        for seed in range(n_traj)
    ]

    # Try parallel; fall back to sequential if Pool fails (e.g. in notebooks)
    try:
        with Pool(processes=n_workers) as pool:
            results = pool.map(_worker, tasks)
    except Exception as e:
        print(f"[run_parallel] multiprocessing failed ({e}), running sequentially.")
        results = [_worker(t) for t in tasks]

    energies_list   = [r[0] for r in results]
    fidelities_list = [r[1] for r in results]

    return {
        "energies":   np.array(energies_list),
        "fidelities": np.array(fidelities_list) if fidelities_list[0] is not None else None,
    }
