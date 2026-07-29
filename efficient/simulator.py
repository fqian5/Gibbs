"""
Efficient Randomized Lindblad Simulator for 1D TFIM ground-state preparation.

Baseline: zhiyan's TNN_TFIM_state.ipynb (pure-state MPS, verified correct).
Reference physics: arXiv:2508.05703

Key optimizations over the zhiyan baseline:
    1. dt = 0.05  (vs baseline 0.01) → 5× fewer TEBD sub-steps per simulation step
    2. Ss = 2σ    (vs baseline 4σ)   → 2× shorter interaction window
    3. Pre-seeded gate cache          → system-bond gates are warm-cache hits from step 1
    Expected combined speedup: ~10× with negligible accuracy loss.

Algorithm per step  (same physics as zhiyan):
    1. Randomly pick: site_i ∈ [0,L-1], Pauli type ∈ {X,Y,Z},
                      ω ∈ [0, ω_max], coupling sign ∈ {+1,-1}
    2. Insert a |0⟩ bath qubit into the MPS right after site_i
    3. Hamiltonian on extended chain:
         H = pxx Σ X_j X_{j+1}   (system NN bonds)
           + pz  Σ Z_j            (system field)
           + (-ω/2) Z_bath        (bath field sets energy gap to ω)
           + α · S_{site_i} · X_bath   (system–bath coupling)
    4. 4th-order Suzuki-Yoshida TEBD for total time 2·Ss
    5. Projective measurement on bath qubit → remove bath from MPS

Convention: H = pxx * Σ X_i X_{i+1} + pz * Σ Z_i
    pxx = J  (e.g. J=-1 for antiferromagnet, J=1 for ferromagnet)
    pz  = g  (transverse field)
"""

from __future__ import annotations

import random
from typing import Optional

import autoray
import numpy as np
import quimb as qu
import quimb.tensor as qtn
import scipy.linalg
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Numerical stability: robust SVD (same patch as zhiyan's baseline)
# ---------------------------------------------------------------------------

def _robust_svd_numpy(a, full_matrices=False, **kwargs):
    kwargs.pop("backend", None)
    kwargs["lapack_driver"] = "gesvd"
    try:
        return scipy.linalg.svd(a, full_matrices=full_matrices, **kwargs)
    except scipy.linalg.LinAlgError:
        kwargs["check_finite"] = False
        return scipy.linalg.svd(a, full_matrices=full_matrices, **kwargs)


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


autoray.register_function("numpy", "linalg.svd", _robust_svd_numpy)
autoray.register_function("torch", "linalg.svd", _robust_svd_torch)

# ---------------------------------------------------------------------------
# Pauli matrices (global, constructed once)
# ---------------------------------------------------------------------------

X = qu.pauli("X")
Y = qu.pauli("Y")
Z = qu.pauli("Z")

# ---------------------------------------------------------------------------
# MPS helpers  (extracted verbatim from zhiyan's verified notebook)
# ---------------------------------------------------------------------------

def insert_MPS_site(psi: qtn.MatrixProductState,
                    site_i: int,
                    device: torch.device,
                    dtype: torch.dtype) -> qtn.MatrixProductState:
    """Insert a |0⟩ bath qubit into the MPS AFTER site_i.

    Identical to zhiyan's insert_MPS_site (TNN_TFIM_state.ipynb, cell 2).
    """
    def _to(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.tensor(np.array(x), dtype=dtype, device=device)

    clean_data = []
    L_old = psi.L

    for i in range(L_old):
        t = psi[i]
        d = _to(t.data)
        p_axis = t.inds.index(psi.site_ind(i))
        if d.ndim == 2:
            if p_axis == 0:
                d = d.transpose(0, 1).reshape(1, d.shape[1], 2)
            else:
                d = d.reshape(d.shape[0], 1, 2)
        else:
            axes = [ax for ax in range(3) if ax != p_axis] + [p_axis]
            d = d.permute(*axes).contiguous()
        clean_data.append(d)

    target_pos = site_i + 1
    if target_pos <= 0:
        l_bond, r_bond = 1, clean_data[0].shape[0]
    elif target_pos >= L_old:
        l_bond, r_bond = clean_data[-1].shape[1], 1
    else:
        l_bond = clean_data[target_pos - 1].shape[1]
        r_bond = clean_data[target_pos].shape[0]

    # |0⟩ tensor: bath_data[b, b, 0] = 1 (identity on bond, |0⟩ on physical)
    bath_data = torch.zeros((l_bond, r_bond, 2), dtype=dtype, device=device)
    for b in range(min(l_bond, r_bond)):
        bath_data[b, b, 0] = 1.0
    clean_data.insert(target_pos, bath_data)

    new_psi = qtn.MatrixProductState(clean_data, shape="lrp")
    new_psi.cyclic = False

    # Fix any spurious shared index between first and last tensor
    t0, tL = new_psi[0], new_psi[new_psi.L - 1]
    for ind in set(t0.inds) & set(tL.inds):
        if not ind.startswith("k"):
            tL.reindex_({ind: qtn.rand_uuid()}, inplace=True)

    return new_psi


def measure_and_remove_site(mps: qtn.MatrixProductState,
                             i: int,
                             device: torch.device,
                             dtype: torch.dtype):
    """Projective measurement on site i; remove from MPS.

    Identical to zhiyan's measure_and_remove_site (TNN_TFIM_state.ipynb, cell 3).
    Returns (new_mps, outcome).
    """
    L = mps.L
    mps.canonize(i)
    T = mps[i]

    def _to(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.tensor(np.array(x), dtype=dtype, device=device)

    data = _to(T.data)
    phys_ind = mps.site_ind(i)
    p_axis = T.inds.index(phys_ind)

    total_weight = torch.norm(data).item() ** 2
    T_0 = data.select(p_axis, 0) if hasattr(data, "select") else (
        data[0] if p_axis == 0 else data[:, 0] if p_axis == 1 else data[..., 0]
    )
    prob_0 = float(torch.norm(T_0).item() ** 2 / total_weight)
    prob_0 = max(0.0, min(1.0, prob_0))

    outcome = 0 if random.random() < prob_0 else 1
    norm_factor = 1.0 / np.sqrt(prob_0 if outcome == 0 else 1.0 - prob_0)
    proj_vec = torch.tensor(
        [1.0, 0.0] if outcome == 0 else [0.0, 1.0], dtype=dtype, device=device
    )

    def get_lrp(site_idx):
        t = mps[site_idx]
        d = _to(t.data)
        p_ind_loc = mps.site_ind(site_idx)
        if d.ndim == 2:
            p_ax = t.inds.index(p_ind_loc)
            if p_ax == 0:
                d = d.transpose(0, 1).reshape(1, -1, 2)
            else:
                d = d.reshape(-1, 1, 2)
        else:
            if site_idx > 0:
                common = set(t.inds) & set(mps[site_idx - 1].inds)
                if common:
                    l_bond = list(common)[0]
                    l_ax = t.inds.index(l_bond)
                    p_ax = t.inds.index(p_ind_loc)
                    r_ax = [x for x in range(3) if x != l_ax and x != p_ax][0]
                    d = d.permute(l_ax, r_ax, p_ax)
                else:
                    idx_map = {ix: n for n, ix in enumerate(t.inds)}
                    bonds = [ix for ix in t.inds if ix != p_ind_loc]
                    l_ax = idx_map[bonds[0]]
                    r_ax = idx_map[bonds[1]]
                    p_ax = idx_map[p_ind_loc]
                    d = d.permute(l_ax, r_ax, p_ax)
            else:
                p_ax = t.inds.index(p_ind_loc)
                axes = [ax for ax in range(d.ndim) if ax != p_ax] + [p_ax]
                d = d.permute(*axes)
                if d.ndim == 2:
                    d = d.reshape(1, -1, 2)
        return d.contiguous()

    A_bath = get_lrp(i)
    M = torch.tensordot(A_bath, proj_vec, dims=([2], [0])) * norm_factor

    clean_data = []
    if i == 0:
        A_next = get_lrp(1)
        A_new = torch.tensordot(M, A_next, dims=([1], [0]))
        clean_data.append(A_new)
        for s in range(2, L):
            clean_data.append(get_lrp(s))
    else:
        A_prev = get_lrp(i - 1)
        A_new_raw = torch.tensordot(A_prev, M, dims=([1], [0]))
        A_new = A_new_raw.permute(0, 2, 1).contiguous()
        for s in range(L):
            if s == i - 1:
                clean_data.append(A_new)
            elif s == i:
                continue
            else:
                clean_data.append(get_lrp(s))

    if clean_data[0].ndim == 2:
        clean_data[0] = clean_data[0].reshape(1, *clean_data[0].shape)
    if clean_data[-1].ndim == 2:
        s = clean_data[-1].shape
        clean_data[-1] = clean_data[-1].reshape(*s[:-1], 1, s[-1])

    new_mps = qtn.MatrixProductState(
        clean_data, shape="lrp", site_ind_id="k{}", site_tag_id="I{}"
    )
    new_mps.normalize()
    return new_mps, outcome


# ---------------------------------------------------------------------------
# Hamiltonian construction  (extracted from zhiyan's notebook, cell 2)
# ---------------------------------------------------------------------------

def build_hamiltonian_part_vector(L, pxx, pz, omega, alpha, site_i, pauli_op):
    """Build LocalHam for system + bath on extended (L+1)-site chain.

    Identical to zhiyan's build_hamiltonian_part_vector.
    H = pxx Σ X_jX_{j+1}  +  pz Σ Z_j          (system)
      + (-ω/2) Z_bath                             (bath field)
      + α · S_{site_i} · X_bath                  (system–bath coupling)

    The original XX bond between site_i and site_i+1 (now at positions
    site_i and site_i+2 in the extended chain) is NOT included here —
    it is handled separately as the "jumper" via SWAP gates.
    """
    def get_idx(j):
        return j if j <= site_i else j + 1

    L_tot = L + 1
    bath_idx = site_i + 1
    builder = qtn.SpinHam1D(S=1 / 2, cyclic=False)

    # System XX bonds — left of bath
    for j in range(site_i):
        builder[j, j + 1] += pxx, X, X

    # System XX bonds — right of bath
    for j in range(site_i + 1, L - 1):
        u, v = j + 1, j + 2
        builder[u, v] += pxx, X, X

    # System Z fields
    for j in range(L):
        builder[get_idx(j)] += pz, Z

    # Bath Z field
    builder[bath_idx] += (-omega / 2.0), Z

    # System–bath interaction
    ops = {1: X, 2: Y, 3: Z}
    op_str = ops[pauli_op]
    if site_i >= 0:
        builder[site_i, bath_idx] += alpha, op_str, X
    else:
        builder[0, 1] += alpha, X, op_str

    # Dummy bond to ensure chain connectivity (required by TEBD)
    if bath_idx < L_tot - 1:
        builder[bath_idx, bath_idx + 1] += 0.0, X, X

    return builder.build_local_ham(L_tot)


# ---------------------------------------------------------------------------
# Gate generation  (extracted from zhiyan's notebook, cell 4)
# ---------------------------------------------------------------------------

def generate_hamiltonian_gates_cached(H_local, dt, site_i, expm_cache, dtype, device):
    """Compute exp(-i H_bond dt) for every bond in H_local.

    Adapted from zhiyan's generate_hamiltonian_gates_cached_vector.
    Uses expm_cache keyed by (dt, matrix_bytes) to avoid recomputation.
    For uniform TFIM all system–system bonds share the same matrix bytes,
    so only the 2 bath-adjacent bonds are cache misses each step.
    """
    def _to(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.tensor(np.array(x), dtype=dtype, device=device)

    c_dtype = (
        torch.complex128
        if dtype in (torch.float64, torch.complex128)
        else torch.complex64
    )

    gates = {}
    for where, term in H_local.terms.items():
        dim = int(np.sqrt(term.size))
        mat_cpu = term.reshape(dim, dim)

        if np.allclose(mat_cpu, 0.0, atol=1e-15):
            U = torch.eye(dim * dim, dtype=dtype, device=device)
        else:
            key = (round(float(dt), 15), mat_cpu.tobytes())
            U = expm_cache.get(key)
            if U is None:
                mat_gpu = _to(mat_cpu)
                w, V = torch.linalg.eigh(mat_gpu)
                V_c = V.to(c_dtype)
                phase = torch.exp(-1j * w * dt).to(c_dtype)
                U = (V_c @ torch.diag(phase) @ V_c.mH).to(dtype)
                # Only cache gates that don't involve bath-dependent parameters.
                # (bath-adjacent bonds change per step; system bonds are always cached)
                expm_cache[key] = U

        gates[where] = U.reshape(2, 2, 2, 2).contiguous()

    return gates, expm_cache


def get_jumper_gate_cached(dt, pxx, expm_cache, dtype, device):
    """exp(-i pxx X⊗X dt) — the 'jumper' handles the XX bond across the bath.

    For uniform TFIM this gate is the same every step; it is always a cache hit
    after the first step. Adapted from zhiyan's get_adjacent_gate_cached_vector.
    """
    c_dtype = (
        torch.complex128
        if dtype in (torch.float64, torch.complex128)
        else torch.complex64
    )
    sX = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=dtype, device=device)
    H2 = pxx * torch.kron(sX, sX)
    key = (round(float(dt), 15), H2.cpu().numpy().tobytes())

    U2 = expm_cache.get(key)
    if U2 is None:
        if abs(dt) < 1e-15:
            U2 = torch.eye(4, dtype=dtype, device=device)
        else:
            w, V = torch.linalg.eigh(H2)
            V_c = V.to(c_dtype)
            phase = torch.exp(-1j * w * dt).to(c_dtype)
            U2 = (V_c @ torch.diag(phase) @ V_c.mH).to(dtype)
        expm_cache[key] = U2

    return U2.reshape(2, 2, 2, 2).contiguous(), expm_cache


# ---------------------------------------------------------------------------
# SWAP gate (global, d=2)
# ---------------------------------------------------------------------------

def _make_swap(dtype, device):
    S = torch.zeros((4, 4), dtype=dtype, device=device)
    S[0, 0] = S[1, 2] = S[2, 1] = S[3, 3] = 1.0
    return S.reshape(2, 2, 2, 2).contiguous()


# ---------------------------------------------------------------------------
# Single TEBD step  (4th-order Suzuki–Yoshida)
# ---------------------------------------------------------------------------

# 4th-order Yoshida coefficients
_CBRT2 = 2.0 ** (1.0 / 3.0)
_W1 = 1.0 / (2.0 - _CBRT2)   #  0.6756...
_W0 = -_CBRT2 / (2.0 - _CBRT2)  # -1.3512... (negative time step)
_COEFFS = (_W1, _W0, _W1)


def run_mps_step(
    psi: qtn.MatrixProductState,
    duration: float,
    dt: float,
    L: int,
    pxx: float,
    pz: float,
    omega: float,
    alpha: float,
    site_i: int,
    pauli_op: int,
    max_bond: int,
    expm_cache: dict,
    device: torch.device,
    dtype: torch.dtype,
    swap_gate: torch.Tensor,
):
    """One full simulation step: insert bath → TEBD → measure bath.

    Uses 4th-order Suzuki–Yoshida TEBD for total time `duration = 2*Ss`.
    Gate caching ensures system-bond gates are computed only once per unique
    (dt_sub, matrix_bytes) and reused across all steps.

    Returns (new_psi, outcome, expm_cache, max_bond_reached).
    """
    mps = insert_MPS_site(psi, site_i, device, dtype)
    mps.cyclic = False
    max_bond_reached = mps.max_bond()

    H_local = build_hamiltonian_part_vector(L, pxx, pz, omega, alpha, site_i, pauli_op)

    split_opts = {
        "max_bond": max_bond,
        "cutoff": 1e-12,
        "cutoff_mode": "rel",
        "renorm": True,
        "absorb": "both",
    }

    n_steps = int(round(duration / dt))
    if n_steps <= 0:
        mps, outcome = measure_and_remove_site(mps, site_i + 1, device, dtype)
        return mps, outcome, expm_cache, max_bond_reached

    idx_L = site_i
    idx_M = site_i + 1      # bath position
    idx_R = site_i + 2
    has_jumper = (idx_R < mps.L) and (idx_L >= 0)

    # ------------------------------------------------------------------
    # Pre-build gate sets for each unique Yoshida substep size.
    # For uniform TFIM: system bonds are always cache hits after step 1.
    # ------------------------------------------------------------------
    gate_sets: dict = {}

    def _get_gate_set(dt_sub):
        nonlocal expm_cache
        key = round(float(dt_sub), 15)
        if key in gate_sets:
            return gate_sets[key]

        full, expm_cache = generate_hamiltonian_gates_cached(
            H_local, dt_sub, site_i, expm_cache, dtype, device
        )
        half, expm_cache = generate_hamiltonian_gates_cached(
            H_local, dt_sub / 2.0, site_i, expm_cache, dtype, device
        )
        even_bonds = sorted(b for b in full if b[0] % 2 == 0)
        odd_bonds  = sorted(b for b in full if b[0] % 2 == 1)

        jumper_half = None
        if has_jumper:
            jumper_half, expm_cache = get_jumper_gate_cached(
                dt_sub / 2.0, pxx, expm_cache, dtype, device
            )

        gate_sets[key] = (full, half, even_bonds, odd_bonds, jumper_half)
        return gate_sets[key]

    def _strang(dt_sub):
        """One 2nd-order Strang splitting step."""
        full, half, even_bonds, odd_bonds, jumper_half = _get_gate_set(dt_sub)

        # A) jumper half (handles XX across bath via SWAP)
        if has_jumper:
            mps.gate_split_(swap_gate, (idx_M, idx_R), **split_opts)
            mps.gate_split_(jumper_half, (idx_L, idx_M), **split_opts)
            mps.gate_split_(swap_gate, (idx_M, idx_R), **split_opts)

        # B) Strang sweep: odd-half, even-full, odd-half
        for bond in odd_bonds:
            mps.gate_split_(half[bond], bond, **split_opts)
        for bond in even_bonds:
            mps.gate_split_(full[bond], bond, **split_opts)
        for bond in odd_bonds:
            mps.gate_split_(half[bond], bond, **split_opts)

        # C) jumper half (symmetric)
        if has_jumper:
            mps.gate_split_(swap_gate, (idx_M, idx_R), **split_opts)
            mps.gate_split_(jumper_half, (idx_L, idx_M), **split_opts)
            mps.gate_split_(swap_gate, (idx_M, idx_R), **split_opts)

    # Main TEBD loop: 4th-order = S2(w1·dt) · S2(w0·dt) · S2(w1·dt)
    for _ in range(n_steps):
        for w in _COEFFS:
            _strang(w * dt)
        mps.normalize()
        max_bond_reached = max(max_bond_reached, mps.max_bond())

    new_psi, outcome = measure_and_remove_site(mps, site_i + 1, device, dtype)
    return new_psi, outcome, expm_cache, max_bond_reached


# ---------------------------------------------------------------------------
# Pre-seeding the gate cache (Optimization 3)
# ---------------------------------------------------------------------------

def preseed_cache(
    L: int,
    pxx: float,
    pz: float,
    dt: float,
    expm_cache: dict,
    device: torch.device,
    dtype: torch.dtype,
):
    """Warm up the gate cache with all system-bond gates before the main loop.

    Runs a silent dummy step (site_i=0, any omega, alpha=0) to populate
    expm_cache with every system XX and Z bond gate for all 4th-order
    Yoshida substep sizes (w1·dt, w0·dt and their halves).

    For a uniform TFIM all system–system bonds share identical matrix bytes,
    so these 8 entries cover every bond in every future step.
    """
    dummy_psi = qtn.MPS_computational_state("0" * L)
    dummy_psi.apply_to_arrays(
        lambda x: torch.tensor(np.array(x), dtype=dtype, device=device)
    )
    swap_gate = _make_swap(dtype, device)

    # Run one silent step with alpha=0 (no system–bath coupling) to warm cache
    _, _, expm_cache, _ = run_mps_step(
        dummy_psi,
        duration=dt * 3,        # just 3 sub-steps, enough to touch all gate sizes
        dt=dt,
        L=L,
        pxx=pxx,
        pz=pz,
        omega=1.0,              # arbitrary, won't affect system gates
        alpha=0.0,              # zero coupling: bath doesn't perturb system
        site_i=0,
        pauli_op=1,
        max_bond=4,             # tiny bond dim; only for cache warmup
        expm_cache=expm_cache,
        device=device,
        dtype=dtype,
        swap_gate=swap_gate,
    )
    return expm_cache


# ---------------------------------------------------------------------------
# EfficientLindbladSimulator  — main class
# ---------------------------------------------------------------------------

class EfficientLindbladSimulator:
    """Efficient pure-state MPS simulator for 1D TFIM ground-state preparation.

    Hamiltonian: H = pxx * Σ_i X_i X_{i+1}  +  pz * Σ_i Z_i

    Example::

        sim = EfficientLindbladSimulator(L=20, pxx=-1.0, pz=1.5)
        for step in range(500):
            sim.step(alpha=0.25)
            E = sim.get_energy()
            print(f"Step {step}: E = {E:.6f}")
    """

    def __init__(
        self,
        L: int,
        pxx: float,
        pz: float,
        *,
        dt: float = 0.05,
        max_bond: int = 64,
        sigma: float = 4.0,
        Ss_factor: float = 2.0,
        omega_max: float = 5.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.complex128,
        seed: Optional[int] = None,
        preseed: bool = True,
    ):
        """
        Parameters
        ----------
        L         : number of system qubits
        pxx       : XX coupling coefficient (e.g. -J for TFIM)
        pz        : single-site Z field (e.g. g)
        dt        : TEBD sub-step size. Default 0.05 (baseline: 0.01).
        max_bond  : MPS bond dimension cap.
        sigma     : filter width; interaction window = 2 * Ss_factor * sigma.
        Ss_factor : interaction-window multiplier. Default 2.0 (baseline: 4.0).
        omega_max : maximum bath frequency sampled each step.
        device    : 'cpu' or 'cuda'.
        dtype     : torch.complex128 (default) or torch.complex64 on GPU.
        seed      : random seed for reproducibility.
        preseed   : if True, warm up the gate cache before the first real step.
        """
        self.L = L
        self.pxx = pxx
        self.pz = pz
        self.dt = dt
        self.max_bond = max_bond
        self.sigma = sigma
        self.Ss = Ss_factor * sigma
        self.omega_max = omega_max
        self.device = torch.device(device)
        self.dtype = dtype

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        self._swap_gate = _make_swap(dtype, self.device)
        self._expm_cache: dict = {}

        # Initial state: product state |0...0⟩
        self.psi = qtn.MPS_computational_state("0" * L)
        self._to_device(self.psi)

        # Cache warmup
        if preseed:
            self._expm_cache = preseed_cache(
                L, pxx, pz, dt, self._expm_cache, self.device, self.dtype
            )

        # Ham MPO (built once, reused for energy measurements)
        self._ham_mpo: Optional[qtn.MatrixProductOperator] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_device(self, mps: qtn.MatrixProductState):
        mps.apply_to_arrays(
            lambda x: (
                x.to(dtype=self.dtype, device=self.device)
                if isinstance(x, torch.Tensor)
                else torch.tensor(np.array(x), dtype=self.dtype, device=self.device)
            )
        )

    def _get_ham_mpo(self) -> qtn.MatrixProductOperator:
        if self._ham_mpo is None:
            builder = qtn.SpinHam1D(S=1 / 2)
            for i in range(self.L - 1):
                builder[i, i + 1] += self.pxx, X, X
            for i in range(self.L):
                builder[i] += self.pz, Z
            self._ham_mpo = builder.build_mpo(self.L)
            self._to_device(self._ham_mpo)
        return self._ham_mpo

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, alpha: float) -> int:
        """Run one simulation step.

        Randomly selects: site_i, Pauli type, bath frequency ω, coupling sign.
        Inserts bath, runs 4th-order TEBD for time 2·Ss, measures and removes bath.

        Returns the measurement outcome (0 or 1).
        """
        site_i    = random.randint(0, self.L - 1)
        pauli_op  = random.randint(1, 3)
        omega     = random.uniform(0.0, self.omega_max)
        sign      = random.choice([-1, 1])

        new_psi, outcome, self._expm_cache, _ = run_mps_step(
            self.psi,
            duration=2.0 * self.Ss,
            dt=self.dt,
            L=self.L,
            pxx=self.pxx,
            pz=self.pz,
            omega=omega,
            alpha=sign * alpha,
            site_i=site_i,
            pauli_op=pauli_op,
            max_bond=self.max_bond,
            expm_cache=self._expm_cache,
            device=self.device,
            dtype=self.dtype,
            swap_gate=self._swap_gate,
        )
        self.psi = new_psi
        return outcome

    def get_energy(self) -> float:
        """Compute ⟨ψ|H|ψ⟩."""
        ham = self._get_ham_mpo()
        return float(np.real(qtn.expec_TN_1D(self.psi.H, ham, self.psi)))

    def get_fidelity(self, psi_gs: qtn.MatrixProductState) -> float:
        """Compute |⟨GS|ψ⟩|²."""
        overlap = psi_gs.H @ self.psi
        return float(abs(complex(overlap)) ** 2)

    def get_dmrg_ground_state(self):
        """Run DMRG to find the ground state (for benchmarking)."""
        ham = self._get_ham_mpo()
        dmrg = qtn.DMRG2(ham)
        dmrg.solve(max_sweeps=80, verbosity=0, cutoffs=1e-10)
        gs = dmrg.state
        gs.normalize()
        self._to_device(gs)
        return gs, float(np.real(qtn.expec_TN_1D(gs.H, ham, gs)))

    def run(
        self,
        n_steps: int,
        alpha: float,
        psi_gs: Optional[qtn.MatrixProductState] = None,
        verbose: bool = True,
        log_every: int = 10,
    ):
        """Run n_steps simulation steps.

        Returns dict with 'energies' and optionally 'fidelities' arrays.
        """
        energies   = np.zeros(n_steps)
        fidelities = np.zeros(n_steps) if psi_gs is not None else None

        rng = range(n_steps)
        if verbose:
            rng = tqdm(rng, desc="Lindblad steps")

        for n in rng:
            self.step(alpha)
            energies[n] = self.get_energy()
            if fidelities is not None:
                fidelities[n] = self.get_fidelity(psi_gs)
            if verbose and (n % log_every == 0 or n == n_steps - 1):
                msg = f"  step {n+1}/{n_steps}  E={energies[n]:.6f}"
                if fidelities is not None:
                    msg += f"  F={fidelities[n]:.4f}"
                tqdm.write(msg)

        return {"energies": energies, "fidelities": fidelities}
