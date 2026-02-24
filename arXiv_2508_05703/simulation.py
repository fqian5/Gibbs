import numpy as np
import quimb as qu
import quimb.tensor as qtn
import random

def manual_swap(mps, i, j, max_bond=None):
    t1 = mps[i]
    t2 = mps[j]

    if not isinstance(t1, qtn.Tensor) or not isinstance(t2, qtn.Tensor):
        # Clean up if tuple?
        # But this implies previous step failed.
        # Try to recover by selecting one? No, safer to fail.
        raise ValueError(f"manual_swap: mps[{i}] or mps[{j}] is not a Tensor. Types: {type(t1)}, {type(t2)}")

    k1 = mps.site_ind(i)
    k2 = mps.site_ind(j)

    T = qtn.tensor_contract(t1, t2)
    left_bonds = [ind for ind in t1.inds if ind not in t2.inds and ind != k1]

    try:
        U, V = T.split(left_inds=left_bonds + [k2], get='tensors', absorb='right', max_bond=max_bond)
    except Exception as e:
        raise ValueError(f"Split failed in swap {i}<->{j}: {e}")

    U.reindex_({k2: k1}, inplace=True)
    V.reindex_({k1: k2}, inplace=True)

    tag1 = f'I{i}'
    tag2 = f'I{j}'

    U.drop_tags()
    U.add_tag(tag1)
    V.drop_tags()
    V.add_tag(tag2)

    mps.delete(tag1)
    mps.delete(tag2)
    mps.add_tensor(U)
    mps.add_tensor(V)

def manual_gate(mps, gate, i, j, max_bond=None):
    t1 = mps[i]
    t2 = mps[j]

    if not isinstance(t1, qtn.Tensor) or not isinstance(t2, qtn.Tensor):
        raise ValueError(f"manual_gate: mps[{i}] or mps[{j}] is not a Tensor. Types: {type(t1)}, {type(t2)}")

    k1 = mps.site_ind(i)
    k2 = mps.site_ind(j)

    T = qtn.tensor_contract(t1, t2)

    d = 2
    if gate.ndim == 2:
        gate_data = gate.reshape(d, d, d, d)
    else:
        gate_data = gate

    k1_out = qtn.rand_uuid()
    k2_out = qtn.rand_uuid()

    G = qtn.Tensor(data=gate_data, inds=(k1_out, k2_out, k1, k2), tags={'GATE'})
    T_gated = qtn.tensor_contract(T, G)
    T_gated.reindex_({k1_out: k1, k2_out: k2}, inplace=True)

    left_bonds = [ind for ind in t1.inds if ind not in t2.inds and ind != k1]

    U, V = T_gated.split(left_inds=left_bonds + [k1], get='tensors', absorb='right', max_bond=max_bond)

    tag1 = f'I{i}'
    tag2 = f'I{j}'

    U.drop_tags()
    U.add_tag(tag1)
    V.drop_tags()
    V.add_tag(tag2)

    mps.delete(tag1)
    mps.delete(tag2)
    mps.add_tensor(U)
    mps.add_tensor(V)

def manual_single_site_gate(mps, gate, i):
    t = mps[i]
    if not isinstance(t, qtn.Tensor):
        raise ValueError(f"manual_single_site_gate: mps[{i}] is not a Tensor. Type: {type(t)}")

    k = mps.site_ind(i)

    # Gate G: (k_out, k_in)
    # T: (..., k, ...)
    # Contract k with k_in.

    d = 2
    gate_data = gate if gate.ndim == 2 else gate.reshape(d, d)

    k_out = qtn.rand_uuid()
    G = qtn.Tensor(data=gate_data, inds=(k_out, k), tags={'GATE'})

    T_gated = qtn.tensor_contract(t, G)
    T_gated.reindex_({k_out: k}, inplace=True)

    # Update tensor in MPS
    # Just replace it.
    tag = f'I{i}'
    T_gated.drop_tags()
    T_gated.add_tag(tag)

    mps.delete(tag)
    mps.add_tensor(T_gated)

class RandomizedLindbladSimulator:
    def __init__(self, L, J, g, dt=0.1, max_bond=32, beta=float('inf'),
                 filter_sigma=1.0, filter_T=None):
        self.L = L
        self.J = J
        self.g = g
        self.dt = dt
        self.max_bond = max_bond
        self.beta = beta
        self.sigma = filter_sigma
        self.T = filter_T if filter_T is not None else 5 * self.sigma

        # Initialize system state (all |0>)
        self.psi = qtn.MPS_computational_state('0' * L)
        self.psi.compress_opts = {'max_bond': max_bond}

        # Operators
        self.X = qu.pauli('X')
        self.Y = qu.pauli('Y')
        self.Z = qu.pauli('Z')
        self.I = qu.pauli('I')
        self.B_E = np.array([[0, 0], [1, 0]], dtype=complex)
        self.B_E_dag = self.B_E.conj().T

    def get_system_hamiltonian_mpo(self):
        # We want H in terms of Pauli matrices (eigenvalues +/- 1).
        # SpinHam1D(S=0.5) uses spin operators (eigenvalues +/- 0.5).
        # sigma_z = 2 * S_z. sigma_x = 2 * S_x.
        # So we scale coefficients: g -> 2g, J -> 4J.
        builder = qtn.SpinHam1D(S=0.5)
        for i in range(self.L):
            builder[i] += -2.0 * self.g, 'Z'
        for i in range(self.L - 1):
            builder[i, i+1] += -4.0 * self.J, 'X', 'X'
        return builder.build_mpo(self.L)

    def apply_system_evolution(self, mps, dt):
        # Single site Z terms: -g Z
        U_Z_half = qu.expm(-1j * (-self.g * self.Z) * dt / 2)
        for i in range(self.L):
            manual_single_site_gate(mps, U_Z_half, i)

        # Two site XX terms: -J XX
        U_XX = qu.expm(-1j * (-self.J * (self.X & self.X)) * dt)

        for i in range(0, self.L - 1, 2):
            manual_gate(mps, U_XX, i, i+1, max_bond=self.max_bond)

        for i in range(1, self.L - 1, 2):
            manual_gate(mps, U_XX, i, i+1, max_bond=self.max_bond)

        for i in range(self.L):
            manual_single_site_gate(mps, U_Z_half, i)

    def gaussian_f(self, t):
        return (1.0 / ((2 * np.pi)**0.25 * self.sigma**0.5)) * np.exp(-t**2 / (4 * self.sigma**2))

    def step(self, alpha, omega_max=None):
        target_site = random.randint(0, self.L - 1)
        op_type = random.choice(['X', 'Z'])
        A_S = self.X if op_type == 'X' else self.Z

        if omega_max is None:
            omega_max = 5.0 * (abs(self.J) + abs(self.g))
        omega = random.uniform(-omega_max, omega_max)

        ancilla_state_idx = 0
        E0, E1 = -0.5 * omega, 0.5 * omega

        if self.beta == float('inf'):
            # Ground state: pick state with lower energy
            if E1 < E0:
                ancilla_state_idx = 1
            else:
                ancilla_state_idx = 0
        else:
            w0, w1 = np.exp(-self.beta * E0), np.exp(-self.beta * E1)
            p1 = w1 / (w0 + w1) if (w0 + w1) > 0 else (0.0 if E1 > E0 else 1.0)
            if random.random() < p1:
                ancilla_state_idx = 1

        sys_arrays = [t.data for t in self.psi]

        # Ensure Site 0 is 2D (Right, Phys) for standard MPS inference
        if sys_arrays[0].ndim == 3 and sys_arrays[0].shape[0] == 1:
             # (1, Right, Phys) -> (Right, Phys)
             sys_arrays[0] = sys_arrays[0].reshape(sys_arrays[0].shape[1], sys_arrays[0].shape[2])

        # Ensure Last System Site (L-1) is 3D (Left, Right, Phys)
        last = sys_arrays.pop()
        if last.ndim == 2: # (Left, Phys) -> (Left, 1, Phys)
            last = last.reshape(last.shape[0], 1, last.shape[1])
        elif last.ndim == 3 and last.shape[1] == 1:
            # Already (Left, 1, Phys)
            pass
        # Note: If last was (Left, 1, Phys) but it was the end of chain, it technically acted as (Left, Phys).
        # We need it to connect to Ancilla.
        # Its Right dimension (dim 1) must match Ancilla Left.
        sys_arrays.append(last)

        if ancilla_state_idx == 0:
            ancilla_data = np.array([1., 0.], dtype=complex)
        else:
            ancilla_data = np.array([0., 1.], dtype=complex)
        # Ancilla acts as boundary (Left, Phys).
        # Left dim must match 'last' Right dim (which is 1).
        ancilla_tensor = ancilla_data.reshape(1, 2)

        combined_arrays = sys_arrays + [ancilla_tensor]

        # Let quimb infer shapes.
        # T0: (R, P). T_mid: (L, R, P). T_last: (L, P).
        phi = qtn.MatrixProductState(combined_arrays)
        phi.compress_opts = {'max_bond': self.max_bond}

        M = int(np.ceil(2 * self.T / self.dt))

        H_int_mat = np.kron(A_S, self.B_E) + np.kron(A_S.conj().T, self.B_E_dag)

        for i in range(self.L - 1, target_site, -1):
            manual_swap(phi, i, i+1, max_bond=self.max_bond)

        anc_pos = target_site + 1

        for m in range(M):
            t = (m + 0.5) * self.dt - self.T

            f_val = self.gaussian_f(t)
            coeff = alpha * f_val
            U_int = qu.expm(-1j * coeff * H_int_mat * self.dt / 2)

            manual_gate(phi, U_int, target_site, anc_pos, max_bond=self.max_bond)

            # Swap back to L
            for i in range(anc_pos, self.L):
                manual_swap(phi, i, i+1, max_bond=self.max_bond)

            self.apply_system_evolution(phi, self.dt)

            # Env Evolution
            U_E = qu.expm(-1j * (-0.5 * omega * self.Z) * self.dt)
            manual_single_site_gate(phi, U_E, self.L)

            # Swap Ancilla back
            for i in range(self.L - 1, target_site, -1):
                manual_swap(phi, i, i+1, max_bond=self.max_bond)

            manual_gate(phi, U_int, target_site, anc_pos, max_bond=self.max_bond)

        for i in range(anc_pos, self.L):
            manual_swap(phi, i, i+1, max_bond=self.max_bond)

        phi.canonicalize_(self.L)
        T = phi[self.L].data

        if T.ndim == 2:
             p0 = np.sum(np.abs(T[:, 0])**2)
             p1 = np.sum(np.abs(T[:, 1])**2)
             norm = p0 + p1

             if random.random() < p0 / norm:
                 outcome = 0
                 projected_vec = T[:, 0] / np.sqrt(p0)
             else:
                 outcome = 1
                 projected_vec = T[:, 1] / np.sqrt(p1)
        else:
             T_flat = T.reshape(-1, 2)
             p0 = np.sum(np.abs(T_flat[:, 0])**2)
             p1 = np.sum(np.abs(T_flat[:, 1])**2)
             norm = p0 + p1
             if random.random() < p0 / norm:
                 outcome = 0
                 projected_vec = T[..., 0] / np.sqrt(p0)
             else:
                 outcome = 1
                 projected_vec = T[..., 1] / np.sqrt(p1)

        # Extract arrays ensuring (Left, Right, Phys) order
        new_arrays = []
        for i in range(self.L + 1):
            t = phi[i]
            # Identify indices
            phys_ind = f'k{i}' # Assuming k{i} tags are maintained?
            # Wait, manual_swap maintains k indices?
            # Yes: U.reindex_({k2: k1}).
            # So site i always has k{i}.

            try:
                phys_axis = t.inds.index(phys_ind)
            except ValueError:
                # Fallback: find index starting with k?
                # or assume last?
                print(f"Warning: k{i} not found in site {i}. Inds: {t.inds}")
                phys_axis = -1

            # If standard internal: 3 inds.
            if t.ndim == 3:
                # (Left, Right, Phys) desired.
                # If Phys is last, check bonds.
                # But we don't know bond names easily without neighbors.
                # However, MatrixProductState(arrays) creates NEW bonds.
                # It just needs consistency.
                # If we ensure Phys is last, quimb usually handles L, R.

                # Transpose phys to last
                if phys_axis != 2:
                    perm = [x for x in range(3) if x != phys_axis] + [phys_axis]
                    new_inds = [t.inds[i] for i in perm]
                    t.transpose_(*new_inds)
            elif t.ndim == 2:
                # (Bond, Phys)
                if phys_axis != 1:
                    perm = [x for x in range(2) if x != phys_axis] + [phys_axis]
                    new_inds = [t.inds[i] for i in perm]
                    t.transpose_(*new_inds)

            new_arrays.append(t.data)

        # Now remove ancilla and update
        ancilla_data = new_arrays.pop() # T_L
        # Ancilla data is likely (Bond, Phys) -> (Bond, 2)

        last_sys_data = new_arrays.pop() # T_L-1
        # (Left, Right, Phys) -> (Left, Bond, Phys)

        # Project
        # projected_vec from ancilla_data
        # p0, p1 calculation
        # shape is (Bond, 2)
        T = ancilla_data
        p0 = np.sum(np.abs(T[:, 0])**2)
        p1 = np.sum(np.abs(T[:, 1])**2)
        norm = p0 + p1
        if random.random() < p0 / norm:
             projected_vec = T[:, 0] / np.sqrt(p0)
        else:
             projected_vec = T[:, 1] / np.sqrt(p1)

        # Contract last_sys_data (Left, Bond, Phys) with projected_vec (Bond)
        # Contract axis 1.
        new_last_tensor = np.tensordot(last_sys_data, projected_vec, axes=([1], [0]))
        # Result (Left, Phys).

        new_arrays.append(new_last_tensor)

        # Reconstruct MPS with shape hints
        # First site might be (Right, Phys).
        # Mid sites (Left, Right, Phys).
        # Last site (Left, Phys).
        # Arrays might be (1, R, P) if we kept dummy?
        # But we extracted from phi.
        # If phi had dummy bonds, they are in data.

        # Force strict shapes for constructor
        # 0: (1, R, P)
        # i: (L, R, P)
        # L-1: (L, 1, P)

        # Check Site 0
        if new_arrays[0].ndim == 2: # (R, P)
             new_arrays[0] = new_arrays[0].reshape(1, *new_arrays[0].shape)

        # Check Last Site
        if new_arrays[-1].ndim == 2: # (L, P)
             new_arrays[-1] = new_arrays[-1].reshape(*new_arrays[-1].shape[:-1], 1, new_arrays[-1].shape[-1])

        self.psi = qtn.MatrixProductState(new_arrays, shape='lrp')
        # Normalize
        n = self.psi.norm()
        if n > 1e-12:
            self.psi /= n

    def get_energy(self):
        ham = self.get_system_hamiltonian_mpo()
        return np.real(qtn.expec_TN_1D(self.psi.H, ham, self.psi))
