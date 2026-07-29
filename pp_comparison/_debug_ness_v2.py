import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from lindblad_exact import (build_hamiltonian, surrogate_jumps,
                            build_liouvillian, build_liouvillian_sparse,
                            _unvec, _vec)

L = 6; gamma = 0.1
H = build_hamiltonian(L, -1.0, 1.5); n = 2 ** L
Lsup = build_liouvillian(H, surrogate_jumps(L, gamma))

# (A) Full eigendecomposition: kernel = eigenvector of eigenvalue ~0
w, V = la.eig(Lsup)
order = np.argsort(np.abs(w))
print("5 smallest |eig| of L:", np.round(np.abs(w[order[:5]]), 6))
for k in range(3):
    rho = _unvec(V[:, order[k]], n)
    tr = np.trace(rho)
    if abs(tr) > 1e-8:
        rho = rho / tr; rho = 0.5*(rho+rho.conj().T)
        e = np.real(np.trace(H @ rho))
        # check it is actually a fixed point: ||L vec(rho)||
        res = np.linalg.norm(Lsup @ _vec(rho))
        print(f"  eig#{k}: lam={w[order[k]]:.3e}  <H>={e:.5f}  ||L rho||={res:.2e}")

# (B) Time-evolve to large T with the SAME tau as benchmark and a small-tau check
for tau in (0.1, 0.02):
    step = la.expm(Lsup * tau)
    v = _vec(np.diag([1.0]+[0.0]*(n-1)).astype(complex))
    nsteps = int(round(600/tau))
    for _ in range(nsteps):
        v = step @ v
    rho = _unvec(v, n)
    print(f"  evolve tau={tau} to T=600: <H>={np.real(np.trace(H@rho)):.5f} tr={np.real(np.trace(rho)):.6f}")
