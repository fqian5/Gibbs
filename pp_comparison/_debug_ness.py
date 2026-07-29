import numpy as np
import scipy.linalg as la
from lindblad_exact import build_hamiltonian, surrogate_jumps, build_liouvillian, _unvec, _vec

L = 6; gamma = 0.1
H = build_hamiltonian(L, -1.0, 1.5)
n = 2 ** L
Ks = surrogate_jumps(L, gamma)
Lsup = build_liouvillian(H, Ks)

# eigen-decomposition of L: steady state = eigenvector with eigenvalue ~0
w, V = la.eig(Lsup)
order = np.argsort(np.abs(w))
print("smallest |eigenvalues| of L:", np.abs(w[order[:5]]))
for k in range(3):
    rho = _unvec(V[:, order[k]], n)
    tr = np.trace(rho)
    if abs(tr) > 1e-9:
        rho = rho / tr
        rho = 0.5 * (rho + rho.conj().T)
        e = np.real(np.trace(H @ rho))
        herm = np.linalg.norm(rho - rho.conj().T)
        print(f"  eig#{k} lam={w[order[k]]:.2e} trace={tr:.3e} <H>={e:.5f}")

# Long-time evolution to true steady state
step = la.expm(Lsup * 1.0)
rho = np.zeros((n, n), dtype=complex); rho[0, 0] = 1.0
v = _vec(rho)
for _ in range(300):  # T=300, deep into steady state
    v = step @ v
rho = _unvec(v, n)
print("evolved to T=300: <H> =", np.real(np.trace(H @ rho)), " trace =", np.real(np.trace(rho)))
