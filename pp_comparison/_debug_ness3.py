import numpy as np, scipy.linalg as la
from lindblad_exact import build_hamiltonian, surrogate_jumps, build_liouvillian, _unvec
L = 4
H = build_hamiltonian(L, -1.0, 1.5); n = 2 ** L
print('L=4 E_GS=%.5f' % np.min(la.eigvalsh(H)))
for gamma in (0.05, 0.1, 0.2, 0.5, 1.0):
    Lsup = build_liouvillian(H, surrogate_jumps(L, gamma))
    w, V = la.eig(Lsup); k = np.argmin(np.abs(w))
    rho = _unvec(V[:, k], n); rho /= np.trace(rho); rho = 0.5 * (rho + rho.conj().T)
    e = np.real(np.trace(H @ rho)); pop = np.real(rho[-1, -1])
    print('  gamma=%.2f E_NESS=%.6f pop|1..1>=%.6f' % (gamma, e, pop))
