import numpy as np
import scipy.linalg as la
from lindblad_exact import build_hamiltonian, surrogate_jumps, build_liouvillian, _unvec

# Does E_NESS depend on gamma for the single-site surrogate?
for L in (4, 6):
    H = build_hamiltonian(L, -1.0, 1.5)
    n = 2 ** L
    print(f"L={L}: E_GS={np.min(la.eigvalsh(H)):.5f}")
    for gamma in (0.05, 0.1, 0.2, 0.5, 1.0):
        Lsup = build_liouvillian(H, surrogate_jumps(L, gamma))
        w, V = la.eig(Lsup)
        k = np.argmin(np.abs(w))
        rho = _unvec(V[:, k], n); rho /= np.trace(rho)
        rho = 0.5*(rho+rho.conj().T)
        e = np.real(np.trace(H @ rho))
        # also report the steady state's diagonal populations on site 0
        print(f"  gamma={gamma:.2f}: E_NESS={e:.6f}  (|lam0|={abs(w[k]):.1e})")
    # Is rho_ss = |1..1><1..1| exactly? check overlap
    Lsup = build_liouvillian(H, surrogate_jumps(L, 0.1))
    w, V = la.eig(Lsup); k = np.argmin(np.abs(w))
    rho = _unvec(V[:, k], n); rho /= np.trace(rho)
    pop_11 = np.real(rho[-1, -1])   # |1...1> is last basis state
    print(f"  rho_ss population on |1...1> = {pop_11:.6f}")
