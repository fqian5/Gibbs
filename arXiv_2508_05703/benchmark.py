import time
import numpy as np
import quimb as qu
from simulation import RandomizedLindbladSimulator

def exact_energy(L, J, g):
    dims = [2] * L
    H = 0
    # SpinHam1D(S=0.5) uses spin operators. We use Pauli.
    # So we used scaled coeffs in MPO: 2g, 4J.
    # Here we build directly with Pauli.
    # Z terms: -g Z
    for i in range(L):
        H += -g * qu.ikron(qu.pauli('Z'), dims, [i])
    # XX terms: -J XX
    for i in range(L - 1):
        H += -J * qu.ikron(qu.pauli('X') & qu.pauli('X'), dims, [i, i+1])
    return qu.eigvalsh(H, k=1)[0]

def run_benchmark(L_list, max_bond_list, steps=10):
    print(f"{'L':<5} {'Max Bond':<10} {'Steps':<8} {'Time (s)':<10} {'Time/Step (s)':<15} {'Exact E':<10} {'Sim E':<10} {'Rel Err':<10}")
    print("-" * 100)

    results = []

    for L in L_list:
        # Calculate exact energy once per L
        try:
            E_exact = exact_energy(L, 1.0, 1.5)
        except Exception:
            E_exact = float('nan') # Too large for ED

        for max_bond in max_bond_list:
            # Setup simulation
            J = 1.0
            g = 1.5
            dt = 0.1
            sigma = 2.0
            T = 5.0

            sim = RandomizedLindbladSimulator(L, J, g, dt=dt, max_bond=max_bond, beta=float('inf'),
                                              filter_sigma=sigma, filter_T=T)

            start_time = time.time()
            for _ in range(steps):
                sim.step(alpha=1.0)
            end_time = time.time()

            total_time = end_time - start_time
            avg_time = total_time / steps

            E_sim = sim.get_energy()

            if not np.isnan(E_exact):
                rel_err = abs(E_sim - E_exact) / abs(E_exact)
            else:
                rel_err = float('nan')

            print(f"{L:<5} {max_bond:<10} {steps:<8} {total_time:<10.4f} {avg_time:<15.4f} {E_exact:<10.4f} {E_sim:<10.4f} {rel_err:<10.4e}")
            results.append({
                'L': L,
                'max_bond': max_bond,
                'steps': steps,
                'total_time': total_time,
                'avg_time': avg_time,
                'E_sim': E_sim,
                'E_exact': E_exact,
                'rel_err': rel_err
            })

    return results

if __name__ == "__main__":
    print("Benchmarking Randomized Lindblad Simulator...")
    # System sizes
    Ls = [6, 8]
    # Bond dimensions
    bonds = [16, 32]

    run_benchmark(Ls, bonds, steps=2)
