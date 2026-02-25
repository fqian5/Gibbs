import time
import numpy as np
import quimb as qu
from simulation import RandomizedLindbladSimulator

def run_benchmark(L_list, max_bond_list, steps=10):
    print(f"{'L':<5} {'Max Bond':<10} {'Steps':<8} {'Time (s)':<10} {'Time/Step (s)':<15}")
    print("-" * 60)

    results = []

    for L in L_list:
        for max_bond in max_bond_list:
            # Setup simulation
            J = 1.0
            g = 1.5
            dt = 0.1
            sigma = 2.0
            T = 5.0

            sim = RandomizedLindbladSimulator(L, J, g, dt=dt, max_bond=max_bond, beta=float('inf'),
                                              filter_sigma=sigma, filter_T=T)

            # Warmup
            # sim.step(alpha=1.0)

            start_time = time.time()
            for _ in range(steps):
                sim.step(alpha=1.0)
            end_time = time.time()

            total_time = end_time - start_time
            avg_time = total_time / steps

            print(f"{L:<5} {max_bond:<10} {steps:<8} {total_time:<10.4f} {avg_time:<15.4f}")
            results.append({
                'L': L,
                'max_bond': max_bond,
                'steps': steps,
                'total_time': total_time,
                'avg_time': avg_time
            })

    return results

if __name__ == "__main__":
    print("Benchmarking Randomized Lindblad Simulator...")
    # System sizes
    Ls = [6, 8]
    # Bond dimensions
    bonds = [16, 32]

    run_benchmark(Ls, bonds, steps=2)
