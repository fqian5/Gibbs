import numpy as np
from simulation import RandomizedLindbladSimulator
import quimb as qu
import quimb.tensor as qtn

def exact_diagonalization(L, J, g):
    # Construct Hamiltonian for TFIM: H = -J sum XX - g sum Z
    # Using quimb
    # XX term
    H = qu.ham_heis(L, j=J, b=g, cyclic=False) # Wait, ham_heis is Heisenberg.
    # Build manually
    dims = [2] * L
    H = 0
    # Z terms
    for i in range(L):
        H += -g * qu.ikron(qu.pauli('Z'), dims, [i])
    # XX terms
    for i in range(L - 1):
        H += -J * qu.ikron(qu.pauli('X') & qu.pauli('X'), dims, [i, i+1])

    return qu.eigvalsh(H, k=1)[0]

def main():
    L = 6
    J = 1.0
    g = 1.5
    dt = 0.1
    max_bond = 32

    print(f"Initializing Simulation for L={L}, J={J}, g={g}")

    # Exact Energy
    E_exact = exact_diagonalization(L, J, g)
    print(f"Exact Ground State Energy: {E_exact:.6f}")

    # Simulator
    # Parameters for cooling
    # alpha should be small? Or 1?
    alpha = 1.0
    # sigma?
    sigma = 2.0
    T = 10.0 # 5*sigma

    sim = RandomizedLindbladSimulator(L, J, g, dt=dt, max_bond=max_bond, beta=float('inf'),
                                      filter_sigma=sigma, filter_T=T)

    steps = 50
    energies = []

    print("Starting Randomized Lindblad Dynamics...")
    for step in range(steps):
        sim.step(alpha=alpha)
        E = sim.get_energy()
        energies.append(E)
        print(f"Step {step+1}/{steps}, Energy: {E:.6f}")

    print("Simulation Complete.")
    print(f"Final Energy: {energies[-1]:.6f}")
    print(f"Exact Energy: {E_exact:.6f}")

if __name__ == "__main__":
    main()
