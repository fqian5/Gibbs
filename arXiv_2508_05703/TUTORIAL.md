# Randomized Lindblad Dynamics Simulator Tutorial

This repository implements the randomized Lindblad dynamics simulation algorithm described in [arXiv:2508.05703](https://arxiv.org/abs/2508.05703). It uses Matrix Product States (MPS) via the `quimb` library to efficiently simulate the cooling of a quantum many-body system (specifically the Transverse Field Ising Model) into its ground state using a transient ancilla qubit.

## Requirements

*   Python 3.8+
*   `numpy`
*   `quimb`
*   `cotengra` (recommended for efficient contraction)

## Core Concept

The simulation proceeds in discrete time steps. In each step:
1.  A random site $i$ in the system is selected.
2.  A random coupling operator $A_S$ (e.g., $X_i$ or $Z_i$) is chosen.
3.  A random frequency $\omega$ is sampled.
4.  An ancilla qubit is initialized in a state depending on the inverse temperature $\beta$ and $\omega$.
5.  The system and ancilla interact via a unitary evolution $U = \mathcal{T} e^{-i \int (H_S + H_E + f(t) H_{int}) dt}$.
6.  The ancilla is measured, and the system state is updated (projected) based on the measurement outcome.

This process effectively realizes a Lindblad master equation that drives the system towards a thermal state (or ground state if $\beta \to \infty$).

## Usage

The main class is `RandomizedLindbladSimulator` in `simulation.py`.

### 1. Initialization

```python
from simulation import RandomizedLindbladSimulator

# System parameters
L = 10          # Number of qubits
J = 1.0         # Ising coupling strength
g = 1.5         # Transverse field strength
max_bond = 32   # MPS bond dimension

# Simulation parameters
dt = 0.1        # Trotter step size
sigma = 2.0     # Filter width (determines interaction window)
T = 10.0        # Half-width of time window (usually 5 * sigma)
beta = float('inf') # Inverse temperature (inf = zero temp/ground state)

# Create simulator
sim = RandomizedLindbladSimulator(L, J, g, dt=dt, max_bond=max_bond, beta=beta,
                                  filter_sigma=sigma, filter_T=T)
```

### 2. Running the Simulation

You can evolve the system step-by-step. The `step` method performs one full randomized interaction cycle.

```python
num_steps = 100
alpha = 1.0 # Coupling strength scaling factor

for step in range(num_steps):
    sim.step(alpha=alpha)

    # Calculate energy or other observables
    energy = sim.get_energy()
    print(f"Step {step}: Energy = {energy}")
```

### 3. Key Parameters

*   **`L`**: System size. Larger L increases simulation cost linearly (MPS) but entanglement growth might require larger bond dimensions.
*   **`max_bond`**: Controls the accuracy of the MPS representation. If the system becomes highly entangled, `quimb` will truncate singular values to keep the bond dimension below this limit. Increase this for higher accuracy at the cost of runtime.
*   **`dt`**: Time step for the Trotterized time evolution within the interaction window. Smaller `dt` is more accurate but slower.
*   **`filter_sigma`**: Controls the spectral resolution of the cooling process.
*   **`beta`**: Target inverse temperature. Set to `float('inf')` for ground state preparation.

## Performance

For a 1D system, the cost per step scales roughly as $O(L \cdot \chi^3)$ where $\chi$ is the bond dimension (due to the long-range gate application implemented via swaps).

To run benchmarks:

```bash
python benchmark.py
```

## Example

See `run_experiment.py` for a complete example comparing the simulation results against exact diagonalization for a small system.
