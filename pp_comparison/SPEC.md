# Benchmark Spec — quimb MPS vs PauliPropagation.jl on 1D TFIM Trotter evolution

## Hamiltonian (lock to ONE convention)
H = pxx * Σ_{i=0}^{L-2} X_i X_{i+1}  +  pz * Σ_{i=0}^{L-1} Z_i   (open boundary, Pauli ±1)
Defaults: pxx = -1.0, pz = 1.5.  Criticality scan: (pxx,pz)=(-1.0,1.0).
Use the `efficient/simulator.py` convention (pxx=-1.0, pz=+1.5), NOT the arXiv file's baked-in minus signs.

## Circuit: one 2nd-order Strang step U(dt) (matches arXiv_2508_05703 apply_system_evolution)
1. Z half-layer: RZ_i = exp(-i·pz·Z·dt/2) on all i
2. XX even bonds: RXX = exp(-i·pxx·(X⊗X)·dt) on bonds (0,1),(2,3),...
3. XX odd bonds:  same on (1,2),(3,4),...
4. Z half-layer: exp(-i·pz·Z·dt/2) on all i
N_steps = round(T/dt). Initial state |0...0⟩.
Angles: RZ generator-angle per half-layer = pz·dt/2 ; RXX generator-angle per bond = pxx·dt.

## Observables (vs time t=n·dt)
- PRIMARY: ⟨Z_{L/2}⟩(t)  (single site, flatters PP light-cone)
- ⟨Σ_i Z_i⟩(t)  (extensive)
- (optional) ⟨H⟩(t)

## Parameter grid
- dt = 0.05 (primary), T = 4.0 (N=80); stress: T=8.0.
- L = 8,10,12 (exact ground truth) ; 16,20,24 ; 30,50 ; 100,200 (PP showcase).
- MPS χ ∈ {16,32,64,128,256}; split opts cutoff=1e-12 cutoff_mode=rel absorb=both renorm=True.
- PP: min_abs_coeff ∈ {1e-2,1e-3,1e-4,1e-5,1e-6}; max_weight ∈ {4,6,8,12,Inf}.

## Ground truth & metrics
- Small L: PRIMARY reference = exact Trotter-circuit statevector (same gate list, no truncation) → isolates simulator truncation error. Also report continuous e^{-iHt}|0⟩ to show the Trotter floor.
- Error: e_O(t)=|⟨O⟩_method−⟨O⟩_ref|; report E_inf=max_t, E_1=mean_t. Normalize ΣZ by L.
- Large L: self-convergence (MPS in χ, PP in min_abs_coeff/max_weight) + cross-method overlay.

## Performance
- wall-clock vs L at fixed accuracy target (E_inf ≤ 1e-3 on ⟨Z_{L/2}⟩ at T=4)
- wall-clock vs accuracy Pareto at (L=12,T=4),(L=30,T=4),(L=50,T=8 critical)
- memory peak vs L
- EXCLUDE Julia JIT warmup: time the SECOND run / precompile first. quimb warmup call too. Pin BLAS threads. Deterministic — no sampling here.

## Sanity checks
- ⟨Z_i⟩(0)=+1 ; ⟨H⟩(0)=pz·L (pz=1.5,L=10 → 15.0). Both must reproduce at t=0.
- Small-L: both converged methods + exact-Trotter agree to ~1e-6.
- ⟨Z_{L/2}⟩ starts at +1, oscillates+decays (XX drives it; Z field commutes with Z).

## Pitfalls
- Sign/convention mismatch is the #1 bug — verify ⟨H⟩(0)=pz·L first.
- Pauli ±1 everywhere (quimb qu.pauli is ±1; SpinHam1D(S=0.5) is spin-1/2, needs 2g/4J rescale).
- Same Trotter order/dt/ordering/boundary in BOTH; benchmark simulators not discretizations.
- PP cost ~ depth × retained-term-count; single-site Z light-cone radius ≈ N_layers → once 2N≥L, PP loses L-independence.
- MPS cost ~ O(L·χ³); near criticality S(t) grows linearly → χ blows up. Report half-chain entropy S(t).

## PauliPropagation.jl 0.7.3 API (verified installed)
- Built-in: `tfitrottercircuit(nqubits, nlayers; topology, start_with_ZZ)` — BUT this is the standard TFI with ZZ+X; OUR model is XX+Z. Either (a) build a custom circuit of PauliRotation gates, or (b) map by a basis rotation. SAFEST: build custom gates.
  - Use `PauliRotation([:Z], [i])` for the Z field; `PauliRotation([:X,:X],[i,i+1])` for XX. Thetas are the rotation angles; PP convention is gate = exp(-i θ/2 P). So to get exp(-i·pz·Z·dt/2) use θ = pz·dt (since exp(-iθZ/2)); for exp(-i·pxx·XX·dt) use θ = 2·pxx·dt. VERIFY this factor-of-2 against PP docs/source by checking a 1-qubit case numerically.
- `propagate(circuit, psum_or_pstr, thetas; max_weight, min_abs_coeff, max_freq, max_sins, heisenberg=true)` → propagated PauliSum.
- `overlapwithzero(psum)` → ⟨0...0| O |0...0⟩ (real). This is the expectation value on |0...0⟩.
- Heisenberg picture: start from observable O (e.g. PauliString Z on site L/2), propagate BACKWARD through circuit (heisenberg=true), then overlapwithzero.
- `PauliString(nqubits, :Z, site)`, `PauliSum(nqubits)`, `add!`. Check exact constructors in installed source: ~/.julia/packages/PauliPropagation/7EY66/src.
