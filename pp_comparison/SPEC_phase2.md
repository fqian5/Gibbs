# Phase 2 Spec — Dissipative Cooling: PauliPropagation.jl vs quimb-MPS

Conventions locked: H = pxx·Σ X_iX_{i+1} + pz·Σ Z_i, pxx=-1.0, pz=+1.5, open boundary, Pauli ±1, init |0...0>.

## Central resolution (honest)
The EXACT filtered jump K = Σ_ij f̂(E_i-E_j) A_ij |ψ_i><ψ_j| (PRR_2024/lindblad.py:62-77) is GLOBAL/dense in
the eigenbasis → ~4^L Pauli strings, no light-cone, no truncatable tail. PP CANNOT scale on it.
=> Two experiments:
- **A (validation only, L<=8):** PP brute-force (FrozenGate/transfer map, no truncation) reproduces exact filtered-jump
  cooling ⟨H⟩(t)→E_GS. Proves PP channel-adjoint is CORRECT, not scalable.
- **B (the real benchmark):** both methods run the SAME local surrogate dissipator (below), scored against the
  SURROGATE's own NESS E_NESS(γ), NOT the true GS. Primary deliverable.

## 1B. Local surrogate Lindbladian (Experiment B — primary)
dρ/dt = -i[H,ρ] + Σ_i γ ( K_i ρ K_i† - ½{K_i†K_i, ρ} ),  K_i = σ_i^+ = |1><0|_i = ½(X_i + iY_i)
Damps each site toward the mean-field single-site GS. SIGN TRAP: pz=+1.5>0 → single-site GS of pz·Z is |1>
(eigenvalue -1), so damp toward |1>, NOT |0>.
Steady state: between E_MF = -pz·L = -1.5L (γ→∞, frozen to |1...1>) and true E_GS (γ→0+). NESS energy E_NESS(γ)
tunable. Scan γ ∈ {0.05,0.1,0.2,0.5}; default γ=0.1. Report E_NESS(γ) vs E_GS (DMRG) as the model's own physics error.
Optional phase-2.5: two-site bond-local jump (weight-2, still tractable) tightens NESS→GS.

## 2. PP implementation (Heisenberg adjoint)
𝓛†(O) = i[H,O] + Σ_i γ ( K_i† O K_i - ½{K_i†K_i,O} ). Propagate O=H BACKWARD N_τ=T/τ steps; ⟨H⟩(t)=overlapwithzero.
Strang split per outer step τ:  e^{τ𝓛†} ≈ e^{(τ/2)𝓛†_coh} · e^{τ 𝓛†_diss} · e^{(τ/2)𝓛†_coh}
- Coherent layer = phase-1 Trotter circuit (RZ half-layers + RXX even/odd), reuse verbatim. Sub-Trotterize the τ/2
  coherent block with n_sub = round((τ/2)/dt), dt=0.05.
- Dissipative layer = per-site AmplitudeDampingNoise, p = 1 - e^{-γτ}. PP's AmplitudeDampingNoise damps toward |0>;
  X-conjugate per site (FrozenGate/transfer map: X_i · AmpDamp_i · X_i) to damp toward |1>.
Single-site amp-damping ADJOINT transfer map, toward |0>:  I→I, Z→(1-p)Z+pI, X→√(1-p)X, Y→√(1-p)Y.
Toward |1> (X-conjugated): I→I, Z→(1-p)Z - pI, X→√(1-p)X, Y→√(1-p)Y.
KEY: dissipative layer is WEIGHT-NON-INCREASING (X,Y scaled by √(1-p); Z splits to Z + I, reducing weight) — it
PRUNES the operator growth the coherent RXX layer creates. Expect PP MORE efficient here than in the unitary phase-1.
Truncation: min_abs_coeff ∈ {1e-3,1e-4,1e-5}, max_weight ∈ {4,6,∞}.
Experiment A in PP: dense K as custom transfer map / FrozenGate on full 2^L, no truncation, L<=8 only. Measure
term-count blowup vs τ to demonstrate why truncated PP fails on exact jump, then stop.

## 3. Reference / MPS side
1. Exact density-matrix Lindblad (L<=8): vectorize ρ, build Liouvillian 4^L×4^L, vec(ρ)(t)=e^{𝓛t}vec(ρ0),
   ⟨H⟩(t)=Tr[Hρ(t)]. Experiment A: exact filtered K (or reuse lindblad.py Lindblad_simulation rho_hist/avg_energy).
   Experiment B: surrogate K_i=σ^+. GOLD STANDARD both PP and MPS must match at small L.
2. MPS trajectory unraveling (scaling, Exp B): reuse efficient/simulator.py TEBD. Quantum-jump MCWF:
   - no-jump evolution under H_eff = H - (i/2)Σ_i γ K_i†K_i = H - (i/2)γ Σ|1><1|_i (single-site non-unitary gate + renorm)
   - stochastic jumps K_i=σ_i^+ at rate γ⟨K_i†K_i⟩
   - average ⟨H⟩(t) over N_traj (start 100-500), report MC stderr ~1/√N_traj.
   χ ∈ {16,32,64,128}. (Alternative: deterministic MPDO, doubles bond index — offer as option.)
3. DMRG true E_GS (efficient/simulator.py get_dmrg_ground_state) — reference line for NESS-to-GS gap.

## 4. Protocol & metrics
1. Validate Exp A (L=6,8): PP-bruteforce vs exact-DM vs MPS-traj on EXACT jump → agree ~1e-3, →E_GS.
2. Validate surrogate Exp B (L=6,8): PP-local vs exact-DM vs MPS-traj on surrogate → agree to E_NESS(γ).
3. Scale Exp B (L=10,16,24,30,50,100): PP-truncated vs MPS-traj-χ. Self-convergence + cross-method (no exact ref).
Metrics:
- accuracy small L: |⟨H⟩_method - ⟨H⟩_exact|(t) max & mean, normalized by L; steady-state |E_NESS_method - E_NESS_exact|.
- STEADY-STATE ENERGY vs TRUNCATION (key dissipative metric): plot E_∞=⟨H⟩(T) vs knob (PP min_abs_coeff/max_weight;
  MPS χ). PP truncation over-prunes correlations → NESS drifts HIGHER (under-cooled); MPS χ caps entanglement (damped
  NESS is LOW-entanglement → favors MPS at modest χ). Report both biases.
- wall-clock vs L at fixed |E_∞ - E_NESS_ref|/L ≤ 1e-3 (cheapest converged setting). MPS pays N_traj tax PP avoids.
- wall-clock vs accuracy Pareto at L=24, L=50.
- memory; Julia JIT excluded (2nd run/@btime), single-thread pinned, versions/hardware logged.
Time params: τ=0.1, dt=0.05, γ=0.1; T≈30-50 (relaxation ~1/γ=10), confirm NESS plateau. Identical τ,dt,γ,Strang both sides.

## 6. Minimal viable first run
L=8, τ=0.1, dt=0.05, γ=0.1, surrogate K_i=σ^+ toward |1>, T=40.
Exact-DM ref → assert ⟨H⟩(0)=pz·L=12.0. PP local + per-site X-conj AmpDamp p=1-e^{-γτ}, propagate O=H.
MPS MCWF N_traj=200, χ∈{32,64}. Overlay 3 curves → agree ~1e-3·L at NESS. Then scale L→{24,50,100} (PP+MPS).
Run L=6 Exp-A slice once to anchor channel-adjoint correctness.

## #1 IMPLEMENTATION RISK: damping direction. pz=+1.5 → MF GS is |1>. AmpDamp must be X-conjugated. ASSERT
E_NESS → -pz·L = -1.5L in γ→∞ limit to catch a flipped channel before it poisons every curve.
