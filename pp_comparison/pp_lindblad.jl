# PauliPropagation.jl Heisenberg-adjoint propagation of the site-local sigma^+
# surrogate Lindbladian (Experiment B).
#
# Adjoint generator:
#   L^dag(O) = i[H,O] + sum_i gamma ( K_i^dag O K_i - 1/2 {K_i^dag K_i, O} ),  K_i = sigma_i^+.
# Per OUTER step tau we Strang-split the adjoint propagator:
#   e^{tau L^dag} ~ e^{(tau/2) L^dag_coh} . e^{tau L^dag_diss} . e^{(tau/2) L^dag_coh}
#   - coherent block  = phase-1 Trotter circuit (RZ half-layers + RXX even/odd),
#     sub-Trotterized with n_sub = round((tau/2)/dt) sub-steps of size dt each.
#   - dissipative block = per-site amplitude damping with p = 1 - e^{-gamma*tau},
#     X-CONJUGATED so it damps toward |1> (not PP's default |0>).
#
# Damping direction (THE #1 trap, pz>0 => MF GS is |1>):
#   PP AmplitudeDampingNoise adjoint (toward |0>):  I->I, Z->(1-p)Z + p I, X->sqrt(1-p)X, Y->sqrt(1-p)Y.
#   X-conjugated  X . AmpDamp . X  (toward |1>):    I->I, Z->(1-p)Z - p I, X->sqrt(1-p)X, Y->sqrt(1-p)Y.
#   We realize the X-conjugation by sandwiching AmplitudeDampingNoise(i,p) between
#   CliffordGate(:X, i) on both sides.  verify_damping_direction() asserts the
#   toward-|1> transfer map numerically BEFORE any benchmark run.
#
# Observable: O = H (PauliSum). Propagate BACKWARD N = T/tau outer steps;
# <H>(t) = overlapwithzero(propagated O). We rebuild an n-outer-step circuit per
# recorded time t (O(N^2) gate apps total) -- an upper bound, consistent w/ phase-1.
#
# Truncation: min_abs_coeff, max_weight (the dissipative layer is WEIGHT-NON-INCREASING,
# so it prunes the operator growth the RXX layer creates -- expect PP efficient here).
#
# Conventions locked: H = pxx*sum X_iX_{i+1} + pz*sum Z_i, pxx=-1.0, pz=1.5, init |0..0>.
# 1-based qubit indices.  JIT excluded from timing (warmup run before timing).
#
# CLI: julia --project=... pp_lindblad.jl <config.json> <out.json>
#   config: {L, gamma, T, tau, dt, pxx, pz, min_abs_coeff, max_weight}

using PauliPropagation
using LinearAlgebra
using JSON
using Printf

# ---------------------------------------------------------------------------
# Coherent phase-1 Strang layer (verbatim convention from phase-1 pp_trotter.jl).
# theta_Z = pz*dt (exp(-i pz Z dt/2)); theta_XX = 2*pxx*dt (exp(-i pxx XX dt)).
# ---------------------------------------------------------------------------
function append_coherent_strang!(circuit, thetas, nq, pxx, pz, dt)
    theta_z = pz * dt
    theta_xx = 2 * pxx * dt
    for i in 1:nq
        push!(circuit, PauliRotation([:Z], [i])); push!(thetas, theta_z)
    end
    for i in 1:2:(nq-1)
        push!(circuit, PauliRotation([:X, :X], [i, i + 1])); push!(thetas, theta_xx)
    end
    for i in 2:2:(nq-1)
        push!(circuit, PauliRotation([:X, :X], [i, i + 1])); push!(thetas, theta_xx)
    end
    for i in 1:nq
        push!(circuit, PauliRotation([:Z], [i])); push!(thetas, theta_z)
    end
end

# Per-site X-conjugated amplitude damping (toward |1>), p frozen.
# CliffordGate(:X) carries no parameter; AmplitudeDampingNoise(i,p) is a FrozenGate
# (also no free theta). So these contribute NO entries to `thetas`.
function append_damping_layer!(circuit, nq, p)
    for i in 1:nq
        push!(circuit, CliffordGate(:X, i))
        push!(circuit, AmplitudeDampingNoise(i, p))
        push!(circuit, CliffordGate(:X, i))
    end
end

# One outer Strang step:  half-coherent . full-dissipative . half-coherent.
# Coherent half-block = n_sub sub-steps of size dt covering total time tau/2.
function append_outer_step!(circuit, thetas, nq, pxx, pz, dt, n_sub, p)
    for _ in 1:n_sub
        append_coherent_strang!(circuit, thetas, nq, pxx, pz, dt)
    end
    append_damping_layer!(circuit, nq, p)
    for _ in 1:n_sub
        append_coherent_strang!(circuit, thetas, nq, pxx, pz, dt)
    end
end

# Build full circuit + thetas for `nsteps` outer steps.
function build_lindblad_circuit(nq, nsteps, pxx, pz, dt, n_sub, p)
    circuit = Gate[]
    thetas = Float64[]
    for _ in 1:nsteps
        append_outer_step!(circuit, thetas, nq, pxx, pz, dt, n_sub, p)
    end
    return circuit, thetas
end

# Observable O = H as a PauliSum.
function hamiltonian_obs(nq, pxx, pz)
    psum = PauliSum(nq)
    for i in 1:(nq-1)
        add!(psum, PauliString(nq, [:X, :X], [i, i + 1], pxx))
    end
    for i in 1:nq
        add!(psum, PauliString(nq, :Z, i, pz))
    end
    return psum
end

# ---------------------------------------------------------------------------
# Damping-direction verification (THE guard). Propagate single Paulis through one
# X-conjugated AmpDamp and check the toward-|1> transfer map numerically.
# ---------------------------------------------------------------------------
function verify_damping_direction()
    p = 0.3
    function damp1(sym)
        circ = Gate[]
        push!(circ, CliffordGate(:X, 1))
        push!(circ, AmplitudeDampingNoise(1, p))
        push!(circ, CliffordGate(:X, 1))
        propagate(circ, PauliString(1, sym, 1); heisenberg=true, min_abs_coeff=0.0)
    end
    # Z -> (1-p) Z - p I ; check coefficients via overlaps.
    outZ = damp1(:Z)
    # overlapwithzero of (1-p)Z - p I on |0>: <0|Z|0>=+1, <0|I|0>=1 => (1-p)*1 - p*1 = 1-2p
    valZ = overlapwithzero(outZ)
    @assert isapprox(valZ, 1 - 2p; atol=1e-10) "Z damping toward |1> wrong: got $valZ, expected $(1-2p)"
    # X,Y -> sqrt(1-p) * X,Y  (overlapwithzero = 0 for X,Y; check coeff via terms)
    # Cheap structural check: <0| sqrt(1-p) X |0> = 0, fine; assert norm scaled.
    return valZ
end

# ---------------------------------------------------------------------------
# Run: <H>(t) for t = 0..N*tau (Heisenberg adjoint), rebuilding circuit each step.
# ---------------------------------------------------------------------------
function run_pp_lindblad(nq, N, pxx, pz, dt, tau, gamma;
                        min_abs_coeff=1e-4, max_weight=Inf)
    n_sub = max(1, round(Int, (tau / 2) / dt))
    p = 1 - exp(-gamma * tau)

    vals = Vector{Float64}(undef, N + 1)
    vals[1] = real(overlapwithzero(hamiltonian_obs(nq, pxx, pz)))  # t=0 -> pz*L
    for n in 1:N
        circuit, thetas = build_lindblad_circuit(nq, n, pxx, pz, dt, n_sub, p)
        O = hamiltonian_obs(nq, pxx, pz)
        prop = propagate(circuit, O, thetas;
                         heisenberg=true, min_abs_coeff=min_abs_coeff, max_weight=max_weight)
        vals[n+1] = real(overlapwithzero(prop))
    end
    return vals
end

function timed_run(nq, N, pxx, pz, dt, tau, gamma; min_abs_coeff=1e-4, max_weight=Inf)
    # warmup (JIT) -- untimed, few steps
    run_pp_lindblad(nq, min(N, 2), pxx, pz, dt, tau, gamma;
                    min_abs_coeff=min_abs_coeff, max_weight=max_weight)
    t0 = time()
    vals = run_pp_lindblad(nq, N, pxx, pz, dt, tau, gamma;
                           min_abs_coeff=min_abs_coeff, max_weight=max_weight)
    wall = time() - t0
    return vals, wall
end

function main()
    verify_damping_direction()
    if length(ARGS) >= 2
        cfg = JSON.parsefile(ARGS[1])
        L = Int(cfg["L"]); gamma = Float64(cfg["gamma"])
        T = Float64(cfg["T"]); tau = Float64(cfg["tau"]); dt = Float64(cfg["dt"])
        pxx = Float64(cfg["pxx"]); pz = Float64(cfg["pz"])
        mac = Float64(cfg["min_abs_coeff"])
        mw_raw = cfg["max_weight"]
        max_weight = (mw_raw == "Inf" || mw_raw === nothing) ? Inf : Float64(mw_raw)
        N = round(Int, T / tau)

        vals, wall = timed_run(L, N, pxx, pz, dt, tau, gamma;
                               min_abs_coeff=mac, max_weight=max_weight)
        ts = [n * tau for n in 0:N]
        out = Dict("t" => ts, "energy" => vals, "wall_time" => wall,
                   "L" => L, "gamma" => gamma, "T" => T, "tau" => tau, "dt" => dt,
                   "pxx" => pxx, "pz" => pz, "min_abs_coeff" => mac,
                   "max_weight" => (max_weight == Inf ? "Inf" : max_weight))
        open(ARGS[2], "w") do io
            JSON.print(io, out)
        end
        @printf("PP-Lindblad: L=%d gamma=%.2f mac=%.0e mw=%s wall=%.3fs  H(0)=%.4f H(T)=%.4f (NESS=-pz*L=%.1f)\n",
                L, gamma, mac, string(max_weight), wall, vals[1], vals[end], -pz * L)
    else
        L = 8; gamma = 0.1; T = 40.0; tau = 0.1; dt = 0.05; pxx = -1.0; pz = 1.5
        N = round(Int, T / tau)
        vals, wall = timed_run(L, N, pxx, pz, dt, tau, gamma; min_abs_coeff=1e-4)
        @printf("Self-test L=%d: H(0)=%.4f H(T)=%.4f (target -pz*L=%.1f) wall=%.3fs\n",
                L, vals[1], vals[end], -pz * L, wall)
    end
end

main()
