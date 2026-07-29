# PauliPropagation.jl (Heisenberg picture) implementation of the
# 2nd-order Strang-Trotter TFIM evolution, matching exact_reference.py / mps_trotter.py.
#
# Hamiltonian (Pauli +/-1):  H = pxx * sum X_i X_{i+1} + pz * sum Z_i  (open boundary)
# One Strang step U(dt):
#     Z half-layer  exp(-i pz Z dt/2)        -> theta_Z  = pz*dt
#     XX even bonds exp(-i pxx XX dt)         -> theta_XX = 2*pxx*dt
#     XX odd bonds  (same)
#     Z half-layer  exp(-i pz Z dt/2)
# (PP convention is gate = exp(-i theta/2 P); the factor-of-2 above is verified
#  numerically in `verify_convention()` below.)
#
# Heisenberg picture: start from observable O (Z at site L/2, or sum Z),
# call propagate(circuit, O, thetas; heisenberg=true) which conjugates O backward
# through the whole circuit, then overlapwithzero(...) gives <0...0| O(t) |0...0>.
# We rebuild the circuit for each time t = n*dt (n layers) and propagate once per t.
#
# Qubit indexing in PP is 1-based; site i (0-based in python) maps to i+1 here.
# We report Z at python-site L/2 == julia index (L div 2) + 1.
#
# Timing EXCLUDES JIT: every timed configuration is run once untimed (warmup)
# then timed on a second run.
#
# Output: JSON to a path given as ARGS, so the Python side can load results.

using PauliPropagation
using LinearAlgebra
using JSON
using Printf

# ---------------------------------------------------------------------------
# Convention check (numeric) -- abort if the factor-of-2 is wrong.
# ---------------------------------------------------------------------------
function verify_convention()
    pz = 1.5; dt = 0.05; pxx = -1.0
    Z = [1.0 0; 0 -1.0]; X = [0.0 1; 1 0]
    Uz = tomatrix(PauliRotation([:Z], [1]), pz * dt)
    Uz_exp = cos(pz * dt / 2) * I - 1im * sin(pz * dt / 2) * Z
    @assert isapprox(Uz, Uz_exp; atol=1e-12) "Z-gate theta convention mismatch"
    Uxx = tomatrix(PauliRotation([:X, :X], [1, 2]), 2 * pxx * dt)
    Uxx_exp = cos(pxx * dt) * I - 1im * sin(pxx * dt) * kron(X, X)
    @assert isapprox(Uxx, Uxx_exp; atol=1e-12) "XX-gate theta convention mismatch"
    return true
end

# ---------------------------------------------------------------------------
# Build one Strang layer of gates + matching thetas, appended to circuit/thetas.
# nq = number of qubits, 1-based indices.
# ---------------------------------------------------------------------------
function append_strang_layer!(circuit, thetas, nq, pxx, pz, dt)
    theta_z = pz * dt          # exp(-i pz Z dt/2)
    theta_xx = 2 * pxx * dt    # exp(-i pxx XX dt)

    # 1. Z half-layer
    for i in 1:nq
        push!(circuit, PauliRotation([:Z], [i]))
        push!(thetas, theta_z)
    end
    # 2. XX even bonds  (python bonds (0,1),(2,3),... -> julia (1,2),(3,4),...)
    for i in 1:2:(nq-1)
        push!(circuit, PauliRotation([:X, :X], [i, i + 1]))
        push!(thetas, theta_xx)
    end
    # 3. XX odd bonds   (python (1,2),(3,4),... -> julia (2,3),(4,5),...)
    for i in 2:2:(nq-1)
        push!(circuit, PauliRotation([:X, :X], [i, i + 1]))
        push!(thetas, theta_xx)
    end
    # 4. Z half-layer
    for i in 1:nq
        push!(circuit, PauliRotation([:Z], [i]))
        push!(thetas, theta_z)
    end
    return circuit, thetas
end

function build_circuit(nq, nlayers, pxx, pz, dt)
    circuit = Gate[]
    thetas = Float64[]
    for _ in 1:nlayers
        append_strang_layer!(circuit, thetas, nq, pxx, pz, dt)
    end
    return circuit, thetas
end

# ---------------------------------------------------------------------------
# Run: for each time step n=0..N, propagate the chosen observable through an
# n-layer circuit and record <O>(t). The observable can be:
#   :zmid  -> single Z at site L/2  (python index L div 2 -> julia (L div 2)+1)
#   :sumz  -> sum_i Z_i
# ---------------------------------------------------------------------------
function run_observable(nq, N, pxx, pz, dt, obs::Symbol;
                        min_abs_coeff=1e-4, max_weight=Inf)
    mid_jl = div(nq, 2) + 1   # python L//2 (0-based) -> julia 1-based

    function make_obs()
        if obs == :zmid
            return PauliString(nq, :Z, mid_jl)
        elseif obs == :sumz
            psum = PauliSum(nq)
            for i in 1:nq
                add!(psum, PauliString(nq, :Z, i))
            end
            return psum
        else
            error("unknown observable $obs")
        end
    end

    vals = Vector{Float64}(undef, N + 1)
    # t=0: observable on |0...0>
    vals[1] = real(overlapwithzero(make_obs()))

    for n in 1:N
        circuit, thetas = build_circuit(nq, n, pxx, pz, dt)
        O = make_obs()
        prop = propagate(circuit, O, thetas;
                         min_abs_coeff=min_abs_coeff, max_weight=max_weight,
                         heisenberg=true)
        vals[n+1] = real(overlapwithzero(prop))
    end
    return vals
end

# Timed single-config run (warmup excluded). Returns (vals, walltime_seconds).
function timed_run(nq, N, pxx, pz, dt, obs; min_abs_coeff=1e-4, max_weight=Inf)
    # warmup (JIT + algorithmic) -- untimed
    run_observable(nq, min(N, 2), pxx, pz, dt, obs;
                   min_abs_coeff=min_abs_coeff, max_weight=max_weight)
    t0 = time()
    vals = run_observable(nq, N, pxx, pz, dt, obs;
                          min_abs_coeff=min_abs_coeff, max_weight=max_weight)
    wall = time() - t0
    return vals, wall
end

# ---------------------------------------------------------------------------
# CLI entry: parse a small JSON config from ARGS[1] (path), write results to ARGS[2].
# config: {L, dt, T, pxx, pz, obs, min_abs_coeff, max_weight}
# When run with no args, executes a self-test sanity run at L=10.
# ---------------------------------------------------------------------------
function main()
    verify_convention()

    if length(ARGS) >= 2
        cfg = JSON.parsefile(ARGS[1])
        L = Int(cfg["L"]); dt = Float64(cfg["dt"]); T = Float64(cfg["T"])
        pxx = Float64(cfg["pxx"]); pz = Float64(cfg["pz"])
        obs = Symbol(cfg["obs"])
        mac = Float64(cfg["min_abs_coeff"])
        mw_raw = cfg["max_weight"]
        max_weight = (mw_raw == "Inf" || mw_raw === nothing) ? Inf : Float64(mw_raw)
        N = round(Int, T / dt)

        vals, wall = timed_run(L, N, pxx, pz, dt, obs;
                               min_abs_coeff=mac, max_weight=max_weight)
        ts = [n * dt for n in 0:N]
        out = Dict("t" => ts, "vals" => vals, "wall_time" => wall,
                   "L" => L, "dt" => dt, "T" => T, "pxx" => pxx, "pz" => pz,
                   "obs" => String(obs), "min_abs_coeff" => mac,
                   "max_weight" => (max_weight == Inf ? "Inf" : max_weight))
        open(ARGS[2], "w") do io
            JSON.print(io, out)
        end
        @printf("PP done: L=%d obs=%s mac=%.0e mw=%s wall=%.3fs  O(T)=%.6f\n",
                L, String(obs), mac, string(max_weight), wall, vals[end])
    else
        # self-test
        L = 10; dt = 0.05; T = 4.0; pxx = -1.0; pz = 1.5
        N = round(Int, T / dt)
        vals, wall = timed_run(L, N, pxx, pz, dt, :zmid; min_abs_coeff=1e-5)
        @printf("Self-test L=%d: <Z_mid>(0)=%.6f <Z_mid>(T)=%.6f wall=%.3fs\n",
                L, vals[1], vals[end], wall)
    end
end

main()
