using PauliPropagation
# Verify AmplitudeDampingNoise adjoint (Heisenberg) transfer map and direction.
# Spec claim toward |0>: I->I, Z->(1-p)Z + p I, X->sqrt(1-p)X, Y->sqrt(1-p)Y.
g = 0.3
println("gamma=p=", g, "  sqrt(1-p)=", sqrt(1-g))
for sym in (:I, :Z, :X, :Y)
    O = PauliString(1, sym, 1)
    circ = [AmplitudeDampingNoise(1, g)]
    out = propagate(circ, O; heisenberg=true, min_abs_coeff=0.0)
    println(sym, " -> ", out)
end
println("---- overlapwithzero of damped Z (should reflect <Z> on |0>) ----")
# On |0>, <Z>=+1. Adjoint AmpDamp toward |0> leaves |0> fixed -> <Z> stays +1.
oZ = propagate([AmplitudeDampingNoise(1, g)], PauliString(1, :Z, 1); heisenberg=true, min_abs_coeff=0.0)
println("overlapwithzero(damped Z) = ", overlapwithzero(oZ))
