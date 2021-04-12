cd(@__DIR__)
using Pkg; Pkg.activate("."); Pkg.instantiate();
using ModelingToolkit, DifferentialEquations

@variables t x(t) v(t) θ(t) ω(t) F(t)
@parameters Lₐ Lₚ mₖ mₗ mₚ g
D = Differential(t)

function L((v, ω), (x, θ), (Lₐ, Lₚ, mₖ, mₗ, mₚ, g), t)
    V = -mₚ*Lₚ*g*cos(θ)
    T = (mₖ+mₗ)*v^2/2 + mₚ*(Lₚ^2*ω^2 + 2*Lₚ*ω*v*cos(θ) + v^2)/2
    T - V
end

a = L((v, ω), (x, θ), (Lₐ, Lₚ, mₖ, mₗ, mₚ, g), t)

function lagrangian2system(
        L, q̇, q, p, t;
        Q = zeros(length(q)),
        defaults = [q̇; q] .=> 0.0,
        kwargs...
    )
    Q_vals = Q
    inds = eachindex(q)

    @variables v[inds] x[inds] Q[inds](t)
    sub = Base.Fix2(substitute, Dict([v.=>q̇; x.=>q]))
    Lf = L(v, x, p, t)

    F = ModelingToolkit.gradient(Lf, x) + Q
    Lᵥ = ModelingToolkit.gradient(Lf, v)
    rhs = sub.(F - ModelingToolkit.jacobian(Lᵥ, x) * q̇ - ModelingToolkit.derivative.(Lᵥ, (t,)))
    M = sub.(ModelingToolkit.jacobian(Lᵥ, v))

    D = Differential(t)

    eqs = [
           D.(q̇) .~ M \ rhs
           D.(q) .~ q̇
           Q .~ Q_vals
          ]

    sys = ODESystem(eqs, t, [q̇; q; Q], p; defaults=defaults, kwargs...)
    return structural_simplify(sys)
end

# Cart input force
F = 1000sin(t)

# Generalized forces
Q = [F, 0]

# Make equations of motion
slosh_cart = lagrangian2system(L, [v, ω], [x, θ], [Lₐ, Lₚ, mₖ, mₗ, mₚ, g], t; Q)

# Initial Conditions
ic = [
    θ => deg2rad(10)
]

# Parameters
p = Dict([
    Lₐ => 10
    Lₚ => 0.5
    mₖ => 100
    mₗ => 0
    mₚ => 25
    g => 9.80665
])


## Simulation
prob = ODEProblem(slosh_cart, ic, (0.0, 10.0), [p...])
sol = solve(prob, Tsit5())
using Plots
Plots.plot(sol)
