include("import.jl")
using Symbolics, ModelingToolkit, DifferentialEquations
using Latexify, Plots

## Lagrangian functionality

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

## Define symbolic variables
n = 3
@variables t θ[1:n](t) ω[1:n](t)
@parameters g l[1:n] m[1:n]

## Define Lagrangian
function L(ω, θ, (g, l, m), t)
    # Moments of inertia
    J  = m .* l.^2 /12

    # Define vertical positions
    y = vcat([l[1] * cos(θ[1]) / 2], map(i -> sum(map(j -> l[j]*cos(θ[j]), 1:i-1)) + l[i] * cos(θ[i]) / 2, 2:n))

    # Define velocities
    vx = vcat([l[1] * cos(θ[1]) * ω[1]/ 2], map(i -> sum(map(j -> l[j]*cos(θ[j])*ω[j], 1:i-1)) + l[i] * cos(θ[i]) * ω[i] / 2, 2:n))
    vy = vcat([-l[1] * sin(θ[1]) * ω[1] / 2], map(i -> sum(map(j -> -l[j]*sin(θ[j])*ω[j], 1:i-1)) - l[i] * sin(θ[i]) * ω[i]  / 2, 2:n))

    # Build Lagrangian
    V = -sum(m.*y.*g)
    T = 1/2 * sum(m.*(vx.^2 + vy.^2) + J.*(ω.^2))
    T - V
end

# Friction terms
Q = zero(ω)

# Define ODESystem
n_pendulum = lagrangian2system(L, ω, θ, [g, l, m], t; Q)

# Initial Conditions
ic = vcat(
    ω .=> zeros(n),
    θ .=> deg2rad.(randn(n)*10)
    )

# Parameters
p = Dict([
    g => 9.806
    l => repeat([1.0], n)
    m => repeat([1.0], n)
])

## Simulation
prob = ODEProblem(n_pendulum, ic, (0.0, 10), [p...])
sol = solve(prob, Tsit5())
Plots.plot(sol)



## Define symbolic variables
n = 2
@variables t θ[1:n](t) ω[1:n](t)
@parameters g l m

## Define Lagrangian
function L(ω, θ, (g, l, m), t)
    # Moments of inertia
    J  = m * l^2 /12

    # Define vertical positions
    y = vcat([l * cos(θ[1]) / 2], map(i -> sum(map(j -> l*cos(θ[j]), 1:i-1)) + l * cos(θ[i]) / 2, 2:n))

    # Define velocities
    vx = vcat([l * cos(θ[1]) * ω[1]/ 2], map(i -> sum(map(j -> l*cos(θ[j])*ω[j], 1:i-1)) + l * cos(θ[i]) * ω[i] / 2, 2:n))
    vy = vcat([-l * sin(θ[1]) * ω[1] / 2], map(i -> sum(map(j -> -l*sin(θ[j])*ω[j], 1:i-1)) - l * sin(θ[i]) * ω[i]  / 2, 2:n))

    # Build Lagrangian
    V = -sum(m.*y.*g)
    T = 1/2 * sum(m.*(vx.^2 + vy.^2) + J.*(ω.^2))
    T - V
end

# Friction terms
# Q = zero(ω)
Q = zero(ω)
# Define ODESystem
n_pendulum = lagrangian2system(L, ω, θ, [g, l, m], t; Q)

# Initial Conditions
ic = vcat(
    ω .=> zeros(n),
    θ .=> deg2rad.(randn(n)*10)
    )

# Parameters
p = Dict([
    g => 9.806
    l => 1.0
    m => 1.0
])

## Simulation
prob = ODEProblem(n_pendulum, ic, (0.0, 10.0), [p...])
sol = solve(prob, Tsit5())
Plots.plot(sol)
