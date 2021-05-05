## incluce dataset object
include("../datasets/dataset.jl")

function ode_func(u,(ω,γ),t)
    A = [0 1; -ω^2  -2*γ]
    return A*u
end

## Define oscillator problem
u0_default = [1.0,0.0]
tspan_default = (0.0,10.0)
oscillator_prob = ODEProblem(ode_func, u0_default, tspan_default, [config["ω"], config["γ"]])
