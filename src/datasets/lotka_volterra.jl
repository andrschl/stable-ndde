## incluce dataset object
include("../datasets/dataset.jl")

# Define the same LV equation
function ode_func(du, u, p, t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = (α   - β*y) * x
  du[2] = dy = (δ*x - γ)   * y
end
# Initial parameters for delayed LV
p_model = Array{Float32}([5/3, 4/3, 1.0, 1.0])
# p_model = Array{Float32}([2.2, 1.0, 2.0, 0.4])
# p_model = Array{Float32}([2/3, 4/3, 1.0, 1.0])
tspan_default = (0.0,10.0)
u0_default = ones(Float64, 2)
LV_prob = ODEProblem(ode_func, u0_default, tspan_default, p_model)
