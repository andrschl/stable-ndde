## incluce dataset object
include("../datasets/dataset.jl")

function ode_func(du,u,p,t)
 du[1] = p[1]*(u[2]-u[1])
 du[2] = u[1]*(p[2]-u[3]) - u[2]
 du[3] = u[1]*u[2] - p[3]*u[3]
end

## Define oscillator problem
u0_default = [3.7144662401391497, 5.132226099526597, 17.85688360743861]
p_model = [10.0,28.0,8/3]
tspan_default = (0.0,100.0)
lorenz_prob = ODEProblem(ode_func, u0_default, tspan_default, p_model)
