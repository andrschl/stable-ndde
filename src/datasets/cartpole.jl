################################################################################
# generate dynamics model
include("../datasets/dataset.jl")

# open loop dynamics (x = x, ẋ, φ, φ̇)
function f_ol(z::Array{T,1},u::Array{T,1}) where {T<:Real}
    l = 0.15
    g = 9.81
    b = 0.0
    m = 0.2
    I = m*l^2/3
    M = 0.3
    x = z[1]
    dx = z[2]
    φ = z[3]
    dφ = z[4]
    return vcat(dx, ((I+m*l^2)*(u[1]+m*l*dφ^2*sin(φ)) - g*m^2*l^2*sin(φ)*cos(φ)) / (I*(M + m) + m*l^2*(M + m*sin(φ)^2)),
            dφ, m*l*(-cos(φ)*u[1] - m*l*dφ^2*sin(φ)*cos(φ) + (M + m)*g*sin(φ)) / (I*(M + m) + m*l^2*(M + m*sin(φ)^2)))
end
function f_ol(z::Array{T,2},u::Array{T,2}) where {T<:Real}
    l = 0.15
    g = 9.81
    b = 0.0
    m = 0.2
    M = 0.3
    I = m*l^2/3
    x = z[1, :]
    dx = z[2, :]
    φ = z[3, :]
    dφ = z[4, :]
    return permutedims(hcat(dx, ((I+m*l^2)*(u[1, :].+m*l*dφ.^2 .*sin.(φ)) .- g*m^2*l^2*sin.(φ).*cos.(φ)) ./ (I*(M + m) .+ m*l^2*(M .+ m*sin.(φ).^2)),
            dφ, m*l*(-cos.(φ).*u[1, :] .- m*l*dφ.^2 .*sin.(φ).*cos.(φ) .+ (M + m)*g*sin.(φ)) ./ (I*(M + m) .+ m*l^2*(M .+ m*sin.(φ).^2))),[2,1])
end

f0 = x-> f_ol(x,zero(x))
# linear feedback layer
struct Feedback{S}
    K::S
end
Flux.@functor Feedback
Feedback(data_dim::Integer) = begin
    Feedback(K)
end
(m::Feedback)(x::AbstractArray) = m.K*x
# closed loop dynamics layer
struct F_CL{S}
    u::S
end
Flux.@functor F_CL
(m::F_CL)(xt::Array{T,1}) where {T<:Real} = begin
    return f_ol(xt[1:data_dim], m.u(xt[1+data_dim:end]))
end
(m::F_CL)(xt::Array{T,2}) where {T<:Real} = begin
    return f_ol(xt[1:data_dim,:], m.u(xt[1+data_dim:end,:]))
end
# define feedback
K = [-1.0000   -1.7124  -14.3836   -2.0819;]
K =  -[-0.999999999999999  -1.835395891173184 -18.625508826472966  -2.540809098278660;]

g = Feedback(K)
# g = Chain(Dense(2,16,tanh),Dense(16,1))
f_cl = F_CL(g)


function dense_predict_reverse_ode(ustart, maxlag, tstart)
    t_span = (tstart, tstart-maxlag)
    ODE_func = (u, p, t) -> f0(u)
    prob = ODEProblem(ODE_func, ustart, t_span)
    alg = Tsit5()
    solve(prob, alg, u0=ustart, dense=true, dt=-1.0)
end
function sample_u0(radius)
    Y = randn(4)
    Y = Y / sqrt(sum(abs2, Y))
    return Y * radius
end

## Define oscillator problem
u0_default = [0.0, 0.0, 0.5, 0.0]
tspan_default = (0.0,3.0)


# ustart = u0_default
# t_span = tspan_default
# ODE_func = (u, p, t) -> f0(u)
# prob = ODEProblem(ODE_func, ustart, t_span)
# alg = Tsit5()
# sol = solve(prob, alg, u0=ustart, dense=true)
# plot(sol)
# plot(0:0.01:3.0, t->sol(t)[1])

fff(x) = f_ol(x, K*x)

prob = ODEProblem((u, p, t) -> fff(u), 0.001*u0_default, (0., 10.0))

sol = solve(prob)

plot(sol)
