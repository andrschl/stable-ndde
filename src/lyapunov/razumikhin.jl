
# Razumikhin function and loss

################################################################################
# some helper functions

function C2_relu(x)
    d = Float32(0.1)
    oftype(x, ifelse(x <= d, ifelse(x>=0, x^3/d^2 - x^4/(2*d^3), zero(x)), x - d/2))
end
d = 0.1
Zygote.@adjoint C2_relu(x) = C2_relu(x), y -> (oftype(x, y*ifelse(x <= d, ifelse(x>=0, 3*x^2/d^2 - 4*x^3/(2*d^3), 0.0), 1.0)),)

# step
function heaviside(t)
   0.5 * (sign(t) + 1.0)
end

################################################################################
# input convex neural network

# ICNN Layer
struct ICNNLayer{R<:AbstractArray,S<:AbstractArray, T<:AbstractArray}
    W::R
    U::S
    b::T
    act
end
Flux.@functor ICNNLayer
# constructor
ICNNLayer(z_in::Integer, x_in::Integer, out::Integer, activation) =
    ICNNLayer(randn(out, x_in), randn(out, z_in), randn(out), activation)
# forward pass
(m::ICNNLayer)(z::AbstractArray, x::AbstractArray) = m.act.(m.W*x + softplus.(m.U)*z + m.b)

# ICNN
struct ICNN{S,T,U}
    InLayer::S
    HLayer1::T
    HLayer2::U
    act
end
# constructor
ICNN(input_dim::Integer, layer_sizes::Vector, activation) = begin
    InLayer = Dense(input_dim, layer_sizes[1])
    HLayer1 = ICNNLayer(layer_sizes[1], input_dim, layer_sizes[2], activation)
    HLayer2 = ICNNLayer(layer_sizes[2], input_dim, 1, activation)
    ICNN(InLayer, HLayer1, HLayer2, activation)
end
Flux.@functor ICNN
# forward pass
(m::ICNN)(x::AbstractArray) = begin
    z = m.act.(m.InLayer(x))
    z = m.HLayer1(z, x)
    z = m.HLayer2(z, x)
    return z
end

################################################################################
# Lyapunov Function

struct Lyapunov{R, S, T<:AbstractFloat, U<:AbstractFloat}
    icnn::R
    act::S
    d::T
    eps::U
end
Flux.@functor Lyapunov
# constructor
Lyapunov(input_dim::Integer; d=0.1, eps=1e-3, layer_sizes::Vector=[64, 64], act=C2_relu) = begin
    icnn = ICNN(input_dim, layer_sizes, act)
    Lyapunov(icnn, act, Float32(d), Float32(eps))
end
# forward pass
(m::Lyapunov)(x::AbstractArray) = begin
    g = m.icnn(x)
    g0 = m.icnn(zero(x))
    z = m.act.(g - g0) .+ m.eps * x'*x
    return z
end

function forwardgrad(m::Lyapunov, x::AbstractArray)
    # inlayer
    W0 = m.icnn.InLayer.weight
    b0 = m.icnn.InLayer.bias
    y0 = W0*x+b0
    z1 = m.act.(y0)
    dz1dx = adjoint(m.act).(y0) .* W0
    # hlayer1
    W1,U1,b1 = m.icnn.HLayer1.W,softplus.(m.icnn.HLayer1.U),m.icnn.HLayer1.b
    y1 = U1*z1 + W1*x + b1
    z2 = m.act.(y1)
    dz2dx = adjoint(m.act).(y1) .* (U1*dz1dx + W1)
    # hlayer2
    W2,U2,b2 = m.icnn.HLayer2.W,softplus.(m.icnn.HLayer2.U),m.icnn.HLayer2.b
    y2 = U2*z2 + W2*x + b2
    z3 = m.act.(y2)
    dz3dx = adjoint(m.act).(y2) .* (U2*dz2dx + W2)
    # output
    yout = z3 - m.icnn(zero(x))
    v = m.act.(yout) .+ m.eps * x'*x
    vx = (adjoint(m.act).(yout) .* dz3dx)[1,:]  + 2*m.eps*x
    return v, vx
end

################################################################################
# dynamics
struct StableDynamics{S, T<:AbstractFloat, U<:AbstractFloat}
    v::Lyapunov
    f_hat::S
    alpha::T
    nfdelays::Integer
    nvdelays::Integer
    p::U
    data_dim::Integer
end
Flux.@functor StableDynamics
# constructor
StableDynamics(data_dim; nfdelays=0, nvdelays=0, α=0.01, q=1.0202, act=C2_relu) = begin
    v = Lyapunov(data_dim, act=act)
    f_hat = Chain(Dense((data_dim) * (nfdelays + 1), 32, swish),
               Dense(32, 64, swish),
               Dense(64, 64, swish),
               Dense(64, 32, swish),
               Dense(32, data_dim))
    StableDynamics(v, f_hat, α, nfdelays, nvdelays, q, data_dim)
end
# forward pass
(m::StableDynamics)(xt::AbstractArray, yt::AbstractArray) = begin
    data_dim = m.data_dim
    x = xt[1:data_dim]
    if x == zero(x)
        return zero(x)
    end
    v, vx = forwardgrad(m.v, x)
    # razumikhin condition
    raz_fac = 1.0
    for i in 1:m.nvdelays
        # raz_fac = raz_fac * heaviside(m.p*v[1] - m.v(yt[i*data_dim + 1:(i+1)*data_dim])[1])
        raz_fac = raz_fac * C2_smooth_step(m.p*v[1] - m.v(yt[i*data_dim + 1:(i+1)*data_dim])[1])
    end
    # return m.f_hat(xt) - vx * relu(vx'*m.f_hat(xt) + m.alpha * v[1]) / sum(abs2, vx) * raz_fac
    return m.f_hat(xt) - vx * C2_relu_shifted(vx'*m.f_hat(xt) + m.alpha * v[1]) / sum(abs2, vx) * raz_fac
end

################################################################################
# soft stability
mutable struct StableDynamicsSoft{O, P, Q<:AbstractArray, R<:AbstractArray,
        S<:AbstractArray, T<:AbstractArray, U<:AbstractFloat,V<:AbstractFloat}
    re_f::O
    re_v::P
    p::Q
    q::R
    flags::S
    vlags::T
    α::U
    β::V
    data_dim::Integer
    nfparams::Integer
    nvparams::Integer
end
# constructor
StableDynamicsSoft(data_dim;flags=[],vlags=[], α=0.1, β=1.1, act=C2_relu) = begin
    nfdelays = length(flags)
    v = Lyapunov(data_dim, act=act)
    f = Chain(Dense((data_dim) * (nfdelays + 1), 32, swish),
               # Dense(32,32,swish),
               Dense(32, 64, swish),
               Dense(64, 64, swish),
               Dense(64, 32, swish),
               Dense(32, data_dim))
    q,re_v = Flux.destructure(v)
    nvparams = length(q)
    p,re_f = Flux.destructure(f)
    nfparams = length(p)
    StableDynamicsSoft(re_f, re_v, p, q, flags, vlags, α, β, data_dim, nfparams, nvparams)
end
StableDynamicsSoft(data_dim, p, q;flags=[],vlags=[], α=0.2, β=1.1, act=C2_relu) = begin
    nfdelays = length(flags)
    v = Lyapunov(data_dim, act=act)
    f = Chain(Dense((data_dim) * (nfdelays + 1), 32, swish),
               # Dense(32,32,swish),
               Dense(32, 64, swish),
               Dense(64, 64, swish),
               Dense(64, 32, swish),
               Dense(32, data_dim))
    _,re_v = Flux.destructure(v)
    nvparams = length(q)
    _,re_f = Flux.destructure(f)
    nfparams = length(p)
    StableDynamicsSoft(re_f, re_v, p, q, flags, vlags, α, β, data_dim, nfparams, nvparams)
end
StableDynamicsSoft(data_dim, re_f, re_v, p, q; flags=[],vlags=[], α=0.2, β=1.1, act=C2_relu) = begin
    nvparams = length(q)
    nfparams = length(p)
    StableDynamicsSoft(re_f, re_v, p, q, flags, vlags, α, β, data_dim, nfparams, nvparams)
end

(m::StableDynamicsSoft)(xt::AbstractArray, yt::AbstractArray, p::AbstractArray, q::AbstractArray) = begin
    data_dim = m.data_dim
    nfdelays = length(m.flags)
    nvdelays = length(m.vlags)
    x = xt[1:data_dim]
    if x == zero(x)
        return zero(x)
    end
    v, vx = forwardgrad(m.re_v(q), x)

    # razumikhin condition
    v_list = []
    for i in 1:nvdelays
        push!(v_list, m.re_v(q)(yt[i*data_dim + 1:(i+1)*data_dim])[1])
    end
    v_max = max(v_list)
    raz_fac = heaviside(m.β*v[1] - v_max)

    # raz_fac = 1.0
    # for i in 1:nvdelays
    #     raz_fac = raz_fac * heaviside(m.β*v[1] - m.re_v(q)(yt[i*data_dim + 1:(i+1)*data_dim])[1])
    # end
    return relu(vx'*m.re_f(p)(xt) + m.α * v[1]) * raz_fac
end




function true_loss_nn(xt::AbstractArray, yt::AbstractArray, p::AbstractArray, q::AbstractArray, m::StableDynamicsSoft)
    x = xt[1:m.data_dim]
    v, vx = forwardgrad(m.re_v(q), x)
    # v,vx = x'*x, 2*x
    # v  = m.re_v(q)(x)
    # vx = dVQuadratic(m.re_v(q),x)

    # raz_fac = 1
    # for i in 1:length(m.vlags)
    #     raz_fac = raz_fac * heaviside(m.β*v[1] - m.re_v(q)(yt[i*m.data_dim + 1:(i+1)*m.data_dim])[1])
    # end

    past_vs = map(i -> m.re_v(q)(yt[i*m.data_dim + 1:(i+1)*m.data_dim])[1], Array(1:length(m.vlags)))
    vmax = maximum(past_vs)
    raz_fac = heaviside(m.β*v[1] - vmax)

    # return relu(vx'*m.re_f(p)(xt) + m.α * v) * raz_fac
    return relu(vx'*m.re_f(p)(xt) + m.α * v[1]) * raz_fac / (v[1] + 1e-3)
    # return leakyrelu(vx'*m.re_f(p)(xt) + m.α * v[1]) * raz_fac
    # return relu(vx'*m.re_f(p)(xt) + m.α * v) * raz_fac
end
