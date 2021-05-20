### Lyapunov function

## some helper functions

function C2_relu(x)
    d = Float32(0.1)
    oftype(x, ifelse(x <= d, ifelse(x>=0, x^3/d^2 - x^4/(2*d^3), zero(x)), x - d/2))
end
d = 0.1
Zygote.@adjoint C2_relu(x) = C2_relu(x), y -> (oftype(x, y*ifelse(x <= d, ifelse(x>=0, 3*x^2/d^2 - 4*x^3/(2*d^3), 0.0), 1.0)),)
dC2_relu(x) = oftype(x, ifelse(x <= d, ifelse(x>=0, 3*x^2/d^2 - 4*x^3/(2*d^3), 0.0), 1.0))

# step
function heaviside(t)
   0.5 * (sign(t) + 1.0)
end

## input convex neural network

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
(m::ICNNLayer)(z::AbstractArray, x::AbstractArray) = begin
    m.act.(m.W*x .+ softplus.(m.U)*z .+ m.b)
end
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

## Lyapunov Function

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
(m::Lyapunov)(x::Array{T,2}) where {T<:Real} = begin
    g = m.icnn(x)
    g0 = m.icnn(zero(x))
    z = m.act.(g - g0) .+ m.eps * reshape(dot.(eachcol(x),eachcol(x)),1,:)
    return z
end
(m::Lyapunov)(x::Array{T,1}) where {T<:Real} = begin
    g = m.icnn(x)
    g0 = m.icnn(zero(x))
    z = m.act.(g - g0) .+ m.eps * dot(x,x)
    return z
end

function forwardgrad(m::Lyapunov, x::AbstractArray)
    # inlayer
    W0 = m.icnn.InLayer.weight
    b0 = m.icnn.InLayer.bias
    # W0 = m.icnn.InLayer.W
    # b0 = m.icnn.InLayer.b
    y0 = W0*x+b0
    z1 = m.act.(y0)
    # dz1dx = adjoint(m.act).(y0) .* W0
    dz1dx = dC2_relu.(y0) .* W0
    # hlayer1
    W1,U1,b1 = m.icnn.HLayer1.W,softplus.(m.icnn.HLayer1.U),m.icnn.HLayer1.b
    y1 = U1*z1 + W1*x + b1
    z2 = m.act.(y1)
    # dz2dx = adjoint(m.act).(y1) .* (U1*dz1dx + W1)
    dz2dx = dC2_relu.(y1) .* (U1*dz1dx + W1)
    # hlayer2
    W2,U2,b2 = m.icnn.HLayer2.W,softplus.(m.icnn.HLayer2.U),m.icnn.HLayer2.b
    y2 = U2*z2 + W2*x + b2
    z3 = m.act.(y2)
    # dz3dx = adjoint(m.act).(y2) .* (U2*dz2dx + W2)
    dz3dx = dC2_relu.(y2) .* (U2*dz2dx + W2)

    # output
    yout = z3 - m.icnn(zero(x))
    v = m.act.(yout) .+ m.eps * x'*x
    # vx = adjoint(m.act).(yout) .* dz3dx)[1,:]  + 2*m.eps*x
    vx = (dC2_relu.(yout) .* dz3dx)[1,:]  + 2*m.eps*x

    return v, vx
end

function forwardgrad_batched(m::Lyapunov, x::AbstractArray)
    # inlayer
    W0 = m.icnn.InLayer.weight
    b0 = m.icnn.InLayer.bias
    # W0 = m.icnn.InLayer.W
    # b0 = m.icnn.InLayer.b
    y0 = W0*x.+b0
    z1 = m.act.(y0)
    # a1 = adjoint(m.act).(y0)
    a1 = dC2_relu.(y0)
    a1 = reshape(a1, size(a1, 1), 1, size(a1, 2))
    dz1dx = a1 .* W0
    # hlayer1
    W1,U1,b1 = m.icnn.HLayer1.W,softplus.(m.icnn.HLayer1.U),m.icnn.HLayer1.b
    y1 = U1*z1 .+ W1*x .+ b1
    z2 = m.act.(y1)
    # a2 = adjoint(m.act).(y1)
    a2 = dC2_relu.(y1)
    a2 = reshape(a2, size(a2, 1), 1, size(a2, 2))
    dz2dx = a2 .* (batched_mul(U1, dz1dx) .+ W1)
    # hlayer2
    W2,U2,b2 = m.icnn.HLayer2.W,softplus.(m.icnn.HLayer2.U),m.icnn.HLayer2.b
    y2 = U2*z2 .+ W2*x .+ b2
    z3 = m.act.(y2)
    # a3 = adjoint(m.act).(y2)
    a3 = dC2_relu.(y2)
    a3 = reshape(a3, size(a3, 1), 1, size(a3, 2))
    # println(size(U2), size(dz2dx))
    dz3dx = a3 .* (batched_mul(U2, dz2dx) .+ W2)
    # output
    yout = z3 - m.icnn(zero(x))
    v = m.act.(yout) .+ m.eps * reshape(dot.(eachcol(x),eachcol(x)),1,:)
    # a4 = adjoint(m.act).(yout)
    a4 = dC2_relu.(yout)
    a4 = reshape(a4, size(a4, 1), 1, size(a4, 2))
    vx = (a4 .* dz3dx)[1,:,:]  .+ 2*m.eps*x
    return v, vx
end
