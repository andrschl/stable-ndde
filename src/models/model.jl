include("../lyapunov/lyapunov.jl")


abstract type AbstractModel end

abstract type AbstractNDEModel <: AbstractModel end
abstract type AbstractNDDEModel <: AbstractNDEModel end

## TODO Neural ODE Model
mutable struct NODE <: AbstractNDEModel

end

## Vanilla NDDE type
mutable struct NDDE{O,P<:AbstractArray,Q<:AbstractArray,R<:Integer,S<:Integer,T<:Function} <: AbstractNDDEModel
    re_f::O
    pf::P
    flags::Q
    data_dim::R
    nfparams::S
    ndde_func!::T
end
# constructors
NDDE(data_dim, re_f, pf; flags=[]) = begin
    function ndde_func!(du, u, h, p, t)
        ut = vcat(u, map(τ -> h(p, t-τ), flags)...)
        du .= re_f(p)(ut)
    end
    NDDE(re_f, pf, flags, data_dim, length(pf), ndde_func!)
end
NDDE(data_dim; flags=[]) = begin
    f = Chain(Dense((data_dim) * (nfdelays + 1), 32, swish),
               Dense(32, 64, swish),
               Dense(64, 64, swish),
               Dense(64, 32, swish),
               Dense(32, data_dim))
    pf,re_f = Flux.destructure(f)
    function ndde_func!(du, u, h, p, t)
        ut = vcat(u, map(τ -> h(p, t-τ), flags)...)
        du .= re_f(p)(ut)
    end
    NDDE(re_f, pf, flags, data_dim, length(pf), ndde_func!)
end

## Razumikhin NDDE type
mutable struct RazNDDE{O,P, Q<:AbstractArray, X<:AbstractArray, R<:AbstractArray, Y<:AbstractArray,
        S<:AbstractFloat, T<:Integer, U<:Integer,V<:AbstractArray, W<:Function} <: AbstractNDDEModel
    re_f::O
    re_v::P
    pf::Q
    pv::X
    flags::R
    vlags::Y
    α::S
    q::S
    data_dim::T
    nfparams::U
    nvparams::U
    fmask::V
    ndde_func!::W
end
# constructors
RazNDDE(data_dim;flags=[],vlags=[], α=0.1, q=1.1, act=C2_relu) = begin
    nfdelays = length(flags)
    v = Lyapunov(data_dim, act=act)
    f = Chain(Dense((data_dim) * (nfdelays + 1), 32, swish),
               Dense(32, 64, swish),
               Dense(64, 64, swish),
               Dense(64, 32, swish),
               Dense(32, data_dim))
    pv,re_v = Flux.destructure(v)
    nvparams = length(pv)
    pf,re_f = Flux.destructure(f)
    nfparams = length(pf)
    # build fmask needed for lyapunov loss
    lags = sort(union(flags,vlags))
    indices_f = sort(union(map(d->findfirst(isequal(d), lags), flags),0))
    fmask = vcat(map(i->Array(i*data_dim+1:(i+1)*data_dim), indices_f)...)
    function ndde_func!(du, u, h, p, t)
        ut = vcat(u, map(τ -> h(p, t-τ), flags)...)
        du .= re_f(p)(ut)
    end
    RazNDDE(re_f, re_v, pf, pv, flags, vlags, α, q, data_dim, nfparams, nvparams, fmask, ndde_func!)
end
RazNDDE(data_dim, pf, pv;flags=[],vlags=[], α=0.2, q=1.1, act=C2_relu) = begin
    nfdelays = length(flags)
    v = Lyapunov(data_dim, act=act)
    f = Chain(Dense((data_dim) * (nfdelays + 1), 32, swish),
               Dense(32, 64, swish),
               Dense(64, 64, swish),
               Dense(64, 32, swish),
               Dense(32, data_dim))
    _,re_v = Flux.destructure(v)
    nvparams = length(pv)
    _,re_f = Flux.destructure(f)
    nfparams = length(pf)
    # build fmask needed for lyapunov loss
    lags = sort(union(flags,vlags))
    indices_f_only = sort(union(map(d->findfirst(isequal(d), lags), flags),0))
    fmask = vcat(map(i->Array(i*data_dim+1:(i+1)*data_dim), indices_f_only)...)
    function ndde_func!(du, u, h, p, t)
        ut = vcat(u, map(τ -> h(p, t-τ), flags)...)
        du .= re_f(p)(ut)
    end
    RazNDDE(re_f, re_v, pf, pv, flags, vlags, α, q, data_dim, nfparams, nvparams,fmask, ndde_func!)
end
RazNDDE(data_dim, re_f, re_v, pf, pv; flags=[],vlags=[], α=0.2, q=1.1, act=C2_relu) = begin
    nvparams = length(pv)
    nfparams = length(pf)
    # build fmask needed for lyapunov loss
    lags = sort(union(flags,vlags))
    indices_f_only = sort(union(map(d->findfirst(isequal(d), lags), flags),0))
    fmask = vcat(map(i->Array(i*data_dim+1:(i+1)*data_dim), indices_f_only)...)
    function ndde_func!(du, u, h, p, t)
        ut = vcat(u, map(τ -> h(p, t-τ), flags)...)
        du .= re_f(p)(ut)
    end
    RazNDDE(re_f, re_v, pf, pv, flags, vlags, α, q, data_dim, nfparams, nvparams,fmask, ndde_func!)
end

## Razumikhin NDDE type
mutable struct KrasNDDE{O,P, Q<:AbstractArray, X<:AbstractArray, R<:AbstractArray, Y<:AbstractArray,
        S<:AbstractFloat, T<:Integer, U<:Integer,V<:AbstractArray, W<:Function} <: AbstractNDDEModel
    re_f::O
    re_v::P
    pf::Q
    pv::X
    flags::R
    vlags::Y
    α::S
    q::S
    data_dim::T
    nfparams::U
    nvparams::U
    fmask::V
    vmask::V
    ndde_func!::W
end
# constructors
KrasNDDE(data_dim;flags=[],vlags=[], α=0.1, q=1.1, act=C2_relu) = begin
    nfdelays = length(flags)
    f = Chain(Dense((data_dim) * (nfdelays + 1), 32, swish),
               Dense(32, 64, swish),
               Dense(64, 64, swish),
               Dense(64, 32, swish),
               Dense(32, data_dim))
    v = Lyapunov(data_dim * (length(vlags)+1), act=act)
    pv,re_v = Flux.destructure(v)
    nvparams = length(pv)
    pf,re_f = Flux.destructure(f)
    nfparams = length(pf)
    # build fmask needed for lyapunov loss
    lags = sort(union(flags,vlags))
    indices_f = sort(union(map(d->findfirst(isequal(d), lags), flags),0))
    indices_v = sort(union(map(d->findfirst(isequal(d), lags), vlags),0))
    fmask = vcat(map(i->Array(i*data_dim+1:(i+1)*data_dim), indices_f)...)
    vmask = vcat(map(i->Array(i*data_dim+1:(i+1)*data_dim), indices_v)...)
    function ndde_func!(du, u, h, p, t)
        ut = vcat(u, map(τ -> h(p, t-τ), flags)...)
        du .= re_f(p)(ut)
    end
    KrasNDDE(re_f, re_v, pf, pv, flags, vlags, α, q, data_dim, nfparams, nvparams, fmask, vmask, ndde_func!)
end
KrasNDDE(data_dim, pf, pv;flags=[],vlags=[], α=0.2, q=1.1, act=C2_relu) = begin
    nfdelays = length(flags)
    f = Chain(Dense((data_dim) * (nfdelays + 1), 32, swish),
               Dense(32, 64, swish),
               Dense(64, 64, swish),
               Dense(64, 32, swish),
               Dense(32, data_dim))
    v = Lyapunov(data_dim * (length(vlags)+1), act=act)
    _,re_v = Flux.destructure(v)
    nvparams = length(pv)
    _,re_f = Flux.destructure(f)
    nfparams = length(pf)
    # build fmask needed for lyapunov loss
    lags = sort(union(flags,vlags))
    indices_f = sort(union(map(d->findfirst(isequal(d), lags), flags),0))
    indices_v = sort(union(map(d->findfirst(isequal(d), lags), vlags),0))
    fmask = vcat(map(i->Array(i*data_dim+1:(i+1)*data_dim), indices_f)...)
    vmask = vcat(map(i->Array(i*data_dim+1:(i+1)*data_dim), indices_v)...)
    function ndde_func!(du, u, h, p, t)
        ut = vcat(u, map(τ -> h(p, t-τ), flags)...)
        du .= re_f(p)(ut)
    end
    KrasNDDE(re_f, re_v, pf, pv, flags, vlags, α, q, data_dim, nfparams, nvparams,fmask,vmask, ndde_func!)
end
KrasNDDE(data_dim, re_f, re_v, pf, pv; flags=[],vlags=[], α=0.2, q=1.1, act=C2_relu) = begin
    nvparams = length(pv)
    nfparams = length(pf)
    # build fmask needed for lyapunov loss
    lags = sort(union(flags,vlags))
    indices_f = sort(union(map(d->findfirst(isequal(d), lags), flags),0))
    indices_v = sort(union(map(d->findfirst(isequal(d), lags), vlags),0))
    fmask = vcat(map(i->Array(i*data_dim+1:(i+1)*data_dim), indices_f)...)
    vmask = vcat(map(i->Array(i*data_dim+1:(i+1)*data_dim), indices_v)...)
    function ndde_func!(du, u, h, p, t)
        ut = vcat(u, map(τ -> h(p, t-τ), flags)...)
        du .= re_f(p)(ut)
    end
    KrasNDDE(re_f, re_v, pf, pv, flags, vlags, α, q, data_dim, nfparams, nvparams,fmask,vmask,  ndde_func!)
end


## prediction
function predict_ndde(u0::Array{T,1}, h0::Function, t::AbstractArray, pf::AbstractArray, m::AbstractNDDEModel; alg=Tsit5()) where {T<:Real}
    t_span = (first(t), last(t))
    prob = DDEProblem(m.ndde_func!, u0, h0, t_span, constant_lags=m.flags)
    alg = MethodOfSteps(alg)
    return Array(solve(prob, alg, p=pf, saveat=t, sensealg = ReverseDiffAdjoint()))
end
function predict_ndde(u0::Array{T,2}, h0::Function, t::AbstractArray, pf::AbstractArray, m::AbstractNDDEModel; alg=Tsit5()) where {T<:Real}
    t_span = (first(t), last(t))
    prob = DDEProblem(m.ndde_func!, u0, h0, t_span, constant_lags=m.flags)
    alg = MethodOfSteps(alg)
    # return cat(reshape.(solve(prob, alg, p=pf, saveat=t, sensealg = ReverseDiffAdjoint()).u, m.data_dim, 1, :)..., dims=2)
    return permutedims(Array(solve(prob, alg, p=pf, saveat=t, sensealg = ReverseDiffAdjoint())), [1,3,2])
end
function dense_predict_ndde(u0::AbstractArray, h0::Function, t_span::Tuple, pf::AbstractArray, m::AbstractNDDEModel; alg=Tsit5())
    prob = DDEProblem(m.ndde_func!, u0, h0, t_span, constant_lags=m.flags)
    alg = MethodOfSteps(alg)
    return solve(prob, alg, p=pf, dense=true)
end

## loss
function LS_loss(x,y)
    # dot(x-y, x-y)
    sum(abs2, x-y)
end
function predict_ndde_loss(u0::AbstractArray, h0::Function, t::AbstractArray, u_data::AbstractArray, pf::AbstractArray, m::AbstractNDDEModel; loss_func=LS_loss, N=1)
    u_pred = predict_ndde(u0, h0, t, pf, m)
    loss = loss_func(u_pred, u_data) / N
    return loss, u_pred
end

## include razumikhin
include("../lyapunov/razumikhin.jl")
include("../lyapunov/krasovskii.jl")
