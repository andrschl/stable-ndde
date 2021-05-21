include("../util/import.jl")

abstract type AbstractDataset end
# type for DDE ground truth dynamics
abstract type AbstractDDEDataset <: AbstractDataset end
# type for ODE ground truth dynamics
abstract type AbstractODEDataset <: AbstractDataset end

# type for dataset for DDE model based on DDE ground truth
# for now hard-coded with constant initial histories
mutable struct DDEDDEDataset <: AbstractDDEDataset
    trajs
    noisy_trajs
    state_dim::Int64
    obs_ids::Array{Int64,1}
    h0s
    tspan::Tuple{Float64,Float64}
    Δt::Float64
    N::Int64
    N_hist::Int64
    p
    prob::DDEProblem
    umax::Float64
    callback
    constant_lags::Array{Float64,1}
    r::Float64
end

DDEDDEDataset(h0s, tspan, Δt, prob, r, constant_lags; obs_ids=Array(1:length(h0s[1](prob.p, tspan[1]))), umax=100.0) = begin
    trajs = []
    state_dim = length(h0s[1](prob.p, first(tspan)))
    N = Int(floor((last(tspan) - first(tspan) - r + constant_lags[end])/Δt)) + 1
    N_hist = Int(r/Δt) + 1
    p = prob.p
    callback = nothing
    @assert r >= constant_lags[end]
    DDEDDEDataset(trajs, [], state_dim, obs_ids, h0s, tspan, Δt, N, N_hist, p, prob, umax, callback, constant_lags, r)
end
# type for dataset for ODE model based on DDE ground truth -> not needed at the moment
mutable struct ODEDDEDataset <: AbstractDDEDataset
    trajs
    noisy_trajs
    σ::Float64
    state_dim::Int64
    obs_ids::Array{Int64,1}
    h0s
    tspan::Tuple{Float64,Float64}
    Δt::Float64
    N::Int64
    N_hist::Int64
    p
    prob::DDEProblem
    umax::Float64
    callback
    constant_lags::Array{Float64,1}
    r::Float64
end
# type for dataset for DDE model based on ODE ground truth
mutable struct DDEODEDataset <: AbstractODEDataset
    trajs
    noisy_trajs
    σ::Float64
    state_dim::Int64
    obs_ids::Array{Int64,1}
    ICs
    tspan::Tuple{Float64,Float64}
    Δt::Float64
    N::Int64
    N_hist::Int64
    p
    prob::ODEProblem
    umax::Float64
    callback
    r::Float64
    const_init::Bool
end
DDEODEDataset(ICs, tspan, Δt, prob, r; obs_ids=1:length(ICs[1]), umax=100.0, const_init=false) = begin
    trajs = []
    state_dim = length(ICs[1])
    N = Int(floor((last(tspan) - first(tspan))/Δt)) + 1
    N_hist = Int(floor(r/Δt)) + 1
    p = prob.p
    callback = nothing
    DDEODEDataset(trajs, [], 0.0, state_dim, obs_ids, ICs, tspan, Δt, N, N_hist, p, prob, umax, callback, r, const_init)
end

# type for dataset for ODE model based on ODE ground truth
mutable struct ODEODEDataset <: AbstractODEDataset
    trajs
    noisy_trajs
    σ::Float64
    state_dim::Int64
    obs_ids::Array{Int64,1}
    ICs
    tspan::Tuple{Float64,Float64}
    Δt::Float64
    N::Int64
    N_hist::Int64
    p
    prob::ODEProblem
    umax::Float64
    callback
    r::Float64
    const_init::Bool
end

## Core functionality:

# update the problem
function remake!(d::AbstractODEDataset; kwargs...)
    for (key, value) in kwargs
        if key != :prob
            setfield!(d, Symbol(key), value)
        end
        if key == :tspan
            d.N = Int(ceil((last(d.tspan)-first(d.tspan))/d.Δt))+1
        end
    end
    d.prob = remake(d.prob, tspan=d.tspan, p=d.p)
end
function remake!(d::AbstractDDEDataset; kwargs...)
    for (key, value) in kwargs
        if key != :prob
            setfield!(d, Symbol(key), value)
        end
        if key == :tspan
            d.N = Int(ceil((last(d.tspan)-first(d.tspan)-d.r+d.constant_lags)/d.Δt))+1
        end
    end
    d.prob = remake(d.prob, tspan=d.tspan, p=d.p)
end
# Generate a new data set. Overwrites previous one.
function gen_dataset!(d::AbstractDataset; alg=Tsit5(),dense=true, kwargs...)

    # change setup
    remake!(d; kwargs...)

    # stopping condition to avoid finite escape
    condition(u,t,integrator) = sum(abs2, u) >= d.umax^2
    affect!(integrator) = terminate!(integrator)
    d.callback = DiscreteCallback(condition, affect!)

    # generate trajectories
    d.trajs = gen_trajs(d; alg=alg, dense=dense)
end
function gen_noise!(d::AbstractDataset, σ::Real)
    d.σ = σ
    d.noisy_trajs = []
    for i in 1:length(d.trajs)
        t = d.trajs[i][1]
        u_noisy = map(j -> d.trajs[i][2][j] + σ*randn(length(d.trajs[i][2][1])), 1:length(d.trajs[i][2]))
        push!(d.noisy_trajs, [t, u_noisy])
    end
end
## utilities

# helper function to generate trajectories
function gen_trajs(d::AbstractODEDataset; alg=Tsit5(), dense=true)
    t = Array(first(d.tspan):d.Δt:last(d.tspan))
    init_t = reverse(Array(first(d.tspan)-d.Δt:-d.Δt:first(d.tspan)-d.r))
    tot_t = vcat(init_t, t)
    tot_tspan = (t[1]-d.r, t[end])
    d.N_hist = length(init_t) + 1
    trajs = []
    if d.const_init
        for u0 in d.ICs
            prob = remake(d.prob, u0=u0, tspan=d.tspan)
            sol = solve(prob, alg, p=d.p, saveat=t, save_idxs=d.obs_ids, callback=d.callback)
            if dense
                dense_sol = solve(prob, alg, p=d.p, save_idxs=d.obs_ids, dense=true, callback=d.callback)
                tot_sol = ξ -> ξ>=sol.t[1]&&ξ<=sol.t[end] ? dense_sol(ξ) : (ξ<sol.t[1] ? u0[d.obs_ids] : zero(u0[d.obs_ids]))
            else
                tot_sol = nothing
            end
            init_u = repeat([u0[d.obs_ids]], d.N_hist-1)
            push!(trajs, [vcat(init_t, sol.t), vcat(init_u, sol.u), tot_sol])
        end
    else
        for u0 in d.ICs
            prob = remake(d.prob, u0=u0, tspan=tot_tspan)
            sol = solve(prob, alg, p=d.p, saveat=tot_t, save_idxs=d.obs_ids, callback=d.callback)
            if dense
                dense_sol = solve(prob, alg, p=d.p, save_idxs=d.obs_ids, dense=true, callback=d.callback)
                tot_sol = ξ -> ξ>=sol.t[1]-d.r&&ξ<=sol.t[end] ? dense_sol(ξ) : zero(u0[d.obs_ids])
            else
                tot_sol = nothing
            end
            push!(trajs, [sol.t, sol.u, tot_sol])
        end
    end
    return trajs
end
function gen_trajs(d::AbstractDDEDataset; alg=Tsit5(), dense=true)
    t = Array(first(d.tspan):d.Δt:last(d.tspan))
    init_t = reverse(Array(first(d.tspan)-d.Δt:-d.Δt:first(d.tspan)-d.constant_lags[end]))
    tot_t = vcat(init_t, t)
    tot_tspan = (t[1]-d.constant_lags[end], t[end])
    trajs = []
    for h0 in d.h0s
        u0 = h0(d.p, t[1])
        prob = remake(d.prob, u0=u0, h0=h0, tspan=d.tspan, constant_lags=d.constant_lags)
        sol = solve(prob, MethodOfSteps(alg), p=d.p, saveat=t, save_idxs=d.obs_ids, callback=d.callback)
        if dense
            dense_sol = solve(prob, MethodOfSteps(alg), p=d.p, save_idxs=d.obs_ids, dense=true, callback=d.callback)
            tot_sol = ξ -> ξ>=sol.t[1]&&ξ<=sol.t[end] ? dense_sol(ξ) : (ξ<sol.t[1]&&ξ>=tot_tspan[1] ? h0(d.p,ξ)[d.obs_ids] : zero(u0[d.obs_ids]))
        else
            tot_sol = nothing
        end
        init_u = map(ξ -> h0(d.p, ξ), init_t)
        tot_u = vcat(init_u, sol.u)
        new_length = min(length(tot_u), length(tot_t))
        push!(trajs, [tot_t[1:new_length], tot_u[1:new_length], tot_sol])
    end
    return trajs
end

## batching utilities
# helper function to create dataset for training
function Base.getindex(d::ODEODEDataset, idx::Int)
    lengths = map(traj->length(traj[1])-d.N_hist+1, d.trajs)
    cum_lengths = map(n -> sum(lengths[1:n]), 1:length(d.trajs))
    traj_idx = findfirst(n->n>=idx, cum_lengths)
    in_traj_idx = d.N_hist-1 + (idx - vcat([0],cum_lengths)[traj_idx])
    t = d.trajs[traj_idx][1][in_traj_idx]
    ut = d.trajs[traj_idx][2][in_traj_idx]
    t, ut
end
function Base.getindex(d::ODEDDEDataset, idx::Int)
    lengths = map(traj->length(traj[1])-d.N_hist+1, d.trajs)
    cum_lengths = map(n -> sum(lengths[1:n]), 1:length(d.trajs))
    traj_idx = findfirst(n->n>=idx, cum_lengths)
    in_traj_idx = d.N_hist-1 + (idx - vcat([0],cum_lengths)[traj_idx])
    t = d.trajs[traj_idx][1][in_traj_idx]
    ut = d.trajs[traj_idx][2][in_traj_idx]
    t, ut
end
function Base.getindex(d::DDEODEDataset, idx::Int)
    lengths = map(traj -> length(traj[1])-d.N_hist+1, d.trajs)
    cum_lengths = map(n -> sum(lengths[1:n]), 1:length(d.trajs))
    traj_idx = findfirst(n -> n>=idx, cum_lengths)
    in_traj_idx = d.N_hist-1 + (idx - vcat([0],cum_lengths)[traj_idx])
    t = d.trajs[traj_idx][1][in_traj_idx]
    # ut = reverse(vcat(d.trajs[traj_idx][2][in_traj_idx-d.N_hist+1:in_traj_idx]...))
    ut = vcat(reverse(d.trajs[traj_idx][2][in_traj_idx-d.N_hist+1:in_traj_idx])...)
    t, ut
end
function Base.getindex(d::DDEDDEDataset, idx::Int)
    lengths = map(traj -> length(traj[1])-d.N_hist+1, d.trajs)
    cum_lengths = map(n -> sum(lengths[1:n]), 1:length(d.trajs))
    traj_idx = findfirst(n -> n>=idx, cum_lengths)
    in_traj_idx = d.N_hist-1 + (idx - vcat([0],cum_lengths)[traj_idx])
    t = d.trajs[traj_idx][1][in_traj_idx]
    # ut = reverse(vcat(d.trajs[traj_idx][2][in_traj_idx-d.N_hist+1:in_traj_idx]...))
    ut = vcat(reverse(d.trajs[traj_idx][2][in_traj_idx-d.N_hist+1:in_traj_idx])...)
    t, ut
end
function Base.getindex(d::AbstractDataset, ids::Array)
    ts = zeros(length(ids))
    uts = zeros(length(d.obs_ids)*d.N_hist, length(ids))
    for (i, id) in enumerate(ids)
            t, ut = d[id]
            ts[i] = t
            uts[:,i] .= ut
    end
    ts, uts
end
Base.IndexStyle(::Type{AbstractDataset}) = IndexLinear()
function Base.length(d::AbstractDataset)
    nobs = 0
    for traj in d.trajs
        nobs += length(traj[1])-d.N_hist+1
    end
    nobs
end
Base.size(d::AbstractDataset) = (length(d),)
function Flux.Data._nobs(d::AbstractDataset)
    length(d)
end
function Flux.Data._getobs(d::AbstractDataset, ids::Array)
    d[ids]
end

## fit GPs
function get_gp(x, y, σ; k0="RBF")
    ks = Dict("RBF"=>SqExponentialKernel(), "Mat32"=>Matern32Kernel(), "Mat52"=>Matern52Kernel())
    k0 = ks[k0]
    function objective_function(x, y)
        function negative_log_likelihood(params)
            kernel =
                (exp(params[1])+1e-10) * (k0 ∘ ScaleTransform(exp(params[2])+1e-10))
            f = GP(kernel)
            fx = f(x, exp(params[3])+1e-10)
            # fx = f(x, σ)
            return -logpdf(fx, y)
        end
        return negative_log_likelihood
    end
    p0 = [0.0,0.0, log(σ)]
    opt = optimize(objective_function(x, y), p0, LBFGS())
    opt_kernel =
        (exp(opt.minimizer[1])+1e-10) *
        (k0 ∘ ScaleTransform(exp(opt.minimizer[2])+1e-10))
    opt_f = GP(opt_kernel)
    opt_fx = opt_f(x, exp(opt.minimizer[3])+1e-10)
    # opt_fx = opt_f(x, σ)
    opt_p_fx =  posterior(opt_fx, y)

    # # log parameter info
    # global ls, ms, αs
    # push!(ls, exp(opt.minimizer[2])+1e-10)
    # push!(ms, exp(opt.minimizer[1])+1e-10)
    # push!(αs, opt_p_fx.data.α...)

    return opt_p_fx
end

## NDDE batching

# get a batch of NDDE training data
function get_ndde_batch(d::AbstractDataset, batchtime::Integer, batchsize::Integer)
    ntrajs = length(d.trajs)
    data_dim = length(d.trajs[1][2][1])
    tot_size = length(d) - (batchtime-1)*ntrajs
    s = sample(Array(1:tot_size), batchsize, replace=false)
    lengths = map(traj -> length(traj[1])-d.N_hist+1-batchtime+1, d.trajs)
    cum_lengths = map(n -> sum(lengths[1:n]), 1:length(d.trajs))

    us = zeros(data_dim, batchtime, batchsize)
    ts = zeros(batchtime, batchsize)
    traj_ids = []

    for (i, idx) in enumerate(s)
        traj_idx = findfirst(n -> n>=idx, cum_lengths)
        in_traj_idx = d.N_hist-1 + (idx - vcat([0],cum_lengths)[traj_idx])
        us[:, :, i] = hcat(d.trajs[traj_idx][2][in_traj_idx:in_traj_idx+batchtime-1]...)
        ts[:, i] = d.trajs[traj_idx][1][in_traj_idx:in_traj_idx+batchtime-1]
        push!(traj_ids, traj_idx)
    end
    return ts, us, traj_ids
end
function get_batch_h0(ts::AbstractArray, us::AbstractArray, traj_ids::AbstractArray, d::AbstractDataset)
    t0 = ts[1,:]
    return (p, ξ) -> hcat(map(i -> d.trajs[traj_ids[i]][3](ξ + t0[i]), 1:length(t0))...)
end
function get_ndde_batch_and_h0(d::AbstractDataset, batchtime::Integer, batchsize::Integer)
    ts, us, traj_ids = get_ndde_batch(d, batchtime, batchsize)
    h0 = get_batch_h0(ts, us, traj_ids, d)
    return ts, us, h0, traj_ids
end
function get_h0(d::AbstractDataset, traj_idx::Integer)
    return (p, ξ) -> d.trajs[traj_idx][3](ξ)
end
function get_noisy_ndde_batch(d::AbstractDataset, batchtime::Integer, batchsize::Integer)
    ntrajs = length(d.trajs)
    data_dim = length(d.trajs[1][2][1])
    tot_size = length(d) - (batchtime-1)*ntrajs
    s = sample(Array(1:tot_size), batchsize, replace=false)
    lengths = map(traj -> length(traj[1])-d.N_hist+1-batchtime+1, d.trajs)
    cum_lengths = map(n -> sum(lengths[1:n]), 1:length(d.trajs))

    us = zeros(data_dim, batchtime, batchsize)
    ts = zeros(batchtime, batchsize)
    traj_ids = []

    for (i, idx) in enumerate(s)
        traj_idx = findfirst(n -> n>=idx, cum_lengths)
        in_traj_idx = d.N_hist-1 + (idx - vcat([0],cum_lengths)[traj_idx])
        us[:, :, i] = hcat(d.noisy_trajs[traj_idx][2][in_traj_idx:in_traj_idx+batchtime-1]...)
        ts[:, i] = d.noisy_trajs[traj_idx][1][in_traj_idx:in_traj_idx+batchtime-1]
        push!(traj_ids, (traj_idx,in_traj_idx))
    end
    return ts, us, traj_ids
end

# only working for 1D data currently -> simply add univariate gps for higher dimensions
# maybe define gps to be zero outside history interval
function get_noisy_batch_h0(ts::AbstractArray, us::AbstractArray, traj_ids::AbstractArray, d::AbstractDataset; k0="RBF")
    t0 = ts[1,:]
    h0s = []
    for (traj_idx, in_traj_idx) in traj_ids

        t_hist = d.trajs[traj_idx][1][in_traj_idx-d.N_hist+1:in_traj_idx]
        u_hist = hcat(d.noisy_trajs[traj_idx][2][in_traj_idx-d.N_hist+1:in_traj_idx]...)

        # fit gp
        h0 = []
        for i in 1:length(u_hist[:,1])
            # mZero = MeanZero()
            # # kern = SE(0.0,0.0)
            # kern = Mat32Iso(0.0,0.0)
            # logObsNoise = log(d.σ)
            # gp = GaussianProcesses.GP(Array{Float64}(t_hist), Array{Float64}(u_hist[i,:]), mZero, kern, logObsNoise)
            # optimize!(gp)
            # push!(h0, t -> predict_y(gp, [t])[1])

            # f= GP(SqExponentialKernel())
            # fx = f(t_hist, d.σ)
            # p_fx =  posterior(fx, u_hist[i,:])
            p_fx = get_gp(t_hist, u_hist[i,:], d.σ,k0=k0)
            push!(h0, t -> mean(p_fx([t])))

            # # show gps
            # pl=plot()
            # plot!(pl, t_hist[1]:0.01:t_hist[end], p_fx)
            # # plot!(pl,gp)
            # scatter!(pl, t_hist, u_hist[i,:])
            # plot!(pl, t->d.trajs[traj_idx][3](t)[i], title=string(traj_idx)*"_"*string(i))
            # display(pl)
        end
        push!(h0s, t -> vcat(map(i -> h0[i](t), 1:length(h0))...))


    end
    return (p, ξ) -> hcat(map(i -> h0s[i](ξ + t0[i]), 1:length(t0))...)
end

# only working for 1D data currently -> simply add univariate gps for higher dimensions
# maybe define gps to be zero outside history interval
function get_noisy_h0(d::AbstractDataset, traj_idx::Integer; k0="RBF")
    t_hist = d.trajs[traj_idx][1][1:d.N_hist]
    u_hist = hcat(d.noisy_trajs[traj_idx][2][1:d.N_hist]...)

    # fit gp
    h0 = []
    for i in 1:length(u_hist[:,1])
        # mZero = MeanZero()
        # kern = SE(0.0,0.0)
        # logObsNoise = log(d.σ)
        # gp = GP(Array{Float64}(t_hist), Array{Float64}(u_hist[i,:]), mZero, kern, logObsNoise)
        # optimize!(gp)
        # push!(h0, t -> predict_y(gp, [t])[1])

        # f= GP(SqExponentialKernel())
        # fx = f(t_hist, d.σ)
        # p_fx =  posterior(fx, u_hist[i,:])
        p_fx = get_gp(t_hist, u_hist[i,:], d.σ,k0=k0)
        push!(h0, t -> mean(p_fx([t])))

        # # show gps
        # pl=plot()
        # plot!(pl, t_hist[1]:0.01:t_hist[end], p_fx)
        # # plot!(pl,gp)
        # scatter!(pl, t_hist, u_hist[i,:])
        # plot!(pl, t->d.trajs[traj_idx][3](t)[i], title=string(traj_idx)*"_"*string(i))
        # display(pl)
    end
    #
    # f= GP(SqExponentialKernel())
    # fx = f(t_hist, d.σ)
    # p_fx =  posterior(fx, u_hist[1,:])

    return (p,t) -> vcat(map(i -> h0[i](t), 1:length(h0))...)
    # return (p,t) -> predict_y(gp, [t])[1]
    # return (p,t) -> mean(p_fx([t]))
end


function get_noisy_ndde_batch_and_h0(d::AbstractDataset, batchtime::Integer, batchsize::Integer; k0="RBF")
    ts, us, traj_ids = get_noisy_ndde_batch(d, batchtime, batchsize)
    h0 = get_noisy_batch_h0(ts, us, traj_ids, d, k0=k0)
    return ts, us, h0, traj_ids
end
