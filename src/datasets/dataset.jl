include("../util/import.jl")

abstract type AbstractDataset end
# type for DDE ground truth dynamics
abstract type AbstractDDEDataset <: AbstractDataset end
# type for ODE ground truth dynamics
abstract type AbstractODEDataset <: AbstractDataset end

# type for dataset for DDE model based on DDE ground truth
mutable struct DDEDDEDataset <: AbstractDDEDataset
    trajs
    state_dim::Int64
    obs_ids::Array{Int64,1}
    ICs
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
# type for dataset for ODE model based on DDE ground truth
mutable struct ODEDDEDataset <: AbstractDDEDataset
    trajs
    state_dim::Int64
    obs_ids::Array{Int64,1}
    ICs
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
end
DDEODEDataset(ICs, tspan, Δt, prob, r; obs_ids=1:length(ICs[1]), umax=100.0) = begin
    trajs = []
    state_dim = length(ICs[1])
    N = Int(floor((last(tspan) - first(tspan))/Δt)) + 1
    N_hist = Int(r/Δt) + 1
    p = prob.p
    callback = nothing
    DDEODEDataset(trajs, state_dim, obs_ids, ICs, tspan, Δt, N, N_hist, p, prob, umax, callback, r)
end

# type for dataset for ODE model based on ODE ground truth
mutable struct ODEODEDataset <: AbstractODEDataset
    trajs
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
end

## Core functionality:

# update the problem
function remake!(d::AbstractDataset; kwargs...)
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
# Generate a new data set. Overwrites previous one.
function gen_dataset!(d::AbstractDataset; alg=Tsit5(), kwargs...)

    # change setup
    remake!(d, kwargs...)

    # stopping condition to avoid finite escape
    condition(u,t,integrator) = sum(abs2, u) >= d.umax^2
    affect!(integrator) = terminate!(integrator)
    d.callback = DiscreteCallback(condition, affect!)

    # generate trajectories
    d.trajs = gen_trajs(d; alg=alg)
end

## utilities

# helper function to generate trajectories
function gen_trajs(d::AbstractODEDataset; alg=Tsit5())
    t = Array(first(d.tspan):d.Δt:last(d.tspan))
    init_t = Array(first(d.tspan)-d.r:d.Δt:first(d.tspan)-d.Δt)
    d.N_hist = length(init_t) + 1
    trajs = []
    for u0 in d.ICs
        prob = remake(d.prob, u0=u0)
        sol = solve(prob, alg, saveat=t, save_idxs=d.obs_ids, callback=d.callback)
        init_u = repeat([u0[d.obs_ids]], d.N_hist)
        push!(trajs, [vcat(init_t, sol.t), vcat(init_u, sol.u)])
    end
    trajs
end
function gen_trajs(d::AbstractDDEDataset; alg=Tsit5())
    t = Array(first(d.tspan):d.Δt:last(d.tspan))
    init_t = Array(first(d.tspan)-d.r:d.Δt:first(d.tspan)-d.Δt)
    d.N_hist = length(init_t) + 1
    trajs = []
    for u0 in d.ICs
        h = (p,t) -> u0
        prob = remake(d.prob, u0=u0, h0=h0)
        sol = solve(prob, MethodOfSteps(alg), saveat=t, callback=d.callback)
        init_u = repeat([u0], d.N_hist)
        push!(trajs, [vcat(init_t, sol.t), vcat(init_u, sol.u)])
    end
    trajs
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
    ut = reverse(vcat(d.trajs[traj_idx][2][in_traj_idx-d.N_hist+1:in_traj_idx]...))
    t, ut
end
function Base.getindex(d::DDEDDEDataset, idx::Int)
    lengths = map(traj -> length(traj[1])-d.N_hist+1, d.trajs)
    cum_lengths = map(n -> sum(lengths[1:n]), 1:length(d.trajs))
    traj_idx = findfirst(n -> n>=idx, cum_lengths)
    in_traj_idx = d.N_hist-1 + (idx - vcat([0],cum_lengths)[traj_idx])
    t = d.trajs[traj_idx][1][in_traj_idx]
    ut = reverse(vcat(d.trajs[traj_idx][2][in_traj_idx-d.N_hist+1:in_traj_idx]...))
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

## Example
ode_func = (u,p,t)-> u
prob = ODEProblem(ode_func, zeros(10), (0,1.0))
df = DDEODEDataset(repeat([randn(10)], 3), (0.0,2.0), 0.1, prob, 0.5, obs_ids=[1,2])
gen_dataset!(df)
remake!(df, tspan=(0.0,10.0))

# test batching
loader = Flux.Data.DataLoader(df, batchsize=5, shuffle=true)
a = zeros(12,5)
for (t,u) in loader
    println(size(t),size(u))
    global a = u
end





## batching with DataLoaders.jl
# function nobs(dataset::AbstractDataset)
#     nobs = 0
#     for traj in dataset.trajs
#         nobs += length(traj[1])-dataset.N_hist
#     end
#     nobs
# end
# function getobs(dataset::ODEODEDataset, idx::Int)
#     lengths = map(traj->length(traj[1])-dataset.N_hist, dataset.trajs)
#     cum_lengths = map(n -> sum(lengths[1:n]), 1:length(dataset.trajs))
#     traj_idx = findfirst(n->n>=idx, cum_lengths)
#     in_traj_idx = dataset.N_hist + (idx - vcat([0],cum_lengths)[traj_idx])
#     obs_t = trajs[traj_idx][1][in_traj_idx]
#     obs_u = trajs[traj_idx][2][in_traj_idx]
#     obs = [obs_t, obs_u]
# end
# function getobs(dataset::ODEDDEDataset, idx::Int)
#     lengths = map(traj->length(traj[1])-dataset.N_hist, dataset.trajs)
#     cum_lengths = map(n -> sum(lengths[1:n]), 1:length(dataset.trajs))
#     traj_idx = findfirst(n->n>=idx, cum_lengths)
#     in_traj_idx = dataset.N_hist + (idx - vcat([0],cum_lengths)[traj_idx])
#     obs_t = trajs[traj_idx][1][in_traj_idx]
#     obs_u = trajs[traj_idx][2][in_traj_idx]
#     obs = [obs_t, obs_u]
# end
# function getobs(dataset::DDEODEDataset, idx::Int)
#     lengths = map(traj -> length(traj[1])-dataset.N_hist, dataset.trajs)
#     cum_lengths = map(n -> sum(lengths[1:n]), 1:length(dataset.trajs))
#     traj_idx = findfirst(n -> n>=idx, cum_lengths)
#     in_traj_idx = dataset.N_hist + (idx - vcat([0],cum_lengths)[traj_idx])
#     obs_t = trajs[traj_idx][1][in_traj_idx]
#     obs_u = vcat(trajs[traj_idx][2][in_traj_idx-dataset.N_hist:in_traj_idx]...)
#     obs = [obs_t, obs_u]
# end
# function getobs(dataset::DDEDDEDataset, idx::Int)
#     lengths = map(traj -> length(traj[1])-dataset.N_hist, dataset.trajs)
#     cum_lengths = map(n -> sum(lengths[1:n]), 1:length(dataset.trajs))
#     traj_idx = findfirst(n -> n>=idx, cum_lengths)
#     in_traj_idx = dataset.N_hist + (idx - vcat([0],cum_lengths)[traj_idx])
#     obs_t = trajs[traj_idx][1][in_traj_idx]
#     obs_u = vcat(trajs[traj_idx][2][in_traj_idx-dataset.N_hist:in_traj_idx]...)
#     obs = [obs_t, obs_u]
# end

## Old stuff...
# function trajs2data(dataset::ODEODEDataset, trajs)
#     for traj in trajs
#         for i in 1:skip_ids:length(traj[1])-length(init_ts)
#             t = traj[1][length(init_ts) + i]
#             xt = vcat(traj[2][length(init_ts) + i .- Array(0:2*length(vlags))]...)
#             push!(data, (t, xt))
#             # push!(data, (t, randn(82)))
#         end
#     end
# end
# function trajs2data(dataset::ODEDDEDataset, trajs)
#     for traj in trajs
#         for i in 1:skip_ids:length(traj[1])-length(init_ts)
#             t = traj[1][length(init_ts) + i]
#             xt = vcat(traj[2][length(init_ts) + i .- Array(0:2*length(vlags))]...)
#             push!(data, (t, xt))
#             # push!(data, (t, randn(82)))
#         end
#     end
# end
# function trajs2data(dataset::DDEDataset, trajs)
#     for traj in trajs
#         for i in 1:skip_ids:length(traj[1])-length(init_ts)
#             t = traj[1][length(init_ts) + i]
#             xt = vcat(traj[2][length(init_ts) + i .- Array(0:2*length(vlags))]...)
#             push!(data, (t, xt))
#             # push!(data, (t, randn(82)))
#         end
#     end
# end
