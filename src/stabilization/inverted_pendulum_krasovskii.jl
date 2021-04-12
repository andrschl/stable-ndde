# import modules
include("import.jl")

# some constants
data_dim = 2
flags = Array([0.03])
vlags = Array(0.0:0.003:0.03)[2:end]
lags = sort(union(flags,vlags))
nfdelays = length(flags)
nvdelays = length(vlags)
max_lag = maximum(lags)


################################################################################
# generate dynamics model

# open loop dynamics
function f_ol(x,u)
    l = 0.3
    g = 9.81
    b = 0.0
    m = 0.2
    x1 = x[1]
    x2 = x[2]
    dx1 = x2
    dx2 = g/l*sin(x1)-1 /(m*l^2)*(b*x2 + u[1])
    return vcat(dx1,dx2)
end
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
(m::F_CL)(xt::AbstractArray) = begin
    return f_ol(xt[1:data_dim], m.u(xt[1+data_dim:end]))
end
# define feedback
g = Feedback(Array([1.749  1.031 ;]))
# g = Chain(Dense(2,16,tanh),Dense(16,1))
f_cl = F_CL(g)

################################################################################
# load Lyapunov Razumikhin function

# include Lyapunov loss function
include("../lyapunov/krasovskii.jl")
# include("model_nn.jl")
model_v = Lyapunov(data_dim*(nvdelays+1), act=C2_relu, layer_sizes=[128,128])
pv,rev = Flux.destructure(model_v)
pf,ref = Flux.destructure(f_cl)

# initialize helper object
# model = StableDynamicsSoft(data_dim, ref, rev, pf, pv, flags=flags,vlags=vlags,α=0.1,β=1.0202)
model = StableDynamicsSoft(data_dim, ref, rev, pf, pv, flags=flags,vlags=vlags,α=0.5,β=1.2)

f_soft = (u,p)->model.re_f(p)(u)

################################################################################
# helper functions for integration

# dde function for prediction
function ndde_func!(du, u, h, p, t)
    global flags
    ut = u
    for tau in flags
        hh = h(p, t - tau)
        ut = cat(ut, hh, dims=1)
    end
    du .= f_soft(ut, p)
end
# interpolated prediction
function dense_predict(u0, h0, p, t; cb=nothing)
    global flags
    t_span = (first(t), last(t))
    prob = DDEProblem(ndde_func!, u0, h0, t_span, p=p; constant_lags=flags)
    alg = MethodOfSteps(Tsit5())
    solve(prob, alg, u0=u0, p=p, dense=true, callback=cb)
end
# discrete prediction
function predict(u0, h0, p, t; cb=nothing)
    global flags
    dims = length(size(u0))
    t_span = (first(t), last(t))
    prob = DDEProblem(ndde_func!, u0, h0, t_span, p=p; constant_lags=flags)
    alg = MethodOfSteps(Tsit5())
    return solve(prob, alg, u0=u0, p=p, saveat=t, callback=cb)
end
# non-smoothed Lyapunov loss
function true_lyapunov_ndde_func!(du, u, h, p, t)
    global flags,vlags
    x = u[1:data_dim]
    pf = p[1:model.nfparams]
    pv = p[model.nfparams+1:end]
    xt = vcat(x, map(τ -> h(pf,t-τ, idxs=1:data_dim), flags)...)
    yt = vcat(x, map(τ -> h(pv,t-τ, idxs=1:data_dim), vlags)...)
    du .= vcat(f_soft(xt, pf), true_loss_nn(xt, yt, pf, pv, model)*ones(1))
end
# loss integral
function predict_true_loss(u0,h0,pf,pv,t)
    z0 = Array(vcat(u0, [0]))
    t_span = (first(t), last(t))
    p = vcat(pf,pv)
    prob = DDEProblem(true_lyapunov_ndde_func!, z0, h0, t_span, p=p; constant_lags=sort(union(flags,vlags)))
    alg = MethodOfSteps(RK4())
    return Array(solve(prob, alg, u0=z0, p=p, saveat=[last(t_span)], sensealg=ReverseDiffAdjoint(), abstol=1e-9,reltol=1e-6))[3,1]
    # return Array(solve(prob, alg, u0=z0, p=p, saveat=[last(t_span)], sensealg=ReverseDiffAdjoint()))[3,1]
    # return Array(solve(prob, alg, u0=z0, p=p, saveat=[last(t_span)], sensealg=ForwardDiffSensitivity()))[3,1]
end

################################################################################
# data generation

using Flux.Data

# some masks
v_mask = Array(1:(length(vlags)+1)*data_dim)
indices_f_only = sort(union(map(d->findfirst(isequal(d), lags), flags),0))
f_mask = vcat(map(i->Array(i*data_dim+1:(i+1)*data_dim), indices_f_only)...)

function generate_data(pf, batch_size, ntrajs; shuffle=true)

    # define set of initial conditions S (use a grid on S)
    max_displacement = 1.0
    min_displacement = 0.1
    s1 = cat(Array(LinRange(-max_displacement, -min_displacement, Int(ceil(ntrajs/2)))),
    Array(LinRange(min_displacement, max_displacement, Int(ceil(ntrajs/2)))), dims=1)
    S = [[x,0.0] for x in s1]

    # define trajectories
    t0 = 0.0
    T = 3
    Δt = 0.1
    t = Array(t0:Δt:T)
    t_span = (t0, T)
    umax = 100.0

    # sampling (this should be the same frequency as in vlags)
    Δts = 0.003
    ts = Array(t0:Δts:T)
    init_ts = Array(t0-maximum(2*lags):Δts:t0)[1:end-1]

    # stopping condition to avoid finite escape (stops at time step where condition is satisfied)
    condition(u,t,integrator) = sum(abs2, u) >= umax^2
    affect!(integrator) = terminate!(integrator)
    cb = DiscreteCallback(condition, affect!)

    trajs = []
    for u0 in S
        sol = predict(u0, (p,t)->u0, pf, ts, cb=cb)
        init_u = repeat([u0], length(init_ts))
        push!(trajs, [vcat(init_ts, sol.t), vcat(init_u, sol.u)])
    end

    data = []
    Δtdata = 3
    for traj in trajs
        for i in 1:Δtdata:length(traj[1])-length(init_ts)
            t = traj[1][length(init_ts) + i]
            xt = vcat(traj[2][length(init_ts) + i .- Array(0:2*length(vlags))]...)
            push!(data, (t, xt))
            # push!(data, (t, randn(82)))
        end
    end
    train_loader = DataLoader(data, batchsize=batch_size, shuffle=shuffle)
    return train_loader
end

loader = generate_data(pf, 32, 4)

# # define loss function
# loss = (x,y) -> (x-y)'*(x-y)/length(t)

################################################################################
# training

# train method
function train!(pf, pv, opt_f, opt_v, iter, batch)

    println("start AD------------")
    @time begin
        dldpf = zero(pf)
        dldpv = zero(pv)
        ls = []
        for (_, xt) in batch
            push!(ls, krasovskii_loss(model, xt, f_mask, v_mask, pf, pv))
            dldpf += Zygote.gradient(pf->krasovskii_loss(model, xt, f_mask, v_mask, pf, pv), pf)[1]
            dldpv += Zygote.gradient(pv->krasovskii_loss(model, xt, f_mask, v_mask, pf, pv), pv)[1]
        end
        Flux.Optimise.update!(opt_f, pf, dldpf)
        Flux.Optimise.update!(opt_v, pv, dldpv)
        println("dlfpf mean is: ", mean(abs,dldpf))
        println("dldpv mean is: ", mean(abs,dldpv))
        println("non-zero fraction: ", length(ls[ls.!=0.0])/length(ls))
        println("max_loss: ", maximum(ls))
    end

    println("stop AD------------")
    println("iteration ", iter)

end

# training loop
@time begin
    nepisodes = 100
    lrsf = repeat([5e-3],nepisodes)
    # lrsf = repeat([0.0],nepisodes)
    lrsv = repeat([5e-3],nepisodes)
    # batch_size = 32
    # nbatches = 2
    # # nbatches = "all"
    # ntrajs = 10
    train_loader = generate_data(pf, batch_size, ntrajs, shuffle=true)
    for episode in 1:nepisodes
        batch_size = 256
        nbatches = 2
        # nbatches = "all"
        ntrajs = 100
        batch_idx = 1
        train_loader = generate_data(pf, batch_size, ntrajs, shuffle=true)
        for batch in train_loader
            println("------------")
            println("episode: ", episode, ", batch_idx: ", batch_idx)
            opt_f = ADAM(lrsf[episode])
            opt_v = ADAM(lrsv[episode])
            train!(pf,pv, opt_f, opt_v, 3*(episode-1) + batch_idx, batch)

            # plot Lyapunov function contout lines
            # if batch_idx % 1 == 0
            #     xs = LinRange(-1, 1, 100)
            #     ys = LinRange(-1, 1, 100)
            #     zs = [rev(pv)([x,y])[1] for x in xs, y in ys]
            #     display(CairoMakie.contour(xs, ys, zs; levels=20))
            # end
            # debugging
            # append!(lyapunovgs, sum(abs,p[1:1186]))
            
            u0 =  [-1.0, 0.0]
            h0 = (p,t) -> u0
            sol = dense_predict(u0, h0, pf, [0.0,3.0])
            display(Plots.plot(sol))
            if nbatches == "all"
                continue
            elseif batch_idx >= nbatches
                break
            else
                batch_idx += 1
            end
        end
    end
end

# test it
u0 =  [-0.5, 0.0]
h0 = (p,t;idxs=nothing) -> u0
sol = dense_predict(u0, h0, pf, [0.0,3.0])
display(Plots.plot(sol))
predict_true_loss(u0,h0,pf,pv,[0.0,3])

lags
