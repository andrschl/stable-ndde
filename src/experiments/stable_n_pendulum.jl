## Load packages
include("../util/import.jl")

## set seed
Random.seed!(10)
rng = MersenneTwister(1234)

## Load pendulum dataset
N_PENDULUM = 3
include("../datasets/n_pendulum.jl")

# initial conditions
θmax = pi/2
distr = Uniform(-θmax, θmax)
ICs = map(i-> vcat(θ.=>rand(rng, distr, N_PENDULUM), ω.=> zeros(N_PENDULUM)), 1:10)
ICs = map(i-> vcat(zeros(N_PENDULUM), rand(rng, distr, N_PENDULUM)), 1:10)

tspan = (0.0, 10.0)
Δt = 0.1
r = 1.0

df = DDEODEDataset(ICs, tspan, Δt, pendulum_prob, r; obs_ids=Array(N_PENDULUM+1:2*N_PENDULUM))
gen_dataset!(df)
batchtime = 100
batchsize = 16

# load batch
ts, batch_u, batch_h0 = get_ndde_batch_and_h0(df, batchtime, batchsize)
batch_t = ts[:,1].-ts[1,1]

## Define model
include("../models/model.jl")
data_dim = 3
flags = Array(Δt:Δt:r)
vlags = flags
model = RazNDDE(data_dim; flags=flags, α=0.1, q=1.01)

## Train method
function train_step!(u0::AbstractArray, u_train::AbstractArray, h0::Function, pf::AbstractArray, t::AbstractArray, m::AbstractNDDEModel, opt)

    println("____________________")
    println("start AD:")
    local train_loss, pred_u
    @time begin
        ps = Flux.params(pf)
        gs = Zygote.gradient(ps) do
            train_loss, pred_u = predict_LS_loss(u0, h0, t, u_train, pf, m; N=batchtime*batchsize)
            return train_loss
        end
    end
    println("stop AD")
    println("____________________")
    println("train_loss: ", train_loss)
    Flux.Optimise.update!(opt, ps, gs)
end

## cyclic learnig schedules
function get_sin_schedule(min, max, period; len=10*period, niter_per_value=1)
    a = (max - min) / 2.0
    b = Array(0:len)
    c = (max + min) / 2.0
    niters = ones(length(b))
    return zip(c .+ a * sin.(2 * pi * b / period), niters)
end
function exp_decays(min, max, period; len=10*period, niter_per_value=1)
    fun = t -> max .* exp.(log(min/max) .* (t .% (period+1)) ./ period)
    iters = Array(0:len)
    return zip(fun(iters), iters)
end
function double_exp_decays(reldecay, locmin, locmax, period; len=10*period, niter_per_value=1)
    fun1 = t -> locmax .* exp.(log(locmin/locmax) .* (t .% (period+1)) ./ period)
    fun2 = t -> exp.(log(reldecay) .* t ./ len)
    iters = Array(0:len)
    return zip(fun1(iters) .* fun2(iters), iters)
end

## training
@time begin
    rel_decay, locmin, locmax, period = 0.01, 1e-5, 5e-3, 100
    lr_args = (rel_decay, locmin, locmax, period)
    lr_kwargs = Dict(:len => 5000)
    lr_schedule_gen = double_exp_decays
    lr_schedule = lr_schedule_gen(lr_args...;lr_kwargs...)
    pf = model.pf
    for (lr, iter) in lr_schedule
        println("==============")
        println("iter: ", iter)
        opt = ADAM(lr)
        ts, batch_u, batch_h0 = get_ndde_batch_and_h0(df, batchtime, batchsize)
        batch_t = ts[:,1].-ts[1,1]
        train_step!(batch_u[:,1,:], batch_u, batch_h0, pf, batch_t, model, opt)

        if iter % 1 == 0
            u_test = hcat(df.trajs[1][2]...)[:, df.N_hist:end]
            h0_test = (p,t)->df.trajs[1][3](t)
            t_test = df.trajs[1][1][df.N_hist:end]
            test_sol = dense_predict_ndde(u_test[:,1], h0_test, (t_test[1], t_test[end]), pf, model)
            pl = plot(test_sol, xlims=(0.0,10.0))
            scatter!(pl, t_test, u_test[1,:], label="θ_true1")
            scatter!(pl, t_test, u_test[2,:], label="θ_true2")
            scatter!(pl, t_test, u_test[3,:], label="θ_true3")
            display(pl)
        end
    end
end


## Evaluate on some trajectories

u_test = hcat(df.trajs[7][2]...)[:, df.N_hist:end]
h0_test = (p,t)->df.trajs[7][3](t)
t_test = df.trajs[7][1][df.N_hist:end]
test_sol = dense_predict_ndde(u_test[:,1], h0_test, (t_test[1], t_test[end]), pf, model)
pl = plot(test_sol, xlims=(0.0,10.0))
scatter!(pl, t_test, u_test[1,:], label="θ_true1")
scatter!(pl, t_test, u_test[2,:], label="θ_true2")
scatter!(pl, t_test, u_test[3,:], label="θ_true3")

# # save params
# dir_name = "/home/andrschl/Documents/MA/stable-time-delay-systems/checkpoints/"
# using BSON: @save
# filename = dir_name * "weights.bson"
# @save filename pf
#
# using BSON: @load
# load_dir ="/home/andrschl/Documents/MA/node_julia/reports/oscillator_node_comparison/ndde_cos/2020-12-25T23:48:03.224"
# filename = load_dir * "/weights.bson"
# @load filename p


# test to DEBUG
u_test = hcat(df.trajs[1][2]...)[:, df.N_hist:end]
h0_test = (p,t)->df.trajs[1][3](t)
t_test = df.trajs[1][1][df.N_hist:end]
tspan_test= (0.0,10.0)
test_pred = predict_ndde(u_test[:,1],h0_test, t_test,pf, model)


pl = Plots.plot(t_test, Base.getindex.(test_pred.u,1), xlims=(-1,10))
Plots.plot!(pl,t->h0_test(nothing,t)[1])
Plots.scatter!(pl, t_test, u_test[1,:])
Plots.plot!(t_test, Base.getindex.(test_pred.u,2), xlims=(-1,10))
Plots.plot!(pl,t->h0_test(nothing,t)[2])
Plots.scatter!(pl, t_test, u_test[2,:])
Plots.plot(dense_predict_ndde(u_test[:,1],h0_test, tspan_test,pf, model), xlims=(-1,10))

ts, batch_u, batch_h0 = get_ndde_batch_and_h0(df, batchtime, batchsize)
batch_t = ts[:,1].-ts[1,1]
batch_pred_u = predict_ndde(batch_u[:,1,:],batch_h0, batch_t,pf, model)
predict_LS_loss(batch_u[:,1,:], batch_h0, batch_t, batch_u, pf, model, N=batchtime*batchsize)[1]
Zygote.gradient(batch_u->predict_LS_loss(batch_u[:,1,:], batch_h0, batch_t, batch_u, pf, model, N=batchtime*batchsize)[1],batch_u)
Zygote.gradient(u_test->predict_LS_loss(u_test[:,1], h0_test, t_test, u_test, pf, model, N=length(t_test))[1],u_test)

ps = Flux.params(pf)
gs = gradient(ps) do
    pred_u = predict_ndde(batch_u[:,1,:], batch_h0, batch_t, pf, model)
    println(size(pred_u))
    return test_loss(pred_u, batch_u)
end
gs[pf]

ps = Flux.params(pf)
gs = Zygote.gradient(ps) do
    pred_u = predict_ndde(u_test[:,1], h0_test, t_test, pf, model)
    println(size(pred_u))
    return test_loss(pred_u, u_test)
end
gs[pf]
predict_ndde(u_test[:,1], h0_test, t_test, pf)
