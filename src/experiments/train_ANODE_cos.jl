## Hyperparameters
config = Dict(
    # on server?
    "server" => false,
    "logging" => false,

    # lr schedule
    "lr_rel_decay" => 0.1,
    "lr_start_max" => 5e-3,
    "lr_start_min" => 1e-5,
    "lr_period" => 40,
    "nepisodes" => 300,

    # ndde training
    "A_train" => 1.0,
    "A_test" => 1.0,
    "ntrain_trajs" => 2,
    "ntest_trajs" => 1,
    "T_train" => 30.0,
    "T_test" => 100.0,
    "datasize" => 151,
    "k0" => "RBF",
    "σ" => 0.1,

    # logging
    "test_eval" => 20,
    "model checkpoint" => 500
)

## argsparse
seed = 1
if length(ARGS) >= 1
    seed = parse(Int64, ARGS[1])
end
if length(ARGS) >= 2
    config["σ"] = parse(Float64, ARGS[2])
end
if length(ARGS) >= 3
    config["aug0"] = parse(Float64, ARGS[3])
else
    config["aug0"] = -1 + (seed-1) * 2.0 / 19
end

println("aug0: ", config["aug0"])

## some server specific stuff..
if config["server"]
    ENV["GKSwstype"] = "nul"
end

## Load packages
cd(@__DIR__)
cd("../../.")
using Pkg; Pkg.activate("."); Pkg.instantiate();
using PyCall
include("../util/import.jl")

## log path
# current_time = Dates.format(now(), "_dd-mm-yy_HH:MM/")
# runname = "stable oscillator"*current_time
# logpath = "reports/"*splitext(basename(@__FILE__))[1]*current_time

project_name = "cos_ANODE"
runname = "seed_"*string(seed)
configname = "/"*string(config["σ"])*"/"
devicename = config["server"] ? "server_" : "blade_"
logpath = "reports/"*project_name*configname*runname*"/"
println(logpath)
println(configname)
mkpath(logpath)
if config["logging"]
    wandb = pyimport("wandb")
    # wandb.init(project=splitext(basename(@__FILE__))[1], entity="andrschl", config=config, name=runname)
    wandb.init(project=project_name, config=config, name=runname, group=devicename*configname)
end

## set seed
Random.seed!(123)
rng = MersenneTwister(123)

## Define dataset
data_dim = 1
aug_dim = 1
A = config["A_train"]
data_size = config["datasize"]
aug0 = config["aug0"]
# aug0 = 0.0
σ = config["σ"]

# generate data
t = Array(LinRange(0.0, config["T_train"], config["datasize"]))
config["Δt"] = t[2]-t[1]
true_sol = t->Array([A*cos(t)])
dtrue_sol = t->Array([-A*sin(t)])
function h0(p, t;idxs=nothing)
    if !isa(t, Array)
        return true_sol(t)
    else
        return hcat(true_sol.(t)...)
    end
end
true_u = h0(nothing,t)
dtrue_u = hcat(dtrue_sol.(t)...)
u_noisy = true_u + σ*randn(1, length(t))
pl = scatter(t, u_noisy[1,:], label="data")
plot!(pl, t->true_sol(t)[1], xlims=(0,12*pi))
display(plot(pl))

# augment dimension (ANODE)
aug = zeros(aug_dim, data_size)   # initialize augmented dim with 0
aug[1, 1] = aug0
aug_u = cat(u_noisy, aug, dims=1)
aug_u0 = aug_u[:, 1]


true_sol2 = t->Array([2*A*sin(t)])
dtrue_sol2 = t->Array([2*A*cos(t)])
function h02(p, t;idxs=nothing)
    if !isa(t, Array)
        return true_sol2(t)
    else
        return hcat(true_sol2.(t)...)
    end
end
true_u2 = h02(nothing,t)
dtrue_u2 = hcat(dtrue_sol2.(t)...)
u_noisy2 = true_u2 + σ*randn(1, length(t))
pl2 = scatter(t, u_noisy2[1,:], label="data2")
plot!(pl2, t->true_sol2(t)[1], xlims=(0,12*pi))
display(plot(pl2))

# augment dimension (ANODE)
aug2 = zeros(aug_dim, data_size)   # initialize augmented dim with 0
aug2[1, 1] = aug0
aug_u2 = cat(u_noisy2, aug2, dims=1)
aug_u02 = aug_u2[:, 1]

# aug_u = zeros(data_dim, data_size)
# aug_u[2,:] = dtrue_u
# aug_u[1,:] = true_u
# aug_u0 = aug_u[:,1]

# define NODE
f = Chain(  Dense(data_dim + aug_dim, 32, swish),
            Dense(32, 64, swish),
            Dense(64, 64, swish),
            Dense(64, 32, swish),
            Dense(32, data_dim+aug_dim))
# f = Chain(  Dense(1 + aug_dim,data_dim+aug_dim))
# f = Chain(  Dense(1 + aug_dim, 32, tanh),
#             Dense(32, 64, tanh),
#             Dense(64, 64, tanh),
#             Dense(64, 32, tanh),
#             Dense(32, data_dim+aug_dim))
# needed for accessing the params
p,re = Flux.destructure(f)
ODE_func(u,p,t) = re(p)(u) # need to restructure for backprop!

# predict node
function predict_node(p, u0, t)
    t_span = (first(t), last(t))
    prob = ODEProblem(ODE_func, u0, t_span)
#    println(size(u_init))
#    println(size(time_values))
    alg = Tsit5()#Vern9()
    # alg = SymplecticEuler()
    return Array(solve(prob, alg, u0=u0, p=p, saveat=t, sensealg=ReverseDiffAdjoint(),abstol=1e-10,reltol=1e-8))
end
function dense_predict_node(p, u0, t)
    t_span = (first(t), last(t))
    prob = ODEProblem(ODE_func, u0, t_span)
#    println(size(u_init))
#    println(size(time_values))
    alg = Tsit5()
    # alg = SymplecticEuler()
    return solve(prob, alg, u0=u0, p=p, dense=true)
end

function predict_loss(p, u0, ud)
    global t
    pred_u = predict_node(p, u0, t) # check this
#    println(size(pred_u))
#    println(size(true_u))
    if length(size(ud)) ==2
        loss = sum(abs2, ud - pred_u[1:data_dim, :]) / (data_size)
    else
        loss = sum(abs2, ud - pred_u[1:data_dim, :, :]) / batchtime*batchsize
    end

    return loss, pred_u
end
predict_loss(p, aug_u0, u_noisy)

function train!(p, aug_u0, aug_u02, opt1, opt2, iter)
    local train_loss
    local pred_u
    ps = Flux.params(p, aug_u0)
    gs = gradient(ps) do
        train_loss, pred_u = predict_loss(p, aug_u0, u_noisy)
        return train_loss
    end
    Flux.Optimise.update!(opt1, ps[1], gs[p])
    Flux.Optimise.update!(opt2, ps[2], gs[aug_u0])
    println(gs[aug_u0], gs)
    println("train loss: ", train_loss)
    push!(train_loss_data, [iter, train_loss])
    ps = Flux.params(p, aug_u02)
    gs = gradient(ps) do
        train_loss, pred_u = predict_loss(p, aug_u02, u_noisy2)
        return train_loss
    end
    Flux.Optimise.update!(opt1, ps[1], gs[p])
    Flux.Optimise.update!(opt2, ps[2], gs[aug_u02])
    println(gs[aug_u02], gs)
    println("train loss: ", train_loss)
    push!(train_loss_data, [iter, train_loss])
    if !config["server"]
        sol = dense_predict_node(p, aug_u0, t)
        pl = scatter(t, u_noisy[1,:], label="data1")
        # scatter!(pl, t, aug_u[2,:], label="data2")

        plot!(pl, sol, xlims=(0,12*pi))

        sol = dense_predict_node(p, aug_u02, t)
        pl2 = scatter(t, u_noisy2[1,:], label="data2")
        # scatter!(pl, t, aug_u[2,:], label="data2")

        plot!(pl2, sol, xlims=(0,12*pi))
        display(plot(pl, pl2))
    end
    # evaluate on whole trajectory
end
# function get_batch(batchtime, batchsize)
#     tot_size = length(t) - (batchtime-1)
#     s = sample(Array(1:tot_size), batchsize, replace=false)
#     us = zeros(data_dim, batchtime, batchsize)
#     t0s = zeros(batchsize)
#     for (i, idx) in enumerate(s)
#         us[1, :, i] = u_noisy[idx:idx+batchtime-1]
#         t0s[i] = t[idx]
#     end
#     return t0s, us
# end

#################################################################################
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
#################################################################################
# train
# show initial trajectory
pred_u = predict_node(p, aug_u0, t)
loss = sum(abs2, true_u - pred_u[1:data_dim, :])/data_size
sol = dense_predict_node(p, [-1,0], t)
pl = plot(t-> -cos(t), xlims=(0,30),label="ground truth")
plot!(pl, t->sol(t)[1], xlims=(0,30))
display(pl)
println("-----------------------------------")
println("Start training with init loss ", loss)

train_loss_data = DataFrame(iter = Int[], train_loss = Float64[])
batchtime = 50
batchsize = 5
aug_u0
# training loop
@time begin
    rel_decay, locmin, locmax, period = config["lr_rel_decay"], config["lr_start_min"], config["lr_start_max"], config["lr_period"]
    lr_args = (rel_decay, locmin, locmax, period)
    lr_kwargs = Dict(:len => config["nepisodes"])
    lr_schedule_gen = double_exp_decays
    lr_schedule = lr_schedule_gen(lr_args...;lr_kwargs...)
    lr_u0_weight = 10.0
    # train loop
    iters = []
    lrs = []
    for (lr, iter) in lr_schedule
        push!(iters, iter)
        push!(lrs, lr)
        println("==============")
        println("iter: ", iter)
        opt1 = ADAM(lr)
        opt2 = ADAM(lr*lr_u0_weight)
        println("aug_u0")
        println(aug_u0)
        println("aug_u02")
        println(aug_u02)
        train!(p, aug_u0,aug_u02, opt1, opt2, iter)
    end
    # pl=plot(iters, lrs)
    # display(pl)
end

# save params
CSV.write(logpath*"train_loss.csv", train_loss_data, header = true)
logpath
using BSON: @save,@load
filename = logpath * "weights.bson"
@save filename p

for seed in 1:10
    logpath = "reports/cos_ANODE/0.1/seed_"*string(seed)*"/"
    filename = logpath * "weights.bson"
    @load filename p
    u_gen = predict_node(p, [1.,0.0], t_pl)
    u_gen_gt = Array(cos.(t_pl))
    prediction_data_gen = DataFrame(t=t_pl, u1 = u_gen[1,:], u2 = u_gen[2,:])
    gt_data_gen = DataFrame(t=t_pl, u1 = u_gen_gt)
    plot(t_pl, u_gen[1,:])
    plot!(t_pl, u_gen_gt)
    CSV.write(logpath*"gen_pred1.csv", prediction_data_gen, header = true)
    CSV.write(logpath*"gen_gt1.csv", gt_data_gen, header = true)
    u_gen = predict_node(p, [0.,-1.0], t_pl)
    u_gen_gt = Array(-sin.(t_pl))
    prediction_data_gen = DataFrame(t=t_pl, u1 = u_gen[1,:], u2 = u_gen[2,:])
    gt_data_gen = DataFrame(t=t_pl, u1 = u_gen_gt)
    plot(t_pl, u_gen[1,:])
    plot!(t_pl, u_gen_gt)
    CSV.write(logpath*"gen_pred2.csv", prediction_data_gen, header = true)
    CSV.write(logpath*"gen_gt2.csv", gt_data_gen, header = true)
end
plot_f = function (z::Point2)
    x = Array(z)
    Point2(re(p)(x))
end
pl=streamplot(plot_f, -2..2, -2..2, arrow_size=3)



# plots
Δtplot=0.02
t_pl = Array(t[1]:Δtplot:t[end])
u_pred = predict_node(p, aug_u0, t_pl)
u_gt = Array(cos.(t_pl))
prediction_data = DataFrame(t=t_pl, u1 = u_pred[1,:], u2 = u_pred[2,:])
training_data = DataFrame(t=t, u1 = u_noisy[1,:])
gt_data = DataFrame(t=t_pl, u1 = u_gt)
CSV.write(logpath*"train_pred.csv", prediction_data, header = true)
CSV.write(logpath*"train_gt.csv", gt_data, header = true)


u_pred2 = predict_node(p, aug_u02, t_pl)
u_gt2 = Array(2*sin.(t_pl))
prediction_data2 = DataFrame(t=t_pl, u1 = u_pred2[1,:], u2 = u_pred2[2,:])
training_data2 = DataFrame(t=t, u1 = u_noisy2[1,:])
gt_data2 = DataFrame(t=t_pl, u1 = u_gt2)
CSV.write(logpath*"train_pred2.csv", prediction_data2, header = true)
CSV.write(logpath*"train_gt2.csv", gt_data2, header = true)


if !config["server"]
    pl = plot()
    scatter!(pl, t, aug_u[1,:], label="data1")
    # scatter!(pl, t, aug_u[2,:], label="data2")
    plot!(pl, t_pl, u_pred[1,:], label="u1")
    plot!(pl, t_pl, u_pred[2,:], label="u2")
end




xs = Array(LinRange(-1, 1, 10))
ys = Array(LinRange(-1, 1, 10))
zs = [re(p)([x,y]) for x in xs, y in ys]
quiver([1,2,3],[3,2,1],quiver=([1,1,1],[1,2,3]))
xxy = [x for x in xs for y in ys]
yxy = [y for x in xs for y in ys]
uxy = [re(p)([x,y])[1] for x in xs for y in ys]
vxy = [re(p)([x,y])[2] for x in xs for y in ys]
# quiver(xxy,yxy,quiver= 0.5 .* (uxy,vxy))

CSV.write(logpath*"quiver.csv", DataFrame(x = xxy, y= yxy, u = uxy, v=vxy), header = true)


re(p)(randn(2))
plot_f = function (z::Point2)
    x = Array(z)
    Point2(re(p)(x))
end
pl=streamplot(plot_f, -2..2, -2..2, arrow_size=3)
