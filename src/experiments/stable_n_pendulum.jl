## Hyperparameters
params = Dict(
    # on server?
    "server" => false,

    # dynamics problem
    "npendulum" => 4,
    "friction" => 0.5,
    "length" => 1.0,
    "mass" => 1.0,

    # training params
    "T_train" => 5.0,
    "T_test" => 20,
    "Δt" => 0.1,
    "r" => 1.0,
    "ntrain_trajs" => 10,
    "ntest_trajs" => 5,
    "θmax" => 2*pi/5,

    # training
    "batchtime" => 50,
    "batchsize" => 16,
    "lr_rel_decay" => 0.01,
    "lr_start_max" => 5e-3,
    "lr_start_min" => 1e-4,
    "lr_period" => 100,
    "nepisodes" => 5000,
    "test_eval" => 100,
    "checkpoint" => 250
)

## some server specific stuff..
if params["server"]
    ENV["GKSwstype"] = "nul"
end

## Load packages
include("../util/import.jl")



## log path
logpath = "../../reports/"*splitext(basename(@__FILE__))[1]*Dates.format(now(), "_dd-mm-yy_HH:MM/")
mkpath(logpath)
wandb = pyimport("wandb")
wandb.init(project="stable_n_pendulum", entity="andrschl", config=params)

## set seed
Random.seed!(10)
rng = MersenneTwister(1234)

## Load pendulum dataset
N_PENDULUM = params["npendulum"]
include("../datasets/n_pendulum.jl")

# initial conditions
θmax = params["θmax"]
distr = Uniform(-θmax, θmax)
# ICs = map(i-> vcat(θ.=>rand(rng, distr, N_PENDULUM), ω.=> zeros(N_PENDULUM)), 1:10)
ICs_train = map(i-> vcat(zeros(N_PENDULUM), rand(rng, distr, N_PENDULUM)), 1:params["ntrain_trajs"])
ICs_test = map(i-> vcat(zeros(N_PENDULUM), rand(rng, distr, N_PENDULUM)), 1:params["ntest_trajs"])

train_tspan = (0.0, params["T_train"])
test_tspan = (0.0, params["T_test"])
Δt = params["Δt"]
r = params["r"]

df_train = DDEODEDataset(ICs_train, train_tspan, Δt, pendulum_prob, r; obs_ids=Array(N_PENDULUM+1:2*N_PENDULUM))
df_test = DDEODEDataset(ICs_test, test_tspan, Δt, pendulum_prob, r; obs_ids=Array(N_PENDULUM+1:2*N_PENDULUM))
gen_dataset!(df_train)
gen_dataset!(df_test)
batchtime = params["batchtime"]
batchsize = params["batchsize"]

## Define model
include("../models/model.jl")
data_dim = N_PENDULUM
flags = Array(Δt:Δt:r)
vlags = flags # not used here
model = RazNDDE(data_dim; flags=flags, α=0.1, q=1.01)

## training
include("../training/training_util.jl")

@time begin
    rel_decay, locmin, locmax, period = params["lr_rel_decay"], params["lr_start_min"], params["lr_start_max"], params["lr_period"]
    lr_args = (rel_decay, locmin, locmax, period)
    lr_kwargs = Dict(:len => params["nepisodes"])
    lr_schedule_gen = double_exp_decays
    lr_schedule = lr_schedule_gen(lr_args...;lr_kwargs...)
    pf = model.pf
    for (lr, iter) in lr_schedule
        println("==============")
        println("iter: ", iter)
        # ndde train step
        opt = ADAM(lr)
        ts, batch_u, batch_h0 = get_ndde_batch_and_h0(df_train, batchtime, batchsize)
        batch_t = ts[:,1].-ts[1,1]
        ndde_train_step!(batch_u[:,1,:], batch_u, batch_h0, pf, batch_t, model, opt)
        # lyapunov train step

        # test evaluation
        if iter % params["test_eval"] == 0
            # log train fit first trajectory
            wandb_plot_ndde_data_vs_prediction(df_train, 1, model, pf, "train fit 1")

            test_losses = []
            for i in 1:params["ntest_trajs"]
                t_test = df_test.trajs[i][1][df_test.N_hist:end]
                u_test = hcat(df_test.trajs[i][2][df_test.N_hist:end]...)
                u0_test = u_test[:,1]
                h0_test = (p,t) -> df_test.trajs[i][3](t)
                test_loss, _ = predict_ndde_loss(u0_test, h0_test, t_test, u_test, pf, model; N=df_test.N)
                push!(test_losses, test_loss)
                # if !params["server"]
                #     test_sol = dense_predict_ndde(u0_test, h0_test, test_tspan, pf, model)
                #     pl = plot(test_sol, xlims=test_tspan, title="Generalization traj " * string(i))
                #     scatter!(pl, t_test, u_test[1,:], label="θ_true1")
                #     scatter!(pl, t_test, u_test[2,:], label="θ_true2")
                #     scatter!(pl, t_test, u_test[3,:], label="θ_true3")
                #     display(pl)
                # end
                wandb_plot_ndde_data_vs_prediction(df_test, i, model, pf, "test fit "*string(i))
            end
            wandb.log(Dict("test loss"=> sum(test_losses)/params["ntest_trajs"]))
        end
        if iter % params["checkpoint"] == 0
            filename = logpath * "weights-" * string(iter) * ".bson"
        end
    end
end

# Logging a custom table of data
mkpath(logpath*"figures/")

# Final plots
wandb_plot_ndde_data_vs_prediction(df_test, 1, model, pf, "test fit 1")
wandb_plot_ndde_data_vs_prediction(df_train, 1, model, pf, "train fit 1")

# save params
using BSON: @save
filename = logpath * "weights.bson"
@save filename pf

# using BSON: @load
# load_dir ="/home/andrschl/Documents/MA/stable-time-delay-systems/reports/stable_n_pendulum_22-04-21_00:49"
# filename = load_dir * "/weights.bson"
# @load filename pf


# # test to DEBUG
# u_test = hcat(df.trajs[1][2]...)[:, df.N_hist:end]
# h0_test = (p,t)->df.trajs[1][3](t)
# t_test = df.trajs[1][1][df.N_hist:end]
# tspan_test= (0.0,10.0)
# test_pred = predict_ndde(u_test[:,1],h0_test, t_test,pf, model)
#
#
# pl = Plots.plot(t_test, Base.getindex.(test_pred.u,1), xlims=(-1,10))
# Plots.plot!(pl,t->h0_test(nothing,t)[1])
# Plots.scatter!(pl, t_test, u_test[1,:])
# Plots.plot!(t_test, Base.getindex.(test_pred.u,2), xlims=(-1,10))
# Plots.plot!(pl,t->h0_test(nothing,t)[2])
# Plots.scatter!(pl, t_test, u_test[2,:])
# Plots.plot(dense_predict_ndde(u_test[:,1],h0_test, tspan_test,pf, model), xlims=(-1,10))
#
# ts, batch_u, batch_h0 = get_ndde_batch_and_h0(df, batchtime, batchsize)
# batch_t = ts[:,1].-ts[1,1]
# batch_pred_u = predict_ndde(batch_u[:,1,:],batch_h0, batch_t,pf, model)
# predict_LS_loss(batch_u[:,1,:], batch_h0, batch_t, batch_u, pf, model, N=batchtime*batchsize)[1]
# Zygote.gradient(batch_u->predict_LS_loss(batch_u[:,1,:], batch_h0, batch_t, batch_u, pf, model, N=batchtime*batchsize)[1],batch_u)
# Zygote.gradient(u_test->predict_LS_loss(u_test[:,1], h0_test, t_test, u_test, pf, model, N=length(t_test))[1],u_test)
#
# ps = Flux.params(pf)
# gs = gradient(ps) do
#     pred_u = predict_ndde(batch_u[:,1,:], batch_h0, batch_t, pf, model)
#     println(size(pred_u))
#     return test_loss(pred_u, batch_u)
# end
# gs[pf]
#
# ps = Flux.params(pf)
# gs = Zygote.gradient(ps) do
#     pred_u = predict_ndde(u_test[:,1], h0_test, t_test, pf, model)
#     println(size(pred_u))
#     return test_loss(pred_u, u_test)
# end
# gs[pf]
# predict_ndde(u_test[:,1], h0_test, t_test, pf)
