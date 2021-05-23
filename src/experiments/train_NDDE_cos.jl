## Hyperparameters
config = Dict(
    # on server?
    "server" => false,
    "logging" => false,

    # lr schedule
    "lr_rel_decay" => 1.0,
    "lr_start_max" => 5e-3,
    "lr_start_min" => 1e-5,
    "lr_period" => 150,
    "nepisodes" => 150,

    # ground truth dynamics
    "ω" => 1.0,
    "γ" => 0.0,

    # ndde model
    "Δtf" => 0.3,
    "rf" => 3.0,

    # ndde training
    "A_train" => 1.0,
    "A_test" => 1.0,
    "ntrain_trajs" => 1,
    "ntest_trajs" => 1,
    "T_train" => 12*pi,
    "T_test" => 100.0,
    "datasize" => 200,
    "k0" => "RBF",
    "σ" => 0.3,

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
# using GaussianProcesses
using AbstractGPs

## log path
# current_time = Dates.format(now(), "_dd-mm-yy_HH:MM/")
# runname = "stable oscillator"*current_time
# logpath = "reports/"*splitext(basename(@__FILE__))[1]*current_time

project_name = "cos_NDDE"
runname = "seed_"*string(seed)
configname = string(config["σ"])*"/"
devicename = config["server"] ? "server_" : "blade_"
logpath = "reports/"*project_name*"/"*configname*runname*"/"
mkpath(logpath)
if config["logging"]
    wandb = pyimport("wandb")
    # wandb.init(project=splitext(basename(@__FILE__))[1], entity="andrschl", config=config, name=runname)
    wandb.init(project=project_name, config=config, name=runname, group=devicename*"longterm_"*configname)
end

## set seed
## set seed
Random.seed!(123)
rng = MersenneTwister(1234)

## Load pendulum dataset
include("../datasets/stable_oscillator.jl")

## Define dataset
data_dim = 1
aug_dim = 1
A = config["A_train"]
data_size = config["datasize"]
σ = config["σ"]
tspan_train = (3.0, config["T_train"])


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
u_noisy = true_u + σ*randn(data_dim, length(t))
pl = scatter(t, u_noisy[1,:], label="data")
plot!(pl, t->true_sol(t)[1], xlims=(0,12*pi))
display(plot(pl))


tspan_train = (3.0, config["T_train"])

df_train
df_train = DDEODEDataset([[1.0,0.0]], tspan_train, config["Δt"], oscillator_prob, config["rf"]; obs_ids=[1])
df_train.trajs = [[t, vcat(true_u, true_sol]]
map(col->col, eachcol(true_u))
gen_dataset!(df_train)
Random.seed!(122+seed)
gen_noise!(df_train, 0.3)
 df_train.trajs
pl = scatter(t, df_train.noisy_trajs[], label="data")
plot!(pl, t->true_sol(t)[1], xlims=(0,12*pi))
display(plot(pl))

## Define model
include("../models/model.jl")
data_dim = 1
flags = Array(config["Δtf"]:config["Δtf"]:config["rf"])
vlags = flags
Random.seed!(seed)
model = KrasNDDE(data_dim; flags=flags, vlags=vlags, α=0.1, q=1.1)
pf = model.pf
pv = model.pv

# iterate(lyap_loader)
## training
include("../training/training_util.jl")

@time begin
    rel_decay, locmin, locmax, period = config["lr_rel_decay"], config["lr_start_min"], config["lr_start_max"], config["lr_period"]
    lr_args = (rel_decay, locmin, locmax, period)
    lr_kwargs = Dict(:len => config["nepisodes"])
    lr_schedule_gen = double_exp_decays
    lr_schedule = lr_schedule_gen(lr_args...;lr_kwargs...)
    for (lr, iter) in lr_schedule
        println("==============")
        println("iter: ", iter)
        # get ndde batch
        optf = ADAM(lr)
        optv=optf
        ts, batch_u, batch_h0,_ = get_noisy_ndde_batch_and_h0(df_train, config["batchtime"], config["batchsize"])
        # ts, batch_u, batch_h0,_ = get_ndde_batch_and_h0(df_train, config["batchtime"], config["batchsize"])
        batch_t = ts[:,1].-ts[1,1]
        # get lyapunov data
        global lyap_loader
        if !config["uncorrelated"]
            gen_dataset!(df_model, p=pf)
            lyap_loader = Flux.Data.DataLoader(df_model, batchsize=config["batchsize_lyap"], shuffle=true)
        end

        # combined train step
        # kras_stable_ndde_train_step!(batch_h0(nothing, batch_t[1]), batch_u, batch_h0, pf, pv, batch_t, model, optf, optv, iter, lyap_loader)
        ndde_train_step!(batch_h0(nothing, batch_t[1]), batch_u, batch_h0, pf, batch_t, model, optf,iter)

        if !config["server"]
            pl_train = scatter(batch_t, batch_u[1,:,1])
            plot!(pl_train, dense_predict_ndde(batch_h0(nothing, batch_t[1])[:,1], (p,t)->batch_h0(p,t)[:,1], (batch_t[1],batch_t[end]), pf, model),xlims=(batch_t[1], batch_t[end]))
            display(pl_train)
        end

        # test evaluation
        if (iter % config["test_eval"] == 0)
            # log train fit
            if config["logging"]
                for i in 1:config["ntrain_trajs"]
                    # if !config["server"]
                    #     wandb_plot_noisy_ndde_data_vs_prediction(df_train, i, model, pf, "train fit "*string(i))
                    # else
                    #     save_plot_noisy_ndde_data_vs_prediction(df_train, i, model, pf, logpath, "train_"*string(i)*"_")
                    # end
                    wandb_plot_noisy_ndde_data_vs_prediction(df_train, i, model, pf, "train fit "*string(i))
                    save_plot_noisy_ndde_data_vs_prediction(df_train, i, model, pf, logpath, "train_"*string(i)*"_")
                end
            end
            # log test fit
            test_losses = []
            for i in 1:config["ntest_trajs"]
                t_test = df_test.noisy_trajs[i][1][df_test.N_hist:end]
                u_test = hcat(df_test.noisy_trajs[i][2][df_test.N_hist:end]...)
                h0_test = get_noisy_h0(df_test, i)
                u0_test = h0_test(pf, t_test[1])
                test_loss, _ = predict_ndde_loss(u0_test, h0_test, t_test, u_test, pf, model; N=df_test.N)
                push!(test_losses, test_loss)
                if !config["server"]
                    test_sol = dense_predict_ndde(u0_test, h0_test, tspan_test, pf, model)
                    pl = plot(test_sol, xlims=tspan_test, title="Generalization traj " * string(i))
                    scatter!(pl, t_test, u_test[1,:], label="θ_true1")
                    display(pl)
                end
                if config["logging"]
                    # if !config["server"]
                    #     wandb_plot_noisy_ndde_data_vs_prediction(df_test, i, model, pf, "test fit "*string(i))
                    # else
                    #     save_plot_noisy_ndde_data_vs_prediction(df_test, i, model, pf, logpath, "test_"*string(i)*"_")
                    # end
                    wandb_plot_noisy_ndde_data_vs_prediction(df_test, i, model, pf, "test fit "*string(i))
                    save_plot_noisy_ndde_data_vs_prediction(df_test, i, model, pf, logpath, "test_"*string(i)*"_")
                end
            end
            if config["logging"]
                wandb.log(Dict("test loss"=> sum(test_losses)/config["ntest_trajs"]), step=iter)
            end
        end
        # if iter % config["model checkpoint"] == 0
        #     using BSON: @save
        #     filename = logpath * "weights-" * string(iter) * ".bson"
        #     @save filename pf
        # end
    end
end

# save params
using BSON: @save
filename_f = logpath * "weights_f.bson"
filename_v = logpath * "weights_v.bson"
@save filename_f pf
@save filename_v pf
