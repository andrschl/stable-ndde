## Hyperparameters
config = Dict(
    # on server?
    "server" => false,
    "logging" => false,

    # ndde training
    "ntrain_trajs" => 2,
    "ntest_trajs" => 2,
    "T_train" => 20.0,
    "T_test" => 40.0,
    "datasize" => 150,
    #"Δt_data" => 0.4,
    "batchtime" => 50,
    "batchsize" => 16,
    "const_init" => false,
    "σ" => 0.00,
    "k0" => "RBF",            # ks ∈ [Mat32, Mat52, RBF]

    # ndde model
    "Δtf" => 0.5,
    "rf" => 5.0,

    # lr schedule
    "lr_rel_decay" => 0.1,
    "lr_start_max" => 5e-3,
    "lr_start_min" => 1e-4,
    "lr_period" => 100,
    "nepisodes" => 1000,

    # logging
    "test_eval" => 20,
    "model checkpoint" => 500
)

## argsparse
seed = 1
if length(ARGS) >= 1
    config["nepisodes"] = parse(Int64, ARGS[1])
end
if length(ARGS) >= 2
    config["lr_period"] = parse(Int64, ARGS[2])
end
if length(ARGS) >= 3
    config["batchtime"] = parse(Int64, ARGS[3])
end
if length(ARGS) >= 4
    config["batchsize"] = parse(Int64, ARGS[4])
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

project_name = "LV_NDDE"
# runname = "seed_"*string(seed)
configname = string(config["nepisodes"])*"/"*string(config["lr_period"])*"/"*string(config["batchtime"])*"/"*string(config["batchsize"])
devicename = config["server"] ? "server_" : "blade_"
logpath = "reports/"*project_name*"/"*configname*"/"
mkpath(logpath)
if config["logging"]
    wandb = pyimport("wandb")
    # wandb.init(project=splitext(basename(@__FILE__))[1], entity="andrschl", config=config, name=runname)
    wandb.init(project=project_name, config=config, name=configname, group=devicename)
end

## set seed
Random.seed!(123)
rng = MersenneTwister(1234)

## Load pendulum dataset
include("../datasets/lotka_volterra.jl")

config["Δt_data"] = config["T_train"]/config["datasize"]

# ICs_train = map(i-> rand(distr_train, 2), 1:config["ntrain_trajs"])
# ICs_test = map(i-> rand(distr_test, 2), 1:config["ntest_trajs"])
u0_init1 = sol=solve(LV_prob, tspan=(0.0,-5.0), u0=[2.0,2.0], dense=true)(-5.0)
u0_init2 = sol=solve(LV_prob, tspan=(0.0,-5.0), u0=[3.0,3.0], dense=true)(-5.0)

ICs_train = [u0_init1, u0_init2]
ICs_test = ICs_train
tspan_train = (0.0, config["T_train"])
tspan_test = (0.0, config["T_test"])

df_train = DDEODEDataset(ICs_train, tspan_train, config["Δt_data"], LV_prob, config["rf"];obs_ids=[1])
df_test = DDEODEDataset(ICs_test, tspan_test, config["Δt_data"], LV_prob, config["rf"];obs_ids=[1])

gen_dataset!(df_train)
Random.seed!(122+seed)
gen_noise!(df_train, config["σ"])
gen_dataset!(df_test)
gen_noise!(df_test, config["σ"])

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
length(df_train)
df_train
get_noisy_ndde_batch_and_h0(df_train, config["batchtime"], config["batchsize"])
# plot(0:0.01:40, t->df_train.trajs[1][3](t)[1])
# scatter!(df_train.noisy_trajs[1][1], vcat(df_train.noisy_trajs[1][2]...))
# plot(0:0.01:40, t->df_train.trajs[2][3](t)[1])
# scatter!(df_train.noisy_trajs[2][1], vcat(df_train.noisy_trajs[2][2]...))

test_loss_data = DataFrame(iter = Int[], test_loss = Float64[])
train_loss_data = DataFrame(iter = Int[], train_loss = Float64[])

3
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
        ts, batch_u, batch_h0,_ = get_noisy_ndde_batch_and_h0(df_train, config["batchtime"], config["batchsize"],k0=config["k0"])
        # ts, batch_u, batch_h0,_ = get_ndde_batch_and_h0(df_train, config["batchtime"], config["batchsize"])
        batch_t = ts[:,1].-ts[1,1]

        # combined train step
        # kras_stable_ndde_train_step!(batch_h0(nothing, batch_t[1]), batch_u, batch_h0, pf, pv, batch_t, model, optf, optv, iter, lyap_loader)
        ndde_train_step!(batch_h0(nothing, batch_t[1]), batch_u, batch_h0, pf, batch_t, model, optf,iter)

        if !config["server"]
            pl_train = plot(title="train")
            for i in 1:length(batch_u[:,1,1])
                scatter!(pl_train, batch_t, batch_u[i,:,1])
            end
            plot!(pl_train, dense_predict_ndde(batch_h0(nothing, batch_t[1])[:,1], (p,t)->batch_h0(p,t)[:,1], (batch_t[1],batch_t[end]), pf, model),xlims=(batch_t[1], batch_t[end]))
            display(pl_train)
        end

        # test evaluation
        if (iter % config["test_eval"] == 0)
            # log train fit
            if config["logging"]
                for i in 1:config["ntrain_trajs"]
                    if !config["server"]
                        # wandb_plot_noisy_ndde_data_vs_prediction(df_train, i, model, pf, "train fit "*string(i), k0=config["k0"])
                        save_plot_noisy_ndde_data_vs_prediction(df_train, i, model, pf, logpath, "train_"*string(i)*"_", k0=config["k0"])

                    else
                        save_plot_noisy_ndde_data_vs_prediction(df_train, i, model, pf, logpath, "train_"*string(i)*"_", k0=config["k0"])
                    end
                end
            end
            # log test fit
            test_losses = []
            for i in 1:config["ntest_trajs"]
                t_test = df_test.noisy_trajs[i][1][df_test.N_hist:end]
                u_test = hcat(df_test.noisy_trajs[i][2][df_test.N_hist:end]...)
                h0_test = get_noisy_h0(df_test, i,k0=config["k0"])
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
                    if !config["server"]
                        # wandb_plot_noisy_ndde_data_vs_prediction(df_test, i, model, pf, "test fit "*string(i), k0=config["k0"])
                        save_plot_noisy_ndde_data_vs_prediction(df_test, i, model, pf, logpath, "test_"*string(i)*"_", k0=config["k0"])
                    else
                        save_plot_noisy_ndde_data_vs_prediction(df_test, i, model, pf, logpath, "test_"*string(i)*"_", k0=config["k0"])
                    end
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
