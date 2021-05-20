## Hyperparameters
config = Dict(
    # on server?
    "server" => false,
    "logging" => true,

    # ndde training
    "ntrain_trajs" => 1,
    "ntest_trajs" => 1,
    "T_train" => 40.0,
    "T_test" => 40.0,
    "datasize" => 1000,
    #"Δt_data" => 0.4,
    "batchtime" => 50,
    "batchsize" => 10,
    "const_init" => false,
    "k0" => "RBF",            # ks ∈ [Mat32, Mat52, RBF]

    # ndde model
    "Δtf" => 0.4,
    "rf" => 4.0,

    # lr schedule
    "lr_rel_decay" => 0.01,
    "lr_start_max" => 5e-4,
    "lr_start_min" => 1e-5,
    "lr_period" => 50,
    "nepisodes" => 5000,

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
    config["npendulum"] = parse(Int64, ARGS[2])
end
if length(ARGS) >= 3
    config["k0"] = ARGS[3]
end
if length(ARGS) >= 4
    config["σ"] = parse(Float64, ARGS[4])
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

project_name = "lorenz"
runname = "seed_"*string(seed)
machine = config["server"] ? "server_" : "blade_"
logpath = "reports/"*project_name*"/seed_"*string(seed)*"/"
mkpath(logpath)
if config["logging"]
    wandb = pyimport("wandb")
    # wandb.init(project=splitext(basename(@__FILE__))[1], entity="andrschl", config=config, name=runname)
    wandb.init(project=project_name, config=config, name=runname, group=machine)
end

## set seed
Random.seed!(123)
rng = MersenneTwister(1234)

## Load pendulum dataset
include("../datasets/lorenz.jl")

config["Δt_data"] = config["T_train"]/config["datasize"]


ICs_train = [[3.7144662401391497, 5.132226099526597, 17.85688360743861]]
ICs_test = [[3.7144662401391497, 5.132226099526597, 17.85688360743861]]

tspan_train = (0.0, config["T_train"])
tspan_test = (0.0, config["T_test"])

df_train = DDEODEDataset(ICs_train, tspan_train, config["Δt_data"], lorenz_prob, config["rf"];obs_ids=[1])
df_test = DDEODEDataset(ICs_test, tspan_test, config["Δt_data"], lorenz_prob, config["rf"];obs_ids=[1])

gen_dataset!(df_train)
Random.seed!(122+seed)
# gen_noise!(df_train, config["σ"])
gen_dataset!(df_test)
# gen_noise!(df_test, config["σ"])

## Define model
include("../models/model.jl")
data_dim = 1
flags = Array(config["Δtf"]:config["Δtf"]:config["rf"])
Random.seed!(seed)
model = KrasNDDE(data_dim; flags=flags, vlags=flags, α=0.1, q=1.1)
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
        # ts, batch_u, batch_h0,_ = get_noisy_ndde_batch_and_h0(df_train, config["batchtime"], config["batchsize"],k0=config["k0"])
        ts, batch_u, batch_h0,_ = get_ndde_batch_and_h0(df_train, config["batchtime"], config["batchsize"])
        batch_t = ts[:,1].-ts[1,1]

        # combined train step
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
                        wandb_plot_ndde_data_vs_prediction(df_train, i, model, pf, "train fit "*string(i))
                        # save_plot_noisy_ndde_data_vs_prediction(df_train, i, model, pf, logpath, "train_"*string(i)*"_", k0=config["k0"])

                    else
                        # save_plot_noisy_ndde_data_vs_prediction(df_train, i, model, pf, logpath, "train_"*string(i)*"_", k0=config["k0"])
                    end
                end
            end
            # log test fit
            test_losses = []
            for i in 1:config["ntest_trajs"]
                t_test = df_test.trajs[i][1][df_test.N_hist:end]
                u_test = hcat(df_test.trajs[i][2][df_test.N_hist:end]...)
                h0_test = get_h0(df_test, i)
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
                        wandb_plot_ndde_data_vs_prediction(df_test, i, model, pf, "test fit "*string(i))
                        # save_plot_noisy_ndde_data_vs_prediction(df_test, i, model, pf, logpath, "test_"*string(i)*"_", k0=config["k0"])
                    else
                        # save_plot_noisy_ndde_data_vs_prediction(df_test, i, model, pf, logpath, "test_"*string(i)*"_", k0=config["k0"])
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