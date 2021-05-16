## Hyperparameters
config = Dict(
    # on server?
    "server" => false,
    "logging" => true,

    # ground truth dynamics
    "npendulum" => 2,
    "friction" => 0.1,
    "length" => 1.0,
    "mass" => 1.0,
    "σ" => 0.05,

    # ndde training
    "θmax_train" => pi/10,
    "θmax_test" => pi/2,
    "ntrain_trajs" => 4,
    "ntest_trajs" => 4,
    "T_train" => 4.0,
    "T_test" => 40.0,
    "datasize" => 200,
    #"Δt_data" => 0.4,
    "batchtime" => 200,
    "batchsize" => 4,
    "const_init" => false,
    "k0" => "RBF",            # ks ∈ [Mat32, Mat52, RBF]

    # ndde model
    "Δtf" => 0.1,
    "rf" => 1.0,

    # lr schedule
    "lr_rel_decay" => 0.01,
    "lr_start_max" => 5e-3,
    "lr_start_min" => 1e-4,
    "lr_period" => 20,
    "nepisodes" => 500,

    # stabilizing training
    "θmax_lyap" => 5.0,
    "nlyap_trajs" => 10,
    "T_lyap" => 30,
    "batchsize_lyap" => 256,
    "nacc_steps_lyap" => 1,
    "uncorrelated" => true,
    "uncorrelated_data_size" => 100000,

    # lyapunov loss
    "Δtv" => 0.1,
    "rv" => 1.0,
    "α" => 0.01,
    "q" => 1.01,
    "weight_f" => 0.1,
    "weight_v" => 1,
    "warmup_steps" => 20,
    "pause_steps" => 60,

    # logging
    "test_eval" => 20,
    "model checkpoint" => 500
)

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
using AbstractGPs, Plots

## argsparse
if length(ARGS)>0
    seed = parse(Int64, ARGS[1])
elseif length(ARGS)>1
    seed = parse(Int64, ARGS[1])
    config["npendulum"] = parse(Int64, ARGS[2])
else
    seed = 1
end
if length(ARGS) == 3
    config["k0"] = ARGS[3]
end
## log path
# current_time = Dates.format(now(), "_dd-mm-yy_HH:MM/")
# runname = "stable oscillator"*current_time
# logpath = "reports/"*splitext(basename(@__FILE__))[1]*current_time

project_name = string(config["npendulum"])*"_pendulum_unstable"
runname = "seed_"*string(seed)
logpath = "reports/"*project_name*"/seed_"*string(seed)*"/"
mkpath(logpath)
if config["logging"]
    wandb = pyimport("wandb")
    # wandb.init(project=splitext(basename(@__FILE__))[1], entity="andrschl", config=config, name=runname)
    wandb.init(project=project_name, config=config, name=runname, group="blade-σ-0.1-episodes-500")
end
## set seed
Random.seed!(123)
rng = MersenneTwister(1234)

## Load pendulum dataset
N_PENDULUM = config["npendulum"]
include("../datasets/n_pendulum.jl")

config["Δt_data"] = config["T_train"]/config["datasize"]

# initial conditions
distr_train = MixtureModel(Uniform, [(-config["θmax_train"], -config["θmax_train"]/2), (config["θmax_train"]/2, config["θmax_train"])])
distr_test = MixtureModel(Uniform, [(-config["θmax_test"], -config["θmax_test"]/2), (config["θmax_test"]/2, config["θmax_test"])])
distr_lyap = Uniform(-config["θmax_lyap"], config["θmax_lyap"])

# ICs_train = map(i-> vcat(rand(rng, distr_train, 2)), 1:config["ntrain_trajs"])
# ICs_test = map(i-> vcat(rand(rng, distr_test, 2)), 1:config["ntest_trajs"])

ICs_train = map(i-> vcat(zeros(N_PENDULUM), rand(distr_train, N_PENDULUM)), 1:config["ntrain_trajs"])
ICs_test = map(i-> vcat(zeros(N_PENDULUM), rand(distr_test, N_PENDULUM)), 1:config["ntest_trajs"])
# ICs_train = map(i-> [rand(distr_train, 1)[1],0.0], 1:config["ntrain_trajs"])
# ICs_test = map(i-> [rand(distr_test, 1)[1],0.0], 1:config["ntest_trajs"])
# ICs_lyap = map(i-> vcat(rand(rng, distr_lyap, 1)), 1:config["nlyap_trajs"])
# h0s_lyap = map(u0 -> (p,t) -> u0, ICs_lyap)

tspan_train = (0.0, config["T_train"])
tspan_lyap = (0.0,config["T_lyap"] )
tspan_test = (0.0, config["T_test"])

df_train = DDEODEDataset(ICs_train, tspan_train, config["Δt_data"], pendulum_prob, config["rf"];obs_ids=Array(N_PENDULUM+1:2*N_PENDULUM))
df_test = DDEODEDataset(ICs_test, tspan_test, config["Δt_data"], pendulum_prob, config["rf"];obs_ids=Array(N_PENDULUM+1:2*N_PENDULUM))

gen_dataset!(df_train)
Random.seed!(122+seed)
gen_noise!(df_train, config["σ"])
gen_dataset!(df_test)
gen_noise!(df_test, config["σ"])

## Define model
include("../models/model.jl")
data_dim = config["npendulum"]
flags = Array(config["Δtf"]:config["Δtf"]:config["rf"])
vlags = Array(config["Δtv"]:config["Δtv"]:config["rv"])
Random.seed!(seed)
model = KrasNDDE(data_dim; flags=flags, vlags=vlags, α=config["α"], q=config["q"])
pf = model.pf
pv = model.pv

## Define model dataset for lyapunov training

if !config["uncorrelated"]
    lyap_prob = DDEProblem(model.ndde_func!, ICs_lyap[1],h0s_lyap[1], tspan_lyap, constant_lags=flags)
    df_model = DDEDDEDataset(h0s_lyap, tspan_lyap, min(config["Δtv"],config["Δtf"]), lyap_prob, max(config["rv"],config["rf"]), flags)
    gen_dataset!(df_model, p=model.pf)
    lyap_loader = Flux.Data.DataLoader(df_model, batchsize=config["batchsize_lyap"], shuffle=true)
else
    data = hcat(map(i-> rand(rng, distr_lyap, 2*data_dim*(length(vlags)+1)), 1:config["uncorrelated_data_size"])...)
    lyap_loader = Flux.Data.DataLoader((Array(1:config["uncorrelated_data_size"]),data), batchsize=config["batchsize_lyap"],shuffle=true)
end

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
        ts, batch_u, batch_h0,_ = get_noisy_ndde_batch_and_h0(df_train, config["batchtime"], config["batchsize"],k0=config["k0"])
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
                        wandb_plot_noisy_ndde_data_vs_prediction(df_train, i, model, pf, "train fit "*string(i), k0=config["k0"])
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
                        wandb_plot_noisy_ndde_data_vs_prediction(df_test, i, model, pf, "test fit "*string(i), k0=config["k0"])
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
