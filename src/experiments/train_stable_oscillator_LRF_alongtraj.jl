config = Dict(
    # on server?
    "server" => false,
    "logging" => true,

    # ground truth dynamics
    "ω" => 1.0,
    "γ" => 0.05,

    # ndde training
    "θmax_train" => 1.0,
    "θmax_test" => 10.0,
    "ntrain_trajs" => 4,
    "ntest_trajs" => 4,
    "T_train" => 4*pi,
    "T_test" => 40*pi,
    "datasize" => 100,
    #"Δt_data" => 0.4,
    "batchtime" => 50,
    "batchsize" => 4,
    "const_init" => false,
    "σ" => 0.02,
    "k0" => "RBF",            # ks ∈ [Mat32, Mat52, RBF]


    # ndde model
    "Δtf" => 0.3,
    "rf" => 3.0,

    # lr schedule
    "lr_rel_decay" => 0.01,
    "lr_start_max" => 5e-3,
    "lr_start_min" => 1e-4,
    "lr_period" => 20,
    "nepisodes" => 200,

    # stabilizing training
    "θmax_lyap" => 5.0,
    "nlyap_trajs" => 10,
    "T_lyap" => 30,
    "batchsize_lyap" => 256,
    "nacc_steps_lyap" => 1,
    "uncorrelated" => false,
    "uncorrelated_data_size" => 100000,
    "resample" => true,

    # lyapunov loss
    "Δtv" => 0.3,
    "rv" => 9.0,
    "α" => 0.01,
    "q" => 1.01,
    "weight_f" => 0.01,
    "weight_v" => 0.1,
    "grad_clipping" => 1e-2,
    "warmup_steps" => 20,
    "pause_steps" => 60,
    "k0" => "RBF",            # ks ∈ [Mat32, Mat52, RBF]


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
using AbstractGPs
## seeding
seed = 1
if length(ARGS) >= 1
    seed = parse(Int64, ARGS[1])
end
if length(ARGS) >= 2
    config["σ"] = parse(Float64, ARGS[2])
end

## log path
project_name = "oscillator_LRF_23_05"
runname = "seed_"*string(seed)
configname = string(config["σ"])*"/"
devicename = config["server"] ? "server_" : "blade_"
logpath = "reports/"*project_name*"/seed_"*string(seed)*"/"
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
rng = MersenneTwister(1234)

## Load pendulum dataset
include("../datasets/stable_oscillator.jl")

config["Δt_data"] = config["T_train"]/config["datasize"]

# initial conditions
distr_train = MixtureModel(Uniform, [(-config["θmax_train"], -config["θmax_train"]/2), (config["θmax_train"]/2, config["θmax_train"])])
distr_test = MixtureModel(Uniform, [(-config["θmax_test"], -config["θmax_test"]/2), (config["θmax_test"]/2, config["θmax_test"])])
distr_lyap = Uniform(-config["θmax_lyap"], config["θmax_lyap"])

# ICs_train = map(i-> vcat(rand(rng, distr_train, 2)), 1:config["ntrain_trajs"])
# ICs_test = map(i-> vcat(rand(rng, distr_test, 2)), 1:config["ntest_trajs"])
ICs_train = map(i-> [rand(distr_train, 1)[1],0.0], 1:config["ntrain_trajs"])
ICs_test = map(i-> [rand(distr_test, 1)[1],0.0], 1:config["ntest_trajs"])


tspan_train = (0.0, config["T_train"])
tspan_lyap = (0.0,config["T_lyap"] )
tspan_test = (0.0, config["T_test"])

df_train = DDEODEDataset(ICs_train, tspan_train, config["Δt_data"], oscillator_prob, config["rf"]; obs_ids=[1])
df_test = DDEODEDataset(ICs_test, tspan_test, config["Δt_data"], oscillator_prob, config["rf"]; obs_ids=[1])

gen_dataset!(df_train)
Random.seed!(122+seed)
gen_noise!(df_train, 0.2)
gen_dataset!(df_test)
gen_noise!(df_test, 0.2)

## Define model
include("../models/model.jl")
data_dim = 1
flags = Array(config["Δtf"]:config["Δtf"]:config["rf"])
vlags = Array(config["Δtv"]:config["Δtv"]:config["rv"])
Random.seed!(seed)
model = RazNDDE(data_dim; flags=flags, vlags=vlags, α=config["α"], q=config["q"])
pf = model.pf
pv = model.pv

## Define model dataset for lyapunov training
function sample_on_RKHS(t; k0="RBF", rng=Random.GLOBAL_RNG,
            α_lim=0.5, α_min_rad=0.5, scaling_lims=(4.0, 10.0), magnitude_lims=(0.01, 1.0))

    ks = Dict("RBF"=>SqExponentialKernel(), "Mat32"=>Matern32Kernel(), "Mat52"=>Matern52Kernel())
    k0 = ks[k0]
    d = length(t)
    Y = randn(rng, d)
    Y = Y ./ sqrt(Y'*Y)
    U = α_lim * rand(rng, Uniform(α_min_rad,1))^(1/d)
    α = U * Y
    # α = rand(rng, Uniform(-α_lim,α_lim), length(t))
    scaling = rand(rng, Uniform(scaling_lims...))
    magnitude = rand(rng, Uniform(magnitude_lims...))
    kernel = magnitude * (k0 ∘ ScaleTransform(scaling))
    return ξ -> (kernelmatrix(kernel, [ξ], t) * α)
end
function sample_RKHS_h0s(init_t, data_dim; k0="RBF", rng=Random.GLOBAL_RNG,
            α_lim=10.0, α_min_rad=0.5, scaling_lims=(0.5, 2.0), magnitude_lims=(0.01, 1.0))

    h0 = []
    for i in 1:data_dim
        push!(h0, sample_on_RKHS(init_t, k0=k0, rng=rng, α_lim=α_lim, α_min_rad=α_min_rad, scaling_lims=scaling_lims, magnitude_lims=magnitude_lims))
    end
    return (p,t) -> vcat(map(i -> h0[i](t), 1:length(h0))...)
end

if !config["uncorrelated"]
    init_t = df_train.trajs[1][1][1:df_train.N_hist]
    h0s_lyap = map(i->sample_RKHS_h0s(init_t, data_dim, rng=rng), 1:config["nlyap_trajs"])
    # u0s = map(i-> rand(distr_test, data_dim), 1:config["nlyap_trajs"])
    # h0s_lyap = map(i-> (p,t)-> u0s[i], 1:config["nlyap_trajs"])
    lyap_prob = DDEProblem(model.ndde_func!, zeros(data_dim), h0s_lyap[1], tspan_lyap, constant_lags=flags)
    df_model = DDEDDEDataset(h0s_lyap, tspan_lyap, min(config["Δtv"],config["Δtf"]), lyap_prob, config["rv"], flags)
    gen_dataset!(df_model, p=model.pf)
    lyap_loader = Flux.Data.DataLoader(df_model, batchsize=config["batchsize_lyap"], shuffle=false)
else
    data = hcat(map(i-> rand(distr_lyap, 2*data_dim*(length(vlags)+1)), 1:config["uncorrelated_data_size"])...)
    lyap_loader = Flux.Data.DataLoader((Array(1:config["uncorrelated_data_size"]),data), batchsize=config["batchsize_lyap"],shuffle=true)
end
if !config["server"]
    pl = plot()
    pl2 = plot()
    for (j, traj) in enumerate(df_model.trajs[1:4])
        h0 = get_noisy_h0(df_test, j)
        for i in 1:data_dim
            # scatter!(pl, traj[1], hcat(traj[2]...)[i,:])
            plot!(pl, traj[1][1]:0.01:0.0,t-> traj[3](t)[i])
            plot!(pl2, traj[1][1]:0.01:0.0, t->h0(nothing,t)[i])
        end
    end
    display(plot(pl,pl2))
end
# iterate(lyap_loader)
## training
include("../training/training_util.jl")

test_loss_data = DataFrame(iter = Int[], test_loss = Float64[])
train_loss_data = DataFrame(iter = Int[], train_loss = Float64[])
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
            if config["resample"]
                init_t = df_train.trajs[1][1][1:df_train.N_hist]
                h0s_lyap = map(i->sample_RKHS_h0s(init_t, data_dim, rng=rng), 1:config["nlyap_trajs"])
                lyap_prob = DDEProblem(model.ndde_func!, zeros(data_dim), h0s_lyap[1], tspan_lyap, constant_lags=flags)
                df_model = DDEDDEDataset(h0s_lyap, tspan_lyap, min(config["Δtv"],config["Δtf"]), lyap_prob, config["rv"], flags)
            end
            gen_dataset!(df_model, p=pf)
            lyap_loader = Flux.Data.DataLoader(df_model, batchsize=config["batchsize_lyap"], shuffle=true)
        end

        # combined train step

        raz_stable_ndde_train_step!(batch_u[:,1,:], batch_u, batch_h0, pf, pv, batch_t, model, optf, optv, iter, lyap_loader)

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
                test_loss = sum(test_losses)/config["ntest_trajs"]
                global test_loss_data
                push!(test_loss_data, [iter, test_loss])
                wandb.log(Dict("test loss"=> test_loss), step=iter)
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
CSV.write(logpath*"test_loss.csv", test_loss_data, header = true)
CSV.write(logpath*"train_loss.csv", train_loss_data, header = true)

# save params
using BSON: @save
filename_f = logpath * "weights_f.bson"
filename_v = logpath * "weights_v.bson"
@save filename_f pf
@save filename_v pf
