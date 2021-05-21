## Hyperparameters
config = Dict(
    # on server?
    "server" => false,
    "logging" => false,

    # lr schedule
    "lr_rel_decay" => 0.01,
    "lr_start_max" => 5e-3,
    "lr_start_min" => 1e-4,
    "lr_period" => 20,
    "nepisodes" => 500,

    # stabilizing training
    "θmax_lyap" => pi/2,
    "nlyap_trajs" => 10,
    "ntest_trajs" => 4,
    "T_lyap" => 3.0,
    "batchsize_lyap" => 256,
    "nacc_steps_lyap" => 1,
    "uncorrelated" => false,
    "uncorrelated_data_size" => 100000,

    # ndde model
    "Δtf" => 0.03,
    "rf" => 0.03,

    # lyapunov loss
    "Δtv" => 0.01,
    "rv" => 0.1,
    "α" => 0.01,
    "q" => 1.01,
    # "α" => 0.1,
    # "q" => 1.1,
    "weight_f" => 0.1,
    "weight_v" => 1,
    "grad_clipping" => 1e10,  # threshold in avg l1 norm after weighting
    "warmup_steps" => 2,
    "pause_steps" => 2,

    # logging
    "test_eval" => 20,
    "model checkpoint" => 500
)
lags
## argsparse
seed = 1
if length(ARGS) >= 1
    seed = parse(Int64, ARGS[1])
end
if length(ARGS) >= 4
    config["weight_f"] = parse(Float64, ARGS[4])
end
if length(ARGS) >= 5
    config["weight_v"] = parse(Float64, ARGS[5])
end
if length(ARGS) >= 6
    config["grad_clipping"] = parse(Float64, ARGS[6])
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

## log path
# current_time = Dates.format(now(), "_dd-mm-yy_HH:MM/")
# runname = "stable oscillator"*current_time
# logpath = "reports/"*splitext(basename(@__FILE__))[1]*current_time

project_name = "inverted_pendulum_LKF_alongtraj"
runname = "seed_"*string(seed)
configname = string(config["weight_f"])*"/"*string(config["weight_v"])*"/"*string(config["grad_clipping"])*"/"
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
rng = MersenneTwister(123)

## Load pendulum dataset
data_dim = 2
include("../datasets/inverted_pendulum.jl")

distr_lyap = Uniform(-config["θmax_lyap"], config["θmax_lyap"])
tspan_lyap = (0.0,config["T_lyap"])
flags = Array(0:config["Δtf"]:config["rf"])[2:end]
vlags = Array(0:config["Δtv"]:config["rv"])[2:end]
# flags = [0.03]
lags = sort(union(flags,vlags))
nfdelays = length(flags)
nvdelays = length(vlags)
max_lag = maximum(lags)
config["Δtv"] = vlags[end] /length(vlags)
config["rv"] = vlags[end]
config["Δtf"] = flags[end] /length(flags)
config["rf"] = flags[end]

## Define model
include("../models/model.jl")

model_v = Lyapunov(data_dim * (length(vlags)+1))
pv,rev = Flux.destructure(model_v)
pf,ref = Flux.destructure(f_cl)

model = KrasNDDE(data_dim, ref, rev, pf, pv, flags=flags,vlags=vlags,α=config["α"],q=config["q"])
model
if !config["uncorrelated"]
    u0s = map(i -> sample_u0(config["θmax_lyap"]), 1:config["nlyap_trajs"])
    h0s = map(u0 -> (p,t)->dense_predict_reverse_ode(u0, flags[end], 0.0)(t), u0s)
    lyap_prob = DDEProblem(model.ndde_func!, zeros(data_dim), h0s[1], tspan_lyap, constant_lags=flags)
    df_model = DDEDDEDataset(h0s, tspan_lyap, min(config["Δtv"],config["Δtf"]), lyap_prob, 2*config["rv"], flags)
    gen_dataset!(df_model, p=model.pf)
    lyap_loader = Flux.Data.DataLoader(df_model, batchsize=config["batchsize_lyap"], shuffle=false)
else
    data = hcat(map(i-> rand(rng, distr_lyap, 2*data_dim*(length(vlags)+1)), 1:config["uncorrelated_data_size"])...)
    lyap_loader = Flux.Data.DataLoader((Array(1:config["uncorrelated_data_size"]),data), batchsize=config["batchsize_lyap"],shuffle=true)


    u0s = map(i -> sample_u0(config["θmax_lyap"]), 1:config["nlyap_trajs"])
    h0s = map(u0 -> (p,t)->dense_predict_reverse_ode(u0, flags[end], 0.0)(t), u0s)
    lyap_prob = DDEProblem(model.ndde_func!, zeros(data_dim), h0s[1], tspan_lyap, constant_lags=flags)
    df_model = DDEDDEDataset(h0s, tspan_lyap, min(config["Δtv"],config["Δtf"]), lyap_prob, 2*config["rv"], flags)
    gen_dataset!(df_model, p=model.pf)
end

# test data
tspan_test = (0.,10.0)
u0s = map(i -> sample_u0(config["θmax_lyap"]), 1:config["ntest_trajs"])
h0s = map(u0 -> (p,t)->dense_predict_reverse_ode(u0, flags[end], 0.0)(t), u0s)
test_prob = DDEProblem(model.ndde_func!, zeros(data_dim), h0s[1], tspan_test, constant_lags=flags)
df_test = DDEDDEDataset(h0s, tspan_test, min(config["Δtv"],config["Δtf"]), test_prob, 2*config["rv"], flags)
gen_dataset!(df_model, p=model.pf)
lyap_loader = Flux.Data.DataLoader(df_model, batchsize=config["batchsize_lyap"], shuffle=false)

if !config["server"]
    pl = plot()
    pl2 = plot()
    for (j, traj) in enumerate(df_model.trajs[1:4])
        for i in 1:data_dim
            # scatter!(pl, traj[1], hcat(traj[2]...)[i,:])
            plot!(pl, traj[1][1]:0.01:3.0,t-> traj[3](t)[i])
        end
    end
    display(plot(pl))
end

## training
include("../training/training_util.jl")


# # check length scaling, magnitudes, and kernel coefficients
# ls = []
# ms = []
# αs = []
# ts, batch_u, batch_h0,_ = get_noisy_ndde_batch_and_h0(df_train, config["batchtime"], config["batchsize"],k0=config["k0"])
# ts, batch_u, batch_h0,_ = get_noisy_ndde_batch_and_h0(df_test, config["batchtime"], config["batchsize"],k0=config["k0"])
# mmax = maximum(ms)
# mmin = minimum(ms)
# lmax = maximum(ls)
# lmax = minimum(ls)
# αmin = minimum(αs)
# αmax = maximum(αs)

test_loss_data = DataFrame(iter = Int[], test_loss = Float64[])
train_loss_data = DataFrame(iter = Int[], train_loss = Float64[])
(_,batch),_ = iterate(lyap_loader)
batch
model.vlags
model.re_f(pf)(randn(4,10))

loss_sum = 0
Random.seed!(1)
for (_,batch) in lyap_loader
    # batch = randn(82,10)
    # loss_sum += sum(kras_loss(batch, pf, pv, model))
    for i in 1:length(batch[1,:])
        loss_sum += sum(kras_loss(reverse(batch[:,i]), pf, pv, model))
    end
end
kras_loss(reverse(batch[:,1]), pf, pv, model)
function predict_true_loss(traj_idx,pf,pv,m, df_model)
    function lyapunov_ndde_func!(du, u, h, p, t)
        x = u[1:data_dim]
        pf = p[1:m.nfparams]
        pv = p[m.nfparams+1:end]
        xt = vcat(x, map(τ -> h(pf,t-τ, idxs=1:data_dim), flags)...)
        ut = vcat(x, map(τ -> h(pv,t-τ, idxs=1:data_dim), vcat(lags, lags.+lags[end]))...)
        du .= vcat(m.re_f(pf)(xt), kras_loss(ut, pf, pv, m)*ones(1))
    end
    h0 = (p,t;idxs=1:data_dim)->df_model.trajs[traj_idx][3](t)
    t_span = (0.17, df_model.trajs[traj_idx][1][end])
    z0 = Array(vcat(h0(nothing, 0.17), [0]))
    p = vcat(pf,pv)
    println(z0)
    prob = DDEProblem(lyapunov_ndde_func!, z0, h0, t_span, p=p; constant_lags=sort(union(flags,vlags)))
    alg = MethodOfSteps(RK4())
    # return Array(solve(prob, alg, u0=z0, p=p, saveat=[last(t_span)], abstol=1e-9,reltol=1e-6))[3,1]
    return Array(solve(prob, alg, u0=z0, p=p, saveat=[last(t_span)]))[3,1]

end
predict_true_loss(1,pf,pv,model, df_model)
batch[:,1]
m = model
m.fmask
batch
xt = batch[m.fmask, :]
yt = ut[m.vmask, :]
df_model.trajs[1]
scatter(Array(0:0.01:0.01*81), batch[:,3])
kras_loss(a, pf, pv, model
model
a
a = vcat(reverse(map(t-> exp(2*t)*ones(2), LinRange(1, 2, 41)))...)
ut = batch[:,3]
xt = ut[m.fmask]
yt = ut[m.vmask]
v, vx = forwardgrad(m.re_v(pv), yt)
m.re_v(pv)(yt) ≈ v
df_model.trajs
reshape(Zygote.jacobian(x->m.re_v(pv)(x), yt)[1],:)≈vx

data = hcat(map(i-> rand(rng, distr_lyap, 2*data_dim*(length(vlags)+1)), 1:config["uncorrelated_data_size"])...)
lyap_loader = Flux.Data.DataLoader((Array(1:config["uncorrelated_data_size"]),data), batchsize=config["batchsize_lyap"],shuffle=true)

loss_sum
4
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
        lr = 1e-3
        optf = ADAM(lr)
        optv=optf

        # sample new lyapunov data
        global lyap_loader
        if !config["uncorrelated"]
            gen_dataset!(df_model, p=pf)
            lyap_loader = Flux.Data.DataLoader(df_model, batchsize=config["batchsize_lyap"], shuffle=true)
        end

        # combined train step
        kras_stabilize_train_step!(pf, pv, model, optf, optv, iter, lyap_loader)

        if !config["server"]
            gen_dataset!(df_model, p=pf)
            pl_train = plot(title="train")
            for (j, traj) in enumerate(df_model.trajs[1:4])
                for i in 1:data_dim
                    # scatter!(pl, traj[1], hcat(traj[2]...)[i,:])
                    plot!(pl_train, traj[1][1]:0.01:3.0,t-> traj[3](t)[i])
                end
            end
            display(pl_train)
        end

        # test evaluation
        if (iter % config["test_eval"] == 0)
            # log train fit
            batch_loss = 0.0
            max_loss = 0.0
            for (_, u_batch) in lyap_loader
                l = sum(kras_loss(u_batch, pf,pv,model))
                batch_loss += l
                max_loss = maximum([max_loss, l])
            end
            println("max in-batch loss: ", max_loss)
        end
        # if iter % config["model checkpoint"] == 0
        #     using BSON: @save
        #     filename = logpath * "weights-" * string(iter) * ".bson"
        #     @save filename pf
        # end
    end
end
model.vmask
model.fmask
# save params
CSV.write(logpath*"test_loss.csv", test_loss_data, header = true)
CSV.write(logpath*"train_loss.csv", train_loss_data, header = true)

using BSON: @save
filename_f = logpath * "weights_f.bson"
filename_v = logpath * "weights_v.bson"
@save filename_f pf
@save filename_v pf
