## Hyperparameters
config = Dict(
    # on server?
    "server" => false,
    "logging" => false,

    # lr schedule
    "lr_rel_decay" => 0.01,
    "lr_start_max" => 5e-2,
    "lr_start_min" => 1e-4,
    "lr_period" => 50,
    "nepisodes" => 300,

    # stabilizing training
    "θmax_lyap" => pi/2,
    "nlyap_trajs" => 4,
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
    "rv" => 0.2,
    # "α" => 0.01,
    # "q" => 1.01,
    "α" => 0.2,
    "q" => 1.2,
    "weight_f" => 1,
    "weight_v" => 5,
    "grad_clipping" => 1e5,  # threshold in avg l1 norm after weighting
    "warmup_steps" => 50,
    "pause_steps" => 50,

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
    config["α"] = parse(Float64, ARGS[2])
end
if length(ARGS) >= 3
    config["q"] = parse(Float64, ARGS[3])
end
if length(ARGS) >= 4
    config["warmup_steps"] = parse(Int64, ARGS[4])
end
if length(ARGS) >= 5
    config["pause_steps"] = parse(Int64, ARGS[5])
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

project_name = "inverted_pendulum_LRF_alongtraj"
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
Random.seed!(122+seed)
rng = MersenneTwister(123)

## Load pendulum dataset
data_dim = 2
include("../datasets/inverted_pendulum.jl")

distr_lyap = Uniform(-config["θmax_lyap"], config["θmax_lyap"])
tspan_lyap = (0.0,config["T_lyap"])
data_dim = 2
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

model_v = Lyapunov(data_dim)
pv,rev = Flux.destructure(model_v)
pf,ref = Flux.destructure(f_cl)

model = RazNDDE(data_dim, ref, rev, pf, pv, flags=flags,vlags=vlags,α=config["α"],q=config["q"])
model.vlags
if !config["uncorrelated"]
    u0s = map(i -> sample_u0(config["θmax_lyap"]), 1:config["nlyap_trajs"])
    h0s = map(u0 -> (p,t)->dense_predict_reverse_ode(u0, flags[end], 0.0)(t), u0s)
    lyap_prob = DDEProblem(model.ndde_func!, zeros(data_dim), h0s[1], tspan_lyap, constant_lags=flags)
    df_model = DDEDDEDataset(h0s, tspan_lyap, min(config["Δtv"],config["Δtf"]), lyap_prob, config["rv"], flags)
    gen_dataset!(df_model, p=model.pf)
    lyap_loader = Flux.Data.DataLoader(df_model, batchsize=config["batchsize_lyap"], shuffle=false)
else
    data = hcat(map(i-> rand(rng, distr_lyap, 2*data_dim*(length(vlags)+1)), 1:config["uncorrelated_data_size"])...)
    lyap_loader = Flux.Data.DataLoader((Array(1:config["uncorrelated_data_size"]),data), batchsize=config["batchsize_lyap"],shuffle=true)
end

# test data
if !config["server"]
    pl = plot()
    for (j, traj) in enumerate(df_model.trajs[1:4])
            for i in 1:data_dim
                # scatter!(pl, traj[1], hcat(traj[2]...)[i,:])
                plot!(pl, -1:0.01:3.0,t-> traj[3](t)[i])
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

train_loss_data = DataFrame(iter = Int[], train_loss = Float64[])


#
function predict_true_loss(traj_idx,pf,pv,m, df_model)
    function lyapunov_ndde_func!(du, u, h, p, t)
        x = u[1:data_dim]
        # pf = p[1:m.nfparams]
        # pv = p[m.nfparams+1:end]
        xt = vcat(x, map(τ -> h(pf,t-τ, idxs=1:data_dim), flags)...)
        ut = vcat(x, map(τ -> h(pv,t-τ, idxs=1:data_dim), vlags)...)
        du .= vcat(m.re_f(pf)(xt), raz_loss(ut, pf, pv, m)*ones(1))
    end
    h0 = (p,t;idxs=1:data_dim)->df_model.trajs[traj_idx][3](t)
    # u0 = [1.0,0.0]
    # h0 = (p,t;idxs=1:data_dim)->u0
    t_span = (0.17, df_model.trajs[traj_idx][1][end])
    z0 = Array(vcat(h0(nothing, 0.17), [0.0]))
    p = vcat(pf,pv)
    prob = DDEProblem(lyapunov_ndde_func!, z0, h0, t_span, p=p; constant_lags=sort(union(flags,vlags)))
    alg = MethodOfSteps(Vern9())
    return Array(solve(prob, alg, u0=z0, p=p, saveat=[last(t_span)]))[3,1]

    # function loss(sol, t)
    #     x = sol(t)
    #     ut = vcat(x, map(τ -> sol(t-τ), vlags)...)
    #     raz_loss(ut, pf, pv, m)
    # end
    # sol = dense_predict_ndde(u0, h0, t_span, pf, m)
    # pl = plot(sol)
    # pl = scatter!(pl,0:0.01:3, t->loss(sol,t))
    # display(pl)
    # return Array(solve(prob, alg, u0=z0, p=p, saveat=[last(t_span)]))[3,1]

end



# (_,x),_=iterate(lyap_loader)
# raz_loss(x, pf, pv, model)
#
# # # debug
# function loss(sol, t)
#     x = sol(t)
#     ut = vcat(x, map(τ -> sol(t-τ), vlags)...)
#     raz_loss(ut, pf, pv, model)
# end
# function myloss(sol, t)
#     x = sol(t)
#     ut = vcat(x, map(τ -> sol(t-τ), vlags)...)
#     xt = vcat(x, map(τ -> sol(t-τ), flags)...)
#     past_vs = map(i -> model.re_v(pv)(ut[i*data_dim + 1:(i+1)*data_dim])[1], Array(1:length(model.vlags)))
#     vmax = maximum(past_vs)
#     vx = gradient(x->model.re_v(pv)(x)[1],sol(t))[1]
#     v = model.re_v(pv)(x)[1]
#     fx = model.re_f(pf)(xt)
#     relu(dot(vx, fx)+model.α *v)*heaviside(v*model.q - vmax)
# end
# function delta_v(sol, t)
#     x = sol(t)
#     ut = vcat(x, map(τ -> sol(t-τ), vlags)...)
#     xt = vcat(x, map(τ -> sol(t-τ), flags)...)
#     past_vs = map(i -> model.re_v(pv)(ut[i*data_dim + 1:(i+1)*data_dim])[1], Array(1:length(model.vlags)))
#     vmax = maximum(past_vs)
#     v = model.re_v(pv)(x)[1]
#     relu(v*model.q - vmax)
# end
# function Lie_v(sol, t)
#     x = sol(t)
#     ut = vcat(x, map(τ -> sol(t-τ), flags)...)
#     vx = gradient(x->model.re_v(pv)(x)[1],sol(t))[1]
#     fx = model.re_f(pf)(ut)
#     relu(dot(vx, fx))
# end
# sol = dense_predict_ndde(u0, h0, (0.0,3.0), pf, model)
# v = ξ->1e-2*rev(pv)(sol(ξ))[1]
# pl = plot()
# plot!(pl, sol)
# plot!(pl, 2.7:0.001:3, t->1e2*loss(sol,t), label="loss")
# plot!(pl, 2.7:0.001:3, v, label="v(x(t))")
# plot!(pl, 2.7:0.001:3, t->1e-4*Lie_v(sol, t), label="Lie_V")
# plot!(pl, 2.7:0.001:3, t->5e-2*delta_v(sol,t),xlims=(2.7,3), label="relu(qV(x(t))-V_max)")
# plot!(pl, 2.7:0.001:3, t->5e-5*myloss(sol,t),xlims=(2.7,3), label="loss2.0")
# display(pl)

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
        optf = ADAM(lr*config["weight_f"])
        optv = ADAM(lr*config["weight_v"])

        # sample new lyapunov data
        global lyap_loader
        if !config["uncorrelated"]
            # gen_dataset!(df_model, p=pf)
            # lyap_loader = Flux.Data.DataLoader(df_model, batchsize=config["batchsize_lyap"], shuffle=true)

            u0s = map(i -> sample_u0(config["θmax_lyap"]), 1:config["nlyap_trajs"])
            h0s = map(u0 -> (p,t)->dense_predict_reverse_ode(u0, flags[end], 0.0)(t), u0s)
            lyap_prob = DDEProblem(model.ndde_func!, zeros(data_dim), h0s[1], tspan_lyap, constant_lags=flags)
            df_model = DDEDDEDataset(h0s, tspan_lyap, min(config["Δtv"],config["Δtf"]), lyap_prob, config["rv"], flags)
            gen_dataset!(df_model, p=model.pf)
            lyap_loader = Flux.Data.DataLoader(df_model, batchsize=config["batchsize_lyap"], shuffle=false)
        end

        # combined train step
        raz_stabilize_train_step!(pf, pv, model, optf, optv, iter, lyap_loader)

        if !config["server"] && (iter % 1==0)
            # gen_dataset!(df_model, p=pf)
            pl_train = plot(title="train")
            for (j, traj) in enumerate(df_model.trajs[1:4])
                for i in 1:data_dim
                    # scatter!(pl, traj[1], hcat(traj[2]...)[i,:])
                    plot!(pl_train, traj[1][1]:0.01:3.0,t-> traj[3](t)[i])
                end
            end
            display(pl_train)
            if iter % 1 == 0
                xs = LinRange(-1, 1, 100)
                ys = LinRange(-1, 1, 100)
                zs = [rev(pv)([x,y])[1] for x in xs, y in ys]
                display(contour(xs, ys, zs; levels=20))
            end
        end

        # test evaluation
        if (iter % config["test_eval"] == 0)
            # log train fit
            batch_loss = 0.0
            max_loss = 0.0
            for (_, u_batch) in lyap_loader
                l = sum(raz_loss(u_batch, pf,pv,model))
                batch_loss += l
                max_loss = maximum([max_loss, l])
            end
            println("max in-batch loss: ", max_loss)
            if config["logging"]
                wandb.log(Dict("train loss" => batch_loss))
            end
        end
        # if iter % config["model checkpoint"] == 0
        #     using BSON: @save
        #     filename = logpath * "weights-" * string(iter) * ".bson"
        #     @save filename pf
        # end
    end
end

xs = LinRange(-5, 5, 200)
ys = LinRange(-5, 5, 200)
zs = [rev(pv)([x,y])[1] for x in xs, y in ys]
CSV.write(logpath*"quiver.csv", DataFrame(zs, :auto), header = true)



# pl = contourf(xs, ys, zs; levels=20)
# display(pl)

u0s = map(i -> 	[0.8822508266912985,	0.47077964994519506], 1:config["ntest_trajs"])
h0s = map(u0 -> (p,t)->dense_predict_reverse_ode(u0, flags[end], 0.0)(t), u0s)
lyap_prob = DDEProblem(model.ndde_func!, zeros(data_dim), h0s[1], tspan_lyap, constant_lags=flags)
df_model = DDEDDEDataset(h0s, tspan_lyap, min(config["Δtv"],config["Δtf"]), lyap_prob, config["rv"], flags)
gen_dataset!(df_model, p=model.pf)

test_loss_data = DataFrame(iter = Int[], test_loss = Float64[])
true_test_loss_data = DataFrame(iter = Int[], test_loss = Float64[])
for (j, traj) in enumerate(df_model.trajs)
    t_pl = Array(0:0.01:traj[1][end])
    pred = DataFrame(t_pred = t_pl)
    u_pred = predict_ndde(traj[3](0), (p,ξ)-> traj[3](ξ), t_pl, pf, model)
    for i in 1:data_dim
        u = Symbol("u_pred",i)
        pred[:, u] = u_pred[i,:]
    end
    CSV.write(logpath*"test_pred"* string(j) *".csv", pred, header = true)
    # push!(test_loss_data, [j, predict_true_loss(j,pf,pv,model, df_model)])
    # push!(true_test_loss_data, [j, predict_true_loss(j,pf,pv,model2, df_model)])
end
display(pl)

# for (j, traj) in enumerate(df_model.trajs)
#         for i in 1:data_dim
#             # scatter!(pl, traj[1], hcat(traj[2]...)[i,:])
#             xx = [traj[3](t)[1] for t in -0.03:0.001:1]
#             yy = [traj[3](t)[2] for t in -0.03:0.001:1]
#             plot!(pl,xx,yy)
#         end
#
# end

# save params
CSV.write(logpath*"test_loss.csv", test_loss_data, header = true)
CSV.write(logpath*"true_loss.csv", test_loss_data, header = true)
CSV.write(logpath*"train_loss.csv", train_loss_data, header = true)

using BSON: @save
filename_f = logpath * "weights_f.bson"
filename_v = logpath * "weights_v.bson"
@save filename_f pf
@save filename_v pf
