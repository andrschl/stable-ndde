## Train step method
function ndde_train_step!(u0::AbstractArray, u_train::AbstractArray, h0::Function, pf::AbstractArray, t::AbstractArray, m::AbstractNDDEModel, opt,iter)
    println("____________________")
    println("start AD:")
    local train_loss, pred_u
    @time begin
        ps = Flux.params(pf)
        gs = Zygote.gradient(ps) do
            train_loss, pred_u = predict_ndde_loss(u0, h0, t, u_train, pf, m; N=config["batchtime"]*config["batchsize"])
            return train_loss
        end
    end
    println("stop AD")
    println("train_loss: ", train_loss)
    println("____________________")
    Flux.Optimise.update!(opt, ps, gs)
    # logging
    if config["logging"]
        wandb.log(Dict("train loss"=> train_loss), step=iter)
        push!(train_loss_data, [iter, train_loss])
    end
end
function clip_grads!(gs, gs_norm; threshold=config["grad_clipping"])
    if gs_norm > threshold
        gs = gs / gs_norm
    end
    return gs
end

function raz_stable_ndde_train_step!(u0::AbstractArray, u_train::AbstractArray, h0::Function, pf::AbstractArray, pv::AbstractArray,
            t::AbstractArray, m::AbstractNDDEModel, optf, optv, iter, lyap_loader)

    println("____________________")
    println("start AD:")
    local train_loss, pred_u
    @time begin
        ps = Flux.params(pf)
        gs = Zygote.gradient(ps) do
            train_loss, pred_u = predict_ndde_loss(u0, h0, t, u_train, pf, m; N=config["batchtime"]*config["batchsize"])
            return train_loss
        end
    end
    @time begin
        dldpf_lyap = zero(pf)
        dldpv_lyap = zero(pv)
        lyap_loss = 0.0
        i = 1
        for (_, u_lyap) in lyap_loader
            curr_loss = sum(raz_loss(u_lyap, pf, pv, model))
            lyap_loss += curr_loss
            if curr_loss != 0.0
                dldpf_lyap += Zygote.gradient(pf->sum(raz_loss(u_lyap, pf, pv, model)), pf)[1]
                dldpv_lyap += Zygote.gradient(pv->sum(raz_loss(u_lyap, pf, pv, model)), pv)[1]
            end
            if config["nacc_steps_lyap"] <= i
                break
            else
                i+=1
            end
        end
        lyap_loss=lyap_loss/i
    end

    println("stop AD")
    dldpf_dyn = gs[pf]
    dldpf_lyap = dldpf_lyap * config["weight_f"]
    dldpv_lyap = dldpv_lyap * config["weight_v"]
    dldpf_dyn_abs1 = mean(abs,dldpf_dyn)
    dldpf_lyap_abs1 = mean(abs,dldpf_lyap)
    dldpv_lyap_abs1 = mean(abs,dldpv_lyap)
    println("dldpf dyn 1-norm is: ", dldpf_dyn_abs1)
    println("dlfpf raz 1-norm is: ", dldpf_lyap_abs1)
    println("dldpv raz 1-norm is: ", dldpv_lyap_abs1)
    # println("non-zero fraction: ", length(ls[ls.!=0.0])/length(ls))
    # println("max_loss: ", maximum(ls))
    println("ndde train_loss: ", train_loss)
    println("raz loss: ", lyap_loss)
    println("____________________")
    clip_grads!(dldpf_lyap, dldpf_lyap_abs1)
    clip_grads!(dldpv_lyap, dldpv_lyap_abs1)
    warmup_fac = min(1.0, max(0,iter-config["pause_steps"])/config["warmup_steps"]) * heaviside(iter-config["pause_steps"])
    Flux.Optimise.update!(optf, pf, dldpf_dyn+warmup_fac*dldpf_lyap)
    Flux.Optimise.update!(optv, pv, dldpv_lyap)
    # logging
    if config["logging"]
        wandb.log(Dict("lr f" => optf.eta, "lr v" => optv.eta, "ndde train loss" => train_loss,
            "raz train loss" => lyap_loss, "dldpf dyn 1-norm" => dldpf_dyn_abs1, "dlfpf raz 1-norm" => dldpf_lyap_abs1,
            "dldpv raz 1-norm" => dldpv_lyap_abs1), step=iter)
        global train_loss_data
        push!(train_loss_data, [iter, train_loss])
    end
end

function kras_stable_ndde_train_step!(u0::AbstractArray, u_train::AbstractArray, h0::Function, pf::AbstractArray, pv::AbstractArray,
            t::AbstractArray, m::AbstractNDDEModel, optf, optv, iter, lyap_loader)
    println("____________________")
    println("start AD:")
    local train_loss, pred_u
    @time begin
        ps = Flux.params(pf)
        gs = Zygote.gradient(ps) do
            train_loss, pred_u = predict_ndde_loss(u0, h0, t, u_train, pf, m; N=config["batchtime"]*config["batchsize"])
            return train_loss
        end
    end
    @time begin
        dldpf_lyap = zero(pf)
        dldpv_lyap = zero(pv)
        lyap_loss = 0.0
        i = 1
        for (_, u_lyap) in lyap_loader
            curr_loss = sum(kras_loss(u_lyap, pf, pv, m))
            lyap_loss += curr_loss
            if curr_loss != 0.0
                dldpf_lyap += Zygote.gradient(pf->sum(kras_loss(u_lyap, pf, pv, m)), pf)[1]
                dldpv_lyap += Zygote.gradient(pv->sum(kras_loss(u_lyap, pf, pv, m)), pv)[1]
            end
            if config["nacc_steps_lyap"] <= i
                break
            else
                i+=1
            end
        end
        lyap_loss=lyap_loss/i
    end
    println("stop AD")
    dldpf_dyn = gs[pf]
    dldpf_lyap = dldpf_lyap * config["weight_f"]
    dldpv_lyap = dldpv_lyap * config["weight_v"]
    dldpf_dyn_abs1 = mean(abs,dldpf_dyn)
    dldpf_lyap_abs1 = mean(abs,dldpf_lyap)
    dldpv_lyap_abs1 = mean(abs,dldpv_lyap)
    println("dldpf dyn 1-norm is: ", dldpf_dyn_abs1)
    println("dlfpf kras 1-norm is: ", dldpf_lyap_abs1)
    println("dldpv kras 1-norm is: ", dldpv_lyap_abs1)
    # println("non-zero fraction: ", length(ls[ls.!=0.0])/length(ls))
    # println("max_loss: ", maximum(ls))
    println("ndde train_loss: ", train_loss)
    println("kras loss: ", lyap_loss)
    println("____________________")
    clip_grads!(dldpf_lyap, dldpf_lyap_abs1)
    clip_grads!(dldpv_lyap, dldpv_lyap_abs1)
    warmup_fac = min(1.0, max(0,iter-config["pause_steps"])/config["warmup_steps"]) * heaviside(iter-config["pause_steps"])
    Flux.Optimise.update!(optf, pf, dldpf_dyn+warmup_fac*dldpf_lyap)
    Flux.Optimise.update!(optv, pv, dldpv_lyap)
    # logging
    if config["logging"]
        wandb.log(Dict("lr f" => optf.eta, "lr v" => optv.eta, "ndde train loss" => train_loss,
            "kras train loss" => lyap_loss, "dldpf dyn 1-norm" => dldpf_dyn_abs1, "dlfpf kras 1-norm" => dldpf_lyap_abs1,
            "dldpv kras 1-norm" => dldpv_lyap_abs1), step=iter)
        global train_loss_data
        push!(train_loss_data, [iter, train_loss])
    end
end
function kras_stabilize_train_step!(pf::AbstractArray, pv::AbstractArray, m::AbstractNDDEModel, optf, optv, iter, lyap_loader)
    println("____________________")
    println("start AD:")
    @time begin
        dldpf_lyap = zero(pf)
        dldpv_lyap = zero(pv)
        lyap_loss = 0.0
        i = 1
        for (_, u_lyap) in lyap_loader
            curr_loss = sum(kras_loss(u_lyap, pf, pv, m))
            lyap_loss += curr_loss
            if curr_loss != 0.0
                dldpf_lyap += Zygote.gradient(pf->sum(kras_loss(u_lyap, pf, pv, m)), pf)[1]
                dldpv_lyap += Zygote.gradient(pv->sum(kras_loss(u_lyap, pf, pv, m)), pv)[1]
            end
            if config["nacc_steps_lyap"] <= i
                break
            else
                i+=1
            end
        end
        lyap_loss=lyap_loss/i
    end
    println("stop AD")
    dldpf_lyap = dldpf_lyap * config["weight_f"]
    dldpv_lyap = dldpv_lyap * config["weight_v"]
    dldpf_lyap_abs1 = mean(abs,dldpf_lyap)
    dldpv_lyap_abs1 = mean(abs,dldpv_lyap)
    println("dlfpf kras 1-norm is: ", dldpf_lyap_abs1)
    println("dldpv kras 1-norm is: ", dldpv_lyap_abs1)
    # println("non-zero fraction: ", length(ls[ls.!=0.0])/length(ls))
    # println("max_loss: ", maximum(ls))
    println("kras loss: ", lyap_loss)
    println("____________________")
    clip_grads!(dldpf_lyap, dldpf_lyap_abs1)
    clip_grads!(dldpv_lyap, dldpv_lyap_abs1)
    warmup_fac = min(1.0, max(0,iter-config["pause_steps"])/config["warmup_steps"]) * heaviside(iter-config["pause_steps"])
    Flux.Optimise.update!(optf, pf, warmup_fac*dldpf_lyap)
    Flux.Optimise.update!(optv, pv, dldpv_lyap)
    # logging
    if config["logging"]
        wandb.log(Dict("lr f" => optf.eta, "lr v" => optv.eta,
            "kras train loss" => lyap_loss, "dlfpf kras 1-norm" => dldpf_lyap_abs1,
            "dldpv kras 1-norm" => dldpv_lyap_abs1), step=iter)
        global train_loss_data
        push!(train_loss_data, [iter, lyap_loss])
    end
end
function raz_stabilize_train_step!(pf::AbstractArray, pv::AbstractArray, m::AbstractNDDEModel, optf, optv, iter, lyap_loader)

    println("____________________")
    println("start AD:")
    @time begin
        dldpf_lyap = zero(pf)
        dldpv_lyap = zero(pv)
        lyap_loss = 0.0
        i = 1
        for (_, u_lyap) in lyap_loader
            curr_loss = sum(raz_loss(u_lyap, pf, pv, model))
            lyap_loss += curr_loss
            if curr_loss != 0.0
                dldpf_lyap += Zygote.gradient(pf->sum(raz_loss(u_lyap, pf, pv, model)), pf)[1]
                dldpv_lyap += Zygote.gradient(pv->sum(raz_loss(u_lyap, pf, pv, model)), pv)[1]
            end
            if config["nacc_steps_lyap"] <= i
                break
            else
                i+=1
            end
        end
        lyap_loss=lyap_loss/i
    end

    println("stop AD")
    dldpf_lyap_abs1 = mean(abs,dldpf_lyap)
    dldpv_lyap_abs1 = mean(abs,dldpv_lyap)
    println("dlfpf raz 1-norm is: ", dldpf_lyap_abs1)
    println("dldpv raz 1-norm is: ", dldpv_lyap_abs1)
    # println("non-zero fraction: ", length(ls[ls.!=0.0])/length(ls))
    # println("max_loss: ", maximum(ls))
    println("raz loss: ", lyap_loss)
    println("____________________")
    # clip_grads!(dldpf_lyap, dldpf_lyap_abs1)
    # clip_grads!(dldpv_lyap, dldpv_lyap_abs1)
    warmup_fac = min(1.0, max(0,iter-config["pause_steps"])/config["warmup_steps"]) * heaviside(iter-config["pause_steps"])
    Flux.Optimise.update!(optf, pf, warmup_fac*dldpf_lyap)
    old_pv = copy(pv)
    Flux.Optimise.update!(optv, pv, dldpv_lyap)
    println("pv change: ",mean(abs, pv - old_pv))
    # logging
    if config["logging"]
        wandb.log(Dict("lr f" => optf.eta, "lr v" => optv.eta,
            "raz train loss" => lyap_loss, "dlfpf raz 1-norm" => dldpf_lyap_abs1,
            "dldpv raz 1-norm" => dldpv_lyap_abs1), step=iter)
        global train_loss_data
        push!(train_loss_data, [iter, lyap_loss])
    end
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
## plotting for callback

# python code
py"""
import matplotlib.pyplot as plt
import wandb
import plotly
# pred: list of tuples (x,y,label) for lineplot
# data: list of tuples (x,y,label) for scatterplot
def wandb_plot(pred, data, title=""):
    plt.plot(title=title)
    for (x,y,label) in pred:
        plt.plot(x,y,label=label)
    for (x,y,label) in data:
        plt.scatter(x,y,label=label)
    wandb.log({title: plt})
"""
function wandb_plot_ndde_data_vs_prediction(df::AbstractDataset, traj_idx::Integer,
        m::AbstractNDDEModel, pf::AbstractArray, title::String; Δtplot=0.02)
    t_data = df.trajs[traj_idx][1][df.N_hist:end]
    u_data = hcat(df.trajs[traj_idx][2][df.N_hist:end]...)
    data = []
    for i in 1:length(u_data[:,1])
        label = "x"*string(i)*"_data"
        push!(data, (t_data, u_data[i,:], label))
    end
    t_pred = Array(t_data[1]:Δtplot:t_data[end])
    u_pred = predict_ndde(u_data[:,1], (p,t)->df.trajs[traj_idx][3](t), t_pred, pf, m)
    pred = []
    for i in 1:length(u_pred[:,1])
        label = "x"*string(i)*"_pred"
        push!(pred, (t_pred, u_pred[i,:], label))
    end
    py"wandb_plot"(pred, data, title=title)
end
function wandb_plot_noisy_ndde_data_vs_prediction(df::AbstractDataset, traj_idx::Integer,
        m::AbstractNDDEModel, pf::AbstractArray, title::String; Δtplot=0.02, k0="RBF")
    t_data = df.trajs[traj_idx][1][df.N_hist:end]
    u_data = hcat(df.trajs[traj_idx][2][df.N_hist:end]...)
    data = []
    for i in 1:length(u_data[:,1])
        label = "x"*string(i)*"_data"
        push!(data, (t_data, u_data[i,:], label))
    end
    t_pred = Array(t_data[1]:Δtplot:t_data[end])
    h0 = get_noisy_h0(df, traj_idx, k0=k0)
    u_pred = predict_ndde(h0(pf, t_data[1]), h0, t_pred, pf, m)
    pred = []
    for i in 1:length(u_pred[:,1])
        label = "x"*string(i)*"_pred"
        push!(pred, (t_pred, u_pred[i,:], label))
    end
    py"wandb_plot"(pred, data, title=title)
end

function save_plot_noisy_ndde_data_vs_prediction(df::AbstractDataset, traj_idx::Integer,
        m::AbstractNDDEModel, pf::AbstractArray, save_dir::String, title::String; Δtplot=0.02, k0="RBF")

    t0 = df.trajs[traj_idx][1][df.N_hist]
    t_data = df.trajs[traj_idx][1][1:end]
    u_data = hcat(df.trajs[traj_idx][2][1:end]...)
    u_data_noisy = hcat(df.noisy_trajs[traj_idx][2][1:end]...)

    t_gt = Array(t0-df.r:Δtplot:t_data[end])
    u_gt = hcat(df.trajs[traj_idx][3].(t_gt)...)

    t_pred = Array(t0:Δtplot:t_data[end])
    h0 = get_noisy_h0(df, traj_idx, k0=k0)
    u_pred = predict_ndde(h0(pf, t0), h0, t_pred, pf, m)

    t_init = Array(t0-df.r:Δtplot:t0)
    u_init = hcat(h0.(nothing, t_init)...)

    # write plot data
    data = DataFrame(t_data = t_data)
    pred = DataFrame(t_pred = t_pred)
    gt = DataFrame(t_gt = t_gt)
    init = DataFrame(t_init= t_init)
    for i in 1:length(u_data[:,1])
        u = Symbol("u_data", i)
        u_noisy = Symbol("u_data_noisy", i)
        data[:, u] = u_data[i,:]
        data[:, u_noisy] = u_data_noisy[i,:]
        u = Symbol("u_pred",i)
        pred[:, u] = u_pred[i,:]
        u = Symbol("u_gt",i)
        gt[:, u] = u_gt[i,:]
        u = Symbol("u_init",i)
        init[:, u] = u_init[i,:]
    end
    CSV.write(save_dir*title*"data.csv", data, header = true)
    CSV.write(save_dir*title*"gt.csv", gt, header = true)
    CSV.write(save_dir*title*"pred.csv", pred, header = true)
    CSV.write(save_dir*title*"init.csv", init, header = true)
end
