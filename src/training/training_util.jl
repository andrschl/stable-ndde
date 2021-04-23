## Train step method
function ndde_train_step!(u0::AbstractArray, u_train::AbstractArray, h0::Function, pf::AbstractArray, t::AbstractArray, m::AbstractNDDEModel, opt)
    println("____________________")
    println("start AD:")
    local train_loss, pred_u
    @time begin
        ps = Flux.params(pf)
        gs = Zygote.gradient(ps) do
            train_loss, pred_u = predict_ndde_loss(u0, h0, t, u_train, pf, m; N=batchtime*batchsize)
            return train_loss
        end
    end
    println("stop AD")
    println("train_loss: ", train_loss)
    println("____________________")
    Flux.Optimise.update!(opt, ps, gs)
    # logging
    wandb.log(Dict("train loss"=> train_loss))
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
    plt.legend()
    wandb.log({title: plt})
"""
function wandb_plot_ndde_data_vs_prediction(df::AbstractDataset, traj_idx::Integer, m::AbstractNDDEModel, pf::AbstractArray, title::String; Δtplot=0.02)
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
