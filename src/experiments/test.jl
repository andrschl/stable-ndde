## Hyperparameters
config = Dict(
    # on server?
    "server" => true,
    "logging" => true,

    # ground truth dynamics
    "npendulum" => 2,
    "friction" => 0.5,
    "length" => 1.0,
    "mass" => 1.0,
    "σ" => 0.05,

    # ndde training
    "θmax_train" => pi/10,
    "θmax_test" => pi/3,
    "ntrain_trajs" => 4,
    "ntest_trajs" => 4,
    "T_train" => 2*pi,
    "T_test" => 20*pi,
    "datasize" => 100,
    #"Δt_data" => 0.4,
    "batchtime" => 50,
    "batchsize" => 4,
    "const_init" => false,

    # ndde model
    "Δtf" => 0.3,
    "rf" => 3.0,

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
    "Δtv" => 0.3,
    "rv" => 3.0,
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

using AbstractGPs, Plots, Optim
σ = 0.1
x = Array(0:0.1:2)
y = sin.(x) + σ*randn(size(x))
f= GP(Matern12Kernel())
fx = f(x, σ)
p_fx =  posterior(fx, y)

pl1 =plot(0:0.01:5,p_fx)
scatter!(pl1, x,y, title="default")


using AbstractGPs, Plots, Optim
σ = 0.1
x = Array(0:0.1:5)
y = sin.(x) + σ*randn(size(x))
function objective_function(x, y)
    function negative_log_likelihood(params)
        kernel =
            softplus(params[1]) * (Matern12Kernel() ∘ ScaleTransform(softplus(params[2])))
        f = GP(kernel)
        fx = f(x, σ)
        return -logpdf(fx, y)
    end
    return negative_log_likelihood
end
p0 = [1.0, 1.0]
opt = optimize(objective_function(x, y), p0, LBFGS())
opt_kernel =
    softplus(opt.minimizer[1]) *
    (Matern32Kernel() ∘ ScaleTransform(softplus(opt.minimizer[2])))
opt_f = GP(opt_kernel)
opt_fx = opt_f(x, σ)
opt_p_fx =  posterior(opt_fx, y)

function get_gp(x, y, σ; k0=Matern12Kernel())
    function objective_function(x, y)
        function negative_log_likelihood(params)
            kernel =
                softplus(params[1]) * (k0 ∘ ScaleTransform(softplus(params[2])))
            f = GP(kernel)
            fx = f(x, σ)
            return -logpdf(fx, y)
        end
        return negative_log_likelihood
    end
    p0 = [1.0,1.0]
    opt = optimize(objective_function(x, y), p0, LBFGS())
    opt_kernel =
        softplus(opt.minimizer[1]) *
        (k0 ∘ ScaleTransform(softplus(opt.minimizer[2])))
    opt_f = GP(opt_kernel)
    opt_fx = opt_f(x, σ)
    opt_p_fx =  posterior(opt_fx, y)
    return opt_p_fx
end

p_fx = get_gp(x,y,0.1, k0=Matern52Kernel())

pl2 =plot(0:0.01:5,p_fx)
scatter!(pl2, x,y, title="optimized")

plot(pl1,pl2)
