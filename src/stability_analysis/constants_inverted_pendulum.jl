################################################################################
# generate time vectors

# t = Array(0:0.2:1.5)
# t0 = 0.0
# T = 1.5
# Δt = 0.1
# t = Array(t0:Δt:T)
# t_span = (t0, T)
# # t= Array{Float32}([3,10])
# # init_t = Array{Float32}(0:0.2:2.8)
# # tot_t = vcat(init_t, t)
#
# # Define some constants (for linear dde data)
#
# # init_t_span = (init_t[1], init_t[end])
# t_span = (t0, T)
# # tot_t_span = (first(init_t_span), last(t_span))
# # data_size = length(t)
# # init_data_size = length(init_t)
# # batch_time = data_size
# batch_size = 64
data_dim = 2

# define lags

# lags=[1]
# lags = Array{Float32}(0.3:0.3:3)
flags = Array([0.03])
vlags = Array(0.01:0.01:0.2)
lags = sort(union(flags,vlags))
nfdelays = length(flags)
nvdelays = length(vlags)
max_lag = maximum(lags)
