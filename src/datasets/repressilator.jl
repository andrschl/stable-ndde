function f_ol(x::Array{T,1},u::Array{T,1}, p) where {T<:Real}
    
    g = 9.81
    b = 0.0
    m = 0.2
    x1 = x[1]
    x2 = x[2]
    dx1 = x2
    dx2 = g/l*sin(x1)-1 /(m*l^2)*(b*x2 + u[1])
    return vcat(dx1,dx2)
end
function f_ol(x::Array{T,2},u::Array{T,2}) where {T<:Real}
    l = 0.3
    g = 9.81
    b = 0.0
    m = 0.2
    x1 = x[1,:]
    x2 = x[2,:]
    dx1 = x2
    dx2 = g/l*sin.(x1).-1 /(m*l^2)*(b*x2 + u[1,:])
    return permutedims(hcat(dx1,dx2),[2,1])
end
# linear feedback layer
struct Feedback{S}
    K::S
end
Flux.@functor Feedback
Feedback(data_dim::Integer) = begin
    Feedback(K)
end
(m::Feedback)(x::AbstractArray) = m.K*x
# closed loop dynamics layer
struct F_CL{S}
    u::S
end
Flux.@functor F_CL
(m::F_CL)(xt::Array{T,1}) where {T<:Real} = begin
    return f_ol(xt[1:data_dim], m.u(xt[1+data_dim:end]))
end
(m::F_CL)(xt::Array{T,2}) where {T<:Real} = begin
    return f_ol(xt[1:data_dim,:], m.u(xt[1+data_dim:end,:]))
end
# define feedback
g = Feedback(Array([1.749  1.031 ;]))
# g = Chain(Dense(2,16,tanh),Dense(16,1))
f_cl = F_CL(g)
