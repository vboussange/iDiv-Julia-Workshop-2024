using UnPack
using LinearAlgebra
include(@__DIR__() * "/../../project_src/utils.jl")

struct DynSDM{LD,KK,DD}
    landscape::LD
    K::KK
    D::DD
end

"""
    (model::DynSDM)(u, p)
    
Returns the change in abundance during a single time step, based on
`model.ecological_dynamics` and dispersal process, calculated based on
`model.landscape` and `p.m` (migration rate).
"""
function (model::DynSDM)(u::Vector{T}, p) where T
    @unpack D, K = model
    @unpack m = p
    du = u .* (one(T) .- u ./ K) + D * u
    return du
end


"""
    step(model::DynSDM, u0, p)

Performs one step ahead prediction with a `DynSDM` model, based on `u0` and
model parametr `p`.
"""
function step(model::DynSDM, u0, p)
    u = u0 + model(u0, p)
    return max.(u, zero(eltype(u)))
end

"""
    simulate(model::DynSDM, u0, ntsteps, p)

Performs `ntsteps` ahead prediction with a `DynSDM` model, based on `u0` and
model parametr `p`. Predictions returned as a vector with entry `i`
corresponding to step `i`.
"""
function simulate(model::DynSDM, u0, ntsteps, p)
    us = [similar(u0) for i in 1:ntsteps+1]
    us[1] .= u0
    for i in 2:ntsteps+1
        us[i] .= step(model, us[i-1], p)
    end
    return us
end