using UnPack
using LinearAlgebra
using Rasters
include(@__DIR__() * "/../../project_src/utils.jl")

"""
    function(x, dd = 4, threshold = 0.1)

Exponential dispersal kernel with mean distance `dd`, returning corresponding
proximity.
"""
disp_kern = function(x::T, α, threshold = 0.1) where T
    prox = exp(-x / α)
    if prox < threshold
        return zero(T)
    end
    return prox
end

struct DynSDM{LD,KK,DD}
    landscape::LD
    K::KK
    D::DD
end

"""
    (model::DynSDM)(u)
    
Returns the change in abundance during a single time step, based on `model.K`
defining carrying capacity and `model.D` defining dispersal process.
"""
function (model::DynSDM)(u::Vector{T}) where T
    @unpack D, K = model
    du = u .* (one(T) .- u ./ K) + D * u
    return du
end

"""
    build_dyn_model(land, p)
Builds a `DynSDM` dynamical model defined from a landscape `land` and parameters `p`. This model
can be simulated with the `simulate` or `step` functions.
"""
function build_dyn_model(landscape::Landscape, p)
    @unpack T0, α, m = p
    Ts = landscape.raster[:]
    K = exp.(-((T0 .- Ts)) .^ 2)

    disp_proximities = sparse(disp_kern.(landscape.dists, α))
    D = - m * (I - (disp_proximities ./ sum(disp_proximities,dims=1)))
    return DynSDM(landscape, K, D)
end


"""
    step(model::DynSDM, u0)

Performs one step ahead prediction with a `DynSDM` model, based on `u0`.
"""
function step(model::DynSDM, u0)
    u = u0 + model(u0)
    return max.(u, zero(eltype(u)))
end

"""
    simulate(model::DynSDM, u0, ntsteps)

Performs `ntsteps` ahead prediction with a `DynSDM` model, based on `u0`.
Predictions returned as a vector with entry `i` corresponding to step `i`.
"""
function simulate(model::DynSDM, u0, ntsteps)
    us = [similar(u0) for i in 1:ntsteps+1]
    us[1] .= u0
    for i in 2:ntsteps+1
        us[i] .= step(model, us[i-1])
    end
    return us
end