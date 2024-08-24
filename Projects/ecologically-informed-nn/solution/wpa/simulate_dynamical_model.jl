#=
Test of a dynamical SDM
1. we import a raster file, which defines the domain over which we solve the dynamical SDM
2. from this raster file, we define a graph, which defines how species can move
3. We simulate dynamics on this graph
4. We plot the dynamics 
=#
using Graphs, SimpleWeightedGraphs
using SparseArrays
using UnPack
using LinearAlgebra
using ComponentArrays
using Rasters, ArchGDAL
using RasterDataSources
using Plots
using JLD2
include("dynamical_model.jl")
include(@__DIR__() * "/../../project_src/utils.jl")


"""
    build_dyn_model(land)
Builds a `DynSDM` dynamical model. This model can be simulated with the
`simulate` or `step` functions.
"""
function build_dyn_model(landscape::Landscape, p)
    @info "building dynamical model"
    @unpack T0, α, m = p
    Ts = landscape.raster[:]
    T0 = 4.0
    K = exp.(-((T0 .- Ts)) .^ 2)

    disp_proximities = sparse(disp_kern.(landscape.dists, α))
    D = - m * (I - (disp_proximities ./ sum(disp_proximities,dims=1)))
    return DynSDM(landscape, K, D)
end

"""
    function(x, dd = 4, threshold = 0.1)

Exponential dispersal kernel with mean distance `dd`, returning corresponding
proximity.
"""
disp_kern = function(x, α, threshold = 0.1)
    prox = exp(-x / α)
    if prox < threshold
        return 0.
    end
    return prox
end

temp_raster = load_raster()

plot(temp_raster)

landscape = Landscape(temp_raster)

p = ComponentVector(m = 0.05, α = 4., T0=4.)

model = build_dyn_model(landscape, p)

# plotting u0
tspan = (0.0, 100.0)
saveat = 0:1:100.


x = lookup(temp_raster, X)
y = lookup(temp_raster, Y)
x_0 = x[floor(Int, length(x)/2)]
y_0 = y[floor(Int, length(y)/2)]
a = 500.
b = 100.
u0_raster = @. exp(- a * ((x - x_0)^2 + (y-y_0)^2)) * exp(- b * (temp_raster - p.T0)^2)
plot(u0_raster)

u0 = u0_raster[:]
plot(get_raster_values(u0, landscape))

sol = simulate(model, u0, 20, p)

sol_rasters = [get_raster_values(sol[i], landscape) for i in 1:length(sol)]

anim = @animate for (i, sol_raster) in enumerate(sol_rasters)
    plot(sol_raster, title="T$i")
end

gif(anim, fps = 5)