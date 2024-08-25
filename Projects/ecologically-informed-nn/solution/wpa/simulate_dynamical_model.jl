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

ntsteps = 20
sol = simulate(model, u0, ntsteps, p)

sol_rasters = [get_raster_values(sol[i], landscape) for i in 1:length(sol)]

anim = @animate for (i, sol_raster) in enumerate(sol_rasters)
    plot(sol_raster, title="T$i")
end

gif(anim, fps = 5)