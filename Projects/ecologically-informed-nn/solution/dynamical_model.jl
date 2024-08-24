using Graphs, SimpleWeightedGraphs
using UnPack
using SparseArrays
using LinearAlgebra

abstract type AbstractDynModel end

const N8 = ((-1, -1,  √2),
            ( 0, -1, 1.0),
            ( 1, -1,  √2),
            (-1,  0, 1.0),
            ( 1,  0, 1.0),
            (-1,  1,  √2),
            ( 0,  1, 1.0),
            ( 1,  1,  √2))

"""
    graph_matrix_from_raster(R; neighbors::Tuple=N8)

Return a sparse matrix corresponding to the adjacency matrix of a graph connecting each pixel to its
neighbour, following the connectivity rule given by `neighbors`.
"""
function graph_matrix_from_raster(
    R;
    neighbors::Tuple=N8,
)
    m, n = size(R)

    # Initialize the buffers of the SparseMatrixCSC
    is, js, vs = Int[], Int[], Float64[]

    for j in 1:n
        for i in 1:m
            # Base node
            for (ki, kj, l) in neighbors
                if !(1 <= i + ki <= m) || !(1 <= j + kj <= n)
                    # Continue when computing edge out of raster image
                    continue
                else
                    # Target node
                    rijk = R[i+ki, j+kj]
                    if iszero(rijk) || isnan(rijk)
                        # Don't include zero or NaN similaritiers
                        continue
                    end
                    push!(is, (j - 1) * m + i)
                    push!(js, (j - 1) * m + i + ki + kj * m)
                    push!(vs, rijk * l)
                end
            end
        end
    end
    return sparse(is, js, vs, m * n, m * n)
end

"""
    Landscape{R,M,AT,D,ID}
Contains a permeability raster, from which is derived a graph representing
distance to closest neighbour, correpsonding to adjacency matrix A. This matrix
is used to calculate a distance matrix. The permeability raster may contain
pixels which are not suited at all for the considered species, so that these
pixels are discarded. The mapping between the landscape graph and the raster is
ensured by `id_to_grid_coordinate_list`.
"""
struct Landscape{R,M,AT,D,ID}
    raster::R
    nrows::M # raster nb rows
    ncols::M # raster nb cols
    A::AT # adjacency matrix
    dists::D # distance matrix
    id_to_grid_coordinate_list::ID
end

function Landscape(raster)
    nrows, ncols = size(raster)
    A = graph_matrix_from_raster(raster)
    id_to_grid_coordinate_list = vec(CartesianIndices((nrows, ncols)))
    
    # pruning
    scci = largest_subgraph(SimpleWeightedDiGraph(A))
    g = SimpleWeightedDiGraph(A[scci, scci])
    id_to_grid_coordinate_list = id_to_grid_coordinate_list[scci]

    # calculating distance
    @info "Calculating shortest paths"

    dists = floyd_warshall_shortest_paths(g, weights(g)).dists
    Landscape(raster, nrows, ncols, A, dists, id_to_grid_coordinate_list)
end


"""
    load_raster()
Loads Chelsa bio1 data as a Rasters.Raster.
"""
function load_raster()
    env_data = Raster(CHELSA{BioClim}, :bio1; version=2, replace_missing=true)[X(6..6.3), Y(45..45.3)]
    env_data = scale(env_data)
    return env_data
end

"""
    largest_subgraph(graph)

    
Returns largest strongly connected subgraph from graph
"""
function largest_subgraph(graph)

    # Find the subgraphs
    scc = strongly_connected_components(graph)

    @info "cost graph contains $(length(scc)) strongly connected subgraphs"

    # Find the largest subgraph
    i = argmax(length.(scc))

    # extract node list and sort it
    scci = sort(scc[i])
    A = adjacency_matrix(graph)

    ndiffnodes = size(A, 1) - length(scci)
    if ndiffnodes > 0
        @info "removing $ndiffnodes nodes from affinity and cost graphs"
    end

    return scci
end

"""
    get_raster_values(val, l::Landscape)
Transforms a list of values into a raster with shape given by `l`.
"""
function get_raster_values(val, l::Landscape)
    canvas = deepcopy(l.raster)
    canvas = canvas .|> eltype(val)
    idxs = DimIndices(canvas)
    for (i, v) in enumerate(val)
        canvas[idxs[l.id_to_grid_coordinate_list[i]]...] = v
    end
    return canvas
end

Base.@kwdef struct DynSDM{L,DX,ED} <: AbstractDynModel
    landscape::L
    Δ_x::DX
    ecological_dynamics::ED 
end

"""
    (model::DynSDM)(u, p)
    
Returns the change in abundance during a single time step, based on
`model.ecological_dynamics` and dispersal process, calculated based on
`model.landscape` and `p.m` (migration rate).
"""
function (model::DynSDM)(u, p)
    @unpack Δ_x, landscape, ecological_dynamics = model
    @unpack m = p
    # u = maximum.(u, zeros(eltype(u)))

    du = ecological_dynamics(u)
    lux = m * Δ_x * u
    du = du - lux
    return du
end


"""
    build_dyn_model(temp_raster)
Builds a `DynSDM` dynamical model. This model can be simulated with the
`simulate` or `step` functions.
"""
function build_dyn_model(temp_raster::Raster)

    @info "building dynamical model"

    permeability = .!isnan.(temp_raster)
    landscape = Landscape(permeability)

    Ts = temp_raster[landscape.id_to_grid_coordinate_list]
    T0 = 4.0
    K0 = exp.(-((T0 .- Ts)) .^ 2)
    plot(get_raster_values(K0, landscape))

    ecological_dynamics = function (x::Array{T}) where {T}
        return x .* (one(T) .- x ./ K0)
    end

    disp_proximities = sparse(disp_kern.(landscape.dists))
    Δₓ = I - (disp_proximities ./ sum(disp_proximities,dims=1))
    
    return DynSDM(landscape, Δₓ, ecological_dynamics)
end

"""
    function(x, dd = 4, threshold = 0.1)

Exponential dispersal kernel with mean distance `dd`, returning corresponding
proximity.
"""
disp_kern = function(x, dd = 4, threshold = 0.1)
    prox = exp(-x / dd)
    if prox < threshold
        return 0.
    end
    return prox
end

"""
    step(model::DynSDM, u0, p)

Performs one step ahead prediction with a `DynSDM` model, based on `u0` and
model parametr `p`.
"""
function step(model::AbstractDynModel, u0, p)
    u = u0 + model(u0, p)
    return max.(u, zero(eltype(u)))
end

"""
    simulate(model::DynSDM, u0, ntsteps, p)

Performs `ntsteps` ahead prediction with a `DynSDM` model, based on `u0` and
model parametr `p`. Predictions returned as a vector with entry `i`
corresponding to step `i`.
"""
function simulate(model::AbstractDynModel, u0, ntsteps, p)
    us = [similar(u0) for i in 1:ntsteps+1]
    us[1] .= u0
    for i in 2:ntsteps+1
        us[i] .= step(model, us[i-1], p)
    end
    return us
end