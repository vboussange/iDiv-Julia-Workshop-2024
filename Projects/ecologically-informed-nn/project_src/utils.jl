
raster_data_path = @__DIR__() * "/../data/raster-data/"
mkpath(raster_data_path)
ENV["RASTERDATASOURCES_PATH"] = raster_data_path

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
    Landscape{R,M,D,ID}
Contains a raster, from which is derived a graph representing distance to
closest neighbour. This graph is used to calculate a distance matrix. The
mapping between the landscape graph and the raster is ensured by
`id_to_grid_coordinate_list`.
"""
struct Landscape{R,M,D,ID}
    raster::R
    nrows::M # raster nb rows
    ncols::M # raster nb cols
    dists::D # distance matrix
    id_to_grid_coordinate_list::ID
end

function Landscape(raster)
    nrows, ncols = size(raster)

    permeability = .!isnan.(raster)

    A = graph_matrix_from_raster(permeability)
    id_to_grid_coordinate_list = vec(CartesianIndices((nrows, ncols)))
    
    # pruning
    scci = largest_subgraph(SimpleWeightedDiGraph(A))
    g = SimpleWeightedDiGraph(A[scci, scci])
    id_to_grid_coordinate_list = id_to_grid_coordinate_list[scci]

    # calculating distance
    @info "Calculating shortest paths"

    dists = floyd_warshall_shortest_paths(g, weights(g)).dists
    Landscape(raster, nrows, ncols, dists, id_to_grid_coordinate_list)
end

"""
    scale(raster::Raster)

Scale raster with offset and scale.
"""
function scale(raster::Raster)
    T = typeof(raster.metadata["scale"])
    (raster .|> T) .* raster.metadata["scale"] .+ raster.metadata["offset"]
end

"""
    load_raster()
Loads Chelsa `bio1` data as a Rasters.Raster.
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