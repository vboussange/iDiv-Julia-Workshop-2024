using Distributions, DataFrames
using CSV
include("simulate_dynamical_model.jl")

"""
    generate_PA_data(; env_data, proba_observation_raster, n_samples, standardizer)
Generate `n_samples` presence absence mock up occurences based on
`proba_observation_raster`, together with associated predictors. Predictors are
standardized witha `standardizer`.
"""
function generate_PA_data(;env_data, proba_observation_raster, n_samples)
    idx_xy = sample(DimIndices(proba_observation_raster), n_samples, replace=false)
    loc_xy = [DimPoints(proba_observation_raster)[idx_xy[i]...] for i in 1:length(idx_xy)]
    s = map(p -> env_data[X(At(p[1])), Y(At(p[2]))], loc_xy)

    proba = [proba_observation_raster[idx_xy[i]...] for i in 1:length(idx_xy)]


    # standardizing pred
    pred = vcat(hcat(collect.(loc_xy)...), s')

    return pred, reshape(rand.(Bernoulli.(proba)), 1, :) .|> Float32
end

n_samples= 500
τ = 100

XY_arr = []
for (i, proba_observation_raster) in enumerate(sol_rasters)
    # simulating expoential increase in data
    n_samples_i = floor(Int, n_samples * exp((i-ntsteps) / τ))
    XY_arr_i = generate_PA_data(;proba_observation_raster, 
                                env_data = temp_raster, 
                                n_samples = n_samples_i)
    push!(XY_arr, XY_arr_i)
end

df = DataFrame("x"=>Float32[], "y" => Float32[], "temp" => Float32[], "PA" => Bool[], "t" => Int[])
for i ∈ 1:length(XY_arr)
    pred_i = XY_arr[i][1]
    y = XY_arr[i][2]
    append!(df, DataFrame(vcat(pred_i, y, fill(i, length(y))')', ["x", "y", "temp" , "PA", "t"]))
end

CSV.write(@__DIR__() * "/../../data/PA_data.csv", df)
