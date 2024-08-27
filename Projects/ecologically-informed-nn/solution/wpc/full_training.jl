using CSV
using Lux, Optimisers, Zygote
import Flux: binarycrossentropy, mse
using DataFrames
using MLJ, MLUtils
using ProgressMeter
using Random; rng = Random.default_rng()
using Printf
using Rasters, RasterDataSources, ArchGDAL
using ComponentArrays
using Plots

Random.seed!(rng, 1234)

include(@__DIR__() * "/../wpa/dynamical_model.jl")

function build_model(; npred, hidden_nodes, dev=cpu)
    model = Chain(Dense(npred, hidden_nodes),
        BatchNorm(hidden_nodes),
        Dense(hidden_nodes, hidden_nodes, tanh), 
        BatchNorm(hidden_nodes),
        Dense(hidden_nodes, hidden_nodes, tanh), 
        BatchNorm(hidden_nodes),
        Dense(hidden_nodes, 1),
        Lux.sigmoid
    )

    return model
end

function process_data(df; train_split=0.9, batchsize=128)

    pred_data = df[:,[:x, :y, :temp, :t]] # .|> Float32
    # normalising predictors:
    standardizer = machine(Standardizer(), pred_data)
    fit!(standardizer)
    x_data = (MLJ.transform(standardizer, pred_data) |> Array)' .|> Float32

    y_data = df[:,:PA]'

    (x_train, y_train), (x_test, y_test) = splitobs((x_data, y_data); at=train_split)

    return (
        # Use DataLoader to automatically minibatch and shuffle the data
        DataLoader(collect.((x_train, y_train)); batchsize, shuffle=true),
        # Don't shuffle the test data
        DataLoader(collect.((x_test, y_test)); batchsize, shuffle=false),
        standardizer)
end

function get_pred_all(temp, t)
    loc_xy = DimPoints(temp)[:]
    temp_vec =  temp[:]
    pred_ti = vcat(hcat(collect.(loc_xy)...), temp_vec') .|> Float32
    pred_all = DataFrame(vcat(pred_ti, fill(t, size(pred_ti, 2))')', [:x, :y, :temp, :t])
    pred_all = (MLJ.transform(standardizer, pred_all) |> Array)' .|> Float32
    return pred_all
end

function project_spatially(nn, ps, st, landscape, t, standardizer)
    temp = landscape.raster
    pred_all = get_pred_all(temp, t)
    ŷ, _ = Lux.apply(nn, pred_all, ps, st)
    predicted = reshape(ŷ, :)
    urast = get_raster_values(predicted, landscape)
    return urast
end

function loss_mech(model, ps, st, data, landscape, pred_all_time)
    T = eltype(pred_all_time[1])
    l = zero(T)
    model_mech = build_dyn_model(landscape, ps)
    for i in 2:length(pred_all_time)
        ŷ₀, _ = Lux.apply(model, pred_all_time[i-1], ps, st)
        ŷ₁, _ = Lux.apply(model, pred_all_time[i], ps, st)
        ŷ₁_mech = reshape(step(model_mech, reshape(ŷ₀, :)), 1, :)
        l += mse(ŷ₁, ŷ₁_mech) #* exp(- (length(pred_all_time) - i) / 100.)
    end
    return l, st, (;)
end

function loss_data(model, ps, st, data)
    x, y = data
    ŷ, st = Lux.apply(model, x, ps, st)
    l = binarycrossentropy(ŷ, y)
    return l, st, (;)
end

"""
    plot_predictions(nn, pred)
Plots model predictions.
"""
function plot_predictions(nn, pred, landscape)
    predicted = reshape(nn(pred), :)
    urast = get_raster_values(predicted, landscape)
    return plot(urast, title="Predictions")
end

"""
    score_model(nns, data_rasters, all_pred)
Returns the MSE between model predictions and full abundance data.
"""
function score_model(nn, ps, st, data_rasters, pred_all_time)
    @assert length(pred_all_time) == length(data_rasters)
    l = 0 
    for (ti, rast) in enumerate(data_rasters)
        ŷ, _ = Lux.apply(nn, pred_all_time[ti], ps, st)
        l += mse(reshape(ŷ, :), rast[:])
    end
    return l
end

"""
    train(; nns, 
            dyn_model, 
            data_rasters, 
            env_data, 
            standardizer, 
            optim, 
            n_epochs, 
            n_samples, 
            p_dyn_model,
            τ,
            mechanistic_constraints = false)

Perform training. `dyn_model` is the dynamical model to enforce the dynamics.
`data_rasters` is the full dataset used to create mock-up PA data. `env_data`
corresponds to the environmental predictor dataset. `standatdizer` is a MLJ
machine to standardize the data. `optim` is used to train the NN. `n_samples` is
the number of mock-up samples to be create at the shallowest time step. This
number is reduced exponentially .`τ` characterises the decay in sample number.
across time steps (smaller `τ` means less data deep in time)
"""
function train(;df,
                model,
                loss_data,
                loss_mech=nothing,
                opt,
                n_epochs,
                print_freq,
                kwargs...)

    @unpack train_split, batchsize = kwargs
    train_dataloader, test_dataloader, _ = process_data(df; train_split, batchsize)
   
    train_state = Training.TrainState(model, ps, st, opt)
    vjp_rule = AutoZygote()

    ### Warmup the model
    @info "Warming up..."
    x_proto = randn(rng, Float32, 4, 2)
    y_proto = rand(rng, Bool, 1, 2)
    Training.compute_gradients(vjp_rule, loss_data, (x_proto, y_proto), train_state)
    if !isnothing(loss_mech)
        Training.compute_gradients(vjp_rule, loss_mech, nothing, train_state)
    end

    @info "Training..."
    for epoch in 1:n_epochs
        stime = time()
        for data in train_dataloader
            (_, l, _, train_state) = Training.single_train_step!(vjp_rule, loss_data, data, train_state)
        end

        if !isnothing(loss_mech)
            (_, l, _, train_state) = Training.single_train_step!(vjp_rule, loss_mech, nothing, train_state)
        end
        ttime = time() - stime

        # Validate the model
        st_ = Lux.testmode(train_state.states)
        tr_loss = mean([loss_data(model, train_state.parameters, st_, data)[1] for data in train_dataloader])
        te_loss = mean([loss_data(model, train_state.parameters, st_, data)[1] for data in test_dataloader])

        # tr_acc = accuracy(model, train_state.parameters, st_, train_dataloader) * 100
        # te_acc = accuracy(model, train_state.parameters, st_, test_dataloader) * 100
        # epoch % 200 == 1 && @show train_state.parameters
        epoch % print_freq == 0 && @printf "[%2d/%2d] \t Time %.2fs \t Training loss: %.2f Test loss: \
                    %.2f\n" epoch n_epochs ttime tr_loss te_loss
    end
    return (train_state.parameters,  Lux.testmode(train_state.states))
end

# data processing
# true data for model scoring
true_data_path =  @__DIR__() * "/../../data/true_abundance_data.jld2"
JLD2.@load true_data_path sol_rasters

df = CSV.read(@__DIR__() * "/../../data/PA_data.csv", DataFrame)
temp_raster = load_raster()
landscape = Landscape(temp_raster)
tsteps = sort!(unique(df.t))

_, _, standardizer = process_data(df)
pred_all_time = [get_pred_all(temp_raster, t) for t in tsteps] 

# model architecture
nn_params = (;npred = 4, hidden_nodes = 32)
model = build_model(;nn_params...)
ps, st = Lux.setup(rng, model)

# training without mechanistic constraints
opt = Adam(5e-3)
ps, st = train(;loss_data,
                df, 
                model,
                ps,
                st,
                opt, 
                print_freq=5, 
                n_epochs= 100, 
                batchsize=128, 
                train_split=0.9)

score = score_model(model, ps, st, sol_rasters, pred_all_time)
@printf "Score model without mech. constraintss: %.2f\n" score
anim = @animate for t in tsteps
    pred = project_spatially(model, ps, st, landscape, t, standardizer)
    plot(pred, title="T = $t")
end
gif(anim, fps=5)

# training with mechanistic constraints
ps, st = Lux.setup(rng, model)

pmech = (m = 0.05, α = 4., T0=4.)
ps = ps |> ComponentArray
ps = ComponentArray(ps; pmech...)

opt = Adam(2e-3)
ps, st = train(;loss_data, 
                loss_mech = (m, ps, st, data) -> loss_mech(m, 
                                                        ps, 
                                                        st,
                                                        data,
                                                        landscape, 
                                                        pred_all_time), 
                df, 
                model,
                ps,
                st,
                opt, 
                print_freq=1, 
                n_epochs= 100, 
                batchsize=128, 
                train_split=0.9)

score = score_model(model, ps, st, sol_rasters, pred_all_time)
@printf "Score model with mech. constraints: %.2f\n" score
anim = @animate for t in tsteps
    pred = project_spatially(model, ps, st, landscape, t, standardizer)
    plot(pred, title="T = $t")
end
gif(anim, fps=5)