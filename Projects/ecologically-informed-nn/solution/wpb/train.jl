using CSV
using Lux, Optimisers, Zygote
import Flux: binarycrossentropy
using DataFrames
using MLJ, MLUtils
using ProgressMeter
using Random; rng = Random.default_rng()
using Printf
using Rasters, RasterDataSources, ArchGDAL
Random.seed!(rng, 1234)

include(@__DIR__() * "/../wpa/dynamical_model.jl")

function build_model(; npred, hidden_nodes, hidden_layers, dev=cpu)
    model = Chain(Dense(npred, hidden_nodes),
        BatchNorm(hidden_nodes),
        [Dense(hidden_nodes, hidden_nodes, tanh) for _ ∈ 1:hidden_layers-1]...,
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

function project_spatially(nn, ps, st, landscape, t, standardizer)
    temp = landscape.raster
    
    loc_xy = DimPoints(temp)[:]
    temp_vec =  temp[:]
    pred_ti = vcat(hcat(collect.(loc_xy)...), temp_vec') .|> Float32
    pred_all = DataFrame(vcat(pred_ti, fill(t, size(pred_ti, 2))')', [:x, :y, :temp, :t])
    pred_all = (MLJ.transform(standardizer, pred_all) |> Array)' .|> Float32

    ŷ, _ = Lux.apply(nn, pred_all, ps, st)
    predicted = reshape(ŷ, :)
    urast = get_raster_values(predicted, landscape)
    return urast
end

function loss(model, ps, st, data)
    x, y = data
    ŷ, st = Lux.apply(model, x, ps, st)
    l = binarycrossentropy(y, ŷ)
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
                opt,
                n_epochs,
                print_freq,
                kwargs...)

    @unpack train_split, batchsize = kwargs
    train_dataloader, test_dataloader, _ = process_data(df; train_split, batchsize)
   
    train_state = Training.TrainState(rng, model, opt)
    vjp_rule = AutoZygote()

    ### Warmup the model
    @info "Warming up..."
    x_proto = randn(rng, Float32, 4, 2)
    y_proto = rand(rng, Bool, 1, 2)
    Training.compute_gradients(vjp_rule, loss, (x_proto, y_proto), train_state)

    @info "Training..."
    for epoch in 1:n_epochs
        stime = time()
        for data in train_dataloader
            (_, l, _, train_state) = Training.single_train_step!(vjp_rule, loss, data, train_state)
        end
        ttime = time() - stime

        # Validate the model
        st_ = Lux.testmode(train_state.states)
        tr_loss = mean([loss(model, train_state.parameters, st_, data)[1] for data in train_dataloader])
        te_loss = mean([loss(model, train_state.parameters, st_, data)[1] for data in test_dataloader])

        # tr_acc = accuracy(model, train_state.parameters, st_, train_dataloader) * 100
        # te_acc = accuracy(model, train_state.parameters, st_, test_dataloader) * 100
        # epoch % 200 == 1 && @show train_state.parameters
        epoch % print_freq == 0 && @printf "[%2d/%2d] \t Time %.2fs \t Training loss: %.2f Test loss: \
                    %.2f\n" epoch n_epochs ttime tr_loss te_loss
    end
    return (train_state.parameters,  Lux.testmode(train_state.states))
end

# data processing
df = CSV.read(@__DIR__() * "/../../data/PA_data.csv", DataFrame)
_, _, standardizer = process_data(df)


# model architecture
nn_params = (;npred = 4, hidden_nodes = 32, hidden_layers=3)
model = build_model(;nn_params...)

# training
opt = Adam(1e-2)
ps, st = train(;df, model, opt, print_freq=4, n_epochs= 50, batchsize=128, train_split=0.9)

# predictions
temp_raster = load_raster()
landscape = Landscape(temp_raster)

using Plots

tsteps = sort!(unique(df.t))
anim = @animate for t in tsteps
    pred = project_spatially(model, ps, st, landscape, t, standardizer)
    plot(pred, title="T = $t")
end

gif(anim, fps=5)
