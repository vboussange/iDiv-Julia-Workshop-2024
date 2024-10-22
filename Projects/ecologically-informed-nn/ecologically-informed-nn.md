# Ecologically-informed neural networks


![](https://files.ipbes.net/ipbes-web-prod-public-files/styles/xl_1920_scale/azblob/2023-09/IAS%20website%20image.png.webp?itok=tVaiqu2q)

Invasive species pose major threats on nature, nature’s contributions to
people, and good quality of life ([IPBES,
2019](https://zenodo.org/records/3553579)), with global annual costs
estimated to exceed 423 billion USD. The management of invasive alien
species requires the modelling of the species range and spread dynamics,
in order to optimally allocate control effort. While current
state-of-the-art modelling approaches such as deep learning species
distribution model (deep SDM) can help in determining the potential
niche of invasive alien species (e.g., [Brun et
al. 2024](https://www.nature.com/articles/s41467-024-48559-9)), they
assume that species are always in equilibrium with their environment,
neglecting key transient ecological processes such as demographic
changes and dispersal. This oversight is problematic, because these
processes are crucial in determining the establishment of a species in a
novel environment.

The proposed project aims to fill this gap by developing an
ecologically-informed neural network, following a physics-informed
neural network approach (see e.g.[Raissi et
al. 2019](https://www.nature.com/articles/s41467-024-48559-9)).

![](https://benmoseley.blog/wp-content/uploads/2021/08/pinn-1536x608.png)

Specifically, we will build a process-based dynamical model, that we
will couple to a deep SDM. The resulting approach will allow to inform
the process-based model with presence-absence data, generating a
data-driven dynamical forecast of species invasion.

Julia is great for physics-informed approaches, because it allows to
quickly build fast process-based models. Because Julia is an
AD-pervasive language, these process-based models can be further
differentiated, so that you could eventually calibrate its parameters
against the ecological data at hand.

## What you will learn

-   Building a simple deep SDM
-   Building a simple process-based model
-   Scientific machine learning

## Prerequisite

-   [Read this tutorial to understand what is a physics-informed neural
    network and how to code it in
    Julia](https://book.sciml.ai/notes/03-Introduction_to_Scientific_Machine_Learning_through_Physics-Informed_Neural_Networks/)
-   [Have a look at this Lux tutorial on how to train a MNIST
    classifier](https://lux.csail.mit.edu/dev/tutorials/beginner/4_SimpleChains)

## Work packages

The different work packages below can be addressed independently and
distributed among different teams. The team could collaborate on a
public repository.

### **WP A**: Process-based model

The goal is to construct a model which simulates the dynamics of species
abundance on a landscape. Given a landscape consisting of *P* patches
{*p*<sub>1</sub>, …, *p*<sub>*P*</sub>}, we want to model the dynamics
of the population abundance vector
${\bf n}\_{t} = (n\_{1, t}, \dots, n\_{P, t})$ where
*n*<sub>*i*, *t*</sub> corresponds to the population density on
*p*<sub>*i*</sub>.

We will assume that the population experiences a logistic growth, and
that it is subject to dispersal. We will use a difference equation model
to express these processes

$$
n\_{i, t+1} = n\_{i, t} + \[F({\bf n_t})\]\_i,
$$
where
$$
\[F({\bf n_t})\]\_i  = \mathcal{E}(n\_{i, t}) + \[\mathcal{D}({\bf n_t})\]\_i.
$$
Here, ℰ coresponds to the ecological dynamics,
$$
\mathcal{E}(n\_{i, t}) = n\_{i, t}(1 - \frac{n\_{i, t}}{K_i})
$$
with *K*<sub>*i*</sub> the carrying capacity on patch *p*<sub>*i*</sub>,
and where 𝒟 is the dispersal kernel
$$
\[\mathcal{D}({\bf n_t})\]\_i = \left\[ \sum_j  u\_{j, t} m\_{j, i} - \sum_j  u\_{i, t} m\_{i, j}\right\]
$$

with *m*<sub>*j*, *i*</sub> the proportion of individuals migrating from
*p*<sub>*j*</sub> to *p*<sub>*i*</sub>. The first term corresponds to
incoming individuals, the second to leaving individuals.

#### Ecology

We will assume that
*K*<sub>*i*</sub> = *e*<sup>−*κ*(*T*<sub>*i*</sub> − *T*<sup>⋆</sup>)<sup>2</sup></sup>
where *T*<sub>*i*</sub> is the temperature on patch *p*<sub>*i*</sub>,
*T*<sup>⋆</sup> is the optimal temperature for the species, and *κ*
defines how the carrying capacity decreases under sub-optimal
conditions.

#### Dispersal

We will assume that
$$
m\_{i,j} = \begin{cases} 
m \\ e^{-d\_{i,j}/\alpha}, & \text{if } i \neq j \\
0, & \text{if } i = j 
\end{cases}
$$
where *m* is the migration rate and *d*<sub>*i*, *j*</sub> is the
distance between *p*<sub>*i*</sub> and *p*<sub>*j*</sub>.

> **Your task**
>
> Implement this model! For this, you’ll need to create a `DynSDM`
> struct. This struct should be used with the `simulate` function:
>
> ``` julia
> model = DynSDM(...)
> simulate(model, u0, ntsteps)
> ```
>
> Given a `model` and an initial abundance vector `u0` and the number of
> time steps `ntsteps`, this function returns a vector of size `ntsteps`
> containing abundance vectors with the same shape as `u0` for each of
> the time steps investigated.
>
> The dynamics should look as follows:
>
> ![](data/anim.gif)

#### Hints

You are given a `utils.jl` file, which contains helper functions to aid
your work. Important functions are detailed in the boxes below.

> **Loading the data**
>
> You will use temperature data from CHELSA. Run the following snippet.
> It will download the `bio1` raster from CHELSA, and load it.
>
> ``` julia
> include("project_src/utils.jl")
>
> temp = load_raster()
>
> plot(temp)
> ```

> **Defining the landscape**
>
> The `utils.jl` file provides you with a `Landscape` struct. This
> struct is very useful, because it calculates the distance between all
> pixels of a raster. Use it in your model!
>
> Here is a snippet of how you can use it:
>
> ``` julia
> landscape = Landscape(temp)
> landscape.dists
> ```
>
> Here, `landscape.dists[i, j]` corresponds to the `d_{i,j}`
> coefficient, i.e. the distance between pixel `i` and `j`.
>
> The method `get_raster_values` allows you to transform a flat vector
> into a raster corresponding to the associated landscape.
>
> For illustration purposes, here we plot the distance of all patch to
> the 400th patch in the landscape:
>
> ``` julia
> distances_to_400 = get_raster_values(landscape.dists[400,:], landscape)
> plot(distances_to_400)
> ```

> **Numerical values**
>
> ``` julia
> T0 = 4. # optimal growth temperature
> κ = 1.
> α = 4. # mean dispersal distance
> ```

> **Initial conditions**
>
> We suggest using the following initial conditions:
>
> ``` julia
> x = lookup(temp, X)
> y = lookup(temp, Y)
> x_0 = x[floor(Int, length(x)/2)]
> y_0 = y[floor(Int, length(y)/2)]
> a = 2.
> b = 1.
> u0_raster = @. exp(- a * ((x - x_0)^2 + (y-y_0)^2)) * exp(- b * (temp - T0)^2)
> plot(u0_raster)
> ```

> **Defining state variables**
>
> It will be easier to handle a flat vector for calculating the
> dynamics.
>
> ``` julia
> u0 = u0_raster[:]
> ```
>
> But you can use the following helper to transform `u0` or any state
> variable vector into a raster
>
> ``` julia
> plot(get_raster_values(u0, landscape))
> ```

> **Dispersal kernel**
>
> Implementing the dispersal kernel is not trivial - you need to do it
> write to get performance. Consider using the following implementation:
>
> ``` julia
> disp_proximities = sparse(disp_kern.(landscape.dists)) # we transform the distances in proximities
> D = - m * (I - (disp_proximities ./ sum(disp_proximities,dims=1)))
> ```
>
> Now you need to convince yourself that `D * u` exactly equals
> $\mathcal{D}({\bf n}\_t)$.

> **`DynSDM` specs**
>
> You probably want to define a structure of the sort
>
> ``` julia
> Base.@kwdef struct DynSDM{LD,KK,DD} <: AbstractDynModel
>     landscape::LD
>     K::KK
>     D:DD
> end
> ```
>
> that can be instantiated with a function
>
> ``` julia
> build_dyn_model(landscape::Landscape)
> ```
>
> This structure will ensure easy access to `K` and `D`, while storing
> the landscape so that you can project flat state variable vectors to a
> raster.

> **Optional task: model differentiability and mechanistic inference**
>
> -   Is your model differentiable? Try whether you can obtain the
>     derivative of its output with respect to its parameters. A good
>     idea is to embed the model’s parameters in a `ComponentArray`.
>
> -   Once the model is differentiable, generate data and use
>     `Optimisers.jl` to find back the parameters that have generated
>     the data.

<!-- 
You could consider any sort of model, but here are some examples
- The discrete reaction diffusion model from [Bonneau et al. 2017](https://esajournals.onlinelibrary.wiley.com/doi/10.1002/ecs2.1979), where individuals experience a logistic growth with a locally defined carrying capacity $K_i$ that depends on the landscape characteristics, and where some individuals disperse to neighboring vertices.
- The metapopulation model from [Hanski et al. 2003](https://www.nature.com/articles/35008063)  -->

### **WP B**: Constructing a deep SDM in Julia

The goal is to construct a deep SDM NN which can predict the species
distribution at time *t*, given an array of absence-presence
observations.

*ŷ* = NN(long, lat, *E*, *t*)

Given that the species range changes over time, spatial coordinates
(*l**o**n**g*, *l**a**t*), environmental predictors *E* (for this
exercise, we’ll use `bio1` from CHELSA), and time *t* will be used as
inputs to the SDM.

> **Your task**
>
> You are given a synthetic presence absence dataset (see green box
> below.) Build a `Lux.jl` model that predicts the species distribution
> over the whole spatial domain defined by `raster = load_data()`, for
> each time steps in the dataset.
>
> <!-- You'll need to build a function 
> ```julia
> train(;nn, 
>         PA_data, 
>         optim, 
>         n_epochs)
> ``` -->

#### Hints

You have access to a `utils.jl` file containing helper functions to
assist in your task. Important functions are detailed in the sections
below.

> **General idea**
>
> Your script should implement the following key functions:
>
> -   **`build_model`**: Constructs and returns a `Lux` model.
> -   **`preprocess_data`**: Normalizes raw data for neural network
>     training.
> -   **`loss`**: Defines the loss function for training.
> -   **`train`**: The main function that iterates over batches to
>     update model parameters.

> **Loading and pre-processing the absence-presence data**
>
> You will work with a synthetic presence-absence (PA) dataset for a
> virtual invasive species, located in `data/PA_data.csv`. This dataset
> is derived from a process-based model’s true abundance data.
>
> 1.  **Load the data**: Start by loading the `.csv` file into a
>     `DataFrame`.
> 2.  **Preprocess the data**: Implement a
>     `process_data(df; train_split=0.9, batchsize=128)` function to:
>     -   Normalize the features.
>     -   Split the data into training and testing sets.
>     -   Return `train_dataloader` and `test_dataloader` for model
>         training.
>
> Use `splitobs` and `DataLoader` from `MLUtils.jl` for data handling,
> and `Standardizer` from `MLJ` for standardization.
>
> <!-- But feel free to work with real data if you have an idea! -->
> <!-- ![](data/PA.gif) -->
> <!-- 
> You may want to create a function that processes this dataframe so that
> 1. the resulting processed data is normalized 
> 2. you can iterate over it -->
> <!-- Notice how the number of samples increases over time. You should correct for this bias! -->
> <!-- ### True population abundance data
> ![](data/anim.gif)
> in `data/true_abundance_data.jld2`
>
> You may use this dataset to evaluate your model, but not for training! -->

> **🆘 *Advanced help*: loading and pre-processing the absence-presence
> data**
>
> -   To normalize `data` (a DataFrame), you can use
>
> ``` julia
> standardizer = machine(Standardizer(), data)
> fit!(standardizer)
> data = (MLJ.transform(standardizer, pred_data) |> Array)' .|> Float32
> ```
>
> -   `splitobs` may be used as such:
>
> ``` julia
> (x_train, y_train), (x_test, y_test) = splitobs((x_data, y_data); at=train_split)
> ```
>
> -   a `DataLoader` can be defined as such
>
> ``` julia
> DataLoader(collect.((x_train, y_train)); batchsize, shuffle=true)
> ```

> **Neural network architecture and loss**
>
> Create a Multi-Layer Perceptron (MLP) using `Lux.Chain` with `Dense`
> layers. Consider using a `BatchNorm` layer for faster training. The
> final activation function should be `Lux.sigmoid`, outputting a
> probability `ŷ`, which should be compared against the binary
> absence-presence target `y` using `Flux.binarycrossentropy`.

> **🆘 *Advanced help*: Neural network architecture and loss**
>
> -   Consider using the following function to create your model.
>
> ``` julia
> function build_model(; npred, hidden_nodes, hidden_layers, dev=cpu)
>     model = Chain(Dense(npred, hidden_nodes),
>         BatchNorm(hidden_nodes),
>         [Dense(hidden_nodes, hidden_nodes, tanh) for _ ∈ 1:hidden_layers-1]...,
>         Dense(hidden_nodes, 1),
>         Lux.sigmoid
>     )
>
>     return model
> end
> ```
>
> -   Observe that `loss` should have the signature
>     `loss(model, ps, st, data)` to be a valid function to be trained
>     with the `Lux.Training` API (see box below):
>
> ``` julia
> Training.single_train_step!(AutoZygote(), loss, data, train_state)
> ```
>
> -   `loss` should return a tuple: `l, st, (;)`, with `l` the loss
>     value and `st` the state of the model
> -   Remember how to evaluate a `Lux` model:
>
> ``` julia
> ŷ, st = Lux.apply(model, x, ps, st)
> ```

> **Training your network**
>
> Within the `train` function:
>
> 1.  **Initialize Training State**:
>
> ``` julia
> train_state = Training.TrainState(rng, model, opt)
> ```
>
> where `opt` could be the Adam optimizer
>
> ``` julia
> using Optimisers
> opt = Adam(0.01)
> ```
>
> and `rng` is a random number generator
>
> ``` julia
> rng = Random.default_generator()
> ```
>
> 1.  **Training loop**:
>
>     -   Iterate over the data batches:
>
>     ``` julia
>     for data in train_dataloader
>         (_, l, _, train_state) = Training.single_train_step!(AutoZygote(), loss, data, train_state)
>     end
>     ```
>
>     which will update the model parameters for each batch, and an
>     outer loop to iterate multiple times over your data (epochs):
>
>     -   Repeat for multiple epochs:
>
>     ``` julia
>     for epoch in 1:n_epochs
>     # ...
>     end
>     ```
>
>     -   Return the trained model parameters and state:
>
>     <!-- -->
>
>         (train_state.parameters,  Lux.testmode(train_state.states))

> **Projecting the model’s predictions spatially**
>
> -   To project the deep SDM spatially, you’ll need the `temp` raster,
>     that you can obtain by running the following snippet. This will
>     download the `bio1` (mean annual temperature) raster from CHELSA,
>     and load it.
>
> ``` julia
> include("project_src/utils.jl")
> temp = load_raster()
> plot(temp)
> ```
>
> -   With this raster, construct a dataframe containing temp values
>     with associated longitude, latitude, and time. The following
>     snippet permits to create a dataframe with (`x, y, temp`).
>
> ``` julia
>     loc_xy = DimPoints(temp)[:]
>     temp_vec =  temp[:]
>     df_xytemp = DataFrame(vcat(hcat(collect.(loc_xy)...)', temp_vec'), [:x, :y, :temp])
> ```
>
> -   With this dataframe, you can predict a flat vector of abundance,
>     which you can map back to a raster with the helper function
>     `get_raster_values`:
>
> ``` julia
> ŷ, _ = Lux.apply(nn, pred_all, ps, st)
> predicted = reshape(ŷ, :)
> landscape = Landscape(temp)
> urast = get_raster_values(predicted, landscape)
> plot(urast)
> ```
>
> <!-- Feel free to use additional predictors, by modifying the `load_raster` function. -->

> **Optional task: Hyperparameter optimisation**
>
> Hyperparameters of a neural network model are parameters that are not
> learned during training but are set by the user before training. The
> hyperparameters define the structure and behavior of the neural
> network and influence how the model learns from the input data. Some
> resources:
>
> -   [Hyperopt.jl](https://github.com/baggepinnen/Hyperopt.jl).
>
> -   [A Conceptual Explanation of Bayesian Hyperparameter Optimization
>     for Machine
>     Learning](https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f)

### **WP C**: Ecologically-informed neural network

The objective is to improve the performance of the deep SDM by
additionally constraining it with a process-based model.

> **Your task**
>
> Extend the `loss` function used to train the deep SDM developed in
> **WP B** by constraining it with the process-based model developed in
> **WP A**. Assume perfect knowledge of the dynamics of the species
> (i.e., use the process-based model with true generating parameters,
> see **WP A**).

> **Advanced task**
>
> Now assume that you only have partial knowledge of the dynamics (i.e.,
> you know the function form of the dynamical system, but not the true
> parameter values). You’ll need to consider the parameters of the
> mechanistic model as additional parameters to be learnt, on top of
> that of the deep SDM.

#### Hints

> **Loss function**
>
> The extended loss could have the form
>
> $$
> \mathcal{L} = \sum_i l(\hat{y\_{i,t_i}}, y\_{i,t_i}) + \sum\_{i,j} \left\[\hat y\_{i, t\_{j+1}} - \[F({\bf \hat y}\_{t_j})\]\_i\right\]^2
> $$
> where *l* is the binary cross entropy loss.

<!-- 
Once this is done, check how good are your predictions outside the training range by predicting with

```julia
simulate(model, f[end](pred), 1)
```

which will predict ${\bf n}_{T+1}$.

 -->
