# Scientific Machine Learning

Here you will learn about different techniques to calibrate process-based models against data.

- Turing.jl: + parameter CI estimate - curse of dimensionality
- ODE solver differentiation: + curse of dimensionality - works with ODE only + parameterization of components with NN
- ABC inference: - curse of dimensionality + works with any type of process-based model
- PINNs: + works with any type of process-based model - requires NN architecture design

Non-parametric approaches
- RNN
- reservoir computing

![](MainLynx.jpg)
> Data taken from http://people.whitman.edu/~hundledr/courses/M250F03/M250.html
> 
> Originally published in E. P. Odum (1953), Fundamentals of Ecology, Philadelphia, W. B. Saunders


## Importing data
We need to import the data
```julia
using DelimitedFiles
using StatsPlots
using DataFrames
using UnPack
cd(@__DIR__)
data_filename = "data/LynxHare.txt"
hudson_bay_data = DataFrame(readdlm(data_filename), [:t, :hare, :lynx])

p = @df hudson_bay_data plot(:t, [:hare, :lynx], xlabel="Year")
```

## Lotka-Volterra equations

```julia
using OrdinaryDiffEq

# Define Lotka-Volterra model.
function lotka_volterra(du, u, p, t)
    # Model parameters.
    @unpack α, β, γ, δ = p
    # Current state.
    x, y = u

    # Evaluate differential equations.
    du[1] = (α - β * y) * x # prey
    du[2] = (δ * x - γ) * y # predator

    return nothing
end

# Define initial-value problem.
u0 = [2.0, 2.0]
p_true = (;α = 1.5, β = 1.0, γ = 3.0, δ = 1.0)
# tspan = (hudson_bay_data[1,:t], hudson_bay_data[end,:t])
tspan = (0., 5.)
saveat = 0.1
alg = Tsit5()

prob = ODEProblem(lotka_volterra, u0, tspan, p_true)

sol_true = solve(prob, alg; saveat)
# Plot simulation.
plot(sol_true)

data_mat = Array(sol_true) .* exp.(0.3 * randn(size(sol_true)))
# Plot simulation and noisy observations.
plot(sol_true; alpha=0.3)
scatter!(sol_true.t, data_mat'; color=[1 2], label="")
```

## Turing: Bayesian inference

We'll first start by using the gold standard for inference: Bayesian inference. Julia has a very strong library for Bayesian inference: [Turing.jl](https://turinglang.org).

Turing makes heavy use of macros, which allows the library to construct the problem under the hood.

Let's declare our first Turing model!

```julia
using Turing
using LinearAlgebra

# data_mat = hudson_bay_data[:, [:hare, :lynx]] |> Matrix

@model function fitlv(data, prob)
    # Prior distributions.
    σ ~ InverseGamma(3, 0.5)
    α ~ truncated(Normal(1.5, 0.5); lower=0.5, upper=2.5)
    β ~ truncated(Normal(1.2, 0.5); lower=0, upper=2)
    γ ~ truncated(Normal(3.0, 0.5); lower=1, upper=4)
    δ ~ truncated(Normal(1.0, 0.5); lower=0, upper=2)

    # Simulate Lotka-Volterra model. 
    p = (;α, β, γ, δ)
    predicted = solve(prob, alg; p, saveat)

    # Observations.
    for i in 1:length(predicted)
        if all(predicted[i] .> 0)
            data[:, i] ~ MvLogNormal(log.(predicted[i]), σ^2 * I)
        end
    end

    return nothing
end

model = fitlv(data_mat, prob)

# Sample 3 independent chains with forward-mode automatic differentiation (the default).
chain = sample(model, NUTS(), MCMCThreads(), 1000, 3; progress=true)
```

> ! Threads
> How many threads do you have running? `Threads.nthreads()` will tell you!


```julia
plot(chain)
```

#### Data retrodiction

Generate simulated data using samples from the posterior distribution, and compare to the original data.

```julia
function plot_predictions(chain, sol, data_mat)
    myplot = plot(; legend=false)
    posterior_samples = sample(chain[[:α, :β, :γ, :δ]], 300; replace=false)
    for parr in eachrow(Array(posterior_samples))
        p = NamedTuple([:α, :β, :γ, :δ] .=> parr)
        sol_p = solve(prob, Tsit5(); p, saveat)
        plot!(sol_p; alpha=0.1, color="#BBBBBB")
    end

    # Plot simulation and noisy observations.
    plot!(sol; color=[1 2], linewidth=1)
    scatter!(sol.t, data_mat'; color=[1 2])
    return myplot
end
plot_predictions(chain, sol, data_mat)
```

#### Exercise: Hey, this is cheating!

Notice that we use the true `u0`, as if we were to know exactly the initial state. In a real situation, we need also to infer the true state!

```julia
@model function fitlv2(data, prob)
    # Prior distributions.
    σ ~ InverseGamma(2, 3)
    α ~ truncated(Normal(1.5, 0.5); lower=0.5, upper=2.5)
    β ~ truncated(Normal(1.2, 0.5); lower=0, upper=2)
    γ ~ truncated(Normal(3.0, 0.5); lower=1, upper=4)
    δ ~ truncated(Normal(1.0, 0.5); lower=0, upper=2)
    u0 ~ MvLogNormal(data[:,1], σ^2 * I)

    # Simulate Lotka-Volterra model but save only the second state of the system (predators).
    p = (;α, β, γ, δ)
    predicted = solve(prob, alg; p, u0, saveat)

    # Observations.
    for i in 2:length(predicted)
        if all(predicted[i] .> 0)
            data[:, i] ~ MvLogNormal(log.(predicted[i]), σ^2 * I)
        end
    end

    return nothing

end

model2 = fitlv2(data_mat, prob)

# Sample 3 independent chains.
chain2 = sample(model2, NUTS(), MCMCThreads(), 3000, 3; progress=true)
plot(chain2)

function plot_predictions2(chain, sol, data_mat)
    myplot = plot(; legend=false)
    posterior_samples = sample(chain, 300; replace=false)
    for i in 1:length(posterior_samples)
        ps = posterior_samples[i]
        p = get(ps, [:α, :β, :γ, :δ], flatten=true)
        u0 = get(ps, :u0)
        u0 = [u0[1][1][1], u0[1][2][1]]

        sol_p = solve(prob, Tsit5(); u0=u0, p=p, saveat=0.1)
        plot!(sol_p; alpha=0.1, color="#BBBBBB")
    end

    # Plot simulation and noisy observations.
    plot!(sol; color=[1 2], linewidth=1)
    scatter!(sol.t, data_mat'; color=[1 2])
    return myplot
end

plot_predictions2(chain2, sol, data_mat)
```

### Partially observed state

```julia
@model function fitlv3(data::AbstractVector, prob)
    # Prior distributions.
    σ ~ InverseGamma(2, 3)
    α ~ truncated(Normal(1.5, 0.5); lower=0.5, upper=2.5)
    β ~ truncated(Normal(1.2, 0.5); lower=0, upper=2)
    γ ~ truncated(Normal(3.0, 0.5); lower=1, upper=4)
    δ ~ truncated(Normal(1.0, 0.5); lower=0, upper=2)
    u0 ~ MvLogNormal([2.0,data[1]], σ^2 * I)

    # Simulate Lotka-Volterra model but save only the second state of the system (predators).
    p = (;α, β, γ, δ)
    predicted = solve(prob, Tsit5(); p, u0, saveat, save_idxs=2)

    # Observations of the predators.
    for i in 2:length(predicted)
        if predicted[i] > 0
            data[i] ~ LogNormal(log.(predicted[i]), σ^2)
        end
    end

    return nothing
end

model3 = fitlv3(data_mat[2, :], prob)

# Sample 3 independent chains.
chain3 = sample(model3, NUTS(), MCMCThreads(), 3000, 3; progress=true)
plot(chain3)
p = plot_predictions2(chain3, sol, data_mat)
plot!(p, yaxis=:log10)
```

### Scaling to Large Models: Adjoint Sensitivities

The `NUTS` sampler uses automatic differentiation under the hood. 

- See https://arxiv.org/abs/2406.09699 for a complete treatment of AD.

### Using Variational Inference
- https://turinglang.org/docs/tutorials/09-variational-inference/

```julia
import Flux
using Turing: Variational
model = fitlv2(data_mat, prob)
q0 = Variational.meanfield(model)
advi = ADVI(10, 10_000)

q = vi(model, advi, q0; optimizer=Flux.ADAM(1e-2))

function plot_predictions_vi(q, sol, data_mat)
    myplot = plot(; legend=false)
    z = rand(q, 300)
    for parr in eachcol(z)
        p = NamedTuple([:α, :β, :γ, :δ] .=> parr[2:5])
        u0 = parr[6:7]
        sol_p = solve(prob, Tsit5(); u0, p, saveat)
        plot!(sol_p; alpha=0.1, color="#BBBBBB")
    end

    # Plot simulation and noisy observations.
    plot!(sol; color=[1 2], linewidth=1)
    scatter!(sol.t, data_mat'; color=[1 2])
    return myplot
end

plot_predictions_vi(q, sol, data_mat)

```

### Bayesian NN
```julia
using Lux
using Random
using ComponentArrays
using Functors

rng = Random.default_rng()

rbf(x) = exp.(-(x.^2))
nn_init = Lux.Chain(
    Lux.Dense(2,2)
)
p_nn_init, st_nn = Lux.setup(rng, nn_init)

nn = StatefulLuxLayer(nn_init, st_nn)

# Define Lotka-Volterra model.
function lotka_volterra_nn(du, u, p, t)
    # Model parameters.
    @unpack α, γ, p_nn = p

    # Current state.
    x, y = u

    û = nn(u, p_nn) # Network prediction


    # Evaluate differential equations.
    du[1] = (α - û[1]) * x # prey
    du[2] = (û[2] - γ) * y # predator

    return nothing
end

# Define initial-value problem.
u0 = [2.0, 2.0]
p = ComponentArray(;α = 1.5, γ = 3.0, p_nn=p_nn_init)
# tspan = (hudson_bay_data[1,:t], hudson_bay_data[end,:t])
tspan = (0., 5.)
saveat = 0.1
alg = Tsit5()

prob_nn = ODEProblem(lotka_volterra_nn, u0, tspan, p)
init_sol = solve(prob_nn, alg; saveat)
# Plot simulation.
plot(init_sol)


# defining inference model

# Create a regularization term and a Gaussian prior variance term.
alpha = 0.09
sigma = sqrt(1.0 / alpha)

function vector_to_parameters(ps_new::AbstractVector, ps::NamedTuple)
    @assert length(ps_new) == Lux.parameterlength(ps)
    i = 1
    function get_ps(x)
        z = reshape(view(ps_new, i:(i + length(x) - 1)), size(x))
        i += length(x)
        return z
    end
    return fmap(get_ps, ps)
end

@model function fitlv_nn(data, prob)
    # Prior distributions.
    σ ~ InverseGamma(3, 0.5)
    α ~ truncated(Normal(1.5, 0.5); lower=0.5, upper=2.5)
    γ ~ truncated(Normal(3.0, 0.5); lower=1, upper=4)

    p_nn_vec ~ MvNormal(zeros(nparameters), σ^2 * I)

    p_nn = vector_to_parameters(p_nn_vec, p_nn_init)


    # Simulate Lotka-Volterra model. 
    p = ComponentArray(;α, γ, p_nn)
    predicted = solve(prob, alg; p, saveat)

    # Observations.
    for i in 1:length(predicted)
        if all(predicted[i] .> 0)
            data[:, i] ~ MvLogNormal(log.(predicted[i]), σ^2 * I)
        end
    end

    return nothing
end

model = fitlv_nn(data_mat, prob_nn)
q0 = Variational.meanfield(model)
advi = ADVI(10, 10_000)

q = vi(model, advi, q0; optimizer=Flux.ADAM(1e-2))

function plot_predictions_vi_nn(q, sol, data_mat)
    myplot = plot(; legend=false)
    z = rand(q, 300)
    for parr in eachcol(z)
        p[[:α, :δ]] .= parr[2:3]
        p[:p_nn] .= parr[4:end]
        # p[:p_nn] .= 0.
        sol_p = solve(prob_nn, Tsit5(); u0, p, saveat)
        plot!(sol_p; alpha=0.1, color="#BBBBBB")
    end

    # Plot simulation and noisy observations.
    plot!(myplot, sol; color=[1 2], linewidth=1)
    scatter!(myplot, sol.t, data_mat'; color=[1 2])
    return myplot
end

myplot = plot_predictions_vi_nn(q, sol_true, data_mat)
plot!(myplot, yaxis=:log10)


function plot_func_resp(q, data)
    # plotting prediction of functional response
    u1 = range(minimum(data[1,:]), maximum(data[1,:]), length=100) 
    u2 = range(minimum(data[2,:]), maximum(data[2,:]), length=100) 
    u = hcat(u1,u2)

    z = rand(q, 300)

    myplot1 = plot(u2,
                    p_true.β .* u2; 
                    legend=false
                    )
    for parr in eachcol(z)
        p[:p_nn] .= parr[6:end]
        func_resp = nn(u', p.p_nn)
        plot!(myplot1,
                u2,
                func_resp[1,:]; 
                alpha=0.1, 
                color="#BBBBBB"
                )

    end

    myplot2 = plot(u1,
                    p_true.δ .* u2; 
                    legend=false
                    )
    for parr in eachcol(z)
        p[:p_nn] .= parr[6:end]
        func_resp = nn(u', p.p_nn)
        plot!(myplot2,
                u1,
                func_resp[2,:]; 
                alpha=0.1, 
                color="#BBBBBB"
                )

    end
    myplot = plot(myplot1, myplot2)

    return myplot

end

plot_func_resp(q, data_mat)

```



## Machine learning approach

```julia
using SciMLSensitivity

function loss(p)
    predicted = solve(prob_nn, alg; p, saveat, abstol=1e-6, reltol = 1e-6, sensealg = ForwardDiffSensitivity())

    l = 0.
    for i in 1:length(predicted)
        if all(predicted[i] .> 0)
            l += sum((log.(data_mat[:, i]) - log.(predicted[i])).^2)
        end
    end

    return l, predicted


end

losses = []
callback = function (p, l, pred; doplot=true)
    push!(losses, l)
    if doplot
        plt = scatter(sol_true.t, data_mat[1, :]; label = "data")
        scatter!(plt, sol_true.t, pred[1, :]; label = "prediction")
        display(plot(plt))
    end
    return false
end

pinit = ComponentArray(;α = 1., δ = 3.0, p_nn)


callback(pinit, loss(pinit)...; doplot = true)

```

### Resources
- https://turinglang.org/docs/tutorials/10-bayesian-differential-equations/
- 