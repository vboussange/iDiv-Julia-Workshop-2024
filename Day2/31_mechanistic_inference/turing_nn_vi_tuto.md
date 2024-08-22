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
using StatsPlots
using DataFrames
using UnPack
using OrdinaryDiffEq
cd(@__DIR__)

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


### Bayesian NN
```julia
using Turing
import Flux
using Lux
using Turing: Variational
using Random
using ComponentArrays
using Functors
using LinearAlgebra

rng = Random.default_rng()

nn_init = Lux.Chain(
    Lux.Dense(2,2, relu)
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
p = ComponentArray(;α = 1.5, γ = 3.0, p_nn=p_nn_init)

prob_nn = ODEProblem(lotka_volterra_nn, u0, tspan, p)
init_sol = solve(prob_nn, alg; saveat)
# Plot simulation.
plot(init_sol)


# defining inference model

# Create a regularization term and a Gaussian prior variance term.
sigma = 0.5

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

    nparameters = Lux.parameterlength(nn)
    p_nn_vec ~ MvNormal(zeros(nparameters), sigma^2 * I)

    p_nn = vector_to_parameters(p_nn_vec, p_nn_init)


    # Simulate Lotka-Volterra model. 
    p = (;α, γ, p_nn)
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

q = vi(model, advi, q0; optimizer=Flux.Adam(1e-2), init_params = [0.1, 1.5, 3.0, ComponentArray(p_nn_init)[:]])

function plot_predictions_vi_nn(q, sol, data_mat)
    myplot = plot(; legend=false)
    z = rand(q, 300)
    for parr in eachcol(z)
        p[[:α, :γ]] .= parr[2:3]
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
        p[:p_nn] .= parr[4:end]
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
        p[:p_nn] .= parr[4:end]
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
