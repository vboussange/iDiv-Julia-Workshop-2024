# A primer on inference with differentiable process-based models in Julia

Here you will learn about different techniques to calibrate differential equations (or any type of differentiable process-based model) against data. We'll keep the best for the end, where we'll see how we can improve those models by parametrizing uncertain parts with neural networks.

- [A primer on inference with differentiable process-based models in Julia](#a-primer-on-inference-with-differentiable-process-based-models-in-julia)
  - [Wait, what is a differentiable model?](#wait-what-is-a-differentiable-model)
  - [On the importance of gradients for inference](#on-the-importance-of-gradients-for-inference)
  - [Automatic differentiation](#automatic-differentiation)
  - [Lotka-Volterra equations](#lotka-volterra-equations)
  - [Machine learning approach](#machine-learning-approach)
    - [Exercise: Hey, this is cheating!](#exercise-hey-this-is-cheating)
  - [Sensitivity methods](#sensitivity-methods)
  - [Turing: Bayesian inference](#turing-bayesian-inference)
    - [Our first Turing model](#our-first-turing-model)
    - [Data retrodiction](#data-retrodiction)
    - [Exercise: Hey, this is cheating!](#exercise-hey-this-is-cheating-1)
    - [Mode estimation](#mode-estimation)
    - [Partially observed state](#partially-observed-state)
    - [AD backends and `sensealg`](#ad-backends-and-sensealg)
    - [Using Variational Inference](#using-variational-inference)
    - [Summary of resources Resources](#summary-of-resources-resources)


## Wait, what is a differentiable model?
One can usually write a model as a map $\mathcal{M}$ that maps some parameters $p$, an initial state $u_0$ and a time $t$ to a future state $u_t$

$$u_t = \mathcal{M}(u_0, t, p).$$

We call differentiable a model $\mathcal{M}$ for which we can calculate its derivative with respect to $p$ or $u_0$. The derivative $\frac{\partial \mathcal{M}}{\partial \theta}$ expresses how much the model output changes with respect to a small change in $\theta$.

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

Let's illustrate this concept with the [logistic equation model](https://www.google.com/search?client=safari&rls=en&q=verhulst+equation&ie=UTF-8&oe=UTF-8).

Consider the logistic growth model, which has an analytic solution given by:

$\mathcal{M}(u_0, p, t) = \frac{u_0}{\exp(-r \cdot (t - t_0)) + b \cdot u_0 \cdot (1 - \exp(-r \cdot (t - t_0)))}
$

Let's code it

```julia
using UnPack
using Plots
using Random
using ComponentArrays
using BenchmarkTools
Random.seed!(0)

function mymodel(u0, p, t)
    T = eltype(u0)
    @unpack r, b = p
    t0 = 0.

    @. u0 / (exp(-r * (t - t0)) + b * u0 * (one(T) - exp(-r * (t - t0))))
end

p = ComponentArray(;r = rand(), b = rand())
u0 = rand()

tsteps = range(0, 30, length=100)
y = mymodel(u0, p, tsteps)

plot(tsteps, y)
```

> Side Note: what is a `ComponentArray`?

A `ComponentArray` is a convenient Array type that allows to access array elements with symbols, similarly to a `NamedTuple`, while behaving like a standard array. For instance, you could do something like

```julia
cv = ComponentVector(;a = 1, b = 2)
cv .= [3, 4]
```
This is useful, because you cannot take a gradient w.r.t a `NamedTuple`, but you can with a `Vector`!

Now let's try to calculate the gradient of this model. While you could in this case derive the gradient analytically, an analytic derivation is generally tricky with complex models. And what about models that can only be simulated numerically, with no analytic expressions? We need to find a more automatized way to calculate gradients. 

How about [the finite difference method](https://en.wikipedia.org/wiki/Finite_difference_method)?

```julia
function ∂mymodel_∂p(h, u0, p, t)
    phat = (; r = p.r, b= p.b + h)
    return (mymodel(u0, phat, t) - mymodel(u0, p, t)) / h
end

∂mymodel_∂p(1e-1, u0, p, 1.)
```

The gradient of the model is useful to understand how a parameter influences the output of the model. Let's calculate the important of the carrying capacity `b` on the model output:

```julia
dm_dp = ∂mymodel_∂p(1e-1, u0, p, tsteps)
plot(tsteps, dm_dp)
```

As you can observe, the carrying capacity has no effect at small $t$ where population is small, and its influence on the dynamics grows as the population grows. We expect the reverse effect for $r$.

## On the importance of gradients for inference

The ability to calculate the derivative of a model is crucial when it comes to inference. Both within a full Bayesian inference context, where one wants to calculate the posterior distribution of parameters $\theta$ given data $u$, $p(\theta| u)$, or when one wants to obtain a point estimate $\theta^\star = \argmax_\theta (p(\theta | u))$, this quantity is required.

Given an intial estimate, this gradient is used to suggest a new, better estimate.

![](https://editor.analyticsvidhya.com/uploads/631731_P7z2BKhd0R-9uyn9ThDasA.png)

Gradient-based methods are usually very efficient in high-dimensional spaces. This is true in a Bayesian inference context with Hamiltonian Markov Chains, such as the NUTS sampler, or in a frequentist or machine learning, with gradient-based optimizer.

## Automatic differentiation

But how do we choose `h` to calculate the derivative? This is a tricky question, because a too small `h` can lead to round off errors ([see more explanations here](https://book.sciml.ai/notes/08-Forward-Mode_Automatic_Differentiation_(AD)_via_High_Dimensional_Algebras/)) while `h` too large also leads to a bad approximation of the asymptotic definition.

Also, can you calculate how many evaluations of the model do you need if your parameter is $d$ dimensionsal?

$\mathcal{O}(2 d)$

Fortunately, Julia is an *AD-pervasive language*! This means that you can **exactly** differentiate any piece of function written in Julia, at low cost.

```julia
using ForwardDiff

@btime ForwardDiff.gradient(p -> mymodel(u0, p, 1.), p)
```

This is what makes Julia great for model calibration and inference! Write your model in Julia, and any inference method using AD will be able to work with your model! This is not the case in Python or R, where you need to write your model in a machine learning framework such as Torch, JAX or Tensorflow, because those libraries do not know how to differentiate code not written in their own language.

To learn more about AD in Julia, check-out this [cool blog-post](https://gdalle.github.io/AutodiffTutorial/) and [a short presentation](https://gdalle.github.io/JuliaCon2024-AutoDiff/#/title-slide).

Now let's get started with inference.


## Lotka-Volterra equations

We'll use a simple dynamical community model, the [Lotka Volterra](https://en.wikipedia.org/wiki/Lotka–Volterra_equations) model, to generate the data. We'll then contaminate this data with noise, and assume that we do not know the exact parameters that have generated this data. 

The goal of the session will be to estimate those parameters from the data, using a bunch of different techniques. 

So let's first generate the data.

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
```

This is the true state of the system. Now let's contaminate it with observational noise. We'll add lognormally distributed noise, because we are observing population abundance which can only be positive.

```julia
data_mat = Array(sol_true) .* exp.(0.3 * randn(size(sol_true)))
# Plot simulation and noisy observations.
plot(sol_true; alpha=0.3)
scatter!(sol_true.t, data_mat'; color=[1 2], label="")
```

Now that we have our data, let's do some inference!


## Machine learning approach
We'll get started with a very crude approach to inference, where we'll treat the calibration of our LV model similarly to a supervised machine learning task. To do so, we'll define a loss function, defining a distance between our model and the data, and we'll try to minimize this loss.

```julia
function loss(p)
    predicted = solve(prob,
                        alg; 
                        p, 
                        saveat,
                        abstol=1e-6, 
                        reltol = 1e-6)

    l = 0.
    for i in 1:length(predicted)
        if all(predicted[i] .> 0)
            l += sum(abs2, log.(data_mat[:, i]) - log.(predicted[i]))
        end
    end

    return l, predicted


end
```

We'll define an initial estimate for the parameters, and we'll use a gradient-based optimizer that will iteratively update the parameters so as to minimize the loss, following a gradient descent approach

$$\theta_n = \theta_{n-1} + \eta \nabla L(\theta_n)$$

Where $\eta$ is the learning rate.

Let's define a helper function, that will plot how good does the model perform across different iterations.

```julia

losses = []
callback = function (p, l, pred; doplot=true)
    push!(losses, l)
    if length(losses)%10==0
      println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    if doplot
        plt = scatter(sol_true.t, data_mat[1,:]; label = "data x", color = :blue, markerstrokewidth=0)
        scatter!(plt, sol_true.t, Array(pred)[1,:]; label = "prediction x", color = :blue, markershape=:star5, markerstrokewidth=0)
        scatter!(plt, sol_true.t, data_mat[2,:]; label = "data y", color = :red, markerstrokewidth=0)
        scatter!(plt, sol_true.t, Array(pred)[2,:]; label = "prediction y", color = :red, markershape=:star5, markerstrokewidth=0)

        display(plot(plt, yaxis = :log10))
    end
    return false
end

```
And let's define a wrong initial guess

```julia
pinit = ComponentArray(;α = 1., β = 1.5, γ = 1.0, δ = 0.5)

callback(pinit, loss(pinit)...; doplot = true)
```
It's bad but that's what we want!

We'll use the library `Optimization` which is a wrapper library around many optimization libraries in Julia, providing you with many different types of optimizers. We'll use the infamous [ADAM optimizer](https://arxiv.org/abs/1412.6980) (187k citations!!!), widely used in ML.

```julia
using Optimization
using OptimizationOptimisers
using SciMLSensitivity

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

@time res_ada = Optimization.solve(optprob, ADAGrad(0.1); callback = callback, maxiters = 500)
res_ada.minimizer
```

Nice! It seems that the optimizer did a reasonable job, and that we found a reasonable estimate of our parameters!

### Exercise: Hey, this is cheating!

Notice that we use the true `u0`, as if we were to know exactly the initial state. In a real situation, we need also to infer the true state!

Can you modify the model to infer the true state?

```julia
function loss2(p)
    predicted = solve(prob,
                        alg; 
                        p,
                        u0 = p.u0,
                        saveat,
                        abstol=1e-6, 
                        reltol = 1e-6)

    l = 0.
    for i in 1:length(predicted)
        if all(predicted[i] .> 0)
            l += sum(abs2, log.(data_mat[:, i]) - log.(predicted[i]))
        end
    end
    return l, predicted
end
losses = []

pinit = ComponentArray(;α = 1., β = 1.5, γ = 1.0, δ = 0.5, u0 = data_mat[:,1])

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss2(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

@time res_ada = Optimization.solve(optprob, ADAGrad(0.1); callback = callback, maxiters = 1000)
res_ada.minimizer

```

## Sensitivity methods
What's `SciMLSensitivity`? and `adtype = Optimization.AutoZygote()`? Well, AD comes in different flavours, with broadly two types of AD - forward mode and backward mode -, and a bunch of different implementations. You can specify which ones to use with `adtype`, see available options [here](https://docs.sciml.ai/Optimization/stable/API/ad/). But when it comes to differentiating the `solve` function, you want to use `AutoZygote()`, because when trying to differentiate `solve`, the `Zygote` a specific adjoint rule provided by the `SciMLSensitivity` package will be used. These adjoint rules can be specificed by the keyword `sensealg` when calling `solve` and are designed for best performance when differentiating numerical solutions to ODEs. There exist a lot of them (see a review [here](https://arxiv.org/abs/2406.09699)), and if `sensealg` is not provided, a smart polyalgorithm is going to pick up one for you.

See [here](https://docs.sciml.ai/SciMLSensitivity/stable/manual/differential_equation_sensitivities/) for how to choose an algorithm, and let's evaluate the performance of two here for the specificity of our problem

```julia
using Zygote
function loss_sensealg(p, sensealg)
    predicted = solve(prob,
                        alg; 
                        sensealg,
                        p,
                        u0 = p.u0,
                        saveat,
                        abstol=1e-6, 
                        reltol = 1e-6)

    l = 0.
    for i in 1:length(predicted)
        if all(predicted[i] .> 0)
            l += sum(abs2, log.(data_mat[:, i]) - log.(predicted[i]))
        end
    end
    return l
end

@btime Zygote.gradient(p -> loss_sensealg(p, ForwardDiffSensitivity()), pinit)
```

```julia
@btime Zygote.gradient(p -> loss_sensealg(p, ReverseDiffAdjoint()), pinit)
```
For a small number of parameters, forward methods tend to perform best, but with higher number of parameters, the other way around is true.


## Turing: Bayesian inference

We'll first start by using the gold standard for inference: Bayesian inference. Julia has a very strong library for Bayesian inference: [Turing.jl](https://turinglang.org).

Let's declare our first Turing model!

Turing makes heavy use of macros, which allows the library to automatically construct the posterior distribution. Essentially, when defining a model, you need to append the macro `@model`, which will permit Turing.jl to adequately treat the random variables, defined with the `~` symbol.


### Our first Turing model

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
```

Now we can instantiate our model, and run the inference!

```julia

model = fitlv(data_mat, prob)

# Sample 3 independent chains with forward-mode automatic differentiation (the default).
chain = sample(model, NUTS(), MCMCThreads(), 1000, 3; progress=true)

```

> ! Threads
> How many threads do you have running? `Threads.nthreads()` will tell you!

Let's see if our chains have converged.

```julia
using StatsPlots
plot(chain)
```

### Data retrodiction

Let's now generate simulated data using samples from the posterior distribution, and compare to the original data.

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
plot_predictions(chain, sol_true, data_mat)
```

### Exercise: Hey, this is cheating!

Notice that we use the true `u0`, as if we were to know exactly the initial state. In a real situation, we need also to infer the true state!

Can you modify the model to infer the true state?

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
```

Here is a small utility function to visualize your results.

```julia

function plot_predictions2(chain, sol, data_mat)
    myplot = plot(; legend=false)
    posterior_samples = sample(chain, 300; replace=false)
    for i in 1:length(posterior_samples)
        ps = posterior_samples[i]
        p = get(ps, [:α, :β, :γ, :δ], flatten=true)
        u0 = get(ps, :u0, flatten = true)
        u0 = [u0[1][1], u0[2][1]]

        sol_p = solve(prob, Tsit5(); u0, p, saveat)
        plot!(sol_p; alpha=0.1, color="#BBBBBB")
    end

    # Plot simulation and noisy observations.
    plot!(sol; color=[1 2], linewidth=1)
    scatter!(sol.t, data_mat'; color=[1 2])
    return myplot
end

plot_predictions2(chain2, sol_true, data_mat)
```


### Mode estimation
Turing allows you to find the maximum likelihood estimate (MLE) or maximum a posteriori estimate (MAP), similarly to what we have done by hand with the Optimization.jl library. These can be obtained by `maximum_likelihood` and `maximum_a_posteriori`.


```julia
Random.seed!(0)
maximum_a_posteriori(model2, maxiters = 1000)
```
Since `Turing` uses under the hood the same Optimization.jl library, you can specify which optimizer youd'd like to use.

```julia
map_res = maximum_a_posteriori(model2, Adam(0.01), maxiters=2000)
```
We can check whether the optimization has converged:

```julia
@show map_res.optim_result
```
What's very nice is that Turing.jl provides you with utility functions to analyse your mode estimation results.

```julia
using StatsBase
coeftable(map_res)
```


### Partially observed state
Let's assume an ever more complicated situation: for some reason, you only have observation data for the predator. Could you still infer all parameters of your model, including those of the prey?

YES! Because the signal of the variation in abundance of the predator contains information on the dynamics of the whole predator-prey system.

Let's see how we can do that with Turing.jl. Here we need to assume so prior state for the prey. We'll just assume that it is the same as that of the predator.

```julia
@model function fitlv3(data::AbstractVector, prob)
    # Prior distributions.
    σ ~ InverseGamma(2, 3)
    α ~ truncated(Normal(1.5, 0.5); lower=0.5, upper=2.5)
    β ~ truncated(Normal(1.2, 0.5); lower=0, upper=2)
    γ ~ truncated(Normal(3.0, 0.5); lower=1, upper=4)
    δ ~ truncated(Normal(1.0, 0.5); lower=0, upper=2)
    u0 ~ MvLogNormal([data[1], data[1]], σ^2 * I)

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

How cool!

Now you need to realise that up to now, we had a relatively simple model. How would this model scale, should we have a much larger model? Let's cook-up some idealised LV model.


### AD backends and `sensealg`
The `NUTS` sampler uses automatic differentiation under the hood. 

By default, `Turing.jl` uses `ForwardDiff.jl` as an AD backend, meaning that the SciML sensitivity methods are not used when the `solve` function is called. However, you could change the AD backend to `Zygote` with `adtype=AutoZygote()`.

```julia
chain2 = sample(model2, NUTS(), MCMCThreads(), adtype=AutoZygote(), 3000, 3; progress=true)
```

See [here](https://turinglang.org/docs/tutorials/docs-10-using-turing-autodiff/index.html) for more information. 

### Using Variational Inference

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

- https://turinglang.org/docs/tutorials/09-variational-inference/

### Summary of resources Resources
- https://turinglang.org/docs/tutorials/10-bayesian-differential-equations/
- https://turinglang.org/docs/tutorials/09-variational-inference/