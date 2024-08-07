# Scientific Machine Learning

Here you will learn about different techniques to calibrate process-based models against data.

- Turing.jl: + parameter CI estimate - curse of dimensionality
- ODE solver differentiation: + curse of dimensionality - works with ODE only + parameterization of components with NN
- ABC inference: - curse of dimensionality + works with any type of process-based model
- PINNs: + works with any type of process-based model - requires NN architecture design

Non-parametric approaches
- RNN
- reservoir computing

## Lotka-Volterra equations

```julia
using OrdinaryDiffEq
using UnPack
using Plots

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

data_mat = Array(sol_true) .* exp.(0.1 * randn(size(sol_true)))
# Plot simulation and noisy observations.
plot(sol_true; alpha=0.3)
scatter!(sol_true.t, data_mat'; color=[1 2], label="")
```


### Bayesian NN
```julia
using Lux
using Random
using ComponentArrays

rng = Random.default_rng()

rbf(x) = exp.(-(x.^2))
nn_init = Lux.Chain(
    Lux.Dense(2,2)
)
p_nn, st_nn = Lux.setup(rng, nn_init)

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
p = ComponentArray(;α = 1.5, γ = 3.0, p_nn)

prob_nn = ODEProblem(lotka_volterra_nn, u0, tspan, p)
init_sol = solve(prob_nn, alg; saveat)
# Plot simulation.
plot(init_sol)
```

## Machine learning approach

```julia
using SciMLSensitivity

function loss(p)
    predicted = solve(prob_nn,
                        alg; 
                        p, 
                        saveat,
                        abstol=1e-6, 
                        reltol = 1e-6,
                        sensealg = ForwardDiffSensitivity())

    l = 0.
    for i in 1:length(predicted)
        if all(predicted[i] .> 0)
            l += sum(abs2, log.(data_mat[:, i]) - log.(predicted[i]))
        end
    end

    return l, predicted


end

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

pinit = ComponentArray(;α = 1., γ = 3.0, p_nn)

callback(pinit, loss(pinit)...; doplot = true)
```



```julia
using Optimization
using OptimizationOptimisers
using OptimizationOptimJL

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, pinit)

res_ada1 = Optimization.solve(optprob, ADAGrad(0.1); callback = callback, maxiters = 500)
optprob = Optimization.OptimizationProblem(optf, res_ada1.minimizer)
res_ada2 = Optimization.solve(optprob, ADAGrad(0.01); callback = callback, maxiters = 300)

optprob = Optimization.OptimizationProblem(optf, res_ada2.minimizer)
res_lbfgs = Optimization.solve(optprob, LBFGS(); callback = callback, maxiters = 300)
```

> Does not seem to work! Let's try multiple shooting

```julia

TS = [1:25, 26:length(sol_true.t)]

function loss_ms(p)
    l = 0.

    for ts_idx in TS
        saveat = sol_true.t[ts_idx]
        u0 = sol_true.u[ts_idx[1]]
        predicted = solve(prob_nn,
                            alg; 
                            tspan = (saveat[1], saveat[end]),
                            u0,
                            p, 
                            saveat,
                            abstol=1e-6, 
                            reltol = 1e-6,
                            sensealg = ForwardDiffSensitivity())

        for i in 1:length(predicted)
            if all(predicted[i] .> 0)
                l += sum(abs2, log.(data_mat[:, ts_idx[i]]) - log.(predicted[i]))
            end
        end
    end
    l += sum(abs2, p.p_nn) # regularizer
    return l
end

callback = function (p, l)
    push!(losses, l)
    if length(losses)%10==0
      println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss_ms(x), adtype)


optprob1 = Optimization.OptimizationProblem(optf, pinit)
res_ada1 = Optimization.solve(optprob1, ADAM(0.1); callback = callback, maxiters = 1000)
optprob2 = Optimization.OptimizationProblem(optf, res_ada1.minimizer)
res_ada2 = Optimization.solve(optprob2, ADAM(0.01); callback = callback, maxiters = 1000)

final_sol = solve(prob_nn, alg; saveat, p=res_ada.minimizer)
plot(final_sol)
```


Now plotting functional response


```julia

function plot_func_resp(p, data)
    # plotting prediction of functional response
    u1 = range(minimum(data[1,:]), maximum(data[1,:]), length=100) 
    u2 = range(minimum(data[2,:]), maximum(data[2,:]), length=100) 
    u = hcat(u1,u2)

    myplot1 = plot(u2,
                    p_true.β .* u2; 
                    label="True func resp"
                    )
    func_resp = nn(u', p.p_nn)
    plot!(myplot1,
            u2,
            func_resp[1,:]; 
            color="#BBBBBB",
            label="Learnt func resp"
            )

    myplot2 = plot(u1,
                    p_true.δ .* u2; 
                    label="True func resp"
                    )
    plot!(myplot2,
            u1,
            func_resp[2,:]; 
            color="#BBBBBB",
            label="Learnt func resp"
            )

    myplot = plot(myplot1, myplot2)

    return myplot

end

plot_func_resp(res_ada.minimizer, data_mat)

```

### Resources
- https://turinglang.org/docs/tutorials/10-bayesian-differential-equations/
- 

