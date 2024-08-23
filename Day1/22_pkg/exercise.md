# Exercise: packages management


## Adding packages to your base environment
Install `BenchmarkTools` and `Test` to your base environment.

## Replicate an environment
Try to run the Julia script `Day1/22_pkg/housing/housing.jl`. Of course, you'll need to instantiate the dependencies!

## Adding a dependency to an environment
As you understood it, `housing.jl` performs a linear regression with the deep learning framework Flux.jl.
Because the model is linear, and the features are normalized, you can interpret the model's weights to investigate each feature importance.

Find the most important features determining Boston housing price. 


> **Hint 1**
> The csv file is loaded with the lightweight base library DelimitedFiles. Use instead CSV.jl and DataFrames.jl, in order to obtain not only the numerical values of each features, but also the feature names.

> **Hint 2** `names(df)` throws the column names of a DataFrame


> **Hint 3** `model.weight[:]` throws the weights of the linear layer.

## Advanced: creating a package
- Create the package `MyPackage`.
- Implement a `greetings` function, that throws `"greetings!"`
- Create a new environment, and add your local package to it
- call `greetings` from this environment
  
> hint: you should export your function `greetings` within your package, so that it is publicly available.

## Advanced 2: write tests
- add a test folder to your package, and test that the function `greetings` returns `"greetings!"`
- run the test from the package manager
