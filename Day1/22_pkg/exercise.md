# Exercise: packages management


## Replicate an environment
Your objective is to run the Julia script `Day1/22_pkg/housing/housing.jl`.

- `activate` the environment in `Day1/22_pkg_and_project_management/housing/`
- `instantiate` 
- run the script!

## Adding a dependency
As you understood it, the script performs a linear regression with the deep learning framework Flux.jl.
Because the model is linear, and the features are normalized, you can interpret the model's weights to investigate each feature importance.

Your task is now to find the most important feature determining Boston housing price. 


> **Hint 1**
> The csv file is loaded with the lightweight base library DelimitedFiles. Use instead CSV.jl and DataFrames.jl, in order to obtain not only the numerical values of each features, but also the feature names.

> **Hint 2** `names(df)` throws the column names of a DataFrame


> **Hint 3** `model.weight[:]` throws the weights of the linear layer.