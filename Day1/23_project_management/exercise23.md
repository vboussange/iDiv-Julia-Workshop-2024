# Exercise: starting a Julia research project

In this exercise, we'll explore how the relationship between the size of a tree and the amount of carbon it stores changes from one species to the other. We'll use a simple linear regression model to estimate the slope of the relationship, and compare it throughout the species investigated.

## Recommended steps

### Part 1: Set up your project

- Set up your project. You can do it by:
  - Forking/cloning the "computational notebooks" repo ([here](https://github.com/FellowsFreiesWissen/computational_notebooks/tree/file_struct2))
  - Download `set_kit.jl` only ([here](https://github.com/FellowsFreiesWissen/computational_notebooks/tree/file_struct2)) 
  - Write your own file structure (preferrably a function for it), for example:

```css
my_project/
├── src/
├── tests/
├── data/
├──── metadata/
├── results/
└── my-analysis /* this can be a "master" script (.jl) or a notebook (.ipynb) */
```

- Optional: If you see yourself using scripts and not notebooks, alter `set_kit.jl` to add a `run_code.sh` file to the current file structure it creates. (Or create it manually, of course.)

### Part 2: Package management

- Activate your environment, and install the following dependencies
  - `GLM`
  - `DataFrames`
  - `CSV`

- In the `src/regression_functions.jl`, create a function `linear_regression` that takes in a string corresponding to the location of a CSV file, and outputs the slope associated p-value of the linear regression between `tree_size` and `carbon_content` variables in the CSV.

>## Hints

- Here is a [cheat sheet](https://www.ahsmart.com/assets/pages/data-wrangling-with-data-frames-jl-cheat-sheet/DataFramesCheatSheet_v1.x_rev1.pdf) for the DataFrames package
> 
> The slope of a `GLM` model may be retrieved as   
>
> `slope = GLM.coeftable(model).cols[1][2]`
>
> And its p-value as 
> 
> `pval = GLM.coeftable(model).cols[4][2]`
    
- Create a unit test for `linear_regression` in `test/regression_functions.jl`
  - Use the `Test` module

- In `my-analysis.jl` file, write a Julia script that
  - loads the necessary packages
  - loads the functions in `src/regression_functions.jl` 
  - creates an empty `DataFrame`
    - ```julia
      df_results = DataFrame(species_name = [], slope = [], pval = [])
      ```
  - loops through the CSV files, runs for each a linear regression using `linear_regression`, and pushes the results to the `DataFrame`
    - Make sure to print some logging information, e.g. `println("processing ", csv_filename)`
  - exports the dataframe as a CSV file in the result folder

- The `CSV` package might not work for you. Here is an alternative to how you can save your CSVs:

```jl
out_filename = "path_to/output.csv"

# Open the file for writing
open(out_filename, "w") do file
    for row in data
        # Join the "cells" in the row with commas and write to the file
        println(file, join(row, ","))
    end
end
```

- **Optional**: Use a shell script to run `my-analysis.jl`

## Solutions

You may find the solutions in the `my_project_solutions/` folder.