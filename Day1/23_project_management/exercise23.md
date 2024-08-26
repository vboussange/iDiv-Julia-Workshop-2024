# Exercise: starting a Julia research project

In this exercise, you will download some data and play around with setting up your project and writing simple functions/scripts to analyse it.

## Recommended steps

### Part 1: Set up your project

- Set up your project. You can do it by:
  - Forking/cloning the "computational notebooks" repo ([here](https://github.com/FellowsFreiesWissen/computational_notebooks/tree/file_struct2))
  - Download `set_kit.jl` only ([here](https://github.com/FellowsFreiesWissen/computational_notebooks/tree/file_struct2)) 
  - Write your own file structure, for example:

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
  - `Plots`
  - `GLM`
  - `DataFrames`
  - `CSV`

### Get the data

Download 5 years of plant phenological data from the PhenObs project:

- [Data from 2017-2018](https://doi.org/10.25829/idiv.1877-4-3160)
- [Data from 2019](https://doi.org/10.25829/idiv.3519-a6r94f)
- [Data from 2020](https://doi.org/10.25829/idiv.3535-6j8cmx)
- [Data from 2021](https://doi.org/10.25829/idiv.3536-o94ra8)
- [Data from 2022](https://doi.org/10.25829/idiv.3550-m3qf86)

**Important**: Don't forget to store the metadata and to check it's licenses.

### Plot and analyse it

Here are some prompts from your to try:

## Solutions

You may find the solutions in the `my_project_solutions/` folder.