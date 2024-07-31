# Practical introduction to Julia for biodiversity research

This repository contains materials for the [yDiv Julia Workshop 2024]() **Practical introduction to Julia for biodiversity research**, held on the 27th to 29th of August 2024 for the yDiv Graduate School & Postdoc Programme at iDiv, Leipzig, Germany. It should contain useful resources and guidelines to curious ecologists who want to get an overview or get started with the Julia language. 

# Content

The repository is organized by days and sessions. Please refer to the [Program](#program) section to navigate within the repo.

# Requirements

To follow the workshop materials, you need to have the following software installed on your computer:
- Julia
- Jupyter

Additionally, we recommend to use
- VSCode

as an IDE, together with the Julia extension.

Please refer to the [installation instructions](Misc/installation_instructions.md) for further information on how to proceed.


# Usage

To use the workshop materials, clone this repository to your local machine:

```sh
git clone https://github.com/vboussange/iDiv-Julia-Workshop-2024.git
```

# Program
How you should read this program
- 🎤 : Talk
- 💻: Hands-on exercises
- 🎤💻: Interactive session
- 
## Day 1: Introduction to Julia, hands on exercises

**8:50** Arrival at Beehive, iDiv (ground floor, opposite the main entrance).

### Morning session 1
**9:00 - 10:30**

- 🎤 [Greetings to the Workshop](Day1/11_welcome.ipynb) (@Victor) **9:00-9:15**
- 🎤 [Overview of the Julia programming language](Day1/12_julia-overview/12_julia-overview.pdf) (@Daniel) **9:15-9:45**
- 🎤💻 [**Practical**: your first lines of Julia code](Day1/13_julia-practical-on-jupyter/13_julia-practical-on-jupyter.ipynb) (@Daniel and @Victor) **9:45-10:30**
  - Basic syntax and data types
  - Control flow and conditional statements

### Coffee break
**10:30 - 11:00**

### Morning session 2
**11-12:30**
- 🎤 [Why I like Julia](Day1/21_why-i-like-julia-VB/21_why-i-like-julia-VB.ipynb) (@Victor) **11:00-11:15**

- Package and project management (@Victor) **11:15-11:45**
  - 💻🎤 [Introduction to `Pkg.jl`](Day1/22_pkg/Pkg.ipynb)
  - 💻[Exercise: activate an environment and run the associated script](Day1/22_pkg/exercise1.md)
  - 💻[Exercise: Develop your first Julia project - Part 1](Day1/22_pkg_and_project_management/exercise2.md)

- 💻🎤 [VS code workflow and remote development](Day1/23_vscode_remote_dev/vs_code_workflow.ipynb) (@Victor) **11:45-12:30**
  - VSCode editor
  - Remote development

### Lunch
**12:30 - 13:30**

### Afternoon session 3
**13:30 - 15:00**
- 🎤 Why I like Julia (@Daniel)

- 🎤 [Overview of the package ecosystem](Day1/31_julia_ecosystem/31_julia_ecosystem.ipynb) (@Victor) **13:30-13:45**

- 🎤 💻 [loading and saving data, `DataFrames`, broadcasting](Day1/32_dataframe_tuto/32_dataframe_tuto.ipynb) (@Victor) **13:45-14:05**


- 💻 [Hands-on exercises](Day1/32_dataframe_tuto/33_dataframe_exercises.md) **14:05-14:30**


- 🎤 💻 [Plotting and visualisation](Day1/33_plotting/33_plotting.ipynb) (@Victor) **14:30-14:45**

- 💻 [Hands-on exercises continued](Day1/32_dataframe_tuto/33_dataframe_exercises.md) **14:45-15:00**

### Coffee break
**15:00 - 15:30**

### Afternoon session 4
- 🎤 [Why I like Julia](https://github.com/luraess/WSLJulia2023) (@Ludmilla) **15:30-15:45**
- 💻 [Coding the game of life](Day1/42_game_of_life/42_game_of_life.md) **15:55-17:00**


### 🍻 Apéro
**17:00-🌙**

## Day 2: Advanced introduction to Julia, hands-on exercises

The goal of this second-day workshop is 

### Morning session 1

- 🎤 💻 [Geospatial data handling](Day2/11_geodata/11_geodata.ipynb) (@Francesco) **9:00-9:30**
  - [Hands-on exercises](Day2/11_geodata/11_geodata-ex.ipynb)

- 🎤 💻 [Performant Julia code and profiling](Day2/12_performance/12_performance.md) (@Daniel) **9:30-10:00**
  - Introduction
  - Hands-on exercises

- 🎤 💻 [Parallel computing](Day2/13_parallel_computing/parallel_computing.ipynb) (@Victor) **10:00-10:30**
  - Multithreading
  - Multiprocessing
  - [Exercise: Parallelizing the Game of Life](Day2/13_parallel_computing/parallel_computing_exercise_with_sol.ipynb)

### Coffee break
**10:30-11:00**

### Morning session 2
- 💻🎤[Julia project management](Day2/22_pkg_and_project_management/practical_guideline.ipynb) (@Victor) **11:00-11:30**
  - 💻[Exercise: Develop your first Julia project - Part 2](Day1/22_pkg_and_project_management/exercise2.md) 

- 🎤 [Interface with Python, R, MATLAB](Day2/21_interface/interface.ipynb) (@Victor) **11:30-12:00**

- 🎤 Track Introductions (@Victor, @Daniel, @Ludmilla, @Francesco) **12:00-12:30**
  - [Spatio-temporal biodiversity modelling track](#biodiversity-track)
  - [Individual-based modelling track](#daniels-track)

### Lunch
**12:30 - 13:30**

### Afternoon session 3
- 🎤 Advanced Julia constructs: type system, functional programming, metaprogramming (@Daniel)
  - Hands-on exercises
- 🎤 Good (Julia) practices (@Ludmilla)

### Coffee break 
**15:00-15:30**

### Afternoon session 4
- Track-specific topics
  - 🎤 Time series modelling (@Francesco)
    - Hands-on exercises
  - 🎤 Differentiable programming (@Victor)
    - Hands-on exercises
  - 🎤Individual-based modelling (@Daniel)

## Day 3: Project-oriented day

The goal of this third-day workshop is to provide participants with an opportunity to deepen their knowledge of the Julia programming language through biodiversity and glaciology-related projects. 

### Morning session 1
- 💻 Project session

### Coffee break 
**10:30-11:00**

### Morning session 2
- 💻 Project session

### Lunch
**12:30 - 13:30**

### Afternoon session 3
- 💻 Project session

### Coffee break 
**15:00-15:30**

### Afternoon session 4

- 💻 Project session

- 🎤 Wrap-up and feedback


# Projects
----
**Victor's track**: [Ecologically-informed neural network](Projects/Victor/ecologically-informed-nn/ecologically-informed-nn.md)

#### Suggested reading
- [Introduction to Scientific Machine Learning through Physics-Informed Neural Networks](https://book.sciml.ai/notes/03-Introduction_to_Scientific_Machine_Learning_through_Physics-Informed_Neural_Networks/)
- [Lagergren et al. (2020)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008462)
----
- **Daniel's track**: Individual-based modelling (tbd)

#### suggested reading

- [Datseris et al. (2022). Agents.jl description](https://doi.org/10.1177/00375497211068820)
- [Günther et al. (2021) Sample use of Agents.jl](https://doi.org/10.3390/land10121366)

---
**Ludmilla's track**

---
**Oskar's track**

----

# Additional resources
- [Julia official list of tutorials](https://julialang.org/learning/tutorials/)
- [Introduction to Julia, JuliaCon 2022](https://github.com/storopoli/Julia-Workshop), and the [YouTube video](https://www.youtube.com/watch?v=uiQpwMQZBTA) (3 hours)
- [Introductory Julia tutorial by Martin D. Maas](https://www.matecdev.com/posts/julia-tutorial-science-engineering.html)
- [Julia Workshop for Data Science](https://crsl4.github.io/julia-workshop/session1-get-started.html)
- 
# Acknowledgments

The workshop materials are based on numerous resources, which have been indicated in the different sections.

<!-- We thank [WSL Biodiversity center](https://www.wsl.ch/en/about-wsl/organisation/programmes-and-initiatives/wsl-biodiversity-center.html), the [Ecosystem and Landscape Evolution](https://ele.ethz.ch) and [The Laboratory of Hydraulics, Hydrology and Glaciology](https://vaw.ethz.ch/en/) for supporting and funding this workshop. -->

# Contact

<!-- If you have any questions or feedback, feel free to contact the main authors of this workshop, @vboussange and 3. You can also create a pull request. -->
