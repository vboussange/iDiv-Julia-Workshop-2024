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

| Time              | Day 1 :  Introduction to Julia                                                                                                                                                                                                                                                                                                                                                                                                                                   | Day 2 : Advanced introduction to Julia, projects kick-off                                                                                                                                                                                                                                                                                                                                                                        | Day 3 : Project-oriented day                        |
|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|
| **8:50**          | Arrival at Beehive, iDiv                                                                                                                                                                                                                                                                                                                                                                                                                                         |                                                                                                                                                                                                                                                                                                                                                                                                                                  |                                                     |
| **9:00 - 10:30**  | - 🎤 [Greetings to the Workshop](Day1/11_welcome.ipynb) (@Victor) **9:00-9:15** <br> - 🎤 [Overview of the Julia programming language](Day1/12_julia-overview/Session_1_Overview_of_Julia.pdf) (@Daniel) **9:15-9:45** <br> - 🎤💻 [**Practical**: your first lines of Julia code](Day1/13_julia-practical-on-jupyter/13_julia-practical-on-jupyter.ipynb) (@Daniel and @Victor) **9:45-10:30**                                                                            | - 🎤  [Why I like Julia](Day2/11_whyjulia-FM.pdf) (@Francesco) **9:00-9:15** <br> - 🎤 💻 [Geospatial data handling](Day2/11_geodata/11_geodata.ipynb) (@Francesco) **9:15-9:45** <br> - 🎤 💻 [Performant Julia code and profiling](Day2/12_performance/12_performance.md) (@Daniel) **9:45-10:30** <br>                                                                                                                                               | - 💻 Project session                                |
| **10:30 - 11:00** | **Coffee break**                                                                                                                                                                                                                                                                                                                                                                                                                                                 | **Coffee break**                                                                                                                                                                                                                                                                                                                                                                                                                 | **Coffee break**                                    |
| **11:00 - 12:30** | - 🎤 [Why I like Julia](Day1/21_why-i-like-julia-VB/21_why-i-like-julia-VB.ipynb) (@Victor) **11:00-11:15** <br> - 💻🎤 [Package management: `Pkg.jl`](Day1/22_pkg/Pkg.ipynb) (@Victor) **11:15-11:30** <br>   - 💻[Exercise: managing dependencies](Day1/22_pkg/exercise.md) <br> - 💻🎤 [VS code workflow and remote development](Day1/24_vscode_remote_dev/vs_code_workflow.ipynb) (@Victor) **12:15-12:30**                  | - 🎤 💻 [Parallel computing](Day2/13_parallel_computing/parallel_computing.ipynb) (@Victor) **11:00-11:15** <br> - 🎤 [Interface with Python, R, MATLAB](Day2/21_interface/interface.ipynb) (@Victor) **11:15-11:30** <br> - 🎤 Track Introductions (@Victor, @Daniel, @Ludmilla, @Francesco, @Oskar) **12:00-12:30** <br> - [Ecologically informed NN](#victors-track-ecologically-informed-neural-network) <br> - [Individual-based modelling](#daniels-track-individual-based-modelling-tbd) <br> - [Coding the Game of Life](#ludmilla-and-oskars-track-coding-the-game-of-life) | - 💻 Project session                                |
| **12:30 - 13:30** | **Lunch**                                                                                                                                                                                                                                                                                                                                                                                                                                                        | **Lunch**                                                                                                                                                                                                                                                                                                                                                                                                                        | **Lunch**                                           |
| **13:30 - 15:00** | - 🎤 [Why I like Julia](Day1/Session_2_Why_I_like_Julia_DV.pdf) (@Daniel) **13:30-13:45**  <br> - 💻🎤[Good (Julia) practices](Day1/23_project_management/good_practices.ipynb) (@Ludmilla) **13:45-14:45** <br> - 💻[Exercise: Develop your first Julia project](Day1/23_project_management/exercise23.md) <br> - 🎤 [Overview of the package ecosystem](Day1/31_julia_ecosystem/31_julia_ecosystem.ipynb) (@Victor, @all) **14:45-15:00**                                                        | **Track-specific topics** <br>   - 🎤 Time series modelling (@Francesco) <br>   - Hands-on exercises <br>   - 🎤 💻 [Mechanistic inference](Day2/31_mechanistic_inference/mechanistic_inference_ex.md) [[solutions](Day2/31_mechanistic_inference/mechanistic_inference_sol.md)] (@Victor) <br>   - Hands-on exercises <br>   - 🎤 Individual-based modelling (@Daniel)                                                                                                                                                                                              | - 💻 Project session                                |
| **15:00 - 15:30** | **Coffee break**                                                                                                                                                                                                                                                                                                                                                                                                                                                 | **Coffee break**                                                                                                                                                                                                                                                                                                                                                                                                                 | **Coffee break**                                    |
| **15:30 - 17:00** | <br> - 🎤 💻 [loading and saving data, `DataFrames`, broadcasting](Day1/32_dataframe_tuto/32_dataframe_tuto.ipynb) (@Victor) **15:30-15:45** <br> - 💻 [Hands-on exercises](Day1/32_dataframe_tuto/33_dataframe_exercises.ipynb) [[solutions](Day1/32_dataframe_tuto/33_dataframe_exercises_with_sols.ipynb)] **15:45-16:00** <br> - 🎤 💻 [Plotting and visualisation](Day1/33_plotting/33_plotting.ipynb) (@Victor) **16:00-16:15** <br> - 💻 [Hands-on exercises continued](Day1/32_dataframe_tuto/33_dataframe_exercises.ipynb) [[solutions](Day1/32_dataframe_tuto/33_dataframe_exercises_with_sols.ipynb)] **16:15-17:00** | - 💻 Project session                                                                                                                                                                                                                                                                                                                                                                                                             | - 💻 Project session <br> - 🎤 Wrap-up and feedback |
| **17:00 - 🌙**    | **🍻 Apéro**                                                                                                                                                                                                                                                                                                                                                                                                                                                     |                                                                                                                                                                                                                                                                                                                                                                                                                                  |                                                     |



# Projects
----
## Victor's track: Ecologically-informed neural network

See project description [here]((Projects/ecologically-informed-nn/ecologically-informed-nn.md)).
#### Suggested reading
- [Introduction to Scientific Machine Learning through Physics-Informed Neural Networks](https://book.sciml.ai/notes/03-Introduction_to_Scientific_Machine_Learning_through_Physics-Informed_Neural_Networks/)
- [Lagergren et al. (2020)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008462)
----
## Daniel's track: Individual-based modelling (tbd)

#### Suggested reading

- [Datseris et al. (2022). Agents.jl description](https://doi.org/10.1177/00375497211068820)
- [Günther et al. (2021) Sample use of Agents.jl](https://doi.org/10.3390/land10121366)

---
## Ludmilla and Oskar's track: [Coding the game of life](Projects/game-of-life/42_game_of_life.md)



----

# Additional resources
- [Julia official list of tutorials](https://julialang.org/learning/tutorials/)
- [Introduction to Julia, JuliaCon 2022](https://github.com/storopoli/Julia-Workshop), and the [YouTube video](https://www.youtube.com/watch?v=uiQpwMQZBTA) (3 hours)
- [Introductory Julia tutorial by Martin D. Maas](https://www.matecdev.com/posts/julia-tutorial-science-engineering.html)
- [Julia Workshop for Data Science](https://crsl4.github.io/julia-workshop/session1-get-started.html)

# Acknowledgments

The workshop materials are based on numerous resources, which have been indicated in the different sections.

<!-- We thank [WSL Biodiversity center](https://www.wsl.ch/en/about-wsl/organisation/programmes-and-initiatives/wsl-biodiversity-center.html), the [Ecosystem and Landscape Evolution](https://ele.ethz.ch) and [The Laboratory of Hydraulics, Hydrology and Glaciology](https://vaw.ethz.ch/en/) for supporting and funding this workshop. -->

# Contact

<!-- If you have any questions or feedback, feel free to contact the main authors of this workshop, @vboussange and 3. You can also create a pull request. -->
