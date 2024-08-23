####
#### Janzen-Connell Model
####
#### A simple forest ecosystem model to investigate the effect of pathogen-induced
#### density-dependent mortality on tree diversity. This file uses the Agents.jl
#### library. For a pure Julia implementation, see the folder `original`.
####
#### (c) 2024 Daniel Vedder <daniel.vedder@idiv.de>
####     Licensed under the terms of the MIT license.
####

###
### To run this code, open a Julia REPL in this folder and type `include("jcm.jl")`.
### Then launch the program using `jcm.openapp()`.
###

module jcm

const jcm_version = v"2.0"

using Agents,
    CairoMakie,
    GLMakie,
    Logging,
    Random

const settings = Dict("species" => 16,              # The number of species that will be created
                      "worldsize" => 1000,          # The length and breadth of the world in m
                      "runtime" => 500,             # The number of updates the simulation will run
                      "datafile" => "jcm_data.csv", # The name of the recorded data file
                      "datafreq" => 50,             # How long between data recordings?
                      "pathogens" => true,          # Include pathogens in the simulation?
                      "transmission" => 40,         # Pathogen infection radius
                      "neutral" => false,           # All species have identical trait values?
                      "verbosity" => Logging.Debug, # The log level (Debug, Info, Warn, Error)
                      "seed" => 0)                  # The seed for the RNG (0 -> random)

include("ecology.jl")

"""
    initworld()

Initialise the model world, doing necessary housekeeping and adding one tree of each species
to the landscape.
"""
function initworld()
    # initialise the RNG and the logger
    @debug "Setting up auxiliaries."
    if settings["seed"] == 0
        settings["seed"] = abs(rand(Random.RandomDevice(), Int))
    end
    rng = Random.Xoshiro(settings["seed"])
    global_logger(ConsoleLogger(stdout, settings["verbosity"]))
    # create the space and model object
    @debug "Creating model objects."
    space = ContinuousSpace((settings["worldsize"], settings["worldsize"]))
    model = StandardABM(Tree, space; agent_step!, rng)
    # create all species and add one tree from each species to the model
    @debug "Adding species."
    for s in 1:settings["species"]
        sp = createspecies(s)
        add_agent!(model, (0,0), sp, Int(round(sp.max_age/2)), sp.max_size, true,
                   settings["pathogens"])
    end
    @info "Model initialised."
    return model
end

"""
    runmodel()

Set up, run, and visualise the model.
"""
function runmodel()
    model = initworld()
    @time run!(model, settings["runtime"])
    CairoMakie.activate!()
    abmvideo(
        "janzen-connell.mp4", model;
        agent_marker = a -> a.infected ? :diamond : :circle,
        agent_color = a -> a.species.id,
        framerate = 20, frames = 150,
        title = "Janzen-Connell Model"
    )
    @info "Ran and visualised model."
end

"""
    openapp()

Set up a model and launch an interactive window to run it.
"""
function openapp()
    model = initworld()
    fig, abmobs = abmexploration(model;
                                 mlabels = ["Number of trees", "Infected trees", "Species"],
                                 mdata = [nagents, m->count(a->a.infected, allagents(m)),
                                          m->length(unique(map(a->a.species.id, allagents(m))))],
                                 agent_marker = a -> a.infected ? :diamond : :circle,
                                 #agent_size = a -> a.size, #FIXME gives an error for some reason
                                 agent_color = a -> a.species.id)
    @info "Created interface."
    return fig
end

if !isinteractive()
    openapp()
end

end
