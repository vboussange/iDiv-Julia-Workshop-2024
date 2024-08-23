####
#### Janzen-Connell Model
####
#### A simple forest ecosystem model to investigate the effect of pathogen-induced
#### density-dependent mortality on tree diversity.
####
####
#### (c) 2024 Daniel Vedder <daniel.vedder@idiv.de>
####     Originally published at https://github.com/veddox/jcm
####     Licensed under the terms of the MIT license.
####

### INSTRUCTIONS
###
### This file is intended for the IBM course project. Copy it and the associated file
### `ecology.jl` to another location and edit it there. Complete all `TODO` statements
### to get a working model.


module jcm

#TODO import the libraries Agents, GLMakie, Logging, and Random

#TODO Once you have the model working, vary these parameters to see how it changes the model
# behaviour (especially "pathogens", "transmission", and "neutral").
const settings = Dict("species" => 16,              # The number of species that will be created
                      "worldsize" => 1000,          # The length and breadth of the world in m
                      "runtime" => 1000,            # The number of updates the simulation will run
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
    
    #TODO create a square continuous space object of width `worldsize` called `space`
    #TODO create a standard ABM object called `model` that has Tree as its agent type,
    # and uses the space variable defined above, the agent_step! function, and the
    # previously defined RNG
    
    # create all species and add one tree from each species to the model
    @debug "Adding species."

    #TODO use the function `createspecies()` to create as many different species
    # as configured in the "species" setting (use a loop for this).
    #TODO add one individual of each species to the model, using the following code
    # (where `sp` is the species in question):
    # add_agent!(model, (0,0), sp, Int(round(sp.max_age/2)),
    #            sp.max_size, true, settings["pathogens"])

    @info "Model initialised."
    return model
end

"""
    openapp()

Set up a model and launch an interactive window to run it.
"""
function openapp()
    model = initworld()
    #TODO change the graph of the infected trees so that it shows the percentage of trees
    # that are infected rather than their absolute number.
    fig, abmobs = abmexploration(model;
                                 mlabels = ["Number of trees", "Infected trees", "Species"],
                                 mdata = [nagents, m->count(a->a.infected, allagents(m)),
                                          m->length(unique(map(a->a.species.id, allagents(m))))],
                                 agent_marker = a -> a.infected ? :diamond : :circle,
                                 agent_color = a -> a.species.id)
    @info "Created interface."
    return fig
end

if !isinteractive()
    openapp()
end

end
