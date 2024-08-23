###
### Janzen-Connell Model
### (c) Daniel Vedder, MIT license
###

#TODO use `@kwdef struct` to create a `Pathogen` type that stores the parameters
# relating to pathogens (see `Species` below for a template).

"""
    Species

A struct storing all species-specific variables.
"""
@kwdef struct Species
    id::Int
    #TODO add all species-specific parameters here with their default values
    pathogen::Pathogen = Pathogen()
end

"""
    Tree

The core agent type of the model, a single tropical tree.
"""
@agent struct Tree(ContinuousAgent{2,Float64})
    species::Species
    age::Int
    size::Int
    mature::Bool
    infected::Bool
end

"""
    vary(i, p)

Vary a number i randomly  by up to +/- p% (helper function for createspecies()).
"""
function vary(i::Number; p::Int=50)
    i == 0 && return 0
    v = (p/100) * i
    s = i/100
    if isinteger(i)
        s = 1
        v = round(typeof(i), v)
    end
    n = i + rand(-v:s:v)
    return n
end

"""
    createspecies(id)

Create a new species. If settings["neutral"] is true, use the standard trait values, otherwise
vary them to create unique traits.
"""
function createspecies(id::Int)
    p = Pathogen(infection_radius=settings["transmission"])
    s = Species(id=id, pathogen=p)

    #TODO if we're running a neutral simulation (see `settings`), return the object `s`.
    # Otherwise, create a new Species object, using the `vary()` function above to
    # randomly shift each parameter value compared to the default.
end

"""
    agent_step!(tree, model)

Carry out each of the four ecological processes (dispersal, competition, infection, growth)
for one tree in the model. This is the stepping function for Agents.jl.
"""
function agent_step!(tree::Tree, model::AgentBasedModel)
    # reproduction and dispersal
    if tree.mature
        dx = tree.species.dispersal_distance
        for s in 1:tree.species.seed_production
            # find a random location within the species' dispersal distance
            sx = tree.pos.x + rand(-dx:dx)
            dy = Int(round(sqrt(abs(dx^2-(sx-tree.pos.x)^2))))
            sy = tree.pos.y + rand(-dy:dy)
            if sx >= -settings["worldsize"] && sx <= settings["worldsize"] &&
                sy >= -settings["worldsize"] && sy <= settings["worldsize"]
                add_agent!((sx, sy), model, (0,0), tree.species, 0, 0, false, false)
            end
        end
    end
    # competition for space
    for competitor in nearby_agents(tree, model, tree.size)
        # check for overlapping trees and kill the smaller one
        !(hasid(model, competitor) && hasid(model, tree)) && continue

        #TODO check whether the tree or its competitor is larger and kill the smaller one.
        # Print a debug message to say that it died because of competition.
    end
    # infection dynamics

    #TODO If the tree is infected, do two things:
    # (1) Check for conspecific neighbours within the infection radius of the pathogen.
    #     If the pathogen's infection rate is larger than a random number (use `rand(Float64)`
    #     to get this, infect the neighbour.
    # (2) If the pathogen's lethality is larger than a random number, remove the tree from
    #     the model and print a debug message saying that it died because of disease.

    # growth and aging

    #TODO if the tree is not yet mature, increase its size by its species-specific growth rate.
    # Once it reaches its maximum size, set its maturity to true. If the tree exceeds its maximum
    # age, remove it from the model and print a debug statement to say that the tree died
    # of old age. Finally, increment the tree's age by 1.
end
