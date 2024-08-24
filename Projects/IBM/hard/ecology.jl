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

#TODO use `@agent struct` to create a Tree type. This should extend ContinuousAgent
# and have five fields: species, age, size, mature, infected.

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


#TODO write a function `createspecies()` that takes an integer variable `id` as input, and
# does the following two things:
#
# (1) Create a Pathogen object `p` with an infection radius set by the "transmission" setting.
# Then, create a species object `s` using the `id` variable passed to this function and
# the previously created pathogen `p`.
#
# (2) If we're running a neutral simulation (see `settings`), return the object `s`.
# Otherwise, create a new Species object, using the `vary()` function above to
# randomly shift each parameter value compared to the default.


"""
    random_nearby_position(agent, radius)

Return a random position within a given radius around the agent.
(Agents.jl has an inbuilt function for this, but it doesn't work with continuous space.)
"""
function random_nearby_position(a::AbstractAgent, r::Int)
    sx = a.pos.x + rand(-r:r)
    dy = Int(round(sqrt(abs(r^2-(sx-a.pos.x)^2))))
    sy = a.pos.y + rand(-dy:dy)
    if sx > settings["worldsize"] || sx <= 0 || sy > settings["worldsize"] || sy <= 0
        random_nearby_position(a, r) # if our position was out of bounds, try again
    else
        return (sx, sy)
    end
end

"""
    agent_step!(tree, model)

Carry out each of the four ecological processes (dispersal, competition, infection, growth)
for one tree in the model. This is the stepping function for Agents.jl.
"""
function agent_step!(tree::Tree, model::AgentBasedModel)
    # reproduction and dispersal

    #TODO If the tree is mature, produce offspring according to its species seed production.
    # To do so, generate a random location within its dispersal distance and then use the
    # following code: `add_agent!(location, model, (0,0), tree.species, 0, 0, false, false)`

    # competition for space
    for competitor in nearby_agents(tree, model, tree.size)
        # check for overlapping trees and kill the smaller one
        # (make sure both are still alive first to avoid errors)
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
