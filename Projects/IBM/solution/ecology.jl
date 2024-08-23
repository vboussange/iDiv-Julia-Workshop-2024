###
### Janzen-Connell Model
### (c) Daniel Vedder, MIT license
###

"""
    Pathogen

A struct storing the variables for a species-specific pathogen.
"""
@kwdef struct Pathogen
    infection_rate::Float64 = 0.8
    infection_radius::Int = 0
    lethality::Float64 = 0.05
end

"""
    Species

A struct storing all species-specific variables.
"""
@kwdef struct Species
    id::Int
    max_age::Int = 150
    max_size::Int = 25
    growth_rate::Int = 2
    seed_production::Int = 10
    dispersal_distance::Int = 200
    pathogen_resistance::Float64 = 0
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
    if settings["neutral"]
        return s
    else
        max_age = vary(s.max_age)
        max_size = vary(s.max_size)
        growth_rate = vary(s.growth_rate)
        seed_production = vary(s.seed_production)
        dispersal_distance = vary(s.dispersal_distance)
        pathogen_resistance = vary(s.pathogen_resistance)
        return Species(id, max_age, max_size, growth_rate, seed_production,
                       dispersal_distance, pathogen_resistance, p)
    end
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
        if competitor.size > tree.size
            @debug "Tree $(tree.id) died because of competition."
            remove_agent!(tree, model)
            return
        else
            @debug "Tree $(tree.id) died because of competition."
            remove_agent!(competitor, model)
        end
    end
    # infection dynamics
    if tree.infected
        pathogen = tree.species.pathogen
        for neighbour in nearby_agents(tree, model, pathogen.infection_radius)
            # infect nearby conspecifics with a certain probability
            if neighbour.species.id == tree.species.id && pathogen.infection_rate > rand(Float64)
                neighbour.infected = true
            end
        end
        if pathogen.lethality > rand(Float64)
            @debug "Tree $(tree.id) died because of disease."
            remove_agent!(tree, model)
            return
        end
    end
    # growth and aging
    if !tree.mature
        tree.size += tree.species.growth_rate
        tree.size >= tree.species.max_size && (tree.mature = true)
    elseif tree.age >= tree.species.max_age
        @debug "Tree $(tree.id) died because of old age."
        remove_agent!(tree, model)
        return
    end
    tree.age += 1
end
