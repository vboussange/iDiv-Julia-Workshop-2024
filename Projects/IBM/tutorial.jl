###
### Simple tutorial for first-time users of Agents.jl
### https://juliadynamics.github.io/Agents.jl/stable/tutorial/
###

using Agents # bring package into scope

# make the space the agents will live in
space = GridSpace((20, 20)) # 20×20 grid cells

# make an agent type appropriate to this space and with the
# properties we want based on the ABM we will simulate
@agent struct Schelling(GridAgent{2}) # inherit all properties of `GridAgent{2}`
    mood::Bool = false # all agents are sad by default :'(
    group::Int # the group does not have a default value!
end

# define the evolution rule: a function that acts once per step on
# all activated agents (acts in-place on the given agent)
function schelling_step!(agent, model)
    # Here we access a model-level property `min_to_be_happy`
    # This will have an assigned value once we create the model
    minhappy = model.min_to_be_happy
    count_neighbors_same_group = 0
    # For each neighbor, get group and compare to current agent's group
    # and increment `count_neighbors_same_group` as appropriately.
    # Here `nearby_agents` (with default arguments) will provide an iterator
    # over the nearby agents one grid cell away, which are at most 8.
    for neighbor in nearby_agents(agent, model)
        if agent.group == neighbor.group
            count_neighbors_same_group += 1
        end
    end
    # After counting the neighbors, decide whether or not to move the agent.
    # If `count_neighbors_same_group` is at least min_to_be_happy, set the
    # mood to true. Otherwise, move the agent to a random position, and set
    # mood to false.
    if count_neighbors_same_group ≥ minhappy
        agent.mood = true
    else
        agent.mood = false
        move_agent_single!(agent, model)
    end
    return
end

# make a container for model-level properties
properties = Dict(:min_to_be_happy => 3)

# Create the central `AgentBasedModel` that stores all simution information
model = StandardABM(
    Schelling, # type of agents
    space; # space they live in
    agent_step! = schelling_step!, properties
)

# populate the model with agents by automatically creating and adding them
# to random position in the space
for n in 1:300
    add_agent_single!(model; group = n < 300 / 2 ? 1 : 2)
end

# run the model for 5 steps, and collect data.
# The data to collect are given as a vector of tuples: 1st element of tuple is
# what property, or what function of agent -> data, to collect. 2nd element
# is how to aggregate the collected property over all agents in the simulation
using Statistics: mean
xpos(agent) = agent.pos[1]
adata = [(:mood, sum), (xpos, mean)]
adf, mdf = run!(model, 5; adata)
adf # a Julia `DataFrame`
