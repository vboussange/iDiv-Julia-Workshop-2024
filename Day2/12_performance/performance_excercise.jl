###
### Learn more about performance profiling and optimisation in Julia.
###
### Daniel Vedder, August 2024
###

#### EXERCISE 1: converting to Celsius

# Imagine you are working with a data source that uses Fahrenheit,
# but your calculations require degrees Celsius. How do you best
# convert a large number of values from one to the other?

using Random, Profile

## Option A: the "naive" function using a for loop and building up the result vector as we go
function converttocelsiusA(fahrenheit::Vector)
    celsius = []
    for f in fahrenheit
        push!(celsius, (f-32)*(5/9))
    end
    celsius
end

## Option B: using an anonymous function and `map`
function converttocelsiusB(fahrenheit::Vector)
    celsius = f -> (f-32)*(5/9)
    map(celsius, fahrenheit)
end

## create a random vector of 1 million numbers
fahrenheits = rand(0:250, 1000000)

## run the functions once to make sure they are compiled
## (the first execution is always much slower!)
converttocelsiusA(fahrenheits);
converttocelsiusB(fahrenheits);

## Note: adding a semi-colon after a line of code suppresses
## the REPL output - in this case, we don't want to see the array
## that is produced, we're just interested in the runtime

## then use the @time macro to find out how efficient each function is
@time(converttocelsiusA(fahrenheits));
@time(converttocelsiusB(fahrenheits));

## Option A2: uses a loop like the first function, but preallocates the vector to avoid using push!()
function converttocelsiusA2(fahrenheit::Vector)
    celsius = Vector{Float64}(undef, length(fahrenheit))
    for f in eachindex(fahrenheit)
        celsius[f] = (fahrenheit[f]-32)*(5/9)
    end
    celsius
end

converttocelsiusA2(fahrenheits);

@time(converttocelsiusA2(fahrenheits));


#### EXERCISE 2: plant reproduction

## A simple plant type
struct Plant
    id::Int
    size::Int
    seeds::Int
end

## Initialise 10 million plants
biome = [Plant(p, rand(1:10000), 1000) for p in 1:10000000]

## Return the indices of the largest plants
function findlargest(plants::Vector{Plant})
    maxsize = 0
    ids = []
    for p in eachindex(plants)
        if plants[p].size > maxsize
            maxsize = plants[p].size
            ids = [p]
        elseif plants[p].size == maxsize
            push!(ids, p)
        end
    end
    return ids
end

## Return a vector filled with one new plant for every seed produced by the input plants
function reproduce!(plants::Vector{Plant})
    offspring = []
    for p in plants
        for s in 1:p.seeds
            push!(offspring, Plant(length(plants)+length(offspring), p.size, p.seeds))
        end
    end
    offspring
end

## Find the largest plants and let only these reproduce
function updateplants(plants::Vector{Plant})
    largest = findlargest(plants)
    reproduce!(plants[largest])
end

## Finally, profile our mini-model and show the results
Profile.clear()
@profile updateplants(biome);
Profile.print(format=:flat)
