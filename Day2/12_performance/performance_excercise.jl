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

precompile(converttocelsiusA, (Vector,))
precompile(converttocelsiusB, (Vector,))

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


#### EXERCISE 4: Euler 3

# (Taken from: https://projecteuler.net/problem=3)
# The prime factors of 13195 are 5, 7, 13, and 29.
# What is the largest prime factor of the number 600851475143?

## First attempt

isfactor(x, y) = iszero(x % y)

function isprime(n)
    n == 1 && return false
    for i in 2:(n-1)
        isfactor(n, i) && return false
    end
    return true
end

function factors(x)
    factorlist = []
    for i in 1:x
        isfactor(x, i) && push!(factorlist, i)
    end
    factorlist
end

function largestprimefactor(n)
    primefactors = []
    for f in factors(n)
        isprime(f) && push!(primefactors, f)
    end
    return maximum(primefactors)
end

@time largestprimefactor(13195)
@time largestprimefactor(600851475143)

## Second attempt

function isprime2(n::Int64)
    (n == 1 || iszero(n % 2)) && return false
    i = 3
    while i < n
        iszero(n % i) && return false
        i += 2
    end
    return true
end

function largestprimefactor2(n::Int64)
    for j in 1:n
        (iszero(n % j) && isprime2(div(n,j))) && return div(n,j)
    end
    return nothing
end

@time largestprimefactor2(13195)
@time largestprimefactor2(600851475143)
