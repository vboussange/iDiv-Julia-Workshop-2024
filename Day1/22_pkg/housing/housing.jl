# # Housing data

# In this example, we create a linear regression model that predicts the price of a house given input features (such as the size of a house). 

# We use a linear model, that we create with the deep learning framework Flux.jl.
# A linear model can be created as a neural network with a single layer. 
# The number of inputs is the same as the features that the data has. 
# Each input is connected to a single output with no activation function. 
# Then, the output of the model is a linear function that predicts unseen data. 

# ![singleneuron](img/singleneuron.svg)

# Source: [Dive into Deep Learning](http://d2l.ai/chapter_linear-networks/linear-regression.html#from-linear-regression-to-deep-networks)

import Flux
import Flux: gradient, train!
import Flux.Optimise: update!
using DelimitedFiles, Statistics

cd(@__DIR__)

# `get_processed_data` to .

"""
    get_processed_data(split_ratio=0.9)

load the housing data, and normalize it.
"""
function get_processed_data(split_ratio=0.9)
    isfile("housing.csv") ||
        download("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
            "housing.csv")

    rawdata = readdlm("housing.csv", ',', skipstart=1)'

    ## The last feature is our target -- the price of the house.

    x = rawdata[1:13,:]
    y = rawdata[14:14,:]

    ## Normalise the data
    x = (x .- mean(x, dims = 2)) ./ std(x, dims = 2)

    ## Split into train and test sets
    split_index = floor(Int,size(x,2)*split_ratio)
    x_train = x[:,1:split_index]
    y_train = y[:,1:split_index]
    x_test = x[:,split_index+1:size(x,2)]
    y_test = y[:,split_index+1:size(x,2)]

    train_data = (x_train, y_train)
    test_data = (x_test, y_test)

    return train_data,test_data
end

# ## Loss function
loss(model, x, y) = mean(abs2.(model(x) .- y));


"""
    train!(model)
Trains a Flux model.
"""
function train!(model)
    
    (x_train,y_train),(x_test,y_test) = get_processed_data()

    ## Training
    opt = Flux.setup(Flux.Adam(0.1), model)

    for epoch in 1:500
        if epoch % 50 == 0
            println("epoch = $epoch")
            err = loss(model, x_test, y_test)
            @show err
        end

        l, (grad,) = Flux.withgradient(model) do m
            loss(m, x_train, y_train)
        end
        Flux.update!(opt, model, grad)
    end    

end


# ## Model
# A Single dense layer with no activation
model = Flux.Dense(13=>1)
train!(model)