# # Housing data

# In this example, we create a linear regression model that predicts housing data. 
# It replicates the housing data example from the [Knet.jl readme](https://github.com/denizyuret/Knet.jl). 
# Although we could have reused more of Flux (see the MNIST example), the library's abstractions are very 
# lightweight and don't force you into any particular strategy.

# A linear model can be created as a neural network with a single layer. 
# The number of inputs is the same as the features that the data has. 
# Each input is connected to a single output with no activation function. 
# Then, the output of the model is a linear function that predicts unseen data. 

# ![singleneuron](img/singleneuron.svg)

# Source: [Dive into Deep Learning](http://d2l.ai/chapter_linear-networks/linear-regression.html#from-linear-regression-to-deep-networks)

# To run this example, we need the following packages:

using Flux
using Flux: gradient, train!
using Flux.Optimise: update!
using DelimitedFiles, Statistics


# ## Data 

# We create the function `get_processed_data` to load the housing data, and normalize it.

function get_processed_data(split_ratio=0.9)
    isfile("housing.data") ||
        download("https://raw.githubusercontent.com/MikeInnes/notebooks/master/housing.data",
            "housing.data")

    rawdata = readdlm("housing.data")'

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


# ## Model
# A Single dense layer with no activation

model = Dense(13=>1)

# ## Loss function

# The most commonly used loss function for Linear Regression is Mean Squared Error (MSE). 
# We define the MSE function as:

loss(model, x, y) = mean(abs2.(model(x) .- y));

# **Note:** An implementation of the MSE function is also available in 
# [Flux](https://fluxml.ai/Flux.jl/stable/models/losses/#Flux.Losses.mse).

# ## Train function
# Finally, we define the `train` function so that the model learns the best parameters (*W* and *b*):


function train()
    
    (x_train,y_train),(x_test,y_test) = get_processed_data()

    ## Training
    opt = Flux.setup(Adam(0.1), model)

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

cd(@__DIR__)
train()
