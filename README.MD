# Objective:
Making a basic feeding-forward nerual net in c++

A feed-forward neural network is a type of artificial neural network where the connections between the nodes do not form a cycle. This is different from recurrent neural networks, which have cycles.

# Steps:

The initialization will set up the neural network with a specified number of inputs, hidden layer neurons, and outputs.

The feed-forward step will take our inputs and push them through the network to get its prediction.

Back-propagation will calculate the error and distribute it back through the network. This is where we calculate how much each neuron in the network contributed to the total error.

The training loop will run a specified number of epochs, or rounds, of feed-forward prediction and back-propagation adjustment.

The network will adjust its weights according to the error from the output of prediction after back-propagation.

## "Double" data type:
The double is a data type in C++. It's used to store decimal numbers (real numbers) with double precision. In other words, it can store very large (or very small) numbers with a high degree of precision. The exact size of a double can vary, but it's usually 8 bytes, which gives it a precision of about 15 decimal places.

## Why use a vector to represent layers:
Yes, a layer in a neural network consists of multiple neurons. We represent a layer as a vector of Neuron objects because a vector is a dynamic array-like data structure provided by the C++ Standard Library that can hold any type of items, in this case, Neuron objects.

Here are a few reasons why we use a vector to represent a layer:

    Dynamic Size: Unlike arrays, vectors in C++ can change their size dynamically at runtime. This is useful because we might not know how many neurons a layer will have until runtime.

    Convenience: The C++ Standard Library provides many built-in functions that make it easy to work with vectors. For example, we can easily add a neuron to a layer, remove a neuron, find out how many neurons are in a layer, etc.

    Efficiency: Vectors are implemented as dynamic arrays, so they are quite efficient. Accessing elements in a vector is as fast as accessing elements in an array.

    Encapsulation: By storing Neuron objects in a vector, we can keep all the data related to a layer (i.e., its neurons) together in one place. This makes the code easier to understand and maintain.

So, in our Layer class, each instance of the class represents a layer in the neural network, and the vector of Neurons represents the neurons in that layer.

## Backpropagation algorithm
The backpropagation algorithm is used in a neural network to adjust the weights of the neurons in order to minimize the error expressed by the cost function. The goal is to find the set of weights that minimize the cost and thus the error

## Cost function (aka loss function)
Cost/loss function - delta between predicted value and actual value produced by the nerual net. Used to optimize paramateres, such as weight, to minimize the cost.
Example: Mean Squared Average. MSE = (1/n) * Σ(actual - prediction)^2

## Weight
A weight in a neural network is a parameter that determines the strength of the influence between two neurons. When an input is fed into the network, each input is multiplied by a weight, and these products are then summed up and passed through an activation function to produce the output. 

