## push_back function for vectors (from the standard c++ library)
push_back is a function provided by the C++ Standard Library for vector objects. It's used to add an element to the end of the vector.

Here's a simple example:

std::vector<int> myVector;
myVector.push_back(1);  // myVector now contains {1}
myVector.push_back(2);  // myVector now contains {1, 2}
myVector.push_back(3);  // myVector now contains {1, 2, 3}

## void
Use void function to modify the state of an object without returning any value

## pow function
The pow function is a standard C++ function that raises its first argument to the power of its second argument. In this case, pow((target[i] - m_layers[outputLayerIndex][i].getValue()), 2) is calculating the square of the difference between the target output and the actual output of the net

## sigmoid function
The sigmoid function is a type of activation function that's commonly used in neural networks. It's defined as sigmoid(x) = 1 / (1 + exp(-x)), where exp(-x) is the exponential function. The sigmoid function takes a real-valued number and "squashes" it into range between 0 and 1. In the context of neural networks, it's used to convert the weighted sum of the inputs of a neuron into a value that can be used as input for the next layer. It's especially useful in the output layer of a binary classification network, where you want the output to be a probability between 0 and 1.