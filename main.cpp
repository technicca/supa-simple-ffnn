#include <vector>
#include <random>
#include <math.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include "utils.hpp"

class Neuron {
private:
    double m_value; // the neuron's value
    double m_activation; // the neuron's activation
    double m_derived; // the derivative of the activation function
    std::vector<double> m_weights; // the weights of the connections from this neuron
    double m_gradient; // gradient for backpropagation

public:
    // Default constructor
    Neuron() : m_value(0.0), m_activation(0.0), m_derived(0.0), m_gradient(0.0) {}

    // Constructor
    Neuron(int numWeights) : m_value(0.0), m_activation(0.0), m_derived(0.0), m_weights(numWeights), m_gradient(0.0) {
      // Initialize weights with random values
      std::random_device rd;
      std::mt19937 gen(rd());
      std::normal_distribution<double> dist(0.0, 0.1); // Change here

      for (double& weight : m_weights) {
          weight = dist(gen);
      }
  }


    // Copy constructor
    Neuron(const Neuron& other) : m_value(other.m_value), m_activation(other.m_activation), m_derived(other.m_derived), m_gradient(other.m_gradient), m_weights(other.m_weights) {}

    // Getters and setters...
    void setValue(double value) { m_value = value; }
    double getValue() const { return m_value; }

    void setActivation(double activation) { m_activation = activation; }
    double getActivation() const { return m_activation; }

    void setDerived(double derived) { m_derived = derived; }
    double getDerived() const { return m_derived; }

    void setWeight(int index, double weight) {
        if (index >= 0 && index < m_weights.size()) {
            m_weights[index] = weight;
        }
    }
    double getWeight(int index) const {
        if (index >= 0 && index < m_weights.size()) {
            return m_weights[index];
        }
        return 0.0; // or some other default value
    }

    void setGradient(double gradient) { m_gradient = gradient; }
    double getGradient() const { return m_gradient; }
};

class Layer {
private:
    std::vector<Neuron> m_neurons; // neurons in this layer

public:
    // Constructor
    Layer(int size, int nextSize) : m_neurons(size, Neuron(nextSize)) {}


    // Getters and setters for the neurons
    Neuron& getNeuron(int index) {
        if (index >= 0 && index < m_neurons.size()) {
            return m_neurons[index];
        }
        return m_neurons[0]; // or some other default value
    }
    void setNeuron(int index, const Neuron& neuron) {
        if (index >= 0 && index < m_neurons.size()) {
            m_neurons[index] = neuron;
        }
    }
     int size() const {
        return m_neurons.size();
    }
};

class NeuralNetwork {
private:
    std::vector<Layer> m_layers; // layers in the network

public:
    // Constructor
  NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
    // create input layer
    m_layers.push_back(Layer(inputSize, hiddenSize));

    // create hidden layer
    m_layers.push_back(Layer(hiddenSize, outputSize));

    // create output layer
    m_layers.push_back(Layer(outputSize, 0)); // output layer doesn't need to connect to another layer
}


// Training functions:

void feedForward(const std::vector<double>& input) {
    // Set the input layer
    for (int i = 0; i < input.size(); ++i)
        m_layers[0].getNeuron(i).setValue(input[i]);

    // Forward propagation till the last hidden layer
    for (int i = 1; i < m_layers.size() - 1; ++i) { // we removed the == operator
        for (int j = 0; j < m_layers[i].size(); ++j) {
            double activation = 0.0;
            for (int k = 0; k < m_layers[i - 1].size(); ++k)
                activation += m_layers[i - 1].getNeuron(k).getValue() * m_layers[i - 1].getNeuron(k).getWeight(j);
            m_layers[i].getNeuron(j).setValue(sigmoid(activation));
        }
    }

    // For the output layer let's apply softmax activation
    std::vector<double> outputValues;
    for (int j = 0; j < m_layers.back().size(); ++j) {
        double activation = 0.0;
        for (int k = 0; k < m_layers[m_layers.size() - 2].size(); ++k)
            activation += m_layers[m_layers.size() - 2].getNeuron(k).getValue() * m_layers[m_layers.size() - 2].getNeuron(k).getWeight(j);
        outputValues.push_back(activation);
    }
    softmax(outputValues);
}

void backPropagate(const std::vector<double>& target){
    // Calculate gradient for output layer
    int outputLayerIndex = m_layers.size() - 1;
    for(int i = 0; i < m_layers[outputLayerIndex].size(); ++i){
        double output = m_layers[outputLayerIndex].getNeuron(i).getValue();
        double targetVal = target[i];
        double gradient = (output - targetVal) * sigmoidDerivative(output);
        m_layers[outputLayerIndex].getNeuron(i).setGradient(gradient);
    }

    // Calculate gradient for hidden layers
    for(int i = outputLayerIndex - 1; i >= 0; --i){
        for(int j = 0; j < m_layers[i].size(); ++j){
            double error = 0.0;
            for(int k = 0; k < m_layers[i+1].size(); ++k)
                if (i != m_layers.size() - 2) // Check if next layer is not the output layer
                    if (k < m_layers[i+1].size()) {
    error += m_layers[i+1].getNeuron(k).getGradient() * m_layers[i].getNeuron(j).getWeight(k);
}
            double gradient = error * sigmoidDerivative(m_layers[i].getNeuron(j).getValue());
            m_layers[i].getNeuron(j).setGradient(gradient);
        }
    }
}

double calculateError(const std::vector<double>& target){
    double totalError = 0;
    int outputLayerIndex = m_layers.size() - 1;
    for(int i = 0; i <m_layers[outputLayerIndex].size(); ++i)
        totalError += target[i] * log(m_layers[outputLayerIndex].getNeuron(i).getValue() + 1e-10);
    return -totalError;
}


void updateWeights(double learningRate) {
    for (int i = 1; i < m_layers.size(); ++i) {
        for (int j = 0; j < m_layers[i].size(); ++j) {
            for (int k = 0; k < m_layers[i-1].size(); ++k) {
                double newWeight = m_layers[i].getNeuron(j).getWeight(k) -
                                   learningRate * m_layers[i].getNeuron(j).getGradient() * m_layers[i-1].getNeuron(k).getValue();
                m_layers[i].getNeuron(j).setWeight(k, newWeight);
            }
        }
    }
}

void train(const std::vector<std::vector<double>>& inputs, const std::vector<std::vector<double>>& targets, int iterations, double learningRate) {
    for (int i = 0; i < iterations; i++) {
        double totalError = 0.0;
        for (int j = 0; j < inputs.size(); j++) {
            feedForward(inputs[j]);
            backPropagate(targets[j]);
            updateWeights(learningRate);
            totalError += calculateError(targets[j]);
        }
        std::cout << "Iteration: " << i + 1 << " / " << iterations << ", Error: " << totalError / inputs.size() << std::endl;
    }
}

std::vector<double> getOutput() {
    std::vector<double> output;
    int outputLayerIndex = m_layers.size() - 1;
    for (int i = 0; i < m_layers[outputLayerIndex].size(); ++i) {
        output.push_back(m_layers[outputLayerIndex].getNeuron(i).getValue());
    }
    return output;
}

};

std::vector<double> normalize(const std::vector<double>& input) {
    double minVal = *std::min_element(input.begin(), input.end());
    double maxVal = *std::max_element(input.begin(), input.end());

    std::vector<double> normalizedInput;
    for (double val : input) {
        normalizedInput.push_back((val - minVal) / (maxVal - minVal));
    }
    
    return normalizedInput;
}


int main() {
    // Initialize your neural network
    NeuralNetwork nn(3, 2, 1); // create your neural network here
    
    // Set the number of iterations and learning rate
    int iterations = 1000;
    double learningRate = 0.1;

    std::vector<std::vector<double>> inputs = {
        {5.1, 3.5, 1.4, 0.2},   // Iris setosa
        {7.0, 3.2, 4.7, 1.4},   // Iris versicolor
        {6.3, 3.3, 6.0, 2.5}    // Iris virginica
    };

    // Normalize your input data
    for (std::vector<double>& input : inputs) {
        input = normalize(input);
    }

    std::vector<std::vector<double>> targets = {
        {1.0, 0.0, 0.0},   // Iris setosa
        {0.0, 1.0, 0.0},   // Iris versicolor
        {0.0, 0.0, 1.0}    // Iris virginica
    };
    

    // Train your neural network
    nn.train(inputs, targets, iterations, learningRate);

    // Get the output of the network
    std::vector<double> output = nn.getOutput();

    // Print out the output
    std::cout << "Output: ";
    for (double val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
