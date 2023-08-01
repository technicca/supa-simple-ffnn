#include <vector>
#include <cmath>

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double sigmoidDerivative(double x) {        
    double sigmoid = 1.0 / (1.0 + std::exp(-x));
    return sigmoid * (1 - sigmoid);
}

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
    Neuron(int numWeights, double value = 0.0) : m_value(value), m_activation(0.0), m_derived(0.0), m_weights(numWeights), m_gradient(0.0) {
        // Initialize weights with random values
        for (double& weight : m_weights) {
            weight = ((double) rand() / (RAND_MAX));
        }
    }

    // Getters and setters for the private variables
    void setValue(double value) { m_value = value; }
    double getValue() { return m_value; }

    void setActivation(double activation) { m_activation = activation; }
    double getActivation() { return m_activation; }

    void setDerived(double derived) { m_derived = derived; }
    double getDerived() { return m_derived; }

    void setWeight(int index, double weight) { m_weights[index] = weight; }
    double getWeight(int index) { return m_weights[index]; }

    void setGradient(double gradient) { m_gradient = gradient; }
    double getGradient() { return m_gradient; }
};



class Layer {
private:
    std::vector<Neuron> m_neurons; // neurons in this layer

public:
    // Constructor
    Layer(int size) : m_neurons(size) {}

    // Getters and setters for the neurons
    Neuron& getNeuron(int index) { return m_neurons[index]; }
    void setNeuron(int index, const Neuron& neuron) { m_neurons[index] = neuron; }

    // Function to get the layer size
    int size() { return m_neurons.size(); }
};

class NeuralNetwork {
private:
    std::vector<Layer> m_layers; // layers in the network

public:
    // Constructor
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        // create input layer
        m_layers.push_back(Layer(inputSize));

        // create hidden layer
        m_layers.push_back(Layer(hiddenSize));

        // create output layer
        m_layers.push_back(Layer(outputSize));
    }

    // Other member functions to define...
void feedForward(const std::vector<double>& input){
    // Set the input layer
    for(int i = 0; i < input.size(); ++i)
        m_layers[0].getNeuron(i).setValue(input[i]); // Assuming setNeuron sets the value of Neuron
  
    // Forward propagation
    for(int i = 1; i < m_layers.size(); ++i)
        for(int j = 0; j < m_layers[i].size(); ++j){
            double activation = 0.0;
            for(int k = 0; k < m_layers[i-1].size(); ++k)
              activation += m_layers[i-1].getNeuron(k).getValue() * m_layers[i-1].getNeuron(k).getWeight(j);
            m_layers[i].getNeuron(j).setValue(sigmoid(activation)); // Assuming sigmoid is the activation function
        }
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
            for(int k = 0; k < m_layers[i+1].size(); ++k){
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
        for(int i = 0; i < m_layers[outputLayerIndex].size(); ++i)
            totalError += pow((target[i] - m_layers[outputLayerIndex].getNeuron(i).getValue()), 2);
        return totalError / m_layers[outputLayerIndex].size();
    }

void updateWeights(double learningRate){
        for(int i = 1; i < m_layers.size(); ++i)
            for(int j = 0; j < m_layers[i].size(); ++j){
                double newWeight = m_layers[i].getNeuron(j).getWeight(0) - 
                                    learningRate * m_layers[i].getNeuron(j).getGradient() * m_layers[i-1].getNeuron(j).getValue();
                m_layers[i].getNeuron(j).setWeight(0, newWeight);
            }
    }
};

int main() {
    NeuralNetwork nn(3, 2, 1); // Create a neural network with 3 inputs, 2 hidden neurons, and 1 output

    // Train the neural network with your data here...

    return 0;
}