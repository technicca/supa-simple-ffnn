#include <cmath>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

// Define the structure of a neuron
struct Neuron {
    double value;
    vector<double> weights;
};

// Define the structure of a layer
typedef vector<Neuron> Layer;

// Define the structure of a network
struct Network {
    vector<Layer> layers;
};

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoid_derivative(double x) {
    double sigmoid_x = sigmoid(x);
    return sigmoid_x * (1 - sigmoid_x);
}

// Forward propagation
void forward(Network& network, vector<double> input) {
    for (int i = 0; i < network.layers[0].size(); i++) {
        network.layers[0][i].value = input[i];
    }

    for (int i = 1; i < network.layers.size(); i++) {
        for (int j = 0; j < network.layers[i].size(); j++) {
            double sum = 0;
            for (int k = 0; k < network.layers[i - 1].size(); k++) {
                sum += network.layers[i - 1][k].value * network.layers[i - 1][k].weights[j];
            }
            network.layers[i][j].value = sigmoid(sum);
        }
    }
}

double learning_rate = 0.023;

// Mean squared error function
double mse(vector<double> target, vector<double> output) {
    double sum = 0;
    for (int i = 0; i < target.size(); i++) {
        sum += pow(target[i] - output[i], 2);
    }
    return sum / target.size();
}

// Backward propagation
void backward(Network& network, vector<double> target) {
    // Calculate output layer errors
    vector<double> errors(network.layers.back().size());
    for (int i = 0; i < network.layers.back().size(); i++) {
        errors[i] = target[i] - network.layers.back()[i].value;
    }

    // Update output layer weights
    for (int i = 0; i < network.layers.back().size(); i++) {
        for (int j = 0; j < network.layers[network.layers.size() - 2].size(); j++) {
            double delta = learning_rate * errors[i] * sigmoid_derivative(network.layers.back()[i].value);
            network.layers[network.layers.size() - 2][j].weights[i] += delta;
        }
    }

    // Propagate the errors and update weights for hidden layers
    for (int i = network.layers.size() - 2; i > 0; i--) {
        vector<double> errors_next(network.layers[i].size());
        for (int j = 0; j < network.layers[i].size(); j++) {
            for (int k = 0; k < network.layers[i + 1].size(); k++) {
                errors_next[j] += errors[k] * network.layers[i][j].weights[k];
            }
        }
        errors = errors_next;

        for (int j = 0; j < network.layers[i].size(); j++) {
            for (int k = 0; k < network.layers[i - 1].size(); k++) {
                double delta = learning_rate * errors[j] * sigmoid_derivative(network.layers[i][j].value);
                network.layers[i - 1][k].weights[j] += delta;
            }
        }
    }
}


// Prediction function
vector<double> predict(Network& network, vector<double> input) {
    forward(network, input);
    vector<double> output(network.layers.back().size());
    for (int i = 0; i < network.layers.back().size(); i++) {
        output[i] = network.layers.back()[i].value;
    }
    return output;
}

void train(Network& network, vector<vector<double>> inputs, vector<vector<double>> targets, int epochs) {
    double first_error, last_error;
    for (int i = 0; i < epochs; i++) {
        double sum_error = 0;
        for (int j = 0; j < inputs.size(); j++) {
            forward(network, inputs[j]);
            vector<double> output = predict(network, inputs[j]);
            double error = mse(targets[j], output);
            sum_error += error;
            if (i == 0 && j == 0) {
                first_error = error;
            }
            backward(network, targets[j]);
        }
        if (i == epochs - 1) {
            last_error = sum_error / inputs.size();
        }
    }
    cout << "First error: " << first_error << endl;
    cout << "Last error: " << last_error << endl;
}

// Function to print the results
void printResults(Network& network, vector<vector<double>> inputs) {
    for (int i = 0; i < inputs.size(); i++) {
        vector<double> output = predict(network, inputs[i]);
        cout << "Input: ";
        for (int j = 0; j < inputs[i].size(); j++) {
            cout << inputs[i][j] << " ";
        }
        cout << "Output: ";
        for (int j = 0; j < output.size(); j++) {
            cout << output[j] << " ";
        }
        cout << endl;
    }
}

int main() {
    srand(time(NULL));
    // Initialize a network
    Network network;
    network.layers = {Layer(2), Layer(2), Layer(1)};

    // Initialize weights
    for (int i = 0; i < network.layers.size() - 1; i++) {
        for (int j = 0; j < network.layers[i].size(); j++) {
            network.layers[i][j].weights = vector<double>(network.layers[i + 1].size());
            for (int k = 0; k < network.layers[i + 1].size(); k++) {
                double limit = sqrt(6.0 / (network.layers[i].size() + network.layers[i + 1].size()));
                network.layers[i][j].weights[k] = ((double) rand() / (RAND_MAX)) * 2 * limit - limit; // Initialize to random values between -limit and limit
            }
        }
    }

    vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> targets = {{0}, {1}, {1}, {0}};
    train(network, inputs, targets, 2000); // epochs
    // Print the results
    printResults(network, inputs);

    return 0;
}
