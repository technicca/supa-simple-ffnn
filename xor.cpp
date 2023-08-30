// XOR problem neural net

#include <iostream>
#include <vector>
#include <cmath>
#include <random>

float random_float(float min, float max) {
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(min, max); // rage 0 - 1
    return dis(e);
}

float sigmoid(float x) {
    return 1 / (1 + exp(-x));
}

float sigmoid_derivative(float x) {
    return x * (1 - x);
}

std::vector<std::vector<float>> initialize_matrix(int rows, int cols) {
    std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols));
    for(int i=0; i<rows; i++)
        for(int j=0; j<cols; j++)
            matrix[i][j] = random_float(-1, 1);
    return matrix;
}

std::vector<std::vector<float>> matrix_multiply(std::vector<std::vector<float>> a, std::vector<std::vector<float>> b) {
    std::vector<std::vector<float>> result(a.size(), std::vector<float>(b[0].size()));
    for(int i=0; i<a.size(); i++)
        for(int j=0; j<b[0].size(); j++)
            for(int k=0; k<a[0].size(); k++)
                result[i][j] += a[i][k] * b[k][j];
    return result;
}

std::vector<std::vector<float>> matrix_add(std::vector<std::vector<float>> a, std::vector<std::vector<float>> b) {
    if (a.size() != b.size() || a[0].size() != b[0].size()) {
        throw std::invalid_argument("Both matrices must be of the same size");
    }
    std::vector<std::vector<float>> result(a.size(), std::vector<float>(a[0].size()));
    for(int i=0; i<a.size(); i++)
        for(int j=0; j<a[0].size(); j++)
            result[i][j] = a[i][j] + b[i][j];
    return result;
}


// sigmoid function to a matrix. This will be used in the forward propagation step
std::vector<std::vector<float>> apply_sigmoid(std::vector<std::vector<float>> a) {
    std::vector<std::vector<float>> result(a.size(), std::vector<float>(a[0].size()));
    for(int i=0; i<a.size(); i++)
        for(int j=0; j<a[0].size(); j++)
            result[i][j] = sigmoid(a[i][j]);
    return result;
}

std::vector<std::vector<float>> matrix_subtract(std::vector<std::vector<float>> a, std::vector<std::vector<float>> b) {
    std::vector<std::vector<float>> result(a.size(), std::vector<float>(a[0].size()));
    for(int i=0; i<a.size(); i++)
        for(int j=0; j<a[0].size(); j++)
            result[i][j] = a[i][j] - b[i][j];
    return result;
}

// Apply the sigmoid derivative function to a matrix
std::vector<std::vector<float>> apply_sigmoid_derivative(std::vector<std::vector<float>> a) {
    std::vector<std::vector<float>> result(a.size(), std::vector<float>(a[0].size()));
    for(int i=0; i<a.size(); i++)
        for(int j=0; j<a[0].size(); j++)
            result[i][j] = sigmoid_derivative(a[i][j]);
    return result;
}

// perform elementwise multiplication of two matrices. This will be used in the backpropagation step
std::vector<std::vector<float>> elementwise_multiply(std::vector<std::vector<float>> a, std::vector<std::vector<float>> b) {
    std::vector<std::vector<float>> result(a.size(), std::vector<float>(a[0].size()));
    for(int i=0; i<a.size(); i++)
        for(int j=0; j<a[0].size(); j++)
            result[i][j] = a[i][j] * b[i][j];
    return result;
}

void update_weights_and_biases(std::vector<std::vector<float>>& weights, std::vector<std::vector<float>>& biases, std::vector<std::vector<float>> delta, std::vector<std::vector<float>> inputs, float lr) {
    for(int i=0; i<weights.size(); i++)
        for(int j=0; j<weights[0].size(); j++)
            for(int k=0; k<delta.size(); k++)
                weights[i][j] += inputs[k][i] * delta[k][j] * lr;
    for(int i=0; i<biases.size(); i++)
        for(int j=0; j<biases[0].size(); j++)
            biases[i][j] += delta[i][j] * lr;
}

int main() {
    std::vector<std::vector<float>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}}; // XOR dataset
    std::vector<std::vector<float>> expected_output = {{0}, {1}, {1}, {0}};

    int epochs = 20000;
    float lr = 0.8;
    int inputLayerNeurons = 2, hiddenLayerNeurons = 19, outputLayerNeurons = 1;

    // Random weights and bias initialization
    std::vector<std::vector<float>> hidden_weights = initialize_matrix(inputLayerNeurons, hiddenLayerNeurons);
    std::vector<std::vector<float>> hidden_bias = initialize_matrix(inputs.size(), hiddenLayerNeurons);
    std::vector<std::vector<float>> output_weights = initialize_matrix(hiddenLayerNeurons, outputLayerNeurons);
    std::vector<std::vector<float>> output_bias = initialize_matrix(inputs.size(), outputLayerNeurons);

    for(int epoch=0; epoch<epochs; epoch++) { // main training loop
        
        // Forward Propagation
        std::vector<std::vector<float>> hidden_layer_activation = matrix_add(matrix_multiply(inputs, hidden_weights), hidden_bias);
        std::vector<std::vector<float>> hidden_layer_output = apply_sigmoid(hidden_layer_activation);

        std::vector<std::vector<float>> output_layer_activation = matrix_add(matrix_multiply(hidden_layer_output, output_weights), output_bias);
        std::vector<std::vector<float>> predicted_output = apply_sigmoid(output_layer_activation);

        // Backward propagation
        std::vector<std::vector<float>> error = matrix_subtract(expected_output, predicted_output);
        std::vector<std::vector<float>> d_predicted_output = elementwise_multiply(error, apply_sigmoid_derivative(predicted_output));
        
        std::vector<std::vector<float>> error_hidden_layer = matrix_multiply(d_predicted_output, output_weights);
        std::vector<std::vector<float>> d_hidden_layer = elementwise_multiply(error_hidden_layer, apply_sigmoid_derivative(hidden_layer_output));

        // Updating Weights and Biases
        update_weights_and_biases(output_weights, output_bias, d_predicted_output, hidden_layer_output, lr);
        update_weights_and_biases(hidden_weights, hidden_bias, d_hidden_layer, inputs, lr);
    }

    std::cout << "Final hidden weights: \n"; // print the output
    for(auto& row : hidden_weights) {
        for(auto& val : row)
            std::cout << val << ' ';
        std::cout << '\n';
    }

    std::cout << "Final hidden bias: \n";
    for(auto& row : hidden_bias) {
        for(auto& val : row)
            std::cout << val << ' ';
        std::cout << '\n';
    }

    std::cout << "Final output weights: \n";
    for(auto& row : output_weights) {
        for(auto& val : row)
            std::cout << val << ' ';
        std::cout << '\n';
    }

    std::cout << "Final output bias: \n";
    for(auto& row : output_bias) {
        for(auto& val : row)
            std::cout << val << ' ';
        std::cout << '\n';
    }

    std::cout << "\nOutput from neural network after " << epochs << " epochs: \n";
    std::vector<std::vector<float>> hidden_layer_activation = matrix_add(matrix_multiply(inputs, hidden_weights), hidden_bias);
    std::vector<std::vector<float>> hidden_layer_output = apply_sigmoid(hidden_layer_activation);

    std::vector<std::vector<float>> output_layer_activation = matrix_add(matrix_multiply(hidden_layer_output, output_weights), output_bias);
    std::vector<std::vector<float>> predicted_output = apply_sigmoid(output_layer_activation);
    for(auto& row : predicted_output) {
        for(auto& val : row)
            std::cout << val << ' ';
        std::cout << '\n';
    }

    return 0;
}
