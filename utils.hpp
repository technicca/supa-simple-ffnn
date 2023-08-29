#ifndef UTILS
#define UTILS

#include <algorithm>
#include <cmath>
#include <vector>
#include <random>

double relu(double x) {
    return std::max(0.0, x);
}

double reluDerivative(double x) {
    return x > 0 ? 1.0 : 0.0;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double sigmoidDerivative(double x) {
    return x * (1 - x);
}

void softmax(std::vector<double>& values) {
    double max = *std::max_element(values.begin(), values.end());
    double sum = 0.0;

    for (int i = 0; i < values.size(); ++i) {
        values[i] = std::exp(values[i] - max);
        sum += values[i];
    }

    for (int i = 0; i < values.size(); ++i) {
        values[i] /= sum;
    }
}

double softmaxDerivative(int i, const std::vector<double>& softmaxOutputs) {
    return softmaxOutputs[i] * (1 - softmaxOutputs[i]);
}

#endif // UTILS
