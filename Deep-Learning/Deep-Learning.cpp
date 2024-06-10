#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>

class NeuralNetwork {
private:
    std::vector<int> layers;  // number of neurons in each layer
    std::vector<std::vector<std::vector<double>>> weights;  // weights between layers
    std::vector<std::vector<double>> biases;  // biases of neurons
    std::vector<std::vector<double>> neurons;  // neuron values
    double learningRate;

    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    double sigmoidDerivative(double x) {
        return x * (1.0 - x);
    }

    void initializeWeightsAndBiases() {
        std::srand(std::time(nullptr));
        for (int i = 0; i < layers.size() - 1; ++i) {
            std::vector<std::vector<double>> layerWeights;
            std::vector<double> layerBiases;
            for (int j = 0; j < layers[i + 1]; ++j) {
                std::vector<double> neuronWeights;
                for (int k = 0; k < layers[i]; ++k) {
                    neuronWeights.push_back(((double)rand() / (RAND_MAX)));
                }
                layerWeights.push_back(neuronWeights);
                layerBiases.push_back(((double)rand() / (RAND_MAX)));
            }
            weights.push_back(layerWeights);
            biases.push_back(layerBiases);
        }
    }

    std::vector<double> feedforward(const std::vector<double>& input) {
        neurons[0] = input;
        for (int i = 1; i < layers.size(); ++i) {
#pragma omp parallel for
            for (int j = 0; j < layers[i]; ++j) {
                double sum = biases[i - 1][j];
                for (int k = 0; k < layers[i - 1]; ++k) {
                    sum += neurons[i - 1][k] * weights[i - 1][j][k];
                }
                neurons[i][j] = sigmoid(sum);
            }
        }
        return neurons.back();
    }

public:
    NeuralNetwork(const std::vector<int>& layers, double learningRate) : layers(layers), learningRate(learningRate) {
        for (int i = 0; i < layers.size(); ++i) {
            neurons.push_back(std::vector<double>(layers[i]));
        }
        initializeWeightsAndBiases();
    }

    void train(const std::vector<double>& input, const std::vector<double>& target) {
        std::vector<double> output = feedforward(input);
        std::vector<std::vector<double>> errors(layers.size());

        // Calculate output error
        for (int i = 0; i < layers.back(); ++i) {
            errors.back().push_back(target[i] - output[i]);
        }

        // Backpropagate
        for (int i = layers.size() - 2; i >= 0; --i) {
#pragma omp parallel for
            for (int j = 0; j < layers[i]; ++j) {
                double error = 0.0;
                for (int k = 0; k < layers[i + 1]; ++k) {
                    error += errors[i + 1][k] * weights[i][k][j];
                }
                errors[i].push_back(error);
            }
        }

        // Update weights
        for (int i = layers.size() - 1; i > 0; --i) {
#pragma omp parallel for
            for (int j = 0; j < layers[i]; ++j) {
                for (int k = 0; k < layers[i - 1]; ++k) {
                    weights[i - 1][j][k] += learningRate * errors[i][j] * sigmoidDerivative(neurons[i][j]) * neurons[i - 1][k];
                }
                biases[i - 1][j] += learningRate * errors[i][j] * sigmoidDerivative(neurons[i][j]);
            }
        }
    }

    std::vector<double> predict(const std::vector<double>& input) {
        return feedforward(input);
    }
};

int main() {
    std::vector<int> layers = { 2, 3, 1 };
    NeuralNetwork nn(layers, 0.5);

    // Example training data
    std::vector<std::vector<double>> inputs = { {0, 0}, {0, 1}, {1, 0}, {1, 1} };
    std::vector<std::vector<double>> targets = { {0}, {1}, {1}, {0} };


    for (int epoch = 0; epoch < 10000; ++epoch) {
        for (int i = 0; i < inputs.size(); ++i) {
            nn.train(inputs[i], targets[i]);
        }
    }

    // Predicting
    for (const auto& input : inputs) {
        std::vector<double> output = nn.predict(input);
        std::cout << "Input: ";
        for (double val : input) std::cout << val << " ";
        std::cout << "Output: ";
        for (double val : output) std::cout << val << " ";
        std::cout << std::endl;
    }

    return 0;
}
