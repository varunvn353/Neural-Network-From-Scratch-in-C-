#include "dataset.hpp"
#include "neural_network.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

void print_xor_predictions(NeuralNetwork& network, const Dataset& dataset) {
    std::cout << "\nXOR predictions:\n";
    std::cout << std::fixed << std::setprecision(4);
    for (std::size_t i = 0; i < dataset.inputs.cols(); ++i) {
        Matrix sample(2, 1);
        sample(0, 0) = dataset.inputs(0, i);
        sample(1, 0) = dataset.inputs(1, i);

        Matrix output = network.forward(sample);
        const double expected = dataset.targets(0, i);
        const double predicted = output(0, 0);

        std::cout << "  [" << sample(0, 0) << ", " << sample(1, 0) << "] -> "
                  << predicted << " (expected " << expected << ")\n";
    }
}

}  // namespace

int main() {
    std::cout << "=== XOR Neural Network (From Scratch) ===\n";
    std::cout << "Architecture: 2 -> 8 (sigmoid) -> 1 (sigmoid)\n";
    std::cout << "Optimizer: SGD | Loss: MSE\n";

    const Dataset xor_data = make_xor_dataset();

    std::vector<LayerSpec> layers = {
        {8, ActivationType::Sigmoid},
        {1, ActivationType::Sigmoid},
    };

    NeuralNetwork network(2, layers, LossType::MSE, 7);

    const double learning_rate = 2.0;
    const int epochs = 5000;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        const double loss = network.train_batch(xor_data.inputs, xor_data.targets, learning_rate);
        if (epoch % 1000 == 0 || epoch == 1) {
            std::cout << "Epoch " << std::setw(4) << epoch << " | Loss: " << loss << "\n";
        }
    }

    print_xor_predictions(network, xor_data);

    const double final_loss = network.evaluate(xor_data.inputs, xor_data.targets);
    std::cout << "\nFinal loss: " << final_loss << "\n";

    bool solved = true;
    for (std::size_t i = 0; i < xor_data.inputs.cols(); ++i) {
        Matrix sample(2, 1);
        sample(0, 0) = xor_data.inputs(0, i);
        sample(1, 0) = xor_data.inputs(1, i);
        const double predicted = network.forward(sample)(0, 0);
        const double expected = xor_data.targets(0, i);
        if (std::abs(predicted - expected) > 0.2) {
            solved = false;
            break;
        }
    }

    std::cout << (solved ? "XOR solved successfully.\n" : "XOR not fully converged (try rerunning).\n");
    return solved ? 0 : 1;
}
