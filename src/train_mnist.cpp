#include "dataset.hpp"
#include "neural_network.hpp"

#include <algorithm>
#include <fstream>
#include <random>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

double classification_accuracy(const NeuralNetwork& network, const Dataset& dataset) {
    std::size_t correct = 0;
    for (std::size_t i = 0; i < dataset.inputs.cols(); ++i) {
        Matrix sample(dataset.inputs.rows(), 1);
        for (std::size_t r = 0; r < dataset.inputs.rows(); ++r) {
            sample(r, 0) = dataset.inputs(r, i);
        }

        const std::size_t predicted = network.predict_class(sample);

        std::size_t actual = 0;
        double best = dataset.targets(0, i);
        for (std::size_t r = 1; r < dataset.targets.rows(); ++r) {
            if (dataset.targets(r, i) > best) {
                best = dataset.targets(r, i);
                actual = r;
            }
        }

        if (predicted == actual) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(dataset.inputs.cols());
}

Matrix get_batch(const Matrix& data, const std::vector<std::size_t>& indices, std::size_t start,
                 std::size_t batch_size) {
    const std::size_t rows = data.rows();
    const std::size_t end = std::min(start + batch_size, indices.size());
    const std::size_t actual_batch = end - start;

    Matrix batch(rows, actual_batch);
    for (std::size_t b = 0; b < actual_batch; ++b) {
        const std::size_t idx = indices[start + b];
        for (std::size_t r = 0; r < rows; ++r) {
            batch(r, b) = data(r, idx);
        }
    }
    return batch;
}

}  // namespace

int main(int argc, char* argv[]) {
    std::cout << "=== MNIST Subset Neural Network (From Scratch) ===\n";
    std::cout << "Architecture: 784 -> 128 (ReLU) -> 64 (ReLU) -> 10 (softmax)\n";
    std::cout << "Optimizer: mini-batch SGD | Loss: cross-entropy\n";

    std::string data_path = "data/mnist_subset.csv";
    if (argc > 1) {
        data_path = argv[1];
    }

    {
        std::ifstream probe(data_path);
        if (!probe.is_open()) {
            std::cerr << "MNIST subset not found at: " << data_path << "\n";
            std::cerr << "Run: python scripts/generate_mnist_subset.py\n";
            return 1;
        }
    }

    const Dataset full_dataset = load_mnist_csv(data_path);
    const auto split = train_test_split(full_dataset, 0.8, 21);
    const Dataset& train_data = split.first;
    const Dataset& test_data = split.second;

    std::cout << "Loaded " << full_dataset.inputs.cols() << " samples ("
              << train_data.inputs.cols() << " train, "
              << test_data.inputs.cols() << " test)\n";

    std::vector<LayerSpec> layers = {
        {128, ActivationType::ReLU},
        {64, ActivationType::ReLU},
        {10, ActivationType::Softmax},
    };

    NeuralNetwork network(784, layers, LossType::CrossEntropy, 42);

    const double learning_rate = 0.5;
    const std::size_t batch_size = 32;
    const int epochs = 15;

    std::vector<std::size_t> train_indices(train_data.inputs.cols());
    for (std::size_t i = 0; i < train_indices.size(); ++i) {
        train_indices[i] = i;
    }

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::shuffle(train_indices.begin(), train_indices.end(), std::mt19937(static_cast<unsigned>(epoch)));

        double epoch_loss = 0.0;
        std::size_t batches = 0;

        for (std::size_t start = 0; start < train_indices.size(); start += batch_size) {
            const Matrix batch_x = get_batch(train_data.inputs, train_indices, start, batch_size);
            const Matrix batch_y = get_batch(train_data.targets, train_indices, start, batch_size);
            epoch_loss += network.train_batch(batch_x, batch_y, learning_rate);
            ++batches;
        }

        const double train_acc = classification_accuracy(network, train_data);
        const double test_acc = classification_accuracy(network, test_data);

        std::cout << "Epoch " << std::setw(2) << epoch
                  << " | Loss: " << std::fixed << std::setprecision(4) << (epoch_loss / batches)
                  << " | Train acc: " << std::setprecision(2) << (train_acc * 100.0) << "%"
                  << " | Test acc: " << (test_acc * 100.0) << "%\n";
    }

    const double final_test_acc = classification_accuracy(network, test_data);
    std::cout << "\nFinal test accuracy: " << std::fixed << std::setprecision(2)
              << (final_test_acc * 100.0) << "%\n";

    return final_test_acc > 0.85 ? 0 : 1;
}
