#include "dataset.hpp"

#include <algorithm>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

Dataset make_xor_dataset() {
    Dataset dataset;
    dataset.inputs = Matrix(2, 4);
    dataset.inputs(0, 0) = 0.0;
    dataset.inputs(1, 0) = 0.0;
    dataset.inputs(0, 1) = 0.0;
    dataset.inputs(1, 1) = 1.0;
    dataset.inputs(0, 2) = 1.0;
    dataset.inputs(1, 2) = 0.0;
    dataset.inputs(0, 3) = 1.0;
    dataset.inputs(1, 3) = 1.0;

    dataset.targets = Matrix(1, 4);
    dataset.targets(0, 0) = 0.0;
    dataset.targets(0, 1) = 1.0;
    dataset.targets(0, 2) = 1.0;
    dataset.targets(0, 3) = 0.0;
    return dataset;
}

Matrix one_hot_encode(const std::vector<int>& labels, std::size_t num_classes) {
    Matrix encoded(num_classes, labels.size(), 0.0);
    for (std::size_t i = 0; i < labels.size(); ++i) {
        if (labels[i] < 0 || static_cast<std::size_t>(labels[i]) >= num_classes) {
            throw std::invalid_argument("Label out of range for one-hot encoding");
        }
        encoded(static_cast<std::size_t>(labels[i]), i) = 1.0;
    }
    return encoded;
}

Dataset load_mnist_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open MNIST CSV: " + path);
    }

    std::vector<int> labels;
    std::vector<double> pixels;
    std::size_t feature_count = 0;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string token;
        if (!std::getline(ss, token, ',')) {
            continue;
        }

        labels.push_back(std::stoi(token));

        std::vector<double> row_pixels;
        while (std::getline(ss, token, ',')) {
            row_pixels.push_back(std::stod(token) / 255.0);
        }

        if (feature_count == 0) {
            feature_count = row_pixels.size();
        } else if (row_pixels.size() != feature_count) {
            throw std::runtime_error("Inconsistent feature count in MNIST CSV");
        }

        pixels.insert(pixels.end(), row_pixels.begin(), row_pixels.end());
    }

    if (labels.empty()) {
        throw std::runtime_error("MNIST CSV is empty: " + path);
    }

    Dataset dataset;
    dataset.inputs = Matrix::from_vector(pixels, feature_count, labels.size());
    dataset.targets = one_hot_encode(labels, 10);
    return dataset;
}

std::pair<Dataset, Dataset> train_test_split(const Dataset& dataset, double train_ratio, unsigned seed) {
    if (train_ratio <= 0.0 || train_ratio >= 1.0) {
        throw std::invalid_argument("train_ratio must be between 0 and 1");
    }

    const std::size_t total = dataset.inputs.cols();
    std::vector<std::size_t> indices(total);
    for (std::size_t i = 0; i < total; ++i) {
        indices[i] = i;
    }

    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    const std::size_t train_count = static_cast<std::size_t>(train_ratio * static_cast<double>(total));
    const std::size_t test_count = total - train_count;
    const std::size_t features = dataset.inputs.rows();

    Dataset train;
    Dataset test;
    train.inputs = Matrix(features, train_count);
    train.targets = Matrix(dataset.targets.rows(), train_count);
    test.inputs = Matrix(features, test_count);
    test.targets = Matrix(dataset.targets.rows(), test_count);

    for (std::size_t i = 0; i < train_count; ++i) {
        const std::size_t src = indices[i];
        for (std::size_t r = 0; r < features; ++r) {
            train.inputs(r, i) = dataset.inputs(r, src);
        }
        for (std::size_t r = 0; r < dataset.targets.rows(); ++r) {
            train.targets(r, i) = dataset.targets(r, src);
        }
    }

    for (std::size_t i = 0; i < test_count; ++i) {
        const std::size_t src = indices[train_count + i];
        for (std::size_t r = 0; r < features; ++r) {
            test.inputs(r, i) = dataset.inputs(r, src);
        }
        for (std::size_t r = 0; r < dataset.targets.rows(); ++r) {
            test.targets(r, i) = dataset.targets(r, src);
        }
    }

    return {train, test};
}
