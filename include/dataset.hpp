#pragma once

#include "matrix.hpp"

#include <string>
#include <utility>
#include <vector>

struct Dataset {
    Matrix inputs;
    Matrix targets;
};

Dataset make_xor_dataset();
Dataset load_mnist_csv(const std::string& path);
std::pair<Dataset, Dataset> train_test_split(const Dataset& dataset, double train_ratio, unsigned seed);
Matrix one_hot_encode(const std::vector<int>& labels, std::size_t num_classes);
