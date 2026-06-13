#pragma once

#include "activations.hpp"

#include <string>
#include <vector>

struct LayerSpec {
    std::size_t output_size;
    ActivationType activation;
};

class NeuralNetwork {
public:
    NeuralNetwork(std::size_t input_size, const std::vector<LayerSpec>& layers, LossType loss_type,
                  unsigned seed = 42);

    Matrix forward(const Matrix& input);
    double train_batch(const Matrix& inputs, const Matrix& targets, double learning_rate);
    double evaluate(const Matrix& inputs, const Matrix& targets) const;

    std::size_t predict_class(const Matrix& sample) const;

    LossType loss_type() const { return loss_type_; }

private:
    struct Layer {
        Matrix weights;
        Matrix bias;
        Matrix z;
        Matrix activation;
        ActivationType activation_type;
    };

    LossType loss_type_;
    std::vector<Layer> layers_;
    Matrix last_input_;
};
