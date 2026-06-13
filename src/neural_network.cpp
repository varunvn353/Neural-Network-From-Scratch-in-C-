#include "neural_network.hpp"

#include <cmath>
#include <stdexcept>

NeuralNetwork::NeuralNetwork(std::size_t input_size, const std::vector<LayerSpec>& specs, LossType loss_type,
                             unsigned seed)
    : loss_type_(loss_type) {
    if (specs.empty()) {
        throw std::invalid_argument("Network must contain at least one layer");
    }

    std::size_t prev_size = input_size;
    layers_.reserve(specs.size());

    for (std::size_t i = 0; i < specs.size(); ++i) {
        const LayerSpec& spec = specs[i];
        Layer layer;
        layer.activation_type = spec.activation;
        layer.weights = Matrix(spec.output_size, prev_size);
        layer.bias = Matrix(spec.output_size, 1);

        const double scale = std::sqrt(1.0 / static_cast<double>(prev_size));
        layer.weights = Matrix::random(spec.output_size, prev_size, scale, seed + static_cast<unsigned>(i + 1));
        layer.bias.fill(0.0);

        layers_.push_back(std::move(layer));
        prev_size = spec.output_size;
    }
}

Matrix NeuralNetwork::forward(const Matrix& input) {
    last_input_ = input;
    Matrix current = input;

    for (Layer& layer : layers_) {
        layer.z = (layer.weights * current).broadcast_add(layer.bias);
        layer.activation = apply_activation(layer.z, layer.activation_type);
        current = layer.activation;
    }

    return current;
}

double NeuralNetwork::train_batch(const Matrix& inputs, const Matrix& targets, double learning_rate) {
    const Matrix& predictions = forward(inputs);

    Matrix delta = loss_gradient(predictions, targets, loss_type_);

    const bool softmax_cross_entropy =
        loss_type_ == LossType::CrossEntropy && layers_.back().activation_type == ActivationType::Softmax;

    if (!softmax_cross_entropy) {
        delta = delta.apply_pair(activation_derivative(layers_.back().z, layers_.back().activation_type),
                                 [](double grad, double deriv) { return grad * deriv; });
    }

    for (int idx = static_cast<int>(layers_.size()) - 1; idx >= 0; --idx) {
        Layer& layer = layers_[static_cast<std::size_t>(idx)];
        const Matrix& prev_activation =
            idx == 0 ? last_input_ : layers_[static_cast<std::size_t>(idx - 1)].activation;

        const Matrix weight_gradient = delta * prev_activation.transpose();
        const Matrix bias_gradient = delta.sum_cols();

        if (idx > 0) {
            const Layer& previous_layer = layers_[static_cast<std::size_t>(idx - 1)];
            delta = layer.weights.transpose() * delta;
            delta = delta.apply_pair(activation_derivative(previous_layer.z, previous_layer.activation_type),
                                     [](double grad, double deriv) { return grad * deriv; });
        }

        layer.weights -= weight_gradient * learning_rate;
        layer.bias -= bias_gradient * learning_rate;
    }

    return compute_loss(predictions, targets, loss_type_);
}

double NeuralNetwork::evaluate(const Matrix& inputs, const Matrix& targets) const {
    Matrix current = inputs;
    for (const Layer& layer : layers_) {
        current = apply_activation((layer.weights * current).broadcast_add(layer.bias), layer.activation_type);
    }
    return compute_loss(current, targets, loss_type_);
}

std::size_t NeuralNetwork::predict_class(const Matrix& sample) const {
    Matrix current = sample;
    for (const Layer& layer : layers_) {
        current = apply_activation((layer.weights * current).broadcast_add(layer.bias), layer.activation_type);
    }

    std::size_t best = 0;
    double best_value = current(0, 0);
    for (std::size_t r = 1; r < current.rows(); ++r) {
        if (current(r, 0) > best_value) {
            best_value = current(r, 0);
            best = r;
        }
    }
    return best;
}
