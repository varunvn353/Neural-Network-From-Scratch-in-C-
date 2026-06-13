#include "activations.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

double stable_sigmoid(double x) {
    if (x >= 0.0) {
        const double exp_neg = std::exp(-x);
        return 1.0 / (1.0 + exp_neg);
    }
    const double exp_pos = std::exp(x);
    return exp_pos / (1.0 + exp_pos);
}

}  // namespace

Matrix sigmoid(const Matrix& z) {
    return z.apply(stable_sigmoid);
}

Matrix sigmoid_derivative(const Matrix& z) {
    Matrix s = sigmoid(z);
    return s.apply_pair(s, [](double sig, double) { return sig * (1.0 - sig); });
}

Matrix relu(const Matrix& z) {
    return z.apply([](double x) { return std::max(0.0, x); });
}

Matrix relu_derivative(const Matrix& z) {
    return z.apply([](double x) { return x > 0.0 ? 1.0 : 0.0; });
}

Matrix softmax(const Matrix& z) {
    Matrix result(z.rows(), z.cols());
    for (std::size_t c = 0; c < z.cols(); ++c) {
        double max_val = -std::numeric_limits<double>::infinity();
        for (std::size_t r = 0; r < z.rows(); ++r) {
            max_val = std::max(max_val, z(r, c));
        }

        double sum_exp = 0.0;
        for (std::size_t r = 0; r < z.rows(); ++r) {
            sum_exp += std::exp(z(r, c) - max_val);
        }

        for (std::size_t r = 0; r < z.rows(); ++r) {
            result(r, c) = std::exp(z(r, c) - max_val) / sum_exp;
        }
    }
    return result;
}

Matrix apply_activation(const Matrix& z, ActivationType type) {
    switch (type) {
        case ActivationType::Sigmoid:
            return sigmoid(z);
        case ActivationType::ReLU:
            return relu(z);
        case ActivationType::Softmax:
            return softmax(z);
    }
    throw std::invalid_argument("Unknown activation type");
}

Matrix activation_derivative(const Matrix& z, ActivationType type) {
    switch (type) {
        case ActivationType::Sigmoid:
            return sigmoid_derivative(z);
        case ActivationType::ReLU:
            return relu_derivative(z);
        case ActivationType::Softmax:
            return Matrix(z.rows(), z.cols(), 1.0);
    }
    throw std::invalid_argument("Unknown activation type");
}

double compute_loss(const Matrix& predictions, const Matrix& targets, LossType type) {
    const std::size_t batch = predictions.cols();
    if (batch == 0) {
        return 0.0;
    }

    switch (type) {
        case LossType::MSE: {
            Matrix diff = predictions - targets;
            Matrix squared = diff.apply([](double x) { return x * x; });
            return 0.5 * squared.sum() / static_cast<double>(batch);
        }
        case LossType::CrossEntropy: {
            double loss = 0.0;
            for (std::size_t c = 0; c < batch; ++c) {
                for (std::size_t r = 0; r < predictions.rows(); ++r) {
                    const double target = targets(r, c);
                    if (target > 0.0) {
                        const double pred = std::max(predictions(r, c), 1e-12);
                        loss -= target * std::log(pred);
                    }
                }
            }
            return loss / static_cast<double>(batch);
        }
    }
    throw std::invalid_argument("Unknown loss type");
}

Matrix loss_gradient(const Matrix& predictions, const Matrix& targets, LossType type) {
    const std::size_t batch = predictions.cols();
    switch (type) {
        case LossType::MSE:
            return (predictions - targets) / static_cast<double>(batch);
        case LossType::CrossEntropy:
            return (predictions - targets) / static_cast<double>(batch);
    }
    throw std::invalid_argument("Unknown loss type");
}
