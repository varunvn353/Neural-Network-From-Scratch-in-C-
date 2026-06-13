#pragma once

#include "matrix.hpp"

enum class ActivationType {
    Sigmoid,
    ReLU,
    Softmax
};

enum class LossType {
    MSE,
    CrossEntropy
};

Matrix sigmoid(const Matrix& z);
Matrix sigmoid_derivative(const Matrix& z);
Matrix relu(const Matrix& z);
Matrix relu_derivative(const Matrix& z);
Matrix softmax(const Matrix& z);

Matrix apply_activation(const Matrix& z, ActivationType type);
Matrix activation_derivative(const Matrix& z, ActivationType type);

double compute_loss(const Matrix& predictions, const Matrix& targets, LossType type);
Matrix loss_gradient(const Matrix& predictions, const Matrix& targets, LossType type);
