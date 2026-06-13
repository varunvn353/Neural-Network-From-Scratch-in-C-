#pragma once

#include <cmath>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols, double value = 0.0);

    static Matrix from_vector(const std::vector<double>& data, std::size_t rows, std::size_t cols);
    static Matrix random(std::size_t rows, std::size_t cols, double scale = 1.0, unsigned seed = 42);

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    double& operator()(std::size_t r, std::size_t c);
    const double& operator()(std::size_t r, std::size_t c) const;

    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(double scalar) const;
    Matrix operator/(double scalar) const;

    Matrix& operator+=(const Matrix& other);
    Matrix& operator-=(const Matrix& other);

    Matrix transpose() const;
    Matrix apply(std::function<double(double)> fn) const;
    Matrix apply_pair(const Matrix& other, std::function<double(double, double)> fn) const;

    double sum() const;
    Matrix sum_cols() const;
    Matrix broadcast_add(const Matrix& bias) const;

    void fill(double value);
    std::string shape_string() const;

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<double> data_;

    void check_same_shape(const Matrix& other, const char* op) const;
    void check_multiply_shapes(const Matrix& other) const;
};
