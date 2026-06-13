#include "matrix.hpp"

#include <sstream>
#include <utility>

Matrix::Matrix(std::size_t rows, std::size_t cols, double value)
    : rows_(rows), cols_(cols), data_(rows * cols, value) {}

Matrix Matrix::from_vector(const std::vector<double>& data, std::size_t rows, std::size_t cols) {
    if (data.size() != rows * cols) {
        throw std::invalid_argument("Vector size does not match matrix dimensions");
    }
    Matrix m(rows, cols);
    m.data_ = data;
    return m;
}

Matrix Matrix::random(std::size_t rows, std::size_t cols, double scale, unsigned seed) {
    Matrix m(rows, cols);
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, scale);
    for (double& value : m.data_) {
        value = dist(rng);
    }
    return m;
}

double& Matrix::operator()(std::size_t r, std::size_t c) {
    if (r >= rows_ || c >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data_[r * cols_ + c];
}

const double& Matrix::operator()(std::size_t r, std::size_t c) const {
    if (r >= rows_ || c >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }
    return data_[r * cols_ + c];
}

Matrix Matrix::operator+(const Matrix& other) const {
    check_same_shape(other, "+");
    Matrix result(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    check_same_shape(other, "-");
    Matrix result(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    check_multiply_shapes(other);
    Matrix result(rows_, other.cols_);
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t j = 0; j < other.cols_; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < cols_; ++k) {
                sum += (*this)(i, k) * other(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

Matrix Matrix::operator/(double scalar) const {
    if (scalar == 0.0) {
        throw std::invalid_argument("Division by zero");
    }
    return (*this) * (1.0 / scalar);
}

Matrix& Matrix::operator+=(const Matrix& other) {
    check_same_shape(other, "+=");
    for (std::size_t i = 0; i < data_.size(); ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    check_same_shape(other, "-=");
    for (std::size_t i = 0; i < data_.size(); ++i) {
        data_[i] -= other.data_[i];
    }
    return *this;
}

Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (std::size_t r = 0; r < rows_; ++r) {
        for (std::size_t c = 0; c < cols_; ++c) {
            result(c, r) = (*this)(r, c);
        }
    }
    return result;
}

Matrix Matrix::apply(std::function<double(double)> fn) const {
    Matrix result(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = fn(data_[i]);
    }
    return result;
}

Matrix Matrix::apply_pair(const Matrix& other, std::function<double(double, double)> fn) const {
    check_same_shape(other, "apply_pair");
    Matrix result(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = fn(data_[i], other.data_[i]);
    }
    return result;
}

double Matrix::sum() const {
    double total = 0.0;
    for (double value : data_) {
        total += value;
    }
    return total;
}

Matrix Matrix::sum_cols() const {
    Matrix result(rows_, 1);
    for (std::size_t r = 0; r < rows_; ++r) {
        double total = 0.0;
        for (std::size_t c = 0; c < cols_; ++c) {
            total += (*this)(r, c);
        }
        result(r, 0) = total;
    }
    return result;
}

Matrix Matrix::broadcast_add(const Matrix& bias) const {
    if (bias.rows() != rows_ || bias.cols() != 1) {
        throw std::invalid_argument("Bias must be a column vector matching row count");
    }
    Matrix result(rows_, cols_);
    for (std::size_t c = 0; c < cols_; ++c) {
        for (std::size_t r = 0; r < rows_; ++r) {
            result(r, c) = (*this)(r, c) + bias(r, 0);
        }
    }
    return result;
}

void Matrix::fill(double value) {
    for (double& entry : data_) {
        entry = value;
    }
}

std::string Matrix::shape_string() const {
    std::ostringstream oss;
    oss << rows_ << "x" << cols_;
    return oss.str();
}

void Matrix::check_same_shape(const Matrix& other, const char* op) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument(std::string("Shape mismatch for ") + op + ": " + shape_string() +
                                    " vs " + other.shape_string());
    }
}

void Matrix::check_multiply_shapes(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("Invalid matmul shapes: " + shape_string() + " * " + other.shape_string());
    }
}
