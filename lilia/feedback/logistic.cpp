#include "logistic.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <glaze/json.hpp>

namespace lilia::feedback {

namespace {

constexpr usize n = chat::k_features;
using Matrix = std::array<std::array<double, n>, n>;

double dot(const Weights &weights, const chat::Features &x) {
    double sum = 0;
    for (usize i = 0; i < n; ++i) sum += weights[i] * x[i];
    return sum;
}

double sigmoid(double z) { return z >= 0 ? 1 / (1 + std::exp(-z)) : std::exp(z) / (1 + std::exp(z)); }

/// Solves a x = b for symmetric positive definite a (Cholesky); false when
/// it is not.
bool solve(Matrix a, std::array<double, n> &b) {
    for (usize j = 0; j < n; ++j) {
        double diagonal = a[j][j];
        for (usize k = 0; k < j; ++k) diagonal -= a[j][k] * a[j][k];
        if (diagonal <= 0) return false;
        a[j][j] = std::sqrt(diagonal);
        for (usize i = j + 1; i < n; ++i) {
            double value = a[i][j];
            for (usize k = 0; k < j; ++k) value -= a[i][k] * a[j][k];
            a[i][j] = value / a[j][j];
        }
    }
    for (usize i = 0; i < n; ++i) { // L y = b
        for (usize k = 0; k < i; ++k) b[i] -= a[i][k] * b[k];
        b[i] /= a[i][i];
    }
    for (usize i = n; i-- > 0;) { // L^T x = y
        for (usize k = i + 1; k < n; ++k) b[i] -= a[k][i] * b[k];
        b[i] /= a[i][i];
    }
    return true;
}

} // namespace

double predict(const Weights &weights, const chat::Features &x) { return sigmoid(dot(weights, x)); }

Weights fit_logistic(std::span<const Sample> samples, const Weights &prior, double l2, usize max_steps) {
    Weights weights = prior;
    if (samples.empty()) return weights;
    for (usize step = 0; step < max_steps; ++step) {
        std::array<double, n> gradient{};
        Matrix hessian{};
        for (usize i = 0; i < n; ++i) {
            gradient[i] = l2 * (weights[i] - prior[i]);
            hessian[i][i] = l2;
        }
        for (const auto &sample : samples) {
            const double p = predict(weights, sample.x);
            const double residual = sample.weight * (p - sample.label);
            const double curvature = sample.weight * p * (1 - p);
            for (usize i = 0; i < n; ++i) {
                gradient[i] += residual * sample.x[i];
                for (usize j = 0; j <= i; ++j) hessian[i][j] += curvature * sample.x[i] * sample.x[j];
            }
        }
        for (usize i = 0; i < n; ++i) {
            for (usize j = i + 1; j < n; ++j) hessian[i][j] = hessian[j][i];
        }
        if (!solve(hessian, gradient)) break;
        double largest = 0;
        for (usize i = 0; i < n; ++i) {
            weights[i] -= gradient[i];
            largest = std::max(largest, std::abs(gradient[i]));
        }
        if (largest < 1e-7) break;
    }
    return weights;
}

double log_loss(const Weights &weights, std::span<const Sample> samples) {
    double loss = 0;
    double total = 0;
    for (const auto &sample : samples) {
        const double p = std::clamp(predict(weights, sample.x), 1e-9, 1 - 1e-9);
        loss -= sample.weight * (sample.label * std::log(p) + (1 - sample.label) * std::log(1 - p));
        total += sample.weight;
    }
    return total > 0 ? loss / total : 0;
}

Weights bootstrap_weights() {
    // bias, @her, quotes her, names her, @other, quotes other, names other,
    // 你, 你们/谁, question, reaction, after her, her partner, her share,
    // since her, two-person run, speakers.
    return {-1.5, 4.0, 4.0, 2.0, -2.5, -2.0, -1.0, 0.3, -0.3, 0.3, -1.0, 1.2, 1.5, 2.65, -1.0, -1.5, -0.5};
}

std::string write_weights(const Weights &weights) {
    const std::vector<double> values(weights.begin(), weights.end());
    return glz::write_json(values).value_or("[]");
}

std::optional<Weights> read_weights(std::string_view json) {
    std::vector<double> values;
    if (glz::read_json(values, json) || values.size() != n) return std::nullopt;
    Weights weights{};
    std::ranges::copy(values, weights.begin());
    return weights;
}

} // namespace lilia::feedback
