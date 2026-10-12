#pragma once

#include <optional>
#include <span>
#include <string>
#include <string_view>

#include <lilia/chat/addressee.h>

namespace lilia::feedback {

using namespace lighter::types;

// The addressee prior: logistic regression over chat::line_features, with
// the weights pulled toward a prior (hand-set at first, then the pooled
// model for each group), after MaiBot's reply_likelihood but labelled by
// whether a line was said to her rather than by whether she answered it.

using Weights = chat::Features;

struct Sample {
    chat::Features x{};
    double label = 0;
    double weight = 1;
};

/// The model's probability that the line was said to her.
double predict(const Weights &weights, const chat::Features &x);

/// MAP fit by Newton's method (iteratively reweighted least squares): log
/// loss plus l2 / 2 * |weights - prior|^2. No samples: the prior.
Weights fit_logistic(std::span<const Sample> samples, const Weights &prior, double l2, usize max_steps = 25);

/// Mean log loss (weighted) of `weights` on `samples`.
double log_loss(const Weights &weights, std::span<const Sample> samples);

/// Hand-set weights for before any are fitted: explicit signals dominate;
/// signs follow MaiBot's fitted gate where it has the feature.
Weights bootstrap_weights();

/// Weights as a JSON array, and back (nullopt when malformed).
std::string write_weights(const Weights &weights);
std::optional<Weights> read_weights(std::string_view json);

} // namespace lilia::feedback
