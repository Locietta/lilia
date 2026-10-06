#pragma once

#include <chrono>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <lighter/types.hpp>

namespace lilia::chat {

using namespace lighter::types;

/// Splits a cleaned reply into at most `max_bubbles` QQ messages: one per
/// line, long lines broken after sentence-ending punctuation, extra pieces
/// merged into the last bubble. A trailing "。" is dropped from each bubble.
std::vector<std::string> split_bubbles(std::string_view text, usize max_bubbles = 3);

/// Characters (UTF-8 code points) in `text`.
usize count_code_points(std::string_view text);

/// How long a person would take to type `bubble`: about 0.2s per character
/// with some jitter, between 0.6s and 6s, scaled by `speed` (0 disables).
std::chrono::milliseconds typing_delay(std::string_view bubble, double speed, std::mt19937 &rng);

} // namespace lilia::chat
