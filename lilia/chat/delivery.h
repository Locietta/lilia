#pragma once

#include <chrono>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <lighter/types.hpp>

namespace lilia::chat {

using namespace lighter::types;

/// Splits a cleaned reply into at most `max_bubbles` QQ messages, the way
/// people type: one per line; a long line becomes one message per sentence,
/// and a long sentence is broken at commas (or spaces between CJK clauses).
/// Extra pieces merge with their shortest neighbour. Bubbles do not end in
/// "。" or a comma.
std::vector<std::string> split_bubbles(std::string_view text, usize max_bubbles = 3);

/// The first `count` non-blank lines of `text`. Each line is a complete
/// message, so this drops add-on messages without cutting anything short.
std::string first_lines(std::string_view text, usize count);

/// The whole text as one message (lines kept, blank lines dropped); empty
/// when there is nothing to send.
std::vector<std::string> whole_message(std::string_view text);

/// Keeps whole sentences (ending in "。！？!?" or a newline) from the start of
/// `text` while they fit in `max_code_points`. A sentence is never cut: the
/// first one is kept even when it alone is longer. This is a safety net
/// behind the style budget in the prompt, not a way to shorten replies.
std::string fit_length(std::string_view text, usize max_code_points);

/// Characters (UTF-8 code points) in `text`.
usize count_code_points(std::string_view text);

/// How long a person would take to type `bubble`: about 0.2s per character
/// with some jitter, between 0.6s and 6s, scaled by `speed` (0 disables).
std::chrono::milliseconds typing_delay(std::string_view bubble, double speed, std::mt19937 &rng);

} // namespace lilia::chat
