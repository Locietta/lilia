#include "delivery.h"

#include <algorithm>
#include <array>

namespace lilia::chat {

namespace {

constexpr usize k_long_line_code_points = 45;

std::string_view trim(std::string_view text) {
    constexpr std::string_view k_space = " \t\r\n";
    const auto first = text.find_first_not_of(k_space);
    if (first == std::string_view::npos) return {};
    const auto last = text.find_last_not_of(k_space);
    return text.substr(first, last - first + 1);
}

/// Breaks after "。！？!?" (keeping the mark) so a long paragraph becomes
/// several messages.
std::vector<std::string> split_sentences(std::string_view line) {
    static constexpr std::array<std::string_view, 5> k_ends{"。", "！", "？", "!", "?"};
    std::vector<std::string> pieces;
    std::string current;
    for (usize i = 0; i < line.size();) {
        bool ended = false;
        for (const auto end : k_ends) {
            if (line.substr(i).starts_with(end)) {
                current += end;
                i += end.size();
                ended = true;
                break;
            }
        }
        if (!ended) {
            current.push_back(line[i++]);
            continue;
        }
        // Keep runs like "？？" or "！？" together.
        bool more = false;
        for (const auto end : k_ends) more = more || line.substr(i).starts_with(end);
        if (more) continue;
        if (auto piece = trim(current); !piece.empty()) pieces.emplace_back(piece);
        current.clear();
    }
    if (auto piece = trim(current); !piece.empty()) pieces.emplace_back(piece);
    return pieces;
}

std::string drop_trailing_period(std::string text) {
    while (text.ends_with("。")) text.resize(text.size() - std::string_view("。").size());
    return text;
}

} // namespace

std::string fit_length(std::string_view text, usize max_code_points) {
    static constexpr std::array<std::string_view, 10> k_breaks{"。", "！", "？", "，", "；", "!", "?", ",", ";", "\n"};
    std::string out;
    usize used = 0;
    usize clause_start = 0;
    for (usize i = 0; i <= text.size();) {
        usize width = 0;
        if (i < text.size()) {
            for (const auto mark : k_breaks) {
                if (text.substr(i).starts_with(mark)) {
                    width = mark.size();
                    break;
                }
            }
        }
        if (i < text.size() && width == 0) {
            ++i;
            continue;
        }
        const auto clause = text.substr(clause_start, i + width - clause_start);
        const auto size = count_code_points(clause);
        if (used + size > max_code_points) {
            if (out.empty()) {
                // Cut the first clause at a code point boundary.
                usize kept = 0;
                usize bytes = 0;
                while (bytes < clause.size() && kept < max_code_points) {
                    ++bytes;
                    while (bytes < clause.size() && (static_cast<unsigned char>(clause[bytes]) & 0xC0) == 0x80) ++bytes;
                    ++kept;
                }
                out = clause.substr(0, bytes);
            }
            break;
        }
        out += clause;
        used += size;
        if (i >= text.size()) break;
        i += width;
        clause_start = i;
    }
    // A clause cut off mid-list should not end on a comma.
    while (out.ends_with("，") || out.ends_with(",") || out.ends_with("；") || out.ends_with(";") || out.ends_with("\n")) {
        out.resize(out.size() - (out.ends_with("，") || out.ends_with("；") ? std::string_view("，").size() : 1));
    }
    return out;
}

usize count_code_points(std::string_view text) {
    return static_cast<usize>(std::ranges::count_if(text, [](char c) { return (static_cast<unsigned char>(c) & 0xC0) != 0x80; }));
}

std::vector<std::string> split_bubbles(std::string_view text, usize max_bubbles) {
    std::vector<std::string> pieces;
    while (!text.empty()) {
        const auto newline = text.find('\n');
        const auto line = trim(text.substr(0, newline));
        text = newline == std::string_view::npos ? std::string_view{} : text.substr(newline + 1);
        if (line.empty()) continue;
        if (count_code_points(line) > k_long_line_code_points) {
            for (auto &piece : split_sentences(line)) pieces.push_back(std::move(piece));
        } else {
            pieces.emplace_back(line);
        }
    }
    if (max_bubbles == 0) max_bubbles = 1;
    while (pieces.size() > max_bubbles) {
        auto last = std::move(pieces.back());
        pieces.pop_back();
        pieces.back() += " " + last;
    }
    std::vector<std::string> bubbles;
    for (auto &piece : pieces) {
        auto bubble = drop_trailing_period(std::move(piece));
        if (!trim(bubble).empty()) bubbles.push_back(std::move(bubble));
    }
    return bubbles;
}

std::chrono::milliseconds typing_delay(std::string_view bubble, double speed, std::mt19937 &rng) {
    if (speed <= 0) return std::chrono::milliseconds(0);
    std::uniform_real_distribution<double> jitter(0.8, 1.25);
    const double seconds = std::clamp(0.2 * static_cast<double>(count_code_points(bubble)) * jitter(rng), 0.6, 6.0) / speed;
    return std::chrono::milliseconds(static_cast<i64>(seconds * 1000));
}

} // namespace lilia::chat
