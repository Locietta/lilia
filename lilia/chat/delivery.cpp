#include "delivery.h"

#include <algorithm>
#include <array>

namespace lilia::chat {

namespace {

/// Lines longer than this are broken up the way people type: one sentence,
/// or a few clauses, per message.
constexpr usize k_bubble_code_points = 16;

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

/// A space between two non-ASCII characters ("工具就是工具 拿来") separates
/// clauses; one next to a Latin word ("用 AI 出图") does not.
bool clause_space(std::string_view text, usize i) {
    if (text[i] != ' ' || i == 0 || i + 1 >= text.size()) return false;
    // A continuation byte before means the previous character was multi-byte;
    // a lead byte after means the next one is.
    return (static_cast<unsigned char>(text[i - 1]) & 0xC0) == 0x80 && static_cast<unsigned char>(text[i + 1]) >= 0xC0;
}

/// Breaks after "，,；;" and after a space between CJK clauses (keeping the
/// separator), then packs the clauses greedily into chunks of `limit` code
/// points; a single longer clause stays whole.
std::vector<std::string> split_clauses(std::string_view sentence, usize limit) {
    static constexpr std::array<std::string_view, 4> k_marks{"，", "；", ",", ";"};
    std::vector<std::string> clauses;
    usize start = 0;
    for (usize i = 0; i < sentence.size();) {
        usize width = 0;
        for (const auto mark : k_marks) {
            if (sentence.substr(i).starts_with(mark)) width = mark.size();
        }
        if (width == 0 && clause_space(sentence, i)) width = 1;
        if (width == 0) {
            ++i;
            continue;
        }
        i += width;
        clauses.emplace_back(sentence.substr(start, i - start));
        start = i;
    }
    if (start < sentence.size()) clauses.emplace_back(sentence.substr(start));

    std::vector<std::string> chunks;
    for (auto &clause : clauses) {
        if (!chunks.empty() && count_code_points(chunks.back()) + count_code_points(clause) <= limit) {
            chunks.back() += clause;
        } else {
            chunks.push_back(std::move(clause));
        }
    }
    return chunks;
}

struct Piece {
    std::string text;          ///< raw, with its trailing separator
    bool mid_sentence = false; ///< cut at a comma; merging it with the next piece reads naturally
    bool line_end = false;     ///< last piece of a line; merging needs a space
};

/// What is actually sent: no trailing comma, semicolon, period or space.
std::string finish(std::string_view text) {
    text = trim(text);
    for (bool trimmed = true; trimmed;) {
        trimmed = false;
        for (const std::string_view mark : {"。", "，", "；", ",", ";"}) {
            if (text.ends_with(mark)) {
                text = trim(text.substr(0, text.size() - mark.size()));
                trimmed = true;
            }
        }
    }
    return std::string(text);
}

} // namespace

std::string fit_length(std::string_view text, usize max_code_points) {
    // Whole sentences only: cutting into one leaves her trailing off
    // ("其实我想说的是"), which reads worse than a reply that runs long.
    static constexpr std::array<std::string_view, 6> k_ends{"。", "！", "？", "!", "?", "\n"};
    std::string out;
    usize used = 0;
    usize start = 0;
    for (usize i = 0; i <= text.size();) {
        usize width = 0;
        if (i < text.size()) {
            for (const auto mark : k_ends) {
                if (text.substr(i).starts_with(mark)) {
                    width = mark.size();
                    break;
                }
            }
            if (width == 0) {
                ++i;
                continue;
            }
            // Keep runs like "？？" with their sentence.
            for (bool more = true; more;) {
                more = false;
                for (const auto mark : k_ends) {
                    if (mark != "\n" && text.substr(i + width).starts_with(mark)) {
                        width += mark.size();
                        more = true;
                    }
                }
            }
        }
        const auto sentence = text.substr(start, i + width - start);
        const auto size = count_code_points(sentence);
        if (!out.empty() && used + size > max_code_points) break;
        out += sentence;
        used += size;
        if (i >= text.size() || used >= max_code_points) break;
        i += width;
        start = i;
    }
    while (out.ends_with("\n") || out.ends_with(" ")) out.pop_back();
    return out;
}

usize count_code_points(std::string_view text) {
    return static_cast<usize>(std::ranges::count_if(text, [](char c) { return (static_cast<unsigned char>(c) & 0xC0) != 0x80; }));
}

std::vector<std::string> split_bubbles(std::string_view text, usize max_bubbles) {
    std::vector<Piece> pieces;
    while (!text.empty()) {
        const auto newline = text.find('\n');
        const auto line = trim(text.substr(0, newline));
        text = newline == std::string_view::npos ? std::string_view{} : text.substr(newline + 1);
        if (line.empty()) continue;
        if (count_code_points(line) <= k_bubble_code_points) {
            pieces.push_back({.text = std::string(line)});
        } else {
            for (auto &sentence : split_sentences(line)) {
                auto chunks = split_clauses(sentence, k_bubble_code_points);
                for (usize i = 0; i < chunks.size(); ++i)
                    pieces.push_back({.text = std::move(chunks[i]), .mid_sentence = i + 1 < chunks.size()});
            }
        }
        pieces.back().line_end = true;
    }
    if (max_bubbles == 0) max_bubbles = 1;
    // Too many: merge the shortest neighbouring pair, preferring pieces of
    // one sentence; ties go to the later pair.
    while (pieces.size() > max_bubbles) {
        usize best = 0;
        std::pair<int, usize> best_cost{2, 0};
        for (usize i = 0; i + 1 < pieces.size(); ++i) {
            const std::pair<int, usize> cost{pieces[i].mid_sentence ? 0 : 1,
                                             count_code_points(pieces[i].text) + count_code_points(pieces[i + 1].text)};
            if (cost <= best_cost) {
                best = i;
                best_cost = cost;
            }
        }
        auto &left = pieces[best];
        auto right = std::move(pieces[best + 1]);
        pieces.erase(pieces.begin() + static_cast<std::ptrdiff_t>(best + 1));
        // Pieces keep their separators ("，", a space), so only lines need one added.
        if (left.line_end) left.text += ' ';
        left.text += right.text;
        left.mid_sentence = right.mid_sentence;
        left.line_end = right.line_end;
    }
    std::vector<std::string> bubbles;
    for (const auto &piece : pieces) {
        if (auto bubble = finish(piece.text); !bubble.empty()) bubbles.push_back(std::move(bubble));
    }
    return bubbles;
}

std::vector<std::string> whole_message(std::string_view text) {
    std::string out;
    while (!text.empty()) {
        const auto newline = text.find('\n');
        const auto line = trim(text.substr(0, newline));
        text = newline == std::string_view::npos ? std::string_view{} : text.substr(newline + 1);
        if (line.empty()) continue;
        if (!out.empty()) out += '\n';
        out += line;
    }
    if (auto message = finish(out); !message.empty()) return {std::move(message)};
    return {};
}

std::chrono::milliseconds typing_delay(std::string_view bubble, double speed, std::mt19937 &rng) {
    if (speed <= 0) return std::chrono::milliseconds(0);
    std::uniform_real_distribution<double> jitter(0.8, 1.25);
    const double seconds = std::clamp(0.2 * static_cast<double>(count_code_points(bubble)) * jitter(rng), 0.6, 6.0) / speed;
    return std::chrono::milliseconds(static_cast<i64>(seconds * 1000));
}

} // namespace lilia::chat
