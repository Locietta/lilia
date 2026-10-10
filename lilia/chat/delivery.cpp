#include "delivery.h"

#include <algorithm>
#include <array>
#include <set>

#include <lilia/learn/segment.h>

namespace lilia::chat {

namespace {

std::string_view trim(std::string_view text) {
    constexpr std::string_view k_space = " \t\r\n";
    const auto first = text.find_first_not_of(k_space);
    if (first == std::string_view::npos) return {};
    const auto last = text.find_last_not_of(k_space);
    return text.substr(first, last - first + 1);
}

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

std::string first_lines(std::string_view text, usize count) {
    std::string out;
    usize kept = 0;
    while (!text.empty() && kept < count) {
        const auto newline = text.find('\n');
        const auto line = trim(text.substr(0, newline));
        text = newline == std::string_view::npos ? std::string_view{} : text.substr(newline + 1);
        if (line.empty()) continue;
        if (!out.empty()) out += '\n';
        out += line;
        ++kept;
    }
    return out;
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

namespace {

bool is_unicode_space(char32_t c) {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
           c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

bool is_quote(char32_t c) {
    return c == U'"' || c == U'\'' || c == U'“' || c == U'”' || c == U'‘' || c == U'’' || c == U'「' || c == U'」' || c == U'『' ||
           c == U'』';
}

bool is_separator(char32_t c) { return c == U'，' || c == U',' || c == U' ' || c == U'。' || c == U';' || c == U'\n'; }

/// str.isdigit() or is_english_letter() in MaiBot.
bool is_latin_or_digit(char32_t c) {
    return (c >= U'0' && c <= U'9') || (c >= 0xFF10 && c <= 0xFF19) || (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z');
}

/// re.sub(r"[^\S\r\n]*[\r\n]+[^\S\r\n]*", " ", text).strip(): line breaks
/// (and the spaces around them) become one space each run.
std::string flatten_lines(std::string_view text) {
    const auto chars = learn::utf8_chars(text);
    std::string out;
    for (usize i = 0; i < chars.size();) {
        if (!is_unicode_space(chars[i].cp)) {
            out += text.substr(chars[i].begin, chars[i].end - chars[i].begin);
            ++i;
            continue;
        }
        auto end = i;
        usize breaks = 0;
        bool in_break = false;
        for (; end < chars.size() && is_unicode_space(chars[end].cp); ++end) {
            const bool newline = chars[end].cp == U'\n' || chars[end].cp == U'\r';
            if (newline && !in_break) ++breaks;
            in_break = newline;
        }
        if (breaks == 0) {
            out += text.substr(chars[i].begin, chars[end - 1].end - chars[i].begin);
        } else {
            out.append(breaks, ' ');
        }
        i = end;
    }
    // .strip()
    const auto stripped = learn::utf8_chars(out);
    usize first = 0;
    usize last = stripped.size();
    while (first < last && is_unicode_space(stripped[first].cp)) ++first;
    while (last > first && is_unicode_space(stripped[last - 1].cp)) --last;
    return first == last ? std::string() : out.substr(stripped[first].begin, stripped[last - 1].end - stripped[first].begin);
}

} // namespace

namespace {

/// What joins two pieces again: a space for any mark (MaiBot restores the
/// mark; people in QQ groups mostly type a space instead of "，"), a newline
/// for a newline.
std::string soft(std::string_view separator) { return separator == "\n" ? "\n" : separator.empty() ? "" : " "; }

/// The pieces of `text` (MaiBot's first pass, before merging), keeping the
/// empty ones that only carry a space or newline.
std::vector<Segment> raw_pieces(std::string_view text) {
    const auto chars = learn::utf8_chars(text);
    const usize n = chars.size();
    const auto piece = [&](usize i) { return std::string(text.substr(chars[i].begin, chars[i].end - chars[i].begin)); };
    // Which characters sit inside paired quotes. A quote closes only on the
    // same character (or any ASCII quote on an ASCII one), as in MaiBot, so
    // “…” never closes.
    std::vector<bool> inside(n, false);
    bool in_quote = false;
    char32_t opened = 0;
    for (usize i = 0; i < n; ++i) {
        const auto c = chars[i].cp;
        if (!is_quote(c)) {
            inside[i] = in_quote;
        } else if (!in_quote) {
            in_quote = true;
            opened = c;
        } else if (c == opened || ((c == U'"' || c == U'\'') && (opened == U'"' || opened == U'\''))) {
            in_quote = false;
            opened = 0;
        }
    }
    std::vector<Segment> segments;
    std::string current;
    for (usize i = 0; i < n; ++i) {
        const auto c = chars[i].cp;
        if (!is_separator(c)) {
            current += piece(i);
            continue;
        }
        bool split = !inside[i];
        if (split && c != U'\n') {
            const auto colon = [](char32_t x) { return x == U':' || x == U'：'; };
            if ((i > 0 && colon(chars[i - 1].cp)) || (i + 1 < n && colon(chars[i + 1].cp))) split = false;
            if (split && c == U' ' && i > 0 && i + 1 < n) {
                const auto prev = chars[i - 1].cp;
                const auto next = chars[i + 1].cp;
                const auto dash = [](char32_t x) { return x == U'-' || x == U'—'; };
                // MaiBot keeps only Latin-Latin spaces; writers that space
                // English words inside Chinese ("move 完") would see the word
                // torn off its sentence, so either side is enough here.
                if (dash(prev) || dash(next) || is_latin_or_digit(prev) || is_latin_or_digit(next)) split = false;
            }
        }
        if (!split) {
            current += piece(i);
        } else {
            if (!current.empty()) {
                segments.push_back({.text = std::move(current), .separator = piece(i)});
            } else if (c == U' ' || c == U'\n') {
                segments.push_back({.separator = piece(i)});
            }
            current.clear();
        }
    }
    if (!current.empty()) segments.push_back({.text = std::move(current)});
    if (segments.empty()) return {{.text = std::string(text)}};
    return segments;
}

/// The last pass: pieces with text, line breaks flattened to spaces.
std::vector<Segment> finish_pieces(std::vector<Segment> merged) {
    std::vector<Segment> out;
    for (auto &segment : merged) {
        if (auto flat = flatten_lines(segment.text); !flat.empty())
            out.push_back({.text = std::move(flat), .separator = segment.separator});
    }
    return out;
}

} // namespace

std::vector<Segment> split_pieces(std::string_view text) {
    if (learn::utf8_chars(text).size() < 3)
        return text.empty() ? std::vector<Segment>{} : std::vector<Segment>{{.text = std::string(text)}};
    return finish_pieces(raw_pieces(text));
}

std::vector<Segment> split_segments(std::string_view text, std::mt19937 &rng) {
    std::uniform_real_distribution<double> chance(0.0, 1.0);
    const auto chars = learn::utf8_chars(text);
    const usize n = chars.size();
    if (n == 0) return {};
    if (n < 3) {
        if (chance(rng) < 0.01) {
            std::vector<Segment> singles;
            for (const auto &c : chars) singles.push_back({.text = std::string(text.substr(c.begin, c.end - c.begin))});
            return singles;
        }
        return {{.text = std::string(text)}};
    }
    auto segments = raw_pieces(text);

    // Merge neighbours at random; short replies merge more.
    const double merge = 1.0 - (n < 12 ? 0.2 : n < 32 ? 0.6 : 0.7);
    std::vector<Segment> merged;
    for (usize i = 0; i < segments.size();) {
        auto &here = segments[i];
        if (i + 1 < segments.size() && !here.text.empty() && here.separator != "\n" && chance(rng) < merge) {
            auto &next = segments[i + 1];
            merged.push_back(
                {.text = next.text.empty() ? here.text : here.text + soft(here.separator) + next.text, .separator = next.separator});
            i += 2;
        } else {
            merged.push_back(std::move(here));
            ++i;
        }
    }
    return finish_pieces(std::move(merged));
}

std::vector<Segment> pack_segments(std::vector<Segment> segments, usize max_count) {
    if (segments.size() <= max_count) return segments;
    if (max_count == 0) return {};
    const usize count = segments.size();
    std::set<usize> starts{0};
    for (usize i = 1; i < count && starts.size() < max_count; ++i) {
        if (segments[i].quote_previous) starts.insert(i);
    }
    // Then MaiBot's even split fills the remaining groups.
    usize start = 0;
    for (usize group = 0; group < max_count && starts.size() < max_count; ++group) {
        const usize remaining = count - start;
        const usize groups_left = max_count - group;
        starts.insert(start);
        start += (remaining + groups_left - 1) / groups_left;
    }
    const std::vector<usize> sorted(starts.begin(), starts.end());
    std::vector<Segment> packed;
    for (usize g = 0; g < sorted.size(); ++g) {
        const usize end = g + 1 < sorted.size() ? sorted[g + 1] : count;
        Segment joined{.separator = segments[end - 1].separator, .quote_previous = segments[sorted[g]].quote_previous};
        for (usize i = sorted[g]; i < end; ++i) {
            joined.text += segments[i].text;
            if (i + 1 < end) joined.text += soft(segments[i].separator);
        }
        packed.push_back(std::move(joined));
    }
    return packed;
}

namespace {

bool is_emoji(char32_t c) { return (c >= 0x1F000 && c <= 0x1FAFF) || c == 0x2764 || c == 0x263A || c == 0x2639; }

} // namespace

std::string strip_emoji(std::string_view text, const std::vector<char32_t> &kept, std::vector<char32_t> &removed) {
    const auto chars = learn::utf8_chars(text);
    std::string out;
    bool dropping = false; // right after a removed emoji: its selectors and joiners go too
    for (const auto &c : chars) {
        if (is_emoji(c.cp) && !std::ranges::contains(kept, c.cp)) {
            removed.push_back(c.cp);
            dropping = true;
            continue;
        }
        if (dropping && (c.cp == 0xFE0F || c.cp == 0x200D)) continue;
        dropping = false;
        out += text.substr(c.begin, c.end - c.begin);
    }
    if (removed.empty()) return std::string(text);
    // Tidy the gaps an emoji leaves ("好耶 😂 真的" -> "好耶 真的").
    std::string tidy;
    for (const char ch : out) {
        if (ch == ' ' && (tidy.empty() || tidy.back() == ' ' || tidy.back() == '\n')) continue;
        if (ch == '\n' && !tidy.empty() && tidy.back() == ' ') tidy.pop_back();
        tidy += ch;
    }
    while (!tidy.empty() && (tidy.back() == ' ' || tidy.back() == '\n')) tidy.pop_back();
    return tidy;
}

std::vector<char32_t> emoji_in(const std::vector<std::string> &texts) {
    std::vector<char32_t> found;
    for (const auto &text : texts) {
        for (const auto &c : learn::utf8_chars(text)) {
            if (is_emoji(c.cp) && !std::ranges::contains(found, c.cp)) found.push_back(c.cp);
        }
    }
    return found;
}

std::string_view emoji_feeling(char32_t emoji) {
    switch (emoji) {
        case U'😂':
        case U'🤣':
        case U'😆':
        case U'😹': return "笑死";
        case U'😭':
        case U'😢':
        case U'🥲': return "哭";
        case U'😅':
        case U'😓':
        case U'😬': return "尴尬";
        case U'🙈':
        case U'😳':
        case U'☺':
        case U'😊': return "害羞";
        case U'👀': return "吃瓜";
        case U'🤔': return "疑惑";
        case U'😡':
        case U'😠':
        case U'💢': return "生气";
        case U'👍':
        case U'👌': return "赞同";
        case U'❤':
        case U'🥰':
        case U'😍': return "喜欢";
        case U'🥺': return "可怜";
        case U'😏': return "得意";
        case U'🙄':
        case U'😑':
        case U'😐': return "无语";
        case U'😮':
        case U'😱':
        case U'😨':
        case U'😲': return "震惊";
        case U'😴':
        case U'🥱': return "困";
        default: return {};
    }
}

std::string replace_haha(std::string_view text, std::mt19937 &rng) {
    constexpr std::string_view k_ha = "哈";
    std::string out;
    for (usize at = 0; at < text.size();) {
        auto end = at;
        while (text.substr(end).starts_with(k_ha)) end += k_ha.size();
        if (end - at >= 2 * k_ha.size()) {
            // In front of more words, half the time the laugh just goes.
            const bool more = end < text.size() && text[end] != '\n';
            if (more && std::bernoulli_distribution(0.5)(rng)) {
                if (end < text.size() && text[end] == ' ') ++end;
            } else {
                out += std::bernoulli_distribution(0.5)(rng) ? "草" : "www";
            }
            at = end;
        } else if (end > at) {
            out += k_ha; // a lone 哈 ("哈？") is surprise, not a laugh
            at = end;
        } else {
            out += text[at++];
        }
    }
    return out;
}

bool laughs(std::string_view text) { return text.find("草") != std::string_view::npos || text.find("www") != std::string_view::npos; }

bool only_laughs(std::string_view text) {
    text = trim(text);
    if (text.empty()) return false;
    while (!text.empty()) {
        bool matched = false;
        for (const std::string_view part : {"草", "w", "W", "h", "H", "笑死", "哈", "笑", " ", "！", "!", "~", "～"}) {
            if (text.starts_with(part)) {
                text.remove_prefix(part.size());
                matched = true;
                break;
            }
        }
        if (!matched) return false;
    }
    return true;
}

std::string drop_leading_laugh(std::string_view text) {
    for (const std::string_view laugh : {"草", "www"}) {
        if (!text.starts_with(laugh)) continue;
        auto rest = text.substr(laugh.size());
        while (rest.starts_with('w')) rest.remove_prefix(1);
        while (rest.starts_with(' ')) rest.remove_prefix(1);
        if (!rest.empty() && !rest.starts_with('\n')) return std::string(rest);
    }
    return std::string(text);
}

std::string tighten_spacing(std::string_view text) {
    const auto chars = learn::utf8_chars(text);
    // "+" and "#" end words like C++ and C#.
    const auto latin = [](char32_t c) {
        return (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z') || (c >= U'0' && c <= U'9') || c == U'+' || c == U'#';
    };
    const auto wide = [](char32_t c) { return c >= 0x2E80; }; // CJK and full-width punctuation
    std::string out;
    for (usize i = 0; i < chars.size(); ++i) {
        if (chars[i].cp == U' ' && i > 0 && i + 1 < chars.size()) {
            const auto prev = chars[i - 1].cp;
            const auto next = chars[i + 1].cp;
            if ((wide(prev) && latin(next)) || (latin(prev) && wide(next))) continue;
        }
        out += text.substr(chars[i].begin, chars[i].end - chars[i].begin);
    }
    return out;
}

std::chrono::milliseconds typing_delay(std::string_view bubble, double speed) {
    if (speed <= 0) return std::chrono::milliseconds(0);
    const auto chars = learn::utf8_chars(bubble);
    const auto han = [](char32_t c) { return c >= 0x4E00 && c <= 0x9FFF; };
    const auto han_count = std::ranges::count_if(chars, [&](const learn::Utf8Char &c) { return han(c.cp); });
    double seconds = 0;
    if (han_count == 1 && count_code_points(trim(bubble)) == 1) {
        seconds = 0.3 * 3 + 0.3;
    } else {
        for (const auto &c : chars) seconds += han(c.cp) ? 0.3 : 0.15;
    }
    return std::chrono::milliseconds(static_cast<i64>(seconds / speed * 1000));
}

} // namespace lilia::chat
