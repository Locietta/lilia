#include "typos.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>
#include <span>
#include <unordered_map>

#include <xmake/bin2obj/pinyin_chars_txt.hpp>
#include <xmake/bin2obj/pinyin_words_txt.hpp>

#include <lilia/learn/segment.h>

namespace lilia::chat {

namespace {

constexpr double k_max_freq_diff = 200; ///< MaiBot's max_freq_diff
constexpr usize k_candidates = 5;

std::string_view as_text(std::span<const std::byte> data) { return {reinterpret_cast<const char *>(data.data()), data.size()}; }

/// _is_chinese_char
bool is_han(char32_t c) { return c >= 0x4E00 && c <= 0x9FFF; }

void append_utf8(std::string &out, char32_t c) {
    if (c < 0x80) {
        out += static_cast<char>(c);
    } else if (c < 0x800) {
        out += static_cast<char>(0xC0 | (c >> 6));
        out += static_cast<char>(0x80 | (c & 0x3F));
    } else if (c < 0x10000) {
        out += static_cast<char>(0xE0 | (c >> 12));
        out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (c & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (c >> 18));
        out += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (c & 0x3F));
    }
}

std::string utf8_of(char32_t c) {
    std::string out;
    append_utf8(out, c);
    return out;
}

/// Calls `visit(fields)` for every line of `data`, split at spaces.
void for_each_line(std::string_view data, const std::function<void(const std::vector<std::string_view> &)> &visit) {
    std::vector<std::string_view> fields;
    for (usize start = 0; start < data.size();) {
        auto end = data.find('\n', start);
        if (end == std::string_view::npos) end = data.size();
        const auto line = data.substr(start, end - start);
        start = end + 1;
        fields.clear();
        for (usize from = 0; from < line.size();) {
            const auto space = std::min(line.find(' ', from), line.size());
            if (space > from) fields.push_back(line.substr(from, space - from));
            from = space + 1;
        }
        if (fields.size() >= 2) visit(fields);
    }
}

struct Pinyin {
    std::unordered_map<char32_t, std::string> first;              ///< character -> reading
    std::unordered_map<std::string, std::vector<char32_t>> chars; ///< _create_pinyin_dict, code point order
    std::unordered_map<std::string_view, std::vector<std::string>> words;
};

const Pinyin &pinyin() {
    static const Pinyin table = [] {
        Pinyin p;
        for_each_line(as_text(xmake::pinyin_chars_txt), [&p](const std::vector<std::string_view> &fields) {
            const auto chars = learn::utf8_chars(fields[0]);
            if (chars.size() != 1) return;
            p.first.emplace(chars[0].cp, fields[1]);
            p.chars[std::string(fields[1])].push_back(chars[0].cp);
        });
        for_each_line(as_text(xmake::pinyin_words_txt), [&p](const std::vector<std::string_view> &fields) {
            p.words.emplace(fields[0], std::vector<std::string>(fields.begin() + 1, fields.end()));
        });
        return p;
    }();
    return table;
}

/// _load_or_create_char_frequency: each Han character's summed word
/// frequencies over jieba's dictionary, scaled so the most frequent is 1000.
const std::unordered_map<char32_t, double> &frequencies() {
    static const auto table = [] {
        std::unordered_map<char32_t, i64> sums;
        learn::for_each_dictionary_entry([&sums](std::string_view word, i64 freq) {
            for (usize at = 0; at < word.size();) {
                if (const auto c = learn::next_char(word, at); is_han(c)) sums[c] += freq;
            }
        });
        i64 most = 1;
        for (const auto &[c, sum] : sums) most = std::max(most, sum);
        std::unordered_map<char32_t, double> scaled;
        for (const auto &[c, sum] : sums) scaled.emplace(c, static_cast<double>(sum) / static_cast<double>(most) * 1000);
        return scaled;
    }();
    return table;
}

/// _calculate_replacement_probability
double replacement_probability(double original, double target) {
    if (target > original) return 1.0;
    const double diff = original - target;
    if (diff > k_max_freq_diff) return 0.0;
    return std::exp(-3 * diff / k_max_freq_diff);
}

/// _get_similar_tone_pinyin: the same syllable in another tone.
std::string other_tone(std::string_view reading, std::mt19937 &rng) {
    if (reading.empty()) return std::string(reading);
    if (reading.back() < '0' || reading.back() > '9') return std::string(reading) + "1";
    const auto base = std::string(reading.substr(0, reading.size() - 1));
    const int tone = reading.back() - '0';
    if (tone < 1 || tone > 4) return base + std::to_string(std::uniform_int_distribution<int>(1, 4)(rng));
    std::vector<int> tones;
    for (int t = 1; t <= 4; ++t) {
        if (t != tone) tones.push_back(t);
    }
    return base + std::to_string(tones[std::uniform_int_distribution<usize>(0, tones.size() - 1)(rng)]);
}

template <typename T>
const T &pick(const std::vector<T> &items, std::mt19937 &rng) {
    return items[std::uniform_int_distribution<usize>(0, items.size() - 1)(rng)];
}

} // namespace

double char_frequency(char32_t c) {
    const auto &table = frequencies();
    const auto found = table.find(c);
    return found == table.end() ? 0.0 : found->second;
}

std::string reading_of(char32_t c) {
    const auto &table = pinyin().first;
    const auto found = table.find(c);
    return found == table.end() ? utf8_of(c) : found->second;
}

std::vector<std::string> readings_of(std::string_view word) {
    const auto &words = pinyin().words;
    if (const auto found = words.find(word); found != words.end()) return found->second;
    std::vector<std::string> readings;
    for (usize at = 0; at < word.size();) readings.push_back(reading_of(learn::next_char(word, at)));
    return readings;
}

std::vector<char32_t> similar_chars(char32_t c, std::string_view reading, std::string_view tone, double min_freq) {
    const auto &chars = pinyin().chars;
    std::vector<char32_t> homophones;
    const auto add = [&](std::string_view key) {
        if (const auto found = chars.find(std::string(key)); found != chars.end())
            homophones.insert(homophones.end(), found->second.begin(), found->second.end());
    };
    if (!tone.empty()) add(tone);
    add(reading);
    const double original = char_frequency(c);
    std::vector<std::pair<char32_t, double>> candidates;
    for (const auto h : homophones) {
        const double freq = char_frequency(h);
        if (h == c || freq < min_freq) continue;
        if (const double p = replacement_probability(original, freq); p > 0) candidates.emplace_back(h, p);
    }
    std::ranges::stable_sort(candidates, std::greater<>{}, &std::pair<char32_t, double>::second);
    std::vector<char32_t> out;
    for (usize i = 0; i < candidates.size() && i < k_candidates; ++i) out.push_back(candidates[i].first);
    return out;
}

std::vector<std::string> word_homophones(std::string_view word, double min_freq) {
    const auto readings = readings_of(word);
    if (learn::utf8_chars(word).size() <= 1) return {};
    const auto &chars = pinyin().chars;
    std::vector<const std::vector<char32_t> *> options;
    for (const auto &reading : readings) {
        const auto found = chars.find(reading);
        if (found == chars.end() || found->second.empty()) return {};
        options.push_back(&found->second);
    }
    const double least = static_cast<double>(learn::dictionary_frequency(word)) * 0.1;
    // itertools.product over the homophones of each syllable, in order; the
    // dictionary trie prunes combinations no word starts with.
    std::vector<std::pair<std::string, double>> found;
    std::vector<char32_t> picked;
    std::string current;
    const std::function<void(usize, u32)> walk = [&](usize depth, u32 node) {
        for (const auto c : *options[depth]) {
            const auto child = learn::dictionary_child(node, c);
            if (child == 0) continue;
            const auto size = current.size();
            append_utf8(current, c);
            picked.push_back(c);
            if (depth + 1 < options.size()) {
                walk(depth + 1, child);
            } else if (current != word) {
                const auto freq = static_cast<double>(learn::dictionary_node_frequency(child));
                if (freq > 0 && freq >= least) {
                    double sum = 0;
                    for (const auto p : picked) sum += char_frequency(p);
                    const double score = freq * 0.7 + sum / static_cast<double>(picked.size()) * 0.3;
                    if (score >= min_freq) found.emplace_back(current, score);
                }
            }
            current.resize(size);
            picked.pop_back();
        }
    };
    walk(0, 0);
    std::ranges::stable_sort(found, std::greater<>{}, &std::pair<std::string, double>::second);
    std::vector<std::string> out;
    for (usize i = 0; i < found.size() && i < k_candidates; ++i) out.push_back(std::move(found[i].first));
    return out;
}

Typoed make_typos(std::string_view sentence, const TypoConfig &config, std::mt19937 &rng) {
    std::uniform_real_distribution<double> chance(0.0, 1.0);
    std::string result;
    std::vector<std::pair<std::string, std::string>> word_typos; ///< (typo, right)
    std::vector<std::pair<char32_t, char32_t>> char_typos;
    // One character of a word: maybe a homophone in its place.
    const auto try_char = [&](char32_t c, std::string_view reading, double rate) -> std::optional<char32_t> {
        if (chance(rng) >= rate) return std::nullopt;
        const auto tone = chance(rng) < config.tone_error_rate ? other_tone(reading, rng) : std::string();
        const auto similar = similar_chars(c, reading, tone, config.min_freq);
        if (similar.empty()) return std::nullopt;
        const auto typo = pick(similar, rng);
        if (chance(rng) >= replacement_probability(char_frequency(c), char_frequency(typo))) return std::nullopt;
        return typo;
    };
    for (const auto word : learn::cut(sentence)) {
        const auto chars = learn::utf8_chars(word);
        if (std::ranges::none_of(chars, [](const learn::Utf8Char &c) { return is_han(c.cp); })) {
            result += word;
            continue;
        }
        const auto readings = readings_of(word);
        if (chars.size() > 1 && chance(rng) < config.word_replace_rate) {
            if (const auto homophones = word_homophones(word, config.min_freq); !homophones.empty()) {
                const auto &typo = pick(homophones, rng);
                result += typo;
                word_typos.emplace_back(typo, std::string(word));
                continue;
            }
        }
        const double rate = config.error_rate * std::pow(0.7, static_cast<double>(chars.size()) - 1);
        for (usize i = 0; i < chars.size(); ++i) {
            const auto reading = i < readings.size() ? readings[i] : reading_of(chars[i].cp);
            if (const auto typo = try_char(chars[i].cp, reading, rate)) {
                append_utf8(result, *typo);
                char_typos.emplace_back(*typo, chars[i].cp);
            } else {
                result += word.substr(chars[i].begin, chars[i].end - chars[i].begin);
            }
        }
    }
    std::string correction;
    if (chance(rng) < 0.5) {
        if (!word_typos.empty()) {
            correction = pick(word_typos, rng).second;
        } else if (!char_typos.empty()) {
            correction = utf8_of(pick(char_typos, rng).second);
        }
    }
    return {.text = std::move(result), .correction = std::move(correction)};
}

std::vector<Segment> add_typos(std::vector<Segment> segments, const TypoConfig &config, std::mt19937 &rng) {
    if (!config.enabled) return segments;
    std::uniform_real_distribution<double> chance(0.0, 1.0);
    std::vector<Segment> out;
    for (auto &segment : segments) {
        auto typoed = make_typos(segment.text, config, rng);
        if (typoed.correction.empty()) {
            out.push_back({.text = std::move(typoed.text), .separator = std::move(segment.separator)});
        } else if (chance(rng) < 0.5) {
            out.push_back({.text = std::move(typoed.text), .separator = std::move(segment.separator)});
            out.push_back({.text = std::move(typoed.correction), .quote_previous = config.correction_quote});
        } else {
            out.push_back(std::move(segment));
        }
    }
    return out;
}

} // namespace lilia::chat
