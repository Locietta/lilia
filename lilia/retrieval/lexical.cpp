#include "lexical.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

#include <lilia/learn/segment.h>

namespace lilia::retrieval {

namespace {

constexpr double k_k1 = 1.2;
constexpr double k_b = 0.75;

bool is_han(char32_t c) { return (c >= 0x3400 && c <= 0x9fff) || (c >= 0xf900 && c <= 0xfaff); }

bool is_ascii_word(char32_t c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

} // namespace

std::vector<std::string> terms(std::string_view text) {
    std::vector<std::string> out;
    for (const auto word : learn::cut(text)) {
        const auto chars = learn::utf8_chars(word);
        if (chars.size() < 2) continue;
        if (std::ranges::all_of(chars, [](const learn::Utf8Char &c) { return is_ascii_word(c.cp); })) {
            std::string lower(word);
            std::ranges::transform(lower, lower.begin(),
                                   [](char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c; });
            out.push_back(std::move(lower));
        } else if (std::ranges::any_of(chars, [](const learn::Utf8Char &c) { return is_han(c.cp); })) {
            out.emplace_back(word);
        }
    }
    const auto chars = learn::utf8_chars(text);
    for (usize i = 0; i + 1 < chars.size(); ++i) {
        if (is_han(chars[i].cp) && is_han(chars[i + 1].cp))
            out.emplace_back(text.substr(chars[i].begin, chars[i + 1].end - chars[i].begin));
    }
    return out;
}

std::vector<double> bm25(const std::vector<std::string> &query, const std::vector<const std::vector<std::string> *> &documents) {
    std::vector<double> scores(documents.size(), 0.0);
    if (query.empty() || documents.empty()) return scores;
    const std::unordered_set<std::string_view> wanted(query.begin(), query.end());
    std::unordered_map<std::string_view, usize> frequency; // documents containing each query term
    double total_length = 0;
    for (const auto *document : documents) {
        total_length += static_cast<double>(document->size());
        std::unordered_set<std::string_view> seen;
        for (const auto &term : *document) {
            if (wanted.contains(term) && seen.insert(term).second) ++frequency[term];
        }
    }
    const double count = static_cast<double>(documents.size());
    const double average = std::max(total_length / count, 1.0);
    for (usize i = 0; i < documents.size(); ++i) {
        const auto &document = *documents[i];
        std::unordered_map<std::string_view, usize> occurrences;
        for (const auto &term : document) {
            if (wanted.contains(term)) ++occurrences[term];
        }
        const double length = static_cast<double>(document.size());
        for (const auto term : wanted) {
            const auto found = occurrences.find(term);
            if (found == occurrences.end()) continue;
            const double df = static_cast<double>(frequency[term]);
            const double idf = std::log(1.0 + (count - df + 0.5) / (df + 0.5));
            const double tf = static_cast<double>(found->second);
            scores[i] += idf * tf * (k_k1 + 1) / (tf + k_k1 * (1 - k_b + k_b * length / average));
        }
    }
    return scores;
}

} // namespace lilia::retrieval
