#pragma once

#include <string_view>
#include <vector>

#include <lilia/error.h>

namespace lilia::learn {

/// One character of UTF-8 text: its code point and byte range.
struct Utf8Char {
    char32_t cp = 0;
    usize begin = 0;
    usize end = 0;
};

/// Decodes UTF-8; a stray byte becomes U+FFFD of one byte.
std::vector<Utf8Char> utf8_chars(std::string_view text);

/// Chinese word segmentation: a port of jieba 0.42.1's `jieba.cut` in its
/// default mode (accurate, with the HMM for runs the dictionary does not
/// know), over jieba's own dictionary and model, vendored in lilia/data/jieba
/// (MIT). The tokens are slices of `text` covering all of it in order;
/// whitespace and punctuation come out as tokens of their own, as in jieba.
/// The dictionary (349k words) is loaded on first use.
std::vector<std::string_view> cut(std::string_view text);

} // namespace lilia::learn
