#pragma once

#include <functional>
#include <string>
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

/// Decodes the character at `text[at]` and moves `at` past it; a stray byte
/// becomes U+FFFD of one byte.
char32_t next_char(std::string_view text, usize &at);

/// Decodes UTF-8 (see next_char).
std::vector<Utf8Char> utf8_chars(std::string_view text);

/// One code point as UTF-8.
std::string encode_utf8(char32_t c);

/// Chinese word segmentation: a port of jieba 0.42.1's `jieba.cut` in its
/// default mode (accurate, with the HMM for runs the dictionary does not
/// know), over jieba's own dictionary and model, vendored in lilia/data/jieba
/// (MIT). The tokens are slices of `text` covering all of it in order;
/// whitespace and punctuation come out as tokens of their own, as in jieba.
/// The dictionary (349k words) is loaded on first use.
std::vector<std::string_view> cut(std::string_view text);

/// The jieba dictionary's frequency for `word` (its last listing when it has
/// two); 0 when it is not a dictionary word.
i64 dictionary_frequency(std::string_view word);

/// Calls `visit(word, frequency)` for every line of the jieba dictionary, in
/// file order (duplicates included).
void for_each_dictionary_entry(const std::function<void(std::string_view, i64)> &visit);

/// Walks the dictionary trie: the node reached from `node` by `c`, or 0
/// (also the root) when no dictionary word continues that way.
u32 dictionary_child(u32 node, char32_t c);

/// The frequency of the word ending at `node`; 0 for prefixes.
i64 dictionary_node_frequency(u32 node);

} // namespace lilia::learn
