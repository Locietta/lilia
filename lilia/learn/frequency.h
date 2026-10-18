#pragma once

#include <string>
#include <string_view>
#include <vector>

#include <lilia/store/store.h>

namespace lilia::learn {

// High-frequency words and phrases of a chat, after MaiBot's
// services/message_word_frequency_service.py: messages are cut with jieba
// (see segment.h), ASCII tech terms are added, stop words, numbers and
// punctuation are dropped, and runs of 2-3 tokens count as phrases. Each
// batch's tallies are merged into the chat's ranking (Store::
// merge_frequent_terms); slang that is also one of the chat's frequent terms
// ranks first in the jargon reference.

/// MaiBot's per-batch limits: terms seen fewer times are dropped, the rest
/// capped; the chat keeps its top `k_max_frequent_terms`.
inline constexpr i64 k_frequent_min_count = 2;
inline constexpr usize k_frequent_batch_limit = 1000;
inline constexpr usize k_max_frequent_terms = 1000;

/// _normalize_token: trimmed, lower-cased, stripped of surrounding
/// punctuation, inner whitespace collapsed to one space.
std::string normalize_term(std::string_view token);

/// _normalize_match_text (jargon_context_matcher.py): lower-cased, whitespace
/// runs collapsed to one space, trimmed.
std::string normalize_match_text(std::string_view text);

/// _tokenize_meaningful_terms: the meaningful jieba tokens of `text` (URLs,
/// CQ codes and @mentions removed first) in order, then ASCII tech terms
/// (e.g. "c++", "node.js") jieba split apart.
std::vector<std::string> meaningful_terms(std::string_view text);

/// _iter_phrases: runs of 2 then 3 tokens, joined without spaces when all
/// are Han, with spaces otherwise; Han phrases need four Han characters,
/// others a Latin letter.
std::vector<std::string> term_phrases(const std::vector<std::string> &tokens);

/// _build_message_word_frequency_terms: words and phrases seen at least
/// `min_count` times in `texts`, most frequent first, at most `limit`.
std::vector<store::TermTally> count_terms(const std::vector<std::string> &texts, usize limit = k_frequent_batch_limit,
                                          i64 min_count = k_frequent_min_count);

} // namespace lilia::learn
