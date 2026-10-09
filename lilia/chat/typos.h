#pragma once

#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <lilia/chat/delivery.h>
#include <lilia/config.h>

namespace lilia::chat {

// Homophone typos, after MaiBot's src/chat/utils/typo_generator.py: words
// come from jieba, readings from pypinyin (vendored in lilia/data/pinyin),
// character frequencies from jieba's dictionary. Readings of words come from
// pypinyin's phrase table for jieba's dictionary words and per character
// otherwise (pypinyin itself would also read unknown words in context).

/// A sentence with typos and, sometimes, the correction to send after it.
struct Typoed {
    std::string text;
    /// The right word or character, to send after the mistyped message;
    /// empty when there is none to send.
    std::string correction;
};

/// create_typo_sentence: each word may become a homophone word
/// (word_replace_rate), else each of its characters a homophone character
/// (error_rate, lowered inside longer words), favouring frequent ones. Half
/// the time a correction is offered for one of the typos.
Typoed make_typos(std::string_view sentence, const TypoConfig &config, std::mt19937 &rng);

/// _apply_response_post_processing: typos for each segment; when there is a
/// correction, half the time the mistyped segment goes out followed by the
/// correction (quoting it), otherwise the segment goes out clean.
std::vector<Segment> add_typos(std::vector<Segment> segments, const TypoConfig &config, std::mt19937 &rng);

/// The TONE3 reading MaiBot uses for `c` (pypinyin's first reading); `c`
/// itself when it has none.
std::string reading_of(char32_t c);

/// The readings of a word: in context for jieba dictionary words, per
/// character otherwise.
std::vector<std::string> readings_of(std::string_view word);

/// _get_similar_frequency_chars: up to five homophones of `c` (also those of
/// `other_tone` when not empty), frequent enough and likely to replace it,
/// most likely first.
std::vector<char32_t> similar_chars(char32_t c, std::string_view reading, std::string_view other_tone, double min_freq);

/// _get_word_homophones: up to five dictionary words that sound like
/// `word`, at least a tenth as frequent, best first.
std::vector<std::string> word_homophones(std::string_view word, double min_freq);

/// The character's frequency, 0..1000 (MaiBot's char_frequency).
double char_frequency(char32_t c);

} // namespace lilia::chat
