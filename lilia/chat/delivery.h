#pragma once

#include <chrono>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <lighter/types.hpp>

namespace lilia::chat {

using namespace lighter::types;

/// One piece of a reply on its way out: the text, the separator that
/// followed it in the reply (restored when pieces are packed together), and
/// whether it is a typo correction that quotes the message before it.
struct Segment {
    std::string text;
    std::string separator;
    bool quote_previous = false;
    bool operator==(const Segment &) const = default;
};

/// MaiBot's rule splitter (_split_into_sentence_segments): breaks at "，"
/// "," "。" ";", spaces and newlines, dropping the mark; never inside quotes,
/// next to a colon, beside a dash, or at a space next to a Latin letter or
/// digit (MaiBot: only between two); a newline always breaks. Neighbouring pieces then merge at random
/// (pairwise, never across a newline) with probability 0.8 for text under 12
/// characters, 0.4 under 32 and 0.3 otherwise. Text under 3 characters stays
/// whole, except 1% of the time it goes out one character at a time.
std::vector<Segment> split_segments(std::string_view text, std::mt19937 &rng);

/// split_segments without the random merging (and without the 1% split into
/// single characters): the pieces before they merge.
std::vector<Segment> split_pieces(std::string_view text);

/// MaiBot's _merge_processed_segments_to_max_count: packs `segments` into at
/// most `max_count` messages of about even size; a typo correction keeps
/// starting a message of its own when possible. Pieces are joined by a space
/// (or their newline), as people type, not by the mark they were split at.
std::vector<Segment> pack_segments(std::vector<Segment> segments, usize max_count);

/// Replaces every "哈哈…" with "草" or "www" (at random), or, in front of
/// more words, half the time drops it: models laugh in 哈哈, people in these
/// groups do not.
std::string replace_haha(std::string_view text, std::mt19937 &rng);

/// `text` without a leading "草"/"www" laugh when more follows it.
std::string drop_leading_laugh(std::string_view text);

/// Whether `text` laughs (草, www).
bool laughs(std::string_view text);

/// Removes the spaces models put between Chinese and Latin letters or digits
/// ("PK 输了" -> "PK输了"); people rarely type them.
std::string tighten_spacing(std::string_view text);

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

/// Pictographic emoji in `text` (with their variation selectors and
/// joiners), except those in `kept`; each removed one is appended to
/// `removed`. People in QQ groups rarely type emoji unless one is a meme of
/// its own, while models sprinkle them.
std::string strip_emoji(std::string_view text, const std::vector<char32_t> &kept, std::vector<char32_t> &removed);

/// Emoji appearing in `texts` (e.g. learned expressions): the ones people
/// actually use.
std::vector<char32_t> emoji_in(const std::vector<std::string> &texts);

/// The feeling an emoji stood for, to send as a sticker instead ("笑死" for
/// 😂); empty when there is no obvious one.
std::string_view emoji_feeling(char32_t emoji);

/// MaiBot's calculate_typing_time, the pause before the second and later
/// messages of a reply: 0.3s per Han character and 0.15s per other
/// character (1.2s for a lone Han character), divided by `speed`; 0 for
/// speed 0.
std::chrono::milliseconds typing_delay(std::string_view bubble, double speed);

} // namespace lilia::chat
