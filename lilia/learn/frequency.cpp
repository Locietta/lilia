#include "frequency.h"

#include <algorithm>
#include <ranges>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include <lilia/learn/segment.h>

namespace lilia::learn {

namespace {

// Python's regexes over str are Unicode-aware; the character classes below
// follow them closely enough for chat text (letters, digits, Han, kana,
// Hangul, full-width forms).

const std::unordered_set<std::string_view> k_stop_words{
    // _CN_STOP_WORDS
    "一些", "一样", "一定", "一直", "一条", "一种", "一起", "一下", "一个", "一会", "一是", "不是", "不过", "不如", "不能", "不要", "不用",
    "不会", "但原", "为了", "为什么", "也是", "也许", "于是", "什么", "他们", "以后", "以前", "以及", "按照", "但是", "你们", "你的",
    "其实", "其它", "其他", "只是", "只要", "可以", "可是", "没有", "各位", "回复", "因为", "因此", "如果", "它们", "对于", "对方", "就是",
    "已经", "并且", "怎么", "怎样", "总之", "我们", "我的", "所以", "所有", "是否", "是不是", "未知", "未知用户", "有人", "有点", "有没有",
    "无法访问", "然后", "现在", "关于", "根据", "由于", "的话", "直接", "真的", "真是", "自己", "虽然", "这是", "通过", "这个", "这些",
    "这么", "这里", "这样", "那个", "那些", "那么", "那里", "那样", "还是", "还有", "或者", "而且", "来说", "知道", "一下子", "一大",
    "不了", "之前", "出来", "只能", "多少", "今天", "图片", "喜欢", "好像", "好看", "哈哈", "哈哈哈", "开始", "小时", "分钟", "应该",
    "我要", "时候", "时间", "消息", "确实", "聊天", "表情", "觉得", "需要", "感觉", "看看", "看到", "群友", "使用", "可能", "问题", "这种",
    "里面", "内容", "用户", "啊啊啊", "东西",
    // _EN_STOP_WORDS
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as", "at", "be", "because", "but", "by", "can", "could", "for",
    "from", "had", "has", "have", "how", "if", "in", "into", "is", "it", "its", "just", "may", "more", "not", "of", "on", "or", "our", "so",
    "that", "the", "their", "then", "there", "this", "to", "was", "we", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "would", "you", "your"};

/// _TOKEN_TRIM_CHARS
constexpr std::u32string_view k_trim_chars = U" \t\r\n.,;:!?()[]{}<>\"'`~@#$%^&*=|\\/，。！？；：、（）【】《》“”‘’…";

bool is_space(char32_t c) {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
           c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

bool is_latin(char32_t c) { return (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z'); }
bool is_cjk(char32_t c) { return c >= 0x4E00 && c <= 0x9FFF; } ///< _CJK_RE
bool is_digit(char32_t c) { return (c >= U'0' && c <= U'9') || (c >= 0xFF10 && c <= 0xFF19); }

/// \w: letters, digits and the underscore.
bool is_word(char32_t c) {
    if (c < 0x80) return is_latin(c) || is_digit(c) || c == U'_';
    const auto in = [c](char32_t from, char32_t to) { return c >= from && c <= to; };
    return c == 0xAA || c == 0xB5 || c == 0xBA || in(0xB2, 0xB3) || c == 0xB9 || in(0xBC, 0xBE) ||
           (in(0xC0, 0x24F) && c != 0xD7 && c != 0xF7) || in(0x250, 0x2C1) || (in(0x370, 0x3FF) && c != 0x37E && c != 0x387) ||
           (in(0x400, 0x52F) && !in(0x482, 0x489)) || in(0x5D0, 0x5EA) || in(0x620, 0x64A) || in(0x660, 0x669) || in(0xE01, 0xE30) ||
           in(0x1100, 0x11FF) || c == 0x3005 || c == 0x3007 || in(0x3041, 0x3096) || in(0x30A1, 0x30FA) || in(0x30FC, 0x30FF) ||
           in(0x3105, 0x312F) || in(0x3131, 0x318E) || in(0x3400, 0x4DBF) || in(0x4E00, 0x9FFF) || in(0xA000, 0xA48C) ||
           in(0xAC00, 0xD7A3) || in(0xF900, 0xFAFF) || in(0xFF10, 0xFF19) || in(0xFF21, 0xFF3A) || in(0xFF41, 0xFF5A) ||
           in(0xFF66, 0xFF9D) || in(0x20000, 0x323AF);
}

/// str.lower() for the scripts chat text uses.
char32_t lower(char32_t c) {
    if ((c >= U'A' && c <= U'Z') || (c >= 0xC0 && c <= 0xDE && c != 0xD7) || (c >= 0x391 && c <= 0x3A9 && c != 0x3A2) ||
        (c >= 0x410 && c <= 0x42F) || (c >= 0xFF21 && c <= 0xFF3A))
        return c + 0x20;
    if (c >= 0x400 && c <= 0x40F) return c + 0x50;
    return c;
}

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

std::u32string code_points(std::string_view text) {
    std::u32string out;
    for (const auto &c : utf8_chars(text)) out += c.cp;
    return out;
}

std::string to_utf8(std::u32string_view text) {
    std::string out;
    for (const auto c : text) append_utf8(out, c);
    return out;
}

/// What the frequency rules need to know about a token, in one pass.
struct Shape {
    usize length = 0; ///< characters
    usize cjk = 0;
    bool latin = false;
    bool numeric = true;     ///< [\d._+-]+
    bool punctuation = true; ///< [\W_]+
};

Shape shape_of(std::string_view text) {
    Shape shape;
    for (usize at = 0; at < text.size();) {
        const auto c = next_char(text, at);
        ++shape.length;
        shape.cjk += is_cjk(c) ? 1 : 0;
        shape.latin = shape.latin || is_latin(c);
        shape.numeric = shape.numeric && (is_digit(c) || c == U'.' || c == U'_' || c == U'+' || c == U'-');
        shape.punctuation = shape.punctuation && (!is_word(c) || c == U'_');
    }
    return shape;
}

/// re.sub(pattern, " ", text) for a pattern given as a matcher: the length
/// of the match at `at`, 0 for none.
template <typename Match>
std::u32string replace_matches(std::u32string_view text, Match match) {
    std::u32string out;
    for (usize i = 0; i < text.size();) {
        if (const usize length = match(text, i); length > 0) {
            out += U' ';
            i += length;
        } else {
            out += text[i++];
        }
    }
    return out;
}

usize non_space_run(std::u32string_view text, usize from) {
    auto at = from;
    while (at < text.size() && !is_space(text[at])) ++at;
    return at - from;
}

bool starts_with_folded(std::u32string_view text, usize at, std::u32string_view prefix) {
    if (text.size() - at < prefix.size()) return false;
    for (usize k = 0; k < prefix.size(); ++k) {
        if (lower(text[at + k]) != prefix[k]) return false;
    }
    return true;
}

/// Whether `text` could hold a URL, CQ code or @mention (all need one of
/// "://", "www." in any case, "[" or "@").
bool may_have_noise(std::string_view text) {
    if (text.find_first_of("@[") != std::string_view::npos || text.find("://") != std::string_view::npos) return true;
    for (auto dot = text.find('.'); dot != std::string_view::npos; dot = text.find('.', dot + 1)) {
        if (dot >= 3 && std::ranges::all_of(text.substr(dot - 3, 3), [](char c) { return c == 'w' || c == 'W'; })) return true;
    }
    return false;
}

/// _remove_tokenization_noise: URLs, then CQ codes, then @mentions.
std::string remove_noise(std::string_view text) {
    if (!may_have_noise(text)) return std::string(text);
    // https?://\S+|www\.\S+ (case-insensitive)
    auto cleaned = replace_matches(code_points(text), [](std::u32string_view t, usize at) -> usize {
        for (const std::u32string_view prefix : {U"https://", U"http://", U"www."}) {
            if (!starts_with_folded(t, at, prefix)) continue;
            if (const auto run = non_space_run(t, at + prefix.size()); run > 0) return prefix.size() + run;
        }
        return 0;
    });
    // \[CQ:[^\]]+\]
    cleaned = replace_matches(cleaned, [](std::u32string_view t, usize at) -> usize {
        if (!t.substr(at).starts_with(U"[CQ:")) return 0;
        const auto close = t.find(U']', at + 4);
        return close == std::u32string_view::npos || close == at + 4 ? 0 : close - at + 1;
    });
    // @\S+
    return to_utf8(replace_matches(cleaned, [](std::u32string_view t, usize at) -> usize {
        if (t[at] != U'@') return 0;
        const auto run = non_space_run(t, at + 1);
        return run > 0 ? run + 1 : 0;
    }));
}

bool is_tech_byte(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '+' || c == '.' || c == '/' ||
           c == '#' || c == '-';
}
bool is_ascii_word_byte(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_'; }

/// _TECH_TERM_RE: (?<![A-Za-z0-9_])([A-Za-z][A-Za-z0-9_+./#-]{1,})(?![A-Za-z0-9_])
/// over bytes: its classes are ASCII, and no byte of a multi-byte character
/// is ASCII.
std::vector<std::string_view> tech_terms(std::string_view text) {
    std::vector<std::string_view> terms;
    for (usize i = 0; i < text.size();) {
        const char c = text[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) || (i > 0 && is_ascii_word_byte(text[i - 1]))) {
            ++i;
            continue;
        }
        auto end = i + 1;
        while (end < text.size() && is_tech_byte(text[end])) ++end;
        if (end - i < 2) {
            ++i;
            continue;
        }
        terms.push_back(text.substr(i, end - i));
        i = end;
    }
    return terms;
}

/// normalize() for tokens that need no change: no whitespace, nothing to
/// lower-case, no trimmable character at either end.
bool already_normal(std::string_view token) {
    if (token.empty()) return true;
    char32_t first = 0;
    char32_t last = 0;
    for (usize at = 0; at < token.size();) {
        const auto c = next_char(token, at);
        if (is_space(c) || lower(c) != c) return false;
        if (first == 0) first = c;
        last = c;
    }
    return !k_trim_chars.contains(first) && !k_trim_chars.contains(last);
}

std::string normalize(std::string_view token) {
    if (already_normal(token)) return std::string(token);
    // strip(), lower(), strip(_TOKEN_TRIM_CHARS), then \s+ -> " ".
    std::u32string_view view;
    const auto points = code_points(token);
    view = points;
    while (!view.empty() && is_space(view.front())) view.remove_prefix(1);
    while (!view.empty() && is_space(view.back())) view.remove_suffix(1);
    std::u32string lowered;
    for (const auto c : view) lowered += lower(c);
    std::u32string_view trimmed = lowered;
    while (!trimmed.empty() && k_trim_chars.contains(trimmed.front())) trimmed.remove_prefix(1);
    while (!trimmed.empty() && k_trim_chars.contains(trimmed.back())) trimmed.remove_suffix(1);
    std::u32string out;
    for (usize i = 0; i < trimmed.size(); ++i) {
        if (!is_space(trimmed[i])) {
            out += trimmed[i];
        } else if (i == 0 || !is_space(trimmed[i - 1])) {
            out += U' ';
        }
    }
    return to_utf8(out);
}

bool meaningful(std::string_view token) {
    if (token.empty() || k_stop_words.contains(token)) return false;
    const auto shape = shape_of(token);
    if (shape.numeric || shape.punctuation) return false;
    if (shape.cjk > 0) return shape.cjk >= 2;
    return shape.length >= 2;
}

} // namespace

std::string normalize_term(std::string_view token) { return normalize(token); }

std::string normalize_match_text(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (usize at = 0; at < text.size();) {
        const auto c = next_char(text, at);
        if (!is_space(c)) {
            append_utf8(out, lower(c));
        } else if (!out.empty() && out.back() != ' ') {
            out += ' ';
        }
    }
    if (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

std::vector<std::string> meaningful_terms(std::string_view text) {
    const auto cleaned = remove_noise(text);
    std::vector<std::string> terms;
    for (const auto raw : cut(cleaned)) {
        auto token = normalize(raw);
        if (meaningful(token)) terms.push_back(std::move(token));
    }
    // Tech terms jieba split apart, unless already there (a message has few terms).
    for (const auto raw : tech_terms(cleaned)) {
        auto token = normalize(raw);
        if (std::ranges::find(terms, token) == terms.end() && meaningful(token)) terms.push_back(std::move(token));
    }
    return terms;
}

std::vector<std::string> term_phrases(const std::vector<std::string> &tokens) {
    std::vector<std::string> phrases;
    std::vector<Shape> shapes;
    shapes.reserve(tokens.size());
    for (const auto &token : tokens) shapes.push_back(shape_of(token));
    for (usize size = 2; size <= 3; ++size) {
        if (tokens.size() < size) break;
        for (usize index = 0; index + size <= tokens.size(); ++index) {
            const std::span<const std::string> window(tokens.data() + index, size);
            const std::span<const Shape> window_shapes(shapes.data() + index, size);
            // _is_meaningful_phrase: not one token repeated
            if (std::ranges::all_of(window, [&](const std::string &token) { return token == window[0]; })) continue;
            // _join_phrase: Han tokens join directly, others with spaces
            const bool all_han = std::ranges::all_of(window_shapes, [](const Shape &s) { return s.cjk > 0 && !s.latin; });
            std::string phrase;
            for (const auto &token : window) {
                if (!all_han && !phrase.empty()) phrase += ' ';
                phrase += token;
            }
            if (k_stop_words.contains(phrase)) continue;
            usize cjk = 0;
            for (const auto &s : window_shapes) cjk += s.cjk;
            if (cjk > 0 ? cjk >= 4 : std::ranges::any_of(window_shapes, &Shape::latin)) phrases.push_back(std::move(phrase));
        }
    }
    return phrases;
}

std::vector<store::TermTally> count_terms(const std::vector<std::string> &texts, usize limit, i64 min_count) {
    struct Count {
        i64 occurrences = 0;
        i64 messages = 0;
        usize last_message = 0; ///< 1-based; a message counts once per term
    };
    std::unordered_map<std::string, Count> words;
    std::unordered_map<std::string, Count> phrases;
    const auto add = [](std::unordered_map<std::string, Count> &counts, std::vector<std::string> terms, usize message) {
        for (auto &term : terms) {
            auto &count = counts[std::move(term)];
            ++count.occurrences;
            if (count.last_message != message) {
                ++count.messages;
                count.last_message = message;
            }
        }
    };
    usize message = 0;
    for (const auto &text : texts) {
        if (std::ranges::all_of(code_points(text), is_space)) continue;
        ++message;
        auto tokens = meaningful_terms(text);
        if (tokens.empty()) continue;
        add(phrases, term_phrases(tokens), message);
        add(words, std::move(tokens), message);
    }
    struct Item {
        store::TermTally tally;
        usize length;
    };
    std::vector<Item> items;
    for (const auto *counts : {&words, &phrases}) {
        for (const auto &[term, count] : *counts) {
            if (count.occurrences >= std::max<i64>(min_count, 1))
                items.push_back({.tally = {.term = term, .count = count.occurrences, .message_count = count.messages},
                                 .length = shape_of(term).length});
        }
    }
    // _frequency_item_sort_key: (-count, -message_count, -len(term), term)
    std::ranges::sort(items, [](const Item &a, const Item &b) {
        if (a.tally.count != b.tally.count) return a.tally.count > b.tally.count;
        if (a.tally.message_count != b.tally.message_count) return a.tally.message_count > b.tally.message_count;
        if (a.length != b.length) return a.length > b.length;
        return a.tally.term < b.tally.term;
    });
    if (items.size() > std::max<usize>(limit, 1)) items.resize(std::max<usize>(limit, 1));
    std::vector<store::TermTally> tallies;
    tallies.reserve(items.size());
    for (auto &item : items) tallies.push_back(std::move(item.tally));
    return tallies;
}

} // namespace lilia::learn
