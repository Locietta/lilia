#include "segment.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>

#include <lilia/error.h>

#include <xmake/bin2obj/jieba_dict_txt.hpp>
#include <xmake/bin2obj/jieba_hmm_txt.hpp>

namespace lilia::learn {

std::vector<Utf8Char> utf8_chars(std::string_view text) {
    std::vector<Utf8Char> chars;
    chars.reserve(text.size());
    for (usize i = 0; i < text.size();) {
        const auto lead = static_cast<unsigned char>(text[i]);
        usize length = lead < 0x80 ? 1 : (lead >> 5) == 0x6 ? 2 : (lead >> 4) == 0xE ? 3 : (lead >> 3) == 0x1E ? 4 : 0;
        char32_t cp = length == 1 ? lead : length == 2 ? lead & 0x1F : length == 3 ? lead & 0x0F : lead & 0x07;
        bool valid = length != 0 && i + length <= text.size();
        for (usize k = 1; valid && k < length; ++k) {
            const auto next = static_cast<unsigned char>(text[i + k]);
            valid = (next & 0xC0) == 0x80;
            cp = (cp << 6) | (next & 0x3F);
        }
        if (!valid) {
            length = 1;
            cp = 0xFFFD;
        }
        chars.push_back({.cp = cp, .begin = i, .end = i + length});
        i += length;
    }
    return chars;
}

namespace {

// Each piece below names the jieba function it follows; where Python breaks
// ties by comparing tuples, the same order is kept so results match exactly.

constexpr double k_min_float = -3.14e100; // finalseg.MIN_FLOAT

std::string_view as_text(std::span<const std::byte> data) { return {reinterpret_cast<const char *>(data.data()), data.size()}; }

bool is_ascii_alnum(char32_t c) { return (c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z') || (c >= U'0' && c <= U'9'); }
bool is_han(char32_t c) { return c >= 0x4E00 && c <= 0x9FD5; }

/// re_han_default: [一-鿕a-zA-Z0-9+#&\._%\-]
bool is_han_default(char32_t c) {
    return is_han(c) || is_ascii_alnum(c) || c == U'+' || c == U'#' || c == U'&' || c == U'.' || c == U'_' || c == U'%' || c == U'-';
}

/// The prefix dictionary (Tokenizer.gen_pfdict), as a sorted word list: a
/// fragment is "in FREQ" when some word starts with it.
struct Dictionary {
    struct Entry {
        std::string_view word;
        i64 freq = 0;
    };
    std::vector<Entry> entries;
    double log_total = 0;

    /// FREQ.get(fragment): the word's frequency, 0 for prefixes and misses;
    /// nullopt when no word starts with it.
    std::optional<i64> find(std::string_view fragment) const {
        const auto at = std::ranges::lower_bound(entries, fragment, {}, &Entry::word);
        if (at == entries.end() || !at->word.starts_with(fragment)) return std::nullopt;
        return at->word == fragment ? at->freq : 0;
    }
};

const Dictionary &dictionary() {
    static const Dictionary dict = [] {
        Dictionary d;
        const auto data = as_text(xmake::jieba_dict_txt);
        i64 total = 0;
        d.entries.reserve(350'000);
        for (usize start = 0; start < data.size();) {
            auto end = data.find('\n', start);
            if (end == std::string_view::npos) end = data.size();
            auto line = data.substr(start, end - start);
            start = end + 1;
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.remove_suffix(1);
            const auto space = line.find(' ');
            if (line.empty() || space == std::string_view::npos) continue;
            const auto word = line.substr(0, space);
            auto rest = line.substr(space + 1);
            rest = rest.substr(0, rest.find(' '));
            i64 freq = 0;
            std::from_chars(rest.data(), rest.data() + rest.size(), freq);
            d.entries.push_back({.word = word, .freq = freq});
            total += freq;
        }
        // A word listed twice keeps its last frequency; both count in the total.
        std::ranges::stable_sort(d.entries, {}, &Dictionary::Entry::word);
        std::vector<Dictionary::Entry> unique;
        unique.reserve(d.entries.size());
        for (const auto &entry : d.entries) {
            if (!unique.empty() && unique.back().word == entry.word)
                unique.back() = entry;
            else
                unique.push_back(entry);
        }
        d.entries = std::move(unique);
        d.log_total = std::log(static_cast<double>(total));
        return d;
    }();
    return dict;
}

/// finalseg's model. States in the order B, E, M, S, which is also the
/// order Python compares their letters in.
enum State : u8 { B, E, M, S };
constexpr std::array<char, 4> k_state_letters{'B', 'E', 'M', 'S'};

struct Hmm {
    std::array<double, 4> start{k_min_float, k_min_float, k_min_float, k_min_float};
    std::array<std::array<double, 4>, 4> trans{};
    std::array<std::unordered_map<char32_t, double>, 4> emit;

    double emission(State state, char32_t c) const {
        const auto found = emit[state].find(c);
        return found == emit[state].end() ? k_min_float : found->second;
    }
};

State state_of(std::string_view letter) { return letter == "B" ? B : letter == "E" ? E : letter == "M" ? M : S; }

double number(std::string_view text) {
    double value = 0;
    std::from_chars(text.data(), text.data() + text.size(), value);
    return value;
}

const Hmm &hmm() {
    static const Hmm model = [] {
        Hmm h;
        for (auto &row : h.trans) row.fill(k_min_float);
        const auto data = as_text(xmake::jieba_hmm_txt);
        for (usize start = 0; start < data.size();) {
            auto end = data.find('\n', start);
            if (end == std::string_view::npos) end = data.size();
            const auto line = data.substr(start, end - start);
            start = end + 1;
            std::array<std::string_view, 4> fields;
            usize count = 0;
            for (usize from = 0; from <= line.size() && count < fields.size();) {
                const auto space = std::min(line.find(' ', from), line.size());
                fields[count++] = line.substr(from, space - from);
                from = space + 1;
            }
            if (fields[0] == "start" && count == 3) {
                h.start[state_of(fields[1])] = number(fields[2]);
            } else if (fields[0] == "trans" && count == 4) {
                h.trans[state_of(fields[1])][state_of(fields[2])] = number(fields[3]);
            } else if (fields[0] == "emit" && count == 4) {
                const auto chars = utf8_chars(fields[2]);
                if (chars.size() == 1) h.emit[state_of(fields[1])][chars.front().cp] = number(fields[3]);
            }
        }
        return h;
    }();
    return model;
}

using Tokens = std::vector<std::string_view>;

std::string_view slice(std::string_view text, std::span<const Utf8Char> chars, usize from, usize to) {
    return text.substr(chars[from].begin, chars[to - 1].end - chars[from].begin);
}

/// finalseg.viterbi: the B/M/E/S tag of each character.
std::vector<State> viterbi(std::span<const Utf8Char> obs) {
    const auto &model = hmm();
    // PrevStatus: the states each state may follow.
    constexpr std::array<std::array<State, 2>, 4> k_prev{{{E, S}, {B, M}, {M, B}, {S, E}}};
    std::array<double, 4> v{};
    for (const State y : {B, E, M, S}) v[y] = model.start[y] + model.emission(y, obs[0].cp);
    std::vector<std::array<State, 4>> back(obs.size());
    for (usize t = 1; t < obs.size(); ++t) {
        std::array<double, 4> next{};
        for (const State y : {B, E, M, S}) {
            const double em = model.emission(y, obs[t].cp);
            // max over (probability, letter): equal probabilities go to the later letter.
            double best = 0;
            State from = k_prev[y][0];
            bool first = true;
            for (const State y0 : k_prev[y]) {
                const double p = v[y0] + model.trans[y0][y] + em;
                if (first || p > best || (p == best && k_state_letters[y0] > k_state_letters[from])) {
                    best = p;
                    from = y0;
                    first = false;
                }
            }
            next[y] = best;
            back[t][y] = from;
        }
        v = next;
    }
    State last = v[S] >= v[E] ? S : E;
    std::vector<State> tags(obs.size());
    for (usize t = obs.size(); t-- > 0;) {
        tags[t] = last;
        if (t > 0) last = back[t][last];
    }
    return tags;
}

/// finalseg.__cut over a run of Han characters.
void hmm_cut_han(std::string_view text, std::span<const Utf8Char> chars, Tokens &out) {
    const auto tags = viterbi(chars);
    usize begin = 0;
    usize next = 0;
    for (usize i = 0; i < chars.size(); ++i) {
        if (tags[i] == B) {
            begin = i;
        } else if (tags[i] == E) {
            out.push_back(slice(text, chars, begin, i + 1));
            next = i + 1;
        } else if (tags[i] == S) {
            out.push_back(slice(text, chars, i, i + 1));
            next = i + 1;
        }
    }
    if (next < chars.size()) out.push_back(slice(text, chars, next, chars.size()));
}

/// finalseg's re_skip, ([a-zA-Z0-9]+(?:\.\d+)?%?), matched at `from`;
/// returns the end, or `from` when it does not match there.
usize skip_match(std::span<const Utf8Char> chars, usize from) {
    auto at = from;
    while (at < chars.size() && is_ascii_alnum(chars[at].cp)) ++at;
    if (at == from) return from;
    const auto is_digit = [&](usize i) { return i < chars.size() && chars[i].cp >= U'0' && chars[i].cp <= U'9'; };
    if (at < chars.size() && chars[at].cp == U'.' && is_digit(at + 1)) {
        ++at;
        while (is_digit(at)) ++at;
    }
    if (at < chars.size() && chars[at].cp == U'%') ++at;
    return at;
}

/// finalseg.cut: Han runs through the HMM; the rest split around re_skip
/// matches, keeping both the matches and the text between them.
void hmm_cut(std::string_view text, std::span<const Utf8Char> chars, Tokens &out) {
    for (usize i = 0; i < chars.size();) {
        auto j = i;
        if (is_han(chars[i].cp)) {
            while (j < chars.size() && is_han(chars[j].cp)) ++j;
            hmm_cut_han(text, chars.subspan(i, j - i), out);
            i = j;
            continue;
        }
        while (j < chars.size() && !is_han(chars[j].cp)) ++j;
        const auto block = chars.subspan(i, j - i);
        usize gap = 0;
        for (usize k = 0; k < block.size();) {
            const auto end = skip_match(block, k);
            if (end == k) {
                ++k;
                continue;
            }
            if (gap < k) out.push_back(slice(text, block, gap, k));
            out.push_back(slice(text, block, k, end));
            k = gap = end;
        }
        if (gap < block.size()) out.push_back(slice(text, block, gap, block.size()));
        i = j;
    }
}

/// Tokenizer.__cut_DAG over one re_han_default block.
void cut_block(std::string_view text, std::span<const Utf8Char> chars, Tokens &out) {
    const auto &dict = dictionary();
    const usize n = chars.size();
    // get_DAG: the ends of every dictionary word starting at each character.
    std::vector<std::vector<usize>> dag(n);
    for (usize k = 0; k < n; ++k) {
        for (usize i = k; i < n; ++i) {
            const auto freq = dict.find(slice(text, chars, k, i + 1));
            if (!freq) break;
            if (*freq) dag[k].push_back(i);
        }
        if (dag[k].empty()) dag[k].push_back(k);
    }
    // calc: the best route from each character to the end.
    std::vector<std::pair<double, usize>> route(n + 1, {0.0, 0});
    for (usize idx = n; idx-- > 0;) {
        bool first = true;
        for (const usize x : dag[idx]) {
            const auto freq = dict.find(slice(text, chars, idx, x + 1)).value_or(0);
            const double p = std::log(static_cast<double>(freq ? freq : 1)) - dict.log_total + route[x + 1].first;
            if (first || p > route[idx].first || (p == route[idx].first && x > route[idx].second)) route[idx] = {p, x};
            first = false;
        }
    }
    // Single characters gather in a buffer; an unknown run goes to the HMM.
    usize buf = 0;
    usize buf_end = 0;
    const auto flush = [&] {
        if (buf_end - buf == 1) {
            out.push_back(slice(text, chars, buf, buf_end));
        } else if (buf_end > buf) {
            if (!dict.find(slice(text, chars, buf, buf_end)).value_or(0)) {
                hmm_cut(text, chars.subspan(buf, buf_end - buf), out);
            } else {
                for (usize i = buf; i < buf_end; ++i) out.push_back(slice(text, chars, i, i + 1));
            }
        }
    };
    for (usize x = 0; x < n;) {
        const usize y = route[x].second + 1;
        if (y - x == 1) {
            if (buf_end == buf) buf = x;
            buf_end = y;
        } else {
            flush();
            buf = buf_end = 0;
            out.push_back(slice(text, chars, x, y));
        }
        x = y;
    }
    flush();
}

} // namespace

std::vector<std::string_view> cut(std::string_view text) {
    const auto chars = utf8_chars(text);
    const std::span<const Utf8Char> all(chars);
    Tokens out;
    for (usize i = 0; i < all.size();) {
        auto j = i;
        if (is_han_default(all[i].cp)) {
            while (j < all.size() && is_han_default(all[j].cp)) ++j;
            cut_block(text, all.subspan(i, j - i), out);
        } else {
            // Outside blocks (re_skip_default) every character is a token,
            // whitespace included; only "\r\n" stays together.
            while (j < all.size() && !is_han_default(all[j].cp)) ++j;
            for (auto k = i; k < j; ++k) {
                if (all[k].cp == U'\r' && k + 1 < j && all[k + 1].cp == U'\n') {
                    out.push_back(slice(text, all, k, k + 2));
                    ++k;
                } else {
                    out.push_back(slice(text, all, k, k + 1));
                }
            }
        }
        i = j;
    }
    return out;
}

} // namespace lilia::learn
