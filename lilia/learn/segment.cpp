#include "segment.h"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include <lilia/error.h>

#include <xmake/bin2obj/jieba_dict_txt.hpp>
#include <xmake/bin2obj/jieba_hmm_txt.hpp>

namespace lilia::learn {

char32_t next_char(std::string_view text, usize &at) {
    const auto lead = static_cast<unsigned char>(text[at]);
    if (lead < 0x80) {
        ++at;
        return lead;
    }
    const usize length = (lead >> 5) == 0x6 ? 2 : (lead >> 4) == 0xE ? 3 : (lead >> 3) == 0x1E ? 4 : 0;
    char32_t cp = length == 2 ? lead & 0x1F : length == 3 ? lead & 0x0F : lead & 0x07;
    bool valid = length != 0 && at + length <= text.size();
    for (usize k = 1; valid && k < length; ++k) {
        const auto next = static_cast<unsigned char>(text[at + k]);
        valid = (next & 0xC0) == 0x80;
        cp = (cp << 6) | (next & 0x3F);
    }
    if (!valid) {
        ++at;
        return 0xFFFD;
    }
    at += length;
    return cp;
}

std::string encode_utf8(char32_t c) {
    std::string out;
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
    return out;
}

std::vector<Utf8Char> utf8_chars(std::string_view text) {
    std::vector<Utf8Char> chars;
    chars.reserve(text.size());
    for (usize i = 0; i < text.size();) {
        const auto begin = i;
        const auto cp = next_char(text, i);
        chars.push_back({.cp = cp, .begin = begin, .end = i});
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

/// The prefix dictionary (Tokenizer.gen_pfdict) as a trie over code points:
/// a fragment is "in FREQ" when a path spells it. Each edge is one 64-bit
/// slot of an open-addressing table, (parent, character) above the child.
/// Word nodes carry log(freq) - log(total), the term the route adds to.
class Dictionary {
public:
    static constexpr u32 k_root = 0;
    static constexpr u32 k_none = 0; ///< the root is never a child

    u32 child(u32 node, char32_t c) const {
        const u64 key = (static_cast<u64>(node) << k_char_bits) | c;
        for (u64 at = (key * 0x9E3779B97F4A7C15ull) >> shift_;; at = (at + 1) & mask_) {
            const u64 slot = slots_[at];
            if (slot == 0) return k_none;
            if ((slot >> k_child_bits) == key) return static_cast<u32>(slot & k_child_mask);
        }
    }

    /// FREQ[fragment] > 0 for the fragment that leads to `node`.
    bool is_word(u32 node) const { return log_freq_[node] < 1.0; }
    double log_freq(u32 node) const { return log_freq_[node]; }
    i64 freq(u32 node) const { return freq_[node]; }
    double log_total() const { return log_total_; }

    static Dictionary load(std::string_view data) {
        struct Entry {
            std::string_view word;
            i64 freq = 0;
        };
        std::vector<Entry> entries;
        entries.reserve(350'000);
        i64 total = 0;
        for (usize start = 0; start < data.size();) {
            auto end = data.find('\n', start);
            if (end == std::string_view::npos) end = data.size();
            auto line = data.substr(start, end - start);
            start = end + 1;
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.remove_suffix(1);
            const auto space = line.find(' ');
            if (line.empty() || space == std::string_view::npos) continue;
            auto rest = line.substr(space + 1);
            rest = rest.substr(0, rest.find(' '));
            i64 freq = 0;
            std::from_chars(rest.data(), rest.data() + rest.size(), freq);
            entries.push_back({.word = line.substr(0, space), .freq = freq});
            total += freq;
        }
        Dictionary d;
        d.log_total_ = std::log(static_cast<double>(total));
        // Size the table for at most one node per character (load factor < 0.5).
        usize characters = 1;
        for (const auto &entry : entries)
            characters += static_cast<usize>(
                std::ranges::count_if(entry.word, [](char b) { return (static_cast<unsigned char>(b) & 0xC0) != 0x80; }));
        const usize capacity = std::bit_ceil(characters * 2);
        d.slots_.assign(capacity, 0);
        d.mask_ = capacity - 1;
        d.shift_ = static_cast<int>(64 - std::countr_zero(capacity));
        d.log_freq_.assign(1, k_no_word);
        d.freq_.assign(1, 0);
        // In file order, so a word listed twice keeps its last frequency
        // while both count in the total, as in gen_pfdict.
        for (const auto &entry : entries) {
            u32 node = k_root;
            for (usize at = 0; at < entry.word.size();) {
                const auto c = next_char(entry.word, at);
                auto next = d.child(node, c);
                if (next == k_none) next = d.add_child(node, c);
                node = next;
            }
            d.log_freq_[node] = entry.freq > 0 ? std::log(static_cast<double>(entry.freq)) - d.log_total_ : k_no_word;
            d.freq_[node] = static_cast<u32>(std::max<i64>(entry.freq, 0));
        }
        return d;
    }

private:
    static constexpr int k_char_bits = 21;
    static constexpr int k_child_bits = 21;
    static constexpr u64 k_child_mask = (u64{1} << k_child_bits) - 1;
    static constexpr double k_no_word = 2.0; ///< log probabilities are <= 0

    u32 add_child(u32 node, char32_t c) {
        const auto created = static_cast<u32>(log_freq_.size());
        if (created > k_child_mask) throw std::length_error("jieba dictionary too large for the trie");
        log_freq_.push_back(k_no_word);
        freq_.push_back(0);
        const u64 key = (static_cast<u64>(node) << k_char_bits) | c;
        u64 at = (key * 0x9E3779B97F4A7C15ull) >> shift_;
        while (slots_[at] != 0) at = (at + 1) & mask_;
        slots_[at] = (key << k_child_bits) | created;
        return created;
    }

    std::vector<u64> slots_;
    u64 mask_ = 0;
    int shift_ = 64;
    std::vector<double> log_freq_;
    std::vector<u32> freq_; ///< raw, for dictionary_frequency (the largest is 318,825)
    double log_total_ = 0;
};

const Dictionary &dictionary() {
    static const Dictionary dict = Dictionary::load(as_text(xmake::jieba_dict_txt));
    return dict;
}

/// finalseg's model. States in the order B, E, M, S, which is also the
/// order Python compares their letters in.
enum State : u8 { B, E, M, S };
constexpr std::array<char, 4> k_state_letters{'B', 'E', 'M', 'S'};

using Emission = std::array<double, 4>;
constexpr Emission k_no_emission{k_min_float, k_min_float, k_min_float, k_min_float};

struct Hmm {
    std::array<double, 4> start{k_min_float, k_min_float, k_min_float, k_min_float};
    std::array<std::array<double, 4>, 4> trans{};
    /// Per character, its emission log probability in each state.
    std::unordered_map<char32_t, Emission> emit;

    const Emission &emission(char32_t c) const {
        const auto found = emit.find(c);
        return found == emit.end() ? k_no_emission : found->second;
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
                if (chars.size() == 1)
                    h.emit.try_emplace(chars.front().cp, k_no_emission).first->second[state_of(fields[1])] = number(fields[3]);
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

/// Scratch space reused across calls (cut is called per message).
struct Scratch {
    std::vector<Utf8Char> chars;
    std::vector<std::array<State, 4>> back;
    std::vector<State> tags;
    std::vector<u32> dag_begin;              ///< per character, into dag
    std::vector<std::pair<u32, double>> dag; ///< (word end, its log_freq)
    std::vector<std::pair<double, u32>> route;
};

Scratch &scratch() {
    // Never freed: destroying thread_local objects at exit crashes MinGW
    // builds (emulated TLS), and the space is reused until then anyway.
    thread_local Scratch *s = new Scratch;
    return *s;
}

/// finalseg.viterbi: the B/M/E/S tag of each character.
const std::vector<State> &viterbi(std::span<const Utf8Char> obs) {
    const auto &model = hmm();
    // PrevStatus: the states each state may follow.
    constexpr std::array<std::array<State, 2>, 4> k_prev{{{E, S}, {B, M}, {M, B}, {S, E}}};
    std::array<double, 4> v{};
    const auto &first_emission = model.emission(obs[0].cp);
    for (const State y : {B, E, M, S}) v[y] = model.start[y] + first_emission[y];
    auto &back = scratch().back;
    back.resize(obs.size());
    for (usize t = 1; t < obs.size(); ++t) {
        std::array<double, 4> next{};
        const auto &emission = model.emission(obs[t].cp);
        for (const State y : {B, E, M, S}) {
            const double em = emission[y];
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
    auto &tags = scratch().tags;
    tags.resize(obs.size());
    for (usize t = obs.size(); t-- > 0;) {
        tags[t] = last;
        if (t > 0) last = back[t][last];
    }
    return tags;
}

/// finalseg.__cut over a run of Han characters.
void hmm_cut_han(std::string_view text, std::span<const Utf8Char> chars, Tokens &out) {
    const auto &tags = viterbi(chars);
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
    auto &s = scratch();
    // get_DAG: the ends of every dictionary word starting at each character,
    // found in one walk down the trie; with its log_freq for calc. A start
    // with no word keeps itself, scored as frequency 1 (log 1 - log total).
    s.dag.clear();
    s.dag_begin.resize(n + 1);
    for (usize k = 0; k < n; ++k) {
        s.dag_begin[k] = static_cast<u32>(s.dag.size());
        auto node = Dictionary::k_root;
        for (usize i = k; i < n; ++i) {
            node = dict.child(node, chars[i].cp);
            if (node == Dictionary::k_none) break;
            if (dict.is_word(node)) s.dag.emplace_back(static_cast<u32>(i), dict.log_freq(node));
        }
        if (s.dag.size() == s.dag_begin[k]) s.dag.emplace_back(static_cast<u32>(k), 0.0 - dict.log_total());
    }
    s.dag_begin[n] = static_cast<u32>(s.dag.size());
    // calc: the best route from each character to the end; equal scores go
    // to the longer word, as Python's max over (score, end) does.
    auto &route = s.route;
    route.assign(n + 1, {0.0, 0});
    for (usize idx = n; idx-- > 0;) {
        bool first = true;
        for (auto e = s.dag_begin[idx]; e < s.dag_begin[idx + 1]; ++e) {
            const auto [x, log_freq] = s.dag[e];
            const double p = log_freq + route[x + 1].first;
            if (first || p > route[idx].first || (p == route[idx].first && x > route[idx].second)) route[idx] = {p, x};
            first = false;
        }
    }
    // FREQ.get(run) > 0: the run is a dictionary word.
    const auto known = [&](usize from, usize to) {
        auto node = Dictionary::k_root;
        for (auto i = from; i < to; ++i) {
            node = dict.child(node, chars[i].cp);
            if (node == Dictionary::k_none) return false;
        }
        return dict.is_word(node);
    };
    // Single characters gather in a buffer; an unknown run goes to the HMM.
    usize buf = 0;
    usize buf_end = 0;
    const auto flush = [&] {
        if (buf_end - buf == 1) {
            out.push_back(slice(text, chars, buf, buf_end));
        } else if (buf_end > buf) {
            if (!known(buf, buf_end)) {
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
    auto &chars = scratch().chars;
    chars.clear();
    for (usize at = 0; at < text.size();) {
        const auto begin = at;
        const auto cp = next_char(text, at);
        chars.push_back({.cp = cp, .begin = begin, .end = at});
    }
    const std::span<const Utf8Char> all(chars);
    Tokens out;
    out.reserve(chars.size() / 2 + 1);
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

i64 dictionary_frequency(std::string_view word) {
    const auto &dict = dictionary();
    auto node = Dictionary::k_root;
    for (usize at = 0; at < word.size();) {
        node = dict.child(node, next_char(word, at));
        if (node == Dictionary::k_none) return 0;
    }
    return node == Dictionary::k_root ? 0 : dict.freq(node);
}

void for_each_dictionary_entry(const std::function<void(std::string_view, i64)> &visit) {
    const auto data = as_text(xmake::jieba_dict_txt);
    for (usize start = 0; start < data.size();) {
        auto end = data.find('\n', start);
        if (end == std::string_view::npos) end = data.size();
        auto line = data.substr(start, end - start);
        start = end + 1;
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.remove_suffix(1);
        const auto space = line.find(' ');
        if (line.empty() || space == std::string_view::npos) continue;
        auto rest = line.substr(space + 1);
        rest = rest.substr(0, rest.find(' '));
        i64 freq = 0;
        std::from_chars(rest.data(), rest.data() + rest.size(), freq);
        visit(line.substr(0, space), freq);
    }
}

u32 dictionary_child(u32 node, char32_t c) { return dictionary().child(node, c); }

i64 dictionary_node_frequency(u32 node) { return node == Dictionary::k_root ? 0 : dictionary().freq(node); }

} // namespace lilia::learn
