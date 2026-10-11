#pragma once

#include <string>
#include <string_view>
#include <vector>

#include <lighter/types.hpp>

namespace lilia::retrieval {

using namespace lighter::types;

// The keyword leg of retrieval: BM25 over words, in memory. SQLite's FTS5
// cannot segment Chinese (unicode61 keeps a run of Han characters as one token,
// and trigram never matches two-character words), and the corpora here are
// small enough to score in a few milliseconds.

/// Index terms of `text`: jieba words of two or more characters (see
/// learn::cut), lower-cased ASCII words of two or more, and every pair of
/// adjacent Han characters, which catches names the dictionary splits.
std::vector<std::string> terms(std::string_view text);

/// Okapi BM25 (k1 1.2, b 0.75) of `query` against each of `documents`, with
/// document frequencies taken over `documents`; 0 for no shared term.
std::vector<double> bm25(const std::vector<std::string> &query, const std::vector<const std::vector<std::string> *> &documents);

} // namespace lilia::retrieval
