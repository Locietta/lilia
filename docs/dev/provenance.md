# Code taken from liminal

## lighter (xmake package)

From 2026-10-03 to the switch the same day, `lighter/` was vendored from liminal
`main` at `d520ce6` with `git subtree`. It is now its own repository,
[Locietta/lighter](https://github.com/Locietta/lighter) (MIT, history preserved
from liminal). lilia consumes it as the xmake package `lighter 0.2.0` (semver; `0.1.0` was first published as `2026.10.03`); the
recipe lives in `Locietta/xmake-repo` (`packages/l/lighter`), our `xmake`
submodule.

- Change lighter in its own repository, release a new version (tag plus
  `add_versions` in the recipe), then bump the `xmake` submodule and the
  `add_requires` version here.
- For local co-development set `LIGHTER_SOURCE_DIR` to a lighter checkout and
  run `pixi run xmake require -f -y lighter`; unset it to return to the pin.
- Candidates to move into lighter: `lilia/net/websocket.*` and
  `lilia/net/codec.*` (websocket client, base64, SHA-1).
- lighter 0.2.0 fixed `Outcome<T, void, void>::value()` copying its value, so a
  `Task<T>` with a move-only `T` now works; the test helper's `shared_ptr`
  workaround was removed.

## Provider layer (copied, renamed `liminal::` to `lilia::`)

Local changes since the copy:

- `anthropic::ClientOptions::disable_thinking` sends `thinking: {"type": "disabled"}`
  (DeepSeek thinks by default); `require_tool` sends `tool_choice: {"type": "any"}`.
- `provider::SchemaProperty::items` (array element schema), the same change as
  on liminal's `codex/skills-questions-task-plan` branch.
- `anthropic::ClientOptions::prompt_cache` asks Claude for prompt caching:
  top-level automatic caching, a breakpoint on the last tool, and `system` sent
  as text blocks (one per instruction item) with a breakpoint after the leading
  SYSTEM part. Without it `system` stays the single string liminal sends.
- `provider::TextPart::cache` marks the end of a stable prefix inside the
  messages; with `prompt_cache` the text block gets a breakpoint (lilia: the
  chat log, before the per-call tail).

From liminal `main` at `d520ce6`, keeping the relative paths so fixes can be
ported in either direction:

- `liminal/error.h`, `liminal/text.h`
- `liminal/tools/outcome.{h,cpp}`
- `liminal/provider/{anthropic,auth,compact}.{h,cpp}`, `common.h`, `history.h`, `provider.h`
- `liminal/provider/detail/anthropic_protocol.h`, `completion_retry.{h,cpp}`

The SQLite statement helpers in `lilia/store/sqlite.*` were extracted from the
file-local helpers in `liminal/session/store.cpp`. The build scaffolding
(root `xmake.lua`, `pixi.toml`, formatting, hooks and remote test scripts) and
`lilia/xmake/rules/version.lua` were adapted from liminal.

# Third-party data

## jieba 0.42.1 (MIT, `lilia/data/jieba/`)

`lilia/learn/segment.cpp` ports jieba's default `jieba.cut` and embeds its
data. `jieba_dict.txt` is jieba's `dict.txt` byte for byte; `jieba_hmm.txt`
holds the HMM tables of `finalseg/prob_*.py`, one entry per line, written by
`scripts/jieba_data.py` from the PyPI sdist. `LICENSE` is jieba's. Verified on
2026-10-09: identical tokens to Python jieba on 136,370 distinct lines (3.55M
tokens) of Chinese docs, prompts and code.

`lilia/learn/frequency.cpp` ports MaiBot's
`src/services/message_word_frequency_service.py` (`f9ec367`); on the same
lines its terms and its per-batch tallies (4,546 batches of 30) match the
Python code exactly.

## pypinyin 0.55.0 (MIT, `lilia/data/pinyin/`)

`lilia/chat/typos.cpp` ports MaiBot's `src/chat/utils/typo_generator.py`
(`f9ec367`), which asks pypinyin for readings. `scripts/pinyin_data.py`
writes them from the PyPI sdist: `pinyin_chars.txt` (each character's first
TONE3 reading, U+4E00..U+9FFE) and `pinyin_words.txt` (jieba dictionary words
whose reading in context differs from their characters'). `LICENSE` is
pypinyin's. Character frequencies are derived from jieba's dictionary, which
equals MaiBot's `depends-data/char_frequency.json`. Verified on 2026-10-10
against MaiBot's code: homophone candidates for all 20,991 characters and
4,000 random dictionary words match.

`lilia/chat/delivery.cpp` ports MaiBot's reply splitter
(`_split_into_sentence_segments`), packing
(`_merge_processed_segments_to_max_count`) and `calculate_typing_time` from
`src/chat/utils/utils.py`. One deliberate difference: a space next to a Latin
letter or digit never splits (MaiBot: only between two), since writers space
English words inside Chinese.

## QQ face names

`lilia/onebot/message.cpp` names the common QQ system faces by id. The table
is lilia's own; on 2026-10-11 each entry was checked against the list in
MaiBot-Napcat-Adapter's `qq_emoji_list.py` and only those that agreed were
kept.
