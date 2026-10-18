# lilia design

## Shape (milestone 1)

```text
NapCat (Podman, OneBot 11 ws server on 127.0.0.1:3001)
   │  forward websocket, Authorization: Bearer
   ▼
net::WebSocket ── onebot::Session ── chat::Responder ── provider::anthropic::Client ── model API
                                          │
                                     store::Store (SQLite: messages, llm_calls)
```

- **`net/`**: RFC 6455 client over `lighter::Tcp`. Plain `ws://` only, with a
  numeric host. The handshake verifies `Sec-WebSocket-Accept`. Writes are
  serialized; pings are answered inside `receive()`.
- **`onebot/`**: one `Session` per connection.
  - Actions are JSON requests tagged with an `echo`; responses complete the
    matching pending call.
  - `run()` reads frames until the connection dies (or no frame arrives for
    95 s, since NapCat sends a heartbeat every 30 s), then fails every pending
    call with `DISCONNECTED`.
  - Segments keep their data as `glz::generic_i64`, so ids stay exact and
    unknown segment types survive.
- **`chat/`**: one turn per decision, modelled on MaiBot's planner/replyer split.
  - `flow` decides only when the planner looks, never what it sees (MaiBot's
    trigger, 2026-10-11). An @, a reply to her, a private message or her name
    (ASCII aliases as whole words, never inside a link) makes it look at once.
    Otherwise it looks every `ceil(1/talk_value)` messages (default 1: every
    batch), the silence before them counting as MaiBot's credit. After two
    silent cycles it backs off (15 s doubling to 5 min) unless someone calls
    her or 6 messages pile up; what waits is looked at when the backoff ends.
  - On every connection she reads each allowed group's newest 100 messages
    and stores what she missed (not for a chat she has never seen). Whoever
    called her in the last half hour is answered late; the next turn is told
    she just came back online.
  - The worker waits for the burst to settle (`debounce_seconds`, 15 s at
    most). Called, she looks at once (2026-10-14; before, 2 s of quiet): the
    lines that follow reach the planner's next round.
  - The planner (`prompts/planner.md`, `planner_model`, thinking off) keeps
    one history per chat, as MaiBot's does (`chat/session.*`, 2026-10-11).
    - It is rendered each round from the stored lines after the window's
      anchor (`#<row id> [HH:MM] name(qq)→addressee: text`) with the planner's
      own rounds among them: its analysis, its tool calls and their results.
      New lines simply follow. Between trims it only grows, so it is a cached
      prefix. The newest window summaries (below) lead it.
    - Its rules depend on the chat (2026-10-12): `planner_group.md` (whom a
      line is said to, when to chip in) or `planner_private.md` (MaiBot's
      private prompt: everything is said to her; mind the other's pace). The
      writer's opening likewise (`writer_group.md` / `writer_private.md`).
    - Who a line is said to (`chat/addressee`, 2026-10-12): `→她` / `→小明` /
      `→全体` from its @s and quotes, quotes as `[回复 #row 小明: …]` (hers
      restored from her stored segments), computed once per row. An @ without a
      name is named when stored. LLMs find the addressee of a group-chat line
      far better with this structure than from text (Tan et al. 2023; Penzo et
      al. 2024): in a four-person group, friends say 你 to each other.
    - Each round it writes a short analysis, then calls tools:
      `reply(target, to_her, quote, style, intent)`, `find_sticker(feeling)`,
      `send_sticker(sticker)`, `repeat` (text or a sticker),
      `wait(seconds)`, `poke(target)`, `search_history(query)`,
      `look_up_person(target)` (the facts she noted about whoever wrote it),
      `search_web(query)` or `stay_silent(reason)`; no call means silence.
      The calls of a round run in order, as MaiBot runs them (2026-10-12): a
      reply and a sticker, or a look-up and a reply, take one round; `wait`
      and `stay_silent` end the round, and the calls after them are answered
      as not run. In a private chat a new message ends a wait (MaiBot's); in
      group chats a wait wakes early when someone @s or replies to her
      (`state.flow.addressed`).
      Results are short ("她回复了 #n", "等了 N 秒", what a search found);
      her bubbles show up as her own lines. Side actions are its calls too:
      it is told that people are repeating a line or that someone poked her,
      and decides whether to join or poke back.
    - Per round only a tail changes: the first new line, facts about the new
      lines (a poke, her name in a line that does not @ or quote her, 复读),
      the addressee prior for new lines that name no one (below), presence,
      self notes, slang (MaiBot's matcher, first sentence of each meaning),
      recall, the time. Whom a line @s or quotes is on the line itself and is
      not repeated. What she knows about people comes on request, through
      `look_up_person`.
    - `to_her` is the planner's reading of the reply's target (called,
      likely, group, unsure, no); for `no` the writer is told it is
      chipping in, for `group` (said to everyone, 2026-10-13) that it
      answers as one of them and was not asked itself. Hand labels showed
      lines to everyone read as `likely` without it.
      Every call is logged in `planner_turns` (tool, argument, target,
      to_her, her messages, the round's analysis), kept 180 days.
    - Hindsight feedback (`feedback/`, 2026-10-12, after MaiBot's
      reply_effect; off by default): ten minutes after she speaks (or once 40
      lines wait) the helper model reads the window with what came after and
      says whom each line was said to (her, other, group, self, unclear) and
      how her messages went down (stance, contribution). Labels keep 17
      features of the line (`chat::line_features`: explicit @/quote/name,
      你/你们, right after her, her share of the last 5 minutes, a run between
      two people…), not its text, and whom it was said to (her, another
      member, everyone, no one; 2026-10-13, before only her or not). The
      prior learns her-or-not; probes report misreads by whom the line was
      for. A logistic prior is fitted on them
      (Newton/IRLS, pulled toward hand-set weights, per group toward the pooled
      model) and shown to the planner for lines that name no one
      (「像是在对她说」的可能：#n 低/中/高) while `feedback.hint` is on (the
      default since 2026-10-13); before a model is fitted, or with judging
      off, the hand-set weights give it, with no model calls. It is always
      logged in `planner_turns.hint`. Never a gate. Reward is the
      reaction (Pang et al. 2023), not how long the chat ran, from a hand-set
      table (nothing labels how her messages went down, so there is nothing
      to fit it to); her stickers' rewards, derived from their judged effects
      each time (2026-10-13), boost them in search.
      `lilia feedback --backfill/--eval/--report/--import-manual`; `--eval`
      fits with the refit's gates and constants (`fit_models`) and scores
      what production would use (2026-10-13).
      `scripts/addressee_label_page.py` for hand labels.
    - `lilia replay <probes.jsonl> --probe` runs one turn per expected line
      of a synthetic set (`tests/replay/addressee.jsonl`: quotes, @s, her
      earlier lines) and counts intrusions, misses and to_her agreement.
      `lilia replay --probe-db <copy.db>` does the same for the lines labelled
      in a copy of lilia.db: each gets its stored context, the last three of
      others' lines new, and one turn; it prints row ids and decisions only,
      never what anyone said (2026-10-13).
    - Past twice `context_messages` / `context_bytes` the anchor moves to
      keep 1×; rounds about lines no longer shown go, older turns keep their
      calls but lose their analyses. The lines it will drop are summarised
      first (2026-10-19): the window keeps growing, still one cached prefix,
      until their window summary is stored, and only then moves, to the row
      the summary ends at (at 2.5× it moves anyway). Before, the summary was
      written after the move, so for a turn or two the dropped lines were
      neither shown nor summarised. After a restart the history is the
      stored lines again.
    - Window summaries (2026-10-12, after MaiBot's mid-term memory): whenever
      the anchor is set, the lines up to it that no window summary covers (at
      most twice `context_messages`) are summarised whole
      (`prompts/window_summary.md`: threads, who thinks what, what was
      settled, with recall cues). The newest 10 lead the planner's log, in
      the cached part; older ones come back through recall. MaiBot shows
      none of them, only the one recalled (a cue above 0.8 against the last
      12 lines); showing the newest is lilia's choice.
    - Limits: 16 rounds per turn, 3 waits in a row (MaiBot's), 3 replies per
      turn (lilia's), waits of 1–60 s. Per turn, 2 web searches, 3 look-ups,
      3 sticker searches and `memory.max_searches` history searches. Replay
      passes waits in simulated time.
    - Each tool is one entry of a table in `run_turn` (2026-10-13): whether
      the chat offers it this turn, its calls per turn, what a call does.
      A mistake gets a receipt and the round goes on, so the planner sees it:
      a tool that is off here ("现在用不了 X。") or does not exist, a limit
      reached ("这次已经用过 N 次 X 了"), her own line as a reply target, a
      wait without seconds, a target not shown. Nothing is silently repaired.
      Replay's `must_speak` is one guard on the tools that stop a turn plus
      the end of the turn.
  - The writer (`prompts/persona.md` + `prompts/writer.md`) sees the log as
    turns (her lines are assistant turns) plus the planner's style and
    intent ("#n" becomes the line it names), never its analysis: that is
    decision talk, and MaiBot stopped passing it on (2026-10-09, `908dfa6e`).
    What the turn found reaches it as facts: the recalled summary, what
    `search_history` and `search_web` found, jargon reference notes
    (`Findings::jargon` -> `WriterInput::jargon`), what she knows about the
    person she answers and anyone `look_up_person` looked up. Then the
    picked expressions. Everything else about how she talks lives in
    `writer.md`. Its answer passes one repair step (`repair_reply`, 2026-10-13):
    a transcript echo, a leading label or typed mention, Markdown, asides, then
    the byte, length and emoji bounds. Each repair is logged by kind as a
    warning; an answer with nothing left goes to the next writer fallback.
  - `delivery` sends replies the way MaiBot does (2026-10-10).
    - Casual replies are split at "，,。;", spaces and newlines, then
      neighbours merge at random. `typos` swaps in a homophone now and then
      (jieba plus pypinyin), sometimes followed by the correction quoting it.
      Everything is packed into at most 3 messages.
    - Messages wait for typing: 0.3 s per Han character. The first was being
      typed since the planner decided, so it goes out once the selector and
      writer are done or once it could have been typed, whichever is later
      (2026-10-14). MaiBot sends it at once. With the schedule on there is
      no other delay while she is awake. The random read moment before the
      planner (3–20 s, 1–8 s when called) was dropped: it did not depend on
      what she said, so "草" waited as long as a sentence. In the friend
      group her first message came a median 14 s after the line before it
      (slowest quarter 24–58 s), 9 s of it before the planner. About 6 s is
      the model calls: planner, selector, writer, 2 s each, one after
      another. A Haiku call through the gateway takes 1.1–1.8 s even for one
      token, and a cache hit on 23k tokens is no faster than a miss, so what
      costs time is calls in a row, not tokens: the selector now runs beside
      the planner when she is called (below).
    - A deliberate long reply (`ReplyStyle::LONG`) goes out whole and
      untouched: it relaxes casual prose and punctuation rules, allows
      concrete facts/dates from `search_web`, preserves CJK parenthetical
      explanations (`keep_asides`), and skips casual expression habits
      (`select_habits`).
    - The first message quotes only when the planner asks.
  - What the models see (2026-10-11). A quote is stored as part of the text
    ("[回复 小明: 前二十字…] 内容"). QQ faces render by name, cards by their
    prompt and title, forwards as a preview of their first nodes (inline, or
    fetched with `get_forward_msg` like an image description). Notices of
    allowed groups become lines from "群通知" (sender 0, negative ids) that
    never start a turn: pokes, reactions, essence, joins and leaves, mutes. A
    recalled message is stored as "[已撤回]" (its segments stay) and is never
    answered. Muted, she stores messages but makes no model calls. Her group
    card, from `get_group_member_info` on connect, is her name in each group.
  - Prompts are laid out for prompt caching (2026-10-11). The log window is
    anchored and only grows between trims (MaiBot's 2× then 1×), so a turn's
    log is a prefix of the next one's. The planner gets the place and
    the numbered log as one cached part, then a tail with what changes per
    turn, the time last. The writer's system prompt is persona plus rules,
    the same in every chat; the log turns end a cached prefix; the place,
    time, style and notes form the final user part. The planner's tools
    never change between rounds; a tool she cannot use right now answers
    with a receipt instead.
    - Claude's cache lives 5 minutes, and group turns are often 5–60
      minutes apart, so nearly every turn wrote its whole log again (9–16k
      tokens at 1.25×). While a chat is quiet, the planner's last prompt up
      to its breakpoint (without the tail) goes out again every 4 minutes
      with one output token (`model.cache_keepalive_minutes`, 2026-10-14):
      a read at 0.1× that restarts the 5 minutes. Through the gateway such a
      call at 4 minutes kept a 13k-token entry alive past 8. On the friend
      group's gaps, warming for 20 minutes costs least, about 30% below none;
      longer costs a little more. The 1-hour TTL would do the same without
      the calls, but cannot be relied on to be offered. A hit is no faster
      than a miss through the gateway: this saves money, not time.
  - `lilia replay <chat.jsonl>` runs the same ingest and turn code on a
    recorded or synthetic log without NapCat and prints every decision;
    `scripts/replay-remote.sh` runs it on cachy with the live keys.
- **`archive/`**: copies the media of allowed chats (pictures, stickers,
  voice, video, files) into `<data_dir>/media/<sha1[:2]>/<sha1><ext>`.
  - **Why:** the QQ client cache on the host is pruned after a few days, and
    Tencent's download links expire.
  - **Sources, in order:** the segment's `url`; a container `path` mapped
    through NapCat's bind mount; `get_group_file_url`; `get_file`.
  - **Bookkeeping:** content addressing deduplicates files. The extension
    comes from magic bytes, and each segment gets a row in the `media` table,
    including failures and `too_large` (default cap 50 MB).
  - **Concurrency:** at most two messages are archived at a time; media from
    other chats is never fetched.
  - **Image reading:** after a picture or sticker is stored, one
    `vision_model` call (DeepSeek, image block plus a short prompt) describes
    it in a sentence or two, including any text in it. Descriptions are cached
    by content hash in `media.description`. GIFs go whole: DeepSeek reads
    their motion (a grid of frames scored no better, 2026-10-12). The stored
    message text is then re-rendered as `[图片: …]` / `[表情: …]`, and a turn
    waits up to
    `image_wait_seconds` for descriptions of its new messages before planning.
- **`store/`**: SQLite in WAL mode. Only allow-listed chats are stored; the bot
  account is also a member of unrelated groups.
- **`main`**:
  - A `TaskGroup` owns the supervisor and every reply worker, so SIGTERM
    cancels everything structurally.
  - The supervisor reconnects with capped exponential backoff (1 s to 60 s).
  - Message events are ignored until `get_login_info` has identified the bot.

Everything runs on one libuv loop thread. SQLite calls are synchronous; their
latency is negligible at this scale. Large media is read, hashed and written a
megabyte at a time with the loop free in between, and every model call has a
deadline (2026-10-11).

Model calls (2026-10-11):

- A planner model that is rate limited, or fails twice in a row, rests for 10
  minutes; its fallbacks answer meanwhile instead of each call first waiting
  out the deadline.
- Independent steps run side by side: the summary recall and her self notes
  before the planner, the web search and the expression pick before the writer.
- `chat.daily_token_budget` counts only calls made for chats.
  `learning.daily_token_budget` (rolling 24 h) holds back learning batches;
  learning from busy groups can no longer silence her.

## Roadmap

1. **Reply pipeline (done 2026-10-07).** Gate, debounce, planner and writer,
   multi-message delivery, persona v2, replay tool.
2. **Memory (in progress).** References: MaiBot's A_Memorix and person
   profiles, and AIRI's #2748 trust labels.
   - Searchable history (done 2026-10-08, `memory/`). Allowed chats are cut
     into chunks of up to 6 messages, stored in `memory_chunks` with their
     search text; the planner's `search_history` tool ranks them and the
     summaries (below) with `retrieval/`.
   - One retrieval layer (2026-10-11, `retrieval/`), after MaiBot, OpenClaw
     and LangChain, for chunks, summaries, facts, expressions and stickers:
     - Embeddings (Qwen3-VL-Embedding-2B through qvl, the sibling OpenVINO
       service, since 2026-10-12; Qwen3-Embedding-0.6B before) live in
       `embeddings`, keyed by profile (model and input format) and the
       text's SHA-1, cached in memory. A startup backfill embeds whatever
       the current profile lacks, then drops other profiles: a model change
       heals itself.
     - Queries carry one task per corpus, inline as Qwen3-Embedding's
       `Instruct:` or as the request's `instruction` field
       (`memory.instruction_mode`, qvl puts it in the system turn);
       documents are bare; vectors are normalized, so similarity is a dot
       product. Stickers are embedded as text only: their pictures (2026-10-12)
       won 1 of 480 query pairs and pulled in look-alikes, and a
       cross-encoder reranker scored below plain fusion at 2–5 s a search.
     - Dense (brute force) and, for chat search and stickers, BM25 over
       jieba words and Han pairs in memory (FTS5 cannot segment Chinese).
       Each leg ranks 4x what is wanted; RRF (k 60, 0.7/0.3) fuses them.
       The gate looks at the dense similarity; recency is a small boost;
       expressions use MMR (0.85).
   - Facts about people (done 2026-10-08), from the person she answers, each
     with a verbatim quote checked in code; never her own lines, jokes,
     rumours, or relationships inferred from sharing a group. Her own facts
     from what she says about herself (2026-10-10). `Memory::person_notes`
     (2026-10-17) isolates private-chat facts from groups, uses semantic
     retrieval for older facts when a person has more than `facts_per_person`
     facts, and replaces stale facts on update (`replace = true` on
     `learn_facts` writeback).
   - Summaries of chat that scrolls out of context, with recall cues (done
     2026-10-12: window summaries, above). Recall's threshold was measured
     on the rolling summaries' cues; re-measure it on window summaries.
   - Then: ranking by recency and use.
3. **lilia's own style.**
   - Expression learning (done 2026-10-07, `learn/`). Every group the
     account is in feeds an in-memory buffer; batches of 30 lines become
     "当[情境]时，可以[说法]" patterns with counts, and the text is dropped.
     Other groups are read-only. A pattern must quote its source line: each
     part of 用"…", cut at ellipses, has to occur in it. Put-downs like 区
     and 菜 count as words of their own in jieba's cut or inside rare ones
     (真菜), not inside common words like 区别 or 菜单 (2026-10-13).
   - MaiBot-style learning (2026-10-09). Patterns are kept per source chat
     (`expression_groups` decides which chats share them), reviewed before
     storing with every verdict logged, embedded, and picked after the
     planner by a selector call: up to 3 of 30 MMR-ranked candidates
     (`learning.max_selected`). The selector is the planner model with its
     tool call forced (2026-10-14: 47 output tokens instead of ~190 of
     narration, ~0.6 s). When she is called it runs beside the planner's
     first round for the calling line, without the planner's intent, and a
     reply to that line uses its picks (about 2 s sooner). Jargon is mined
     from the same batches: each sighting keeps an anonymised ±3-line
     snippet (at most 10 per term), filtering out common `jieba` dictionary
     words (`dictionary_frequency >= 500`) and pure numbers
     (`ordinary_jargon_word`), and the meaning is inferred at 4, 8, 25
     and 100 sightings, with and without context; a term counts as slang
     only when the two readings differ. Known terms in the recent chat reach
     both the planner and the writer as reference notes (`Findings::jargon` ->
     `WriterInput::jargon`). Not ported: clustering, and behaviour learning,
     which MaiBot itself removed on 2026-10-10.
   - High-frequency terms (2026-10-09). `learn/segment.*` ports jieba's
     default `cut` (dictionary DAG plus HMM, jieba's own data embedded);
     `learn/frequency.*` ports MaiBot's word-frequency service. Batches of
     the chats she talks in keep a ranked table of their frequent words and
     phrases (top 1000); slang that is also a frequent term leads the
     jargon notes. Other groups get no table: MaiBot only reads the current
     chat's.
   - Next: persona revisions proposed by reflection and approved by an admin,
     with every version kept.
4. **Stickers.**
   - Learn stickers from every group the account is in (`sticker/`): the
     image (deduplicated by hash), a vision model's description and tags,
     a capped pool of the fit ones.
   - The pool turns over (2026-10-13). Each sticker has a heat: a sighting
     adds 1, her send 2, and it halves every `stickers.half_life_days` (14).
     Once the pool is full, a spare hotter than the coldest member by more
     than one sighting takes its place, so only a spare seen again gets in;
     a sticker a person edited stays. Before, a newcomer could only
     replace a never-sent member and counts never decayed, so the pool froze.
     Heat also weights the sample used without embeddings and orders word
     matches. Schema 18 started heat at seen + 2 × used.
   - Reward (2026-10-13): n of her sends of a sticker were judged, s of them
     with a positive hindsight reward. The Beta(1, 1) posterior mean
     m = (1 + s) / (2 + n) adds `feedback.reward_weight` × 6 × (m − 0.5) to
     its search score (at most ±0.06 at the default 0.02, as the clamped
     reward sum it replaces), and the sample weight is (heat + 1) × 2m.
   - Usage (2026-10-12, after MaiBot's jargon cycle; MaiBot itself freezes
     sticker tags): each sighting, hers included, keeps an anonymised snippet
     of up to 3 lines before and 3 after (within 3 minutes) in
     `sticker_evidence`. At 3, 6, 12, 24, 48 and 96 snippets the helper model
     infers what the sticker is sent for (`prompts/sticker_usage.md`, the
     previous usage as a prior, `no_info` when unclear); the usage and its
     tags are searched and shown to the planner (`｜常用于：…`). A person's
     tags stay; the usage is still learned.
   - Privacy: like jargon evidence, snippets come from every group not
     excluded. Speakers become 群友N (hers [SELF]); their names, typed
     @mentions, numbers of 5+ digits and links are scrubbed; lines are cut at
     80 code points; only the newest 12 per sticker stay, and a rejected
     sticker loses them. `lilia stickers --forget-group <id>` purges a group.
     No other text from other groups is stored.
   - Tagging (2026-10-12): the prompt asks for the action (for a GIF, the
     motion) and the situations it is sent in, with no example tags (the old
     list primed 无语 onto 44% of the pool; now 16%). A person's
     edits (`modified_by = 'USER'`, from `scripts/apply_sticker_edits.py`)
     are never overwritten. `lilia stickers --retag --db <copy>` re-tags a copy
     so a change can be scored with `lilia retrieval --labels` first, and
     `--apply-from` copies the winning tags.
   - The planner sends them itself (2026-10-11): `find_sticker(feeling)`
     shows the closest few of the whole pool (`retrieval/` over tags and
     description; words alone without memory), marking ones
     she sent within a day; `send_sticker(s<id>)` sends the one it picks.
     `repeat` resends someone's sticker (复读) by its URL.
5. **Replace the ad hoc parts** (review against MaiBot `0f94cfa8`,
   2026-10-12). About a fifth of `chat/` and half the rules in `planner.md`
   and `writer.md` patch symptoms. Each change is measured in replay first.
   - Writer output (measured 2026-10-13). The scrubbers came from older
     writers and log formats. Now that they are one counted step
     (`repair_reply`), 131 replayed answers from the live writer needed no
     format repair at all (one length cap), so the `name: text` log format
     stays. Steps that never fire in the live warnings go.
   - Addressee (done 2026-10-13). It was worked out five times: planner
     prose about 你/你们, tail facts repeating `→她`, `interaction_summary`,
     `line_features`, and two writer patches. The arrows and quotes stay;
     the prior is the one derived signal (addressee recognition: Ouchi &
     Tsuboi 2016, Le et al. 2019, MPC-BERT). Measured with `--probe-db` on
     243 judge-labelled real lines: dropping the tail facts, the summary and
     most of the prose while showing the prior matched the old setup (misses
     5 of 52 against 6–9, misreads 5 against 5–6); without the prior, misses
     rose to 13. The writer's two patches stay: probes do not see the writer.
   - Planner dispatch (done 2026-10-13). An if-chain on tool names with
     per-tool counters became a table of tool handlers; silent repairs of
     planner mistakes (a self target became the newest line, a missing wait
     30 s) became receipts, and replay's `must_speak` one guard. A tool that
     was off ended the turn, and `send_sticker` and `poke` were not gated.
   - Prompt rules (done 2026-10-13). Friendliness, not pretending to know,
     the interests, the wait range and "only that line" each live in one
     place; the planner no longer offers 草/确实 as good replies. Replayed
     (about 280 replies per variant), permitting 草 primed it (8 against 0);
     naming a stock phrase as one to avoid made no clear difference either
     way. The writer names them as habits to avoid and permits none. A/B a
     rule in replay before adding it.
   - No anti-loop guard: MaiBot replaces an analysis over 90% like the
     previous one with a reflection prompt.
   - People reach the planner through `look_up_person` and the writer through
     `Memory::person_notes` (resolved 2026-10-17: semantic retrieval of older
     facts when over `facts_per_person`, near-duplicate updates replace stale
     facts with `replace = true` on `learn_facts` writeback, and private-chat
     facts are kept out of groups). MaiBot (with A_Memorix) shows sectioned
     profiles of up to 3 people per round.
   - Smaller: she cannot @ anyone; quoting is opt-in, where MaiBot quotes
     by default (its hint says for older targets).
   - Wait for more data. Whether to speak is prose rate control (fit
     P(reply) under a budget, as MaiBot's dynamic mode, or a bandit on the
     hindsight reward); on 2026-10-13 there were 85 judged replies in 14
     days. Of the stacked delays, debounce and the image wait are mechanics,
     typing follows the length of what she says, and the busy late reply is
     drawn from a hand-set range and could come from real gaps. But the
     friend group had 82 turn-taking gaps under 10 minutes (median 29 s,
     middle half 13–74 s) and 6 quote replies, and on 2026-10-14 49 answers
     within a live exchange showed no trend with length.
   - Keep: delivery and typos, the planner history and caching, edges and
     quotes, fact grounding, the jargon and sticker usage cycles, the
     retriever core, the prior fitting, privacy scrubbing.

Not planned now: server-management tools, art feeds.
