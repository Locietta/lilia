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
  - The worker waits for the burst to settle (`debounce_seconds`, 2 s when
    addressed, 15 s at most).
  - The planner (`prompts/planner.md`, `planner_model`, thinking off) keeps
    one history per chat, as MaiBot's does (`chat/session.*`, 2026-10-11).
    - It is rendered each round from the stored lines after the window's
      anchor (`#<row id> [HH:MM] name(qq)→addressee: text`) with the planner's
      own rounds among them: its analysis, its tool call and the result. New
      lines simply follow. Between trims it only grows, so it is a cached prefix.
    - Who a line is said to (`chat/addressee`, 2026-10-12): `→她` / `→小明` /
      `→全体` from its @s and quotes, quotes as `[回复 #row 小明: …]` (hers
      restored from her stored segments), computed once per row. An @ without a
      name is named when stored. LLMs find the addressee of a group-chat line
      far better with this structure than from text (Tan et al. 2023; Penzo et
      al. 2024): in a four-person group, friends say 你 to each other.
    - Each round it writes a short analysis, then calls one tool:
      `reply(target, to_her, quote, style, intent)`, `find_sticker(feeling)`,
      `send_sticker(sticker)`, `repeat` (text or a sticker),
      `wait(seconds)`, `poke(target)`, `search_history(query)`,
      `look_up_person(target)` (the facts she noted about whoever wrote it),
      `search_web(query)` or `stay_silent(reason)`; no call means silence.
      Results are short ("她回复了 #n", "等了 N 秒", what a search found);
      her bubbles show up as her own lines. Side actions are its calls too:
      it is told that people are repeating a line or that someone poked her,
      and decides whether to join or poke back.
    - Per round only a tail changes: the first new line, facts about the new
      lines ("新消息里 #n 是对她说的", "#m 是对小明说的", a poke, her name),
      who has talked to whom in the last 15 minutes and a run between two
      people ("阿明和小红在一来一回地聊"), presence, self notes, slang (MaiBot's
      matcher, first sentence of each meaning), recall, the time. What she
      knows about people comes on request, through `look_up_person`.
    - `to_her` is the planner's reading of the reply's target (called,
      likely, unsure, no); for `no` the writer is told it is chipping in.
      Every round is logged in `planner_turns` (tool, argument, target,
      to_her, her messages, a short analysis), kept 180 days.
    - Hindsight feedback (`feedback/`, 2026-10-12, after MaiBot's
      reply_effect; off by default): ten minutes after she speaks (or once 40
      lines wait) the helper model reads the window with what came after and
      says whom each line was said to (her, other, group, self, unclear) and
      how her messages went down (stance, contribution). Labels keep 17
      features of the line (`chat::line_features`: explicit @/quote/name,
      你/你们, right after her, her share of the last 5 minutes, a run between
      two people…), not its text. A logistic prior is fitted on them
      (Newton/IRLS, pulled toward hand-set weights, per group toward the pooled
      model) and shown to the planner for lines that name no one
      ("#n 像是在对她说的可能：低/中/高") when `feedback.hint` is on; it is
      always logged in `planner_turns.hint`. Never a gate. Reward is the
      reaction (Pang et al. 2023), not how long the chat ran; her stickers'
      rewards boost them in search. `lilia feedback --backfill/--eval/--report/
--import-manual`; `scripts/addressee_label_page.py` for hand labels.
    - `lilia replay <probes.jsonl> --probe` runs one turn per expected line
      of a synthetic set (`tests/replay/addressee.jsonl`: quotes, @s, her
      earlier lines) and counts intrusions, misses and to_her agreement.
    - Past twice `context_messages` / `context_bytes` the anchor moves to
      keep 1×; rounds about lines no longer shown go, older turns keep their
      calls but lose their analyses. After a restart the history is the
      stored lines again.
    - Limits: 8 rounds and 3 replies per turn, 3 waits in a row (MaiBot's),
      waits of 1–60 s. Replay passes waits in simulated time.
  - The writer (`prompts/persona.md` + `prompts/writer.md`) sees the log as
    turns (her lines are assistant turns) plus the planner's style, intent
    and analysis (MaiBot's reply reason; "#n" becomes the line it names),
    what the planner looked up this turn, and picked expressions.
    Everything else about how she talks lives in `writer.md`.
  - `delivery` sends replies the way MaiBot does (2026-10-10).
    - Casual replies are split at "，,。;", spaces and newlines, then
      neighbours merge at random. `typos` swaps in a homophone now and then
      (jieba plus pypinyin), sometimes followed by the correction quoting it.
      Everything is packed into at most 3 messages.
    - Only later messages wait for typing: 0.3 s per Han character.
    - A deliberate long reply goes out whole and untouched.
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
    by content hash in `media.description`. With `model.vision_gif_frames`, a
    GIF goes as one JPEG grid of up to 9 distinct frames in time order
    (`image/frames`), for vision models that read only the first frame;
    DeepSeek reads GIFs whole, so it is off. The stored message text is then
    re-rendered as `[图片: …]` / `[表情: …]`, and a turn waits up to
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
       product. With `memory.embed_images` a sticker is also embedded by
       its picture (a tagged file path), and a search takes the better of
       its text and picture.
     - Dense (brute force) and, for chat search and stickers, BM25 over
       jieba words and Han pairs in memory (FTS5 cannot segment Chinese).
       Each leg ranks 4x what is wanted; RRF (k 60, 0.7/0.3) fuses them.
       The gate looks at the dense similarity; recency is a small boost;
       expressions use MMR (0.85).
   - Next: incremental facts and per-person profiles. Facts must quote the
     person; never store lilia's own lines, jokes, rumours, or relationships
     inferred from sharing a group. Record who said what.
   - Then: summaries of chat that scrolls out of context, with recall cues;
     ranking by recency and use; private memories never reach groups.
3. **lilia's own style.**
   - Expression learning (done 2026-10-07, `learn/`). Every group the
     account is in feeds an in-memory buffer; batches of 30 lines become
     "当[情境]时，可以[说法]" patterns with counts, and the text is dropped.
     Other groups are read-only. The planner picks up to two of about 15
     candidates per reply.
   - MaiBot-style learning (2026-10-09). Patterns are kept per source chat
     (`expression_groups` decides which chats share them), reviewed before
     storing with every verdict logged, embedded, and picked after the
     planner by a selector call over MMR-ranked candidates. Jargon is mined
     from the same batches: each sighting keeps an anonymised ±3-line
     snippet (at most 10 per term), and the meaning is inferred at 4, 8, 25
     and 100 sightings, with and without context; a term counts as slang
     only when the two readings differ. Known terms in the recent chat reach
     the planner as notes. Not ported: clustering, behaviour learning.
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
     list primed 无语 onto 44% of the pool; now 16%). Animated stickers can go
     as a grid of frames (`model.vision_gif_frames`, off: DeepSeek reads GIFs
     whole and the grid scored no better on the hand labels). A person's
     edits (`modified_by = 'USER'`, from `scripts/apply_sticker_edits.py`)
     are never overwritten. `lilia stickers --retag --db <copy>` re-tags a copy
     so a change can be scored with `lilia retrieval --labels` first, and
     `--apply-from` copies the winning tags.
   - The planner sends them itself (2026-10-11): `find_sticker(feeling)`
     shows the closest few of the whole pool (`retrieval/` over tags and
     description; words alone without memory), marking ones
     she sent within a day; `send_sticker(s<id>)` sends the one it picks.
     `repeat` resends someone's sticker (复读) by its URL.

Not planned now: server-management tools, art feeds.
