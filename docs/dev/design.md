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
  - `flow` is the cheap gate. Being addressed (@, a reply to lilia, her
    name, an allowed private chat) always opens it. For `conversation_window_seconds`
    after she speaks every new message opens it. Otherwise it opens every
    `ceil(1/talk_value)` messages. Two silent decisions in a row back off
    (15 s doubling to 5 min) unless a burst of 6 messages arrives.
  - The worker waits for the burst to settle (`debounce_seconds`, 2 s when
    addressed, 15 s at most).
  - The planner (`prompts/planner.md`, `planner_model`, thinking off) sees the
    numbered log with the new messages marked and must call `reply(target,
quote, style, search, intent)` or `stay_silent(reason)`. If she was
    addressed, silence is overridden.
  - The writer (`prompts/persona.md` + `prompts/writer.md`) sees the log as
    turns (her lines are assistant turns) plus the planner's style and
    intent; `alt_style_probability` swaps in a one-off style. Web search is
    offered only when the planner asks for it.
  - `delivery` splits the reply into at most 3 messages (one per line, long
    lines after sentence ends, no final "。") and sleeps a typing delay
    before each. Only the first message quotes, and only when the planner asks.
  - `lilia replay <chat.jsonl>` runs the same ingest and turn code on a
    recorded or synthetic log without NapCat and prints every decision;
    `scripts/replay-remote.sh` runs it on cachy with the live keys.- **`archive/`**: copies the media of allowed chats (pictures, stickers,
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
    by content hash in `media.description`. The stored message text is then
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
latency is negligible at this scale.

## Roadmap

1. **Reply pipeline (done 2026-10-07).** Gate, debounce, planner and writer,
   multi-message delivery, persona v2, replay tool.
2. **Memory (in progress).** References: MaiBot's A_Memorix and person
   profiles, and AIRI's #2748 trust labels.
   - Searchable history (done 2026-10-08, `memory/`). Allowed chats are cut
     into chunks of up to 6 messages, embedded by a local model
     (Qwen3-Embedding-0.6B via llama.cpp on cachy's iGPU) and stored in
     `memory_chunks`; the planner's `search_history` tool ranks by cosine
     plus a literal-keyword bonus. Brute force is fine at this size.
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
     the planner as notes. Not ported: jieba word-frequency ranking,
     clustering, behaviour learning.
   - Next: persona revisions proposed by reflection and approved by an admin,
     with every version kept.
4. **Stickers (later).**
   - Learn stickers from every group the account is in.
   - Keep the image (deduplicated by hash) and a short usage note written
     by a vision model from the surrounding messages, then drop the other
     groups' text.
   - Pick by tags first, embeddings once the collection is large; send
     through a `send_sticker` planner tool.

Not planned now: server-management tools, art feeds.
