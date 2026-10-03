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
- **`chat/`**:
  - `trigger` decides whether a message needs an answer: an @-mention, a reply
    to lilia's own message, an alias, or an allowed private chat.
  - `context` lowers the persona, runtime facts and a byte-bounded recent log
    into SYSTEM / DEVELOPER / USER items.
  - `Responder` stores every message from allowed chats and runs at most one
    reply worker per chat. Triggers that arrive while a reply is being
    generated are merged into the next reply.
  - Guards: a minimum interval between replies and a daily token budget,
    counted from `llm_calls`.
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

2. **Tools.**
   - A tool-round loop in `Responder` (the provider already speaks
     `tool_use` / `tool_result`).
   - First tools: cachy service status and allow-listed start/stop
     (`systemctl --user` through `lighter::Process`, with admin checks and
     confirmation), and web search through a self-hosted SearXNG plus a
     fetch-and-extract tool.
   - An MCP client so tools can live out of process.
3. **Memory and persona.**
   - Periodic summarization of chat logs into memories (people, events,
     running jokes).
   - Retrieval scored by relevance plus recency decay plus importance: FTS5
     with a trigram tokenizer first, embeddings later.
   - Persona revisions that are proposed by reflection and approved or rolled
     back by an admin, keeping every version.
4. **Emotes.**
   - Collect stickers (image `sub_type` 1 and `mface`), deduplicated by hash
     with their surrounding context.
   - Tag them with an image-capable model (`deepseek-flash` or MiniMax M3); this
     needs an `ImagePart` in the provider history.
   - A `send_emote` tool.
5. **Feeds.** Pixiv rankings first (RSSHub or the Pixiv API), with NSFW
   filtering before anything is posted. Twitter/X only through a throwaway
   account.
