# MaiBot reference notes

Read on 2026-10-08 from MaiBot `f9ec367` (2026-10-07,
https://github.com/MaiM-with-u/MaiBot). Paths are relative to that repo.
This version runs the "Maisaka" planner/replyer; the older JSON planner,
`plan_style`, `states` and mood system are gone (old defaults come from tag
`0.11.6-beta`). Notes cover what matters for lilia; prompts are quoted where
the wording is the point.

## Persona

- Three fields, each used by one model:
  - `personality` (replyer): "是一个大二女大学生，现在正在上网和群友聊天。善于用人类的角度思考问题，聊天偏日常。"
  - `behavior_style` (planner only): the same plus "不会没话题硬找话题，".
  - `reply_style` (replyer): "你的风格平淡简短，可以参考贴吧的回复风格。不滥用比喻或者生硬句子。视情况省略主语或者进行倒装，风格较为随意。"
- The replyer's identity line is `你的名字是{nick}，也有人叫你{aliases}。\n{personality}`. The fallback is "是人类。" The voice model is never told it is an AI; only the focus-mode planner says "AI".
- `group_chat_prompt` goes to both planner and replyer:
  > 你正在qq群里聊天，下面是群里正在聊的内容，聊天中包含文字，图片和表情包等消息。
  > 回复尽量简短一些。最好一次对一个话题进行回复，但必须考虑不同群友发言之间的交互，免得啰嗦或者回复内容太乱。请注意把握聊天内容。
  > 不要总是提及自己的身份背景，根据聊天内容自由发挥，但是要日常不浮夸，不要刻意找话题。
  > 不用刻意回复其他人发送的表情包，只要关注表情包表达的含义。你可以适当发送表情包表达情绪。控制回复的频率，不要每个人的消息都回复，优先回复你感兴趣的或者主动提及你的，适当回复其他话题。
- Optional alternative styles are chosen with `multiple_probability` (default 0): "用1-2个字进行回复", "用1-2个符号进行回复", "带点翻译腔，但不要太长", and so on.
- `emotion_trait` adds a fixed suffix, for example "你整体理性冷静……".
- Old 0.11 extras:
  - `states`: personas swapped in at random with p=0.3.
  - `interest`: "对技术相关话题，游戏和动漫相关话题感兴趣……不喜欢太过沉重严肃的话题".
  - `plan_style`: included "如果有人对你感到厌烦，请减少回复".
  - A mood text, updated every 45 s and calming down after 200 s. It was removed later.

## Planner (`src/maisaka`, `prompts/zh-CN/maisaka_chat.prompt`)

- **An agent loop with native tools**: `reply`, `wait(seconds)`, `send_emoji`, `query_memory`, `query_person_profile`, and others. Calling no tool means "done, stay quiet". The model writes a free-text 分析 (analysis) and then tool calls.
  - Its earlier analyses and tool results **stay in its history**, so it knows what it did and why.
  - After `reply` it runs again with its own message in context, and may add more, `wait`, or stop. The cap is 10 internal rounds.
  - `wait` is capped at 3 in a row. In groups, new messages do not cut a wait short.
- Prompt core:
  - "你的任务是分析聊天和聊天中的互动情况，然后做出下一步动作。"
  - "请注意，你不是 {bot_name} 本人，不要替 {bot_name} 发言"
  - "现在的上下文和聊天记录只是与参与聊天的用户当前的互动，你们之间可能有更多过去的关系和信息没有展现在上下文中。"
  - "默认直接输出你当前的最新分析，不要重复之前的分析内容"
  - The final user turn: "你需要输出对{bot_name}发言的分析，视情况输出文本内容的分析，思考是否进行工具调用".
- **Anti-loop**: an analysis that is more than 90% similar to the previous one is replaced with "我应该根据我上面思考的内容进行反思，重新思考……".
- **`reply` tool arguments**:
  - `msg_id`
  - `set_quote`
  - `reply_reference`: "有助于回复的信息，包括当前聊天状态、人物关系、事实信息、回忆信息"
  - `reply_style`: 简短表达 / 正常回复 / 长回复
  - optional `expression_intent{focus,reply_act,scene,tone,prefer,avoid}`
  - The replyer receives `reply_reference`, or the planner's analysis if that is empty. It never gets a scripted "intent".
- **Trigger** (`turn_trigger/`):
  - Threshold of `ceil(1/talk_value)` pending messages.
  - Silence earns credit: idle seconds divided by the average interval (at least 30 s) count as virtual messages, up to threshold−1.
  - Idle backoff of 15 s doubling to 300 s after 2 idle cycles, bypassed at 6 pending messages.
  - 1 s debounce.
  - An @ forces a planner turn, but the planner may still decide not to reply.
  - Optional time-of-day `talk_value` rules.
  - A "dynamic" mode uses a fitted logistic P(reply) held to a per-hour budget.
- **Duplicate guard**: "你刚刚已经回复过这条消息，你刚刚的发言是：“…”……注意请不要和之前你的发言重复。"
- **Restart note**: "你离线休息了 {elapsed}，醒来后仍记得上次关机前的聊天内容。"

## Replyer (`src/chat/replyer/maisaka_generator_base.py`, `prompts/zh-CN/maisaka_replyer.prompt`)

- System prompt, short:
  ```
  {identity}
  现在请你读读之前的聊天记录，把握当前的话题，然后给出日常且口语化的回复，
  {reply_style}
  你可以参考【回复信息参考】中的信息，但是视情况而定，不用完全遵守。
  {group_chat_attention_block}
  请注意不要输出多余内容(包括不必要的前后缀，冒号，括号，表情包，@等 )，只输出发言内容就好。
  ```
- Message order:
  1. History: others as user turns with an XML-ish header (msg_id, time, nickname); its own past replies as assistant turns.
  2. A separate user turn: "【表达习惯参考，请视情况自然的使用】\n- 当"{situation}"时，可以用"{style}"来表达。" (0–5 items).
  3. An optional temporary style: "你的说话风格可以尝试：…".
  4. The reference: the planner's `reply_reference`, or its analysis.
  5. The final turn: "当前时间：… 你想要回复的消息是 {sender} 发送的…目标消息，不要把其他历史消息当成当前回复对象。- 发言内容：… 请自然地回复。不要输出多余说明、括号、@ 或额外标记，只输出实际要发言的内容。"
  6. For 简短表达: "请简短的回复，允许句子残缺，奇怪表达，倒装，省略，符合口语习惯，符合省力随意回复习惯". For 长回复: "可以针对问题做出较为详细的评论和说明".
- Memory, person profiles and tool results are **filtered out** of the replyer's context. Only what the planner puts in `reply_reference` reaches it.
- The older "retro" template ends with "现在，你说：". Its light and private variants add "不要回复的太有条理" and "平淡一些".
- **Models**:
  - replyer: deepseek-v4-pro, randomly with or without thinking, temperature 1
  - planner: v4-flash, temperature 0.7
  - utilities: flash

## Post-processing (`src/chat/utils/utils.py`, `typo_generator.py`)

- **Strip brackets** that contain CJK (stage directions). If nothing is left, send "呃呃".
- **Overlong output** (over 1024 characters, mostly CJK) is replaced by a lazy fallback: "不知道" / "懒得说" / "()".
- **Split** at `，`, `,`, space, `。`, `;` and newlines, but never inside quotes or between alphanumerics.
  - Neighbouring pieces are then **merged at random** with probability 1 − strength, where strength is 0.2 for under 12 characters, 0.6 for under 32, and 0.7 otherwise.
  - At most 8 pieces before falling back; then they are packed into at most 3 roughly even messages.
  - An optional LLM splitter must return the identical characters.
- **Typos** (jieba + pinyin): about 1% of characters become homophones. Half the time a typo is followed by a quoted correction message.
- **Typing delay**: 0.3 s per CJK character and 0.15 s per other character, only before the second and later messages.
- Nothing calls `random_remove_punctuation` any more.

## Expression learning (`src/learners/expression_learner.py`, `prompts/zh-CN/learn_style.prompt`)

- **When**: runs on context trims, roughly every 40 messages per chat, at least 30 s apart, with at least 10 messages.
  - Its own messages are included as context, marked `speaker:SELF`; the prompt says "不要总结 SELF 的发言".
- **Prompt** asks for 3–5 items (at most 10) in the form `当 "AAAAA" 时，可以 "BBBBB"`, each part at most 20 characters, with `source_id`, plus "思考有没有特殊的梗，一并总结成语言风格".
- **LLM review before storing** (`expression_evaluation.prompt`): returns suitable true/false with these criteria:
  1. 表达方式或言语风格是否与使用条件或使用情景匹配
  2. 允许部分语法错误或口语化或缺省出现
  3. 表达方式不能太过特指，需要具有泛用性
  4. 一般不涉及具体的人名或名称

  Rejected items are logged, and a person can rescue them in the WebUI.

- **Storage**: per chat (sharing is configurable), `count`, `content_list` of merged situations, `checked`, `modified_by`. Only exact matches merge, and nothing decays.
- **Selection**:
  - Candidates: about 10 sampled with weights by count (1–5), or a vector search over "情景/风格" embeddings with MMR.
  - A sub-agent then sees the last 10 messages and picks 0–5: "优先考虑自然、贴合上下文、不生硬、不模板化。如果没有明显合适的，就返回空数组。"
  - The picks are injected into the **replyer**, not the planner.

## Jargon (`src/learners/jargon_*.py`)

- **Extraction**: from the same batches, terms the model "无法理解含义、或者需要当前聊天圈内语境才能理解" (pinyin abbreviations, English and Chinese abbreviations, in-group phrases).
  - Each term is stored with ±3 lines of context.
  - Already-known terms are counted by substring match.
- **Meaning inference** runs when the count reaches 4, 8, 25 and 100.
  - It is inferred twice: from context, and from the term alone. If the two meanings match, the term is ordinary vocabulary, not jargon.
- **Use**: when a known term appears in others' messages, its meaning goes to the **planner**: "以下黑话来自当前上下文中其他用户消息的机械匹配，仅作理解聊天语境的参考：1. {term}：{meaning}".

## Stickers (`src/emoji_system`)

- **Collection**: every received sticker is hashed with SHA-256 and saved.
  - A VLM tags it: "请提取这个表情包主要表达的情绪、语气或常见使用场景标签，最多 5 个，使用逗号分隔". GIFs are sent as a strip of frames.
  - In chat the sticker appears as `[表情包: tags]`.
- **Registration**: one sticker per 10 minutes, with a pool of 64.
  - When the pool is full, an LLM chooses which old sticker to drop.
  - Unregistered files unused for 30 days are deleted.
- **Sending**: a grid of 25 random stickers with index badges goes to a vision sub-agent, which picks one. The text-only fallback matches the requested emotion against tags with Levenshtein distance.

## Memory (`src/A_memorix`, `src/maisaka/memory`)

- **Person facts**: after each bot reply, facts about the person it answered are extracted. They must be "能被目标用户原始发言直接支持", with a verbatim quote that is checked in code, and must be stable (no jokes, guesses or temporary plans).
- **Chat summaries**: incremental, every 36 messages. The model sees the previous 2 summaries and may return an empty result: "宁可少提取，也不要把噪声写进记忆". It also extracts entities and relations.
- **Profiles**: built from facts into fixed sections (身份 / 关系 / 稳定了解 / 相处偏好 / 近期互动 / 不确定). Cached for 6 h or until the evidence fingerprint changes.
  - Injected for up to 3 people as a user turn: "【人物画像-内部参考】以下内容仅供内部推理，不要向用户逐字复述。……若与当前对话冲突，以当前对话为准。"
- **Retrieval**: no LLM. Vectors, BM25 and graph are combined with RRF and PPR.
  - Exposed to the planner as a `query_memory` tool: "只有回复明显依赖群内过去对话、共同经历、公开约定……时使用；不要为了寒暄、即时情绪回应、轻松接话……而检索。不要把私聊或个人隐私记忆带到群聊里。"
- **Mid-term recall**: messages trimmed from context are summarised into `{summary, recall_cues[3-5]}`. Cues are embedded, and the best one with cosine ≥ 0.8 against the last 12 messages is injected.
- A_Memorix is off by default; mid-term recall is on. All memory calls use the cheap flash model, plus embeddings.
