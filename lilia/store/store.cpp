#include "store.h"

#include <algorithm>
#include <array>
#include <format>
#include <utility>

#include <sqlite3.h>

namespace lilia::store {

namespace {

// 2: media table. 3: expressions table. 4: media.description. 5: memory_chunks.
// 6: person_facts. 7: chat_summaries. 8: stickers. 9: expressions per chat
// (rebuilt), expression_reviews, jargon. 12: embeddings in their own table
// (keyed by profile and text hash), memory_chunks.search_text. 13:
// stickers.modified_by. 14: sticker usage (stickers.usage and its counts,
// sticker_evidence). 15: planner_turns. 16: hindsight feedback (stickers.reward,
// addressee_labels, reply_effects, addressee_models, feedback_progress).
constexpr int k_schema_version = 16;

constexpr std::string_view k_expressions_table = R"sql(
CREATE TABLE expressions (
    id INTEGER PRIMARY KEY,
    chat_type TEXT NOT NULL DEFAULT '',
    chat_id INTEGER NOT NULL DEFAULT 0,
    situation TEXT NOT NULL,
    style TEXT NOT NULL,
    situations TEXT NOT NULL DEFAULT '',
    count INTEGER NOT NULL DEFAULT 1,
    first_seen INTEGER NOT NULL,
    last_active INTEGER NOT NULL,
    checked INTEGER NOT NULL DEFAULT 0,
    modified_by TEXT NOT NULL DEFAULT 'AI',
    UNIQUE (chat_type, chat_id, style)
);
)sql";

constexpr std::string_view k_schema = R"sql(
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY,
    chat_type TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    sender_id INTEGER NOT NULL,
    sender_name TEXT NOT NULL,
    time INTEGER NOT NULL,
    segments TEXT NOT NULL,
    text TEXT NOT NULL,
    from_self INTEGER NOT NULL DEFAULT 0,
    UNIQUE (chat_type, chat_id, message_id)
);
CREATE INDEX IF NOT EXISTS messages_by_chat_time ON messages (chat_type, chat_id, time, id);

CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY,
    time INTEGER NOT NULL,
    chat_type TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cache_read_tokens INTEGER NOT NULL,
    cache_write_tokens INTEGER NOT NULL,
    stop TEXT NOT NULL,
    request_id TEXT NOT NULL,
    error TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS llm_calls_by_time ON llm_calls (time);

CREATE TABLE IF NOT EXISTS media (
    id INTEGER PRIMARY KEY,
    chat_type TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    segment_index INTEGER NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    sha1 TEXT NOT NULL,
    path TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    name TEXT NOT NULL,
    source TEXT NOT NULL,
    error TEXT NOT NULL,
    time INTEGER NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    UNIQUE (chat_type, chat_id, message_id, segment_index)
);
CREATE INDEX IF NOT EXISTS media_by_sha1 ON media (sha1);

CREATE TABLE IF NOT EXISTS expression_reviews (
    id INTEGER PRIMARY KEY,
    time INTEGER NOT NULL,
    chat_type TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    situation TEXT NOT NULL,
    style TEXT NOT NULL,
    verdict TEXT NOT NULL,
    reason TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jargon (
    id INTEGER PRIMARY KEY,
    content TEXT NOT NULL UNIQUE,
    meaning TEXT NOT NULL DEFAULT '',
    is_jargon INTEGER NOT NULL DEFAULT -1,
    count INTEGER NOT NULL DEFAULT 0,
    last_inference_count INTEGER NOT NULL DEFAULT 0,
    complete INTEGER NOT NULL DEFAULT 0,
    created_by TEXT NOT NULL DEFAULT 'AI',
    first_seen INTEGER NOT NULL,
    last_seen INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS jargon_chats (
    jargon_id INTEGER NOT NULL,
    chat_type TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (jargon_id, chat_type, chat_id)
);

CREATE TABLE IF NOT EXISTS jargon_evidence (
    id INTEGER PRIMARY KEY,
    jargon_id INTEGER NOT NULL,
    chat_type TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    snippet TEXT NOT NULL,
    time INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS jargon_evidence_by_term ON jargon_evidence (jargon_id, time);

CREATE TABLE IF NOT EXISTS learned_spans (
    group_id INTEGER PRIMARY KEY,
    first_time INTEGER NOT NULL,
    last_time INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS frequent_terms (
    id INTEGER PRIMARY KEY,
    chat_type TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    term TEXT NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 0,
    message_count INTEGER NOT NULL DEFAULT 0,
    rank INTEGER NOT NULL DEFAULT 0,
    frequency REAL NOT NULL DEFAULT 0,
    message_frequency REAL NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    UNIQUE (chat_type, chat_id, term)
);

CREATE TABLE IF NOT EXISTS memory_chunks (
    id INTEGER PRIMARY KEY,
    chat_type TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    first_row INTEGER NOT NULL,
    last_row INTEGER NOT NULL,
    start_time INTEGER NOT NULL,
    end_time INTEGER NOT NULL,
    text TEXT NOT NULL,
    search_text TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS memory_chunks_by_row ON memory_chunks (chat_type, chat_id, last_row);

CREATE TABLE IF NOT EXISTS person_facts (
    id INTEGER PRIMARY KEY,
    person_id INTEGER NOT NULL,
    fact TEXT NOT NULL,
    quote TEXT NOT NULL,
    chat_type TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    time INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS person_facts_by_person ON person_facts (person_id, time);

CREATE TABLE IF NOT EXISTS chat_summaries (
    id INTEGER PRIMARY KEY,
    chat_type TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    first_row INTEGER NOT NULL,
    last_row INTEGER NOT NULL,
    start_time INTEGER NOT NULL,
    end_time INTEGER NOT NULL,
    summary TEXT NOT NULL,
    cues TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chat_summaries_by_chat ON chat_summaries (chat_type, chat_id, last_row);

CREATE TABLE IF NOT EXISTS stickers (
    id INTEGER PRIMARY KEY,
    sha1 TEXT NOT NULL UNIQUE,
    path TEXT NOT NULL,
    media_type TEXT NOT NULL,
    bytes INTEGER NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    seen INTEGER NOT NULL DEFAULT 1,
    used INTEGER NOT NULL DEFAULT 0,
    first_seen INTEGER NOT NULL,
    last_seen INTEGER NOT NULL,
    last_used INTEGER NOT NULL DEFAULT 0,
    modified_by TEXT NOT NULL DEFAULT 'AI',
    usage TEXT NOT NULL DEFAULT '',
    usage_count INTEGER NOT NULL DEFAULT 0,
    usage_inferred_count INTEGER NOT NULL DEFAULT 0,
    tags_by TEXT NOT NULL DEFAULT 'VISION',
    reward REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS stickers_by_status ON stickers (status, seen);

CREATE TABLE IF NOT EXISTS sticker_evidence (
    id INTEGER PRIMARY KEY,
    sticker_id INTEGER NOT NULL,
    chat_type TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    snippet TEXT NOT NULL,
    own INTEGER NOT NULL DEFAULT 0,
    time INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS sticker_evidence_by_sticker ON sticker_evidence (sticker_id, time);

CREATE TABLE IF NOT EXISTS planner_turns (
    id INTEGER PRIMARY KEY,
    time INTEGER NOT NULL,
    chat_type TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    round INTEGER NOT NULL,
    first_new_row INTEGER NOT NULL,
    shown_row INTEGER NOT NULL,
    tool TEXT NOT NULL,
    argument TEXT NOT NULL DEFAULT '',
    target_message_id INTEGER NOT NULL DEFAULT 0,
    to_her TEXT NOT NULL DEFAULT '',
    sent_message_ids TEXT NOT NULL DEFAULT '',
    analysis TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    hint TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS planner_turns_by_chat ON planner_turns (chat_type, chat_id, time);

CREATE TABLE IF NOT EXISTS addressee_labels (
    id INTEGER PRIMARY KEY,
    chat_type TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    row_id INTEGER NOT NULL,
    time INTEGER NOT NULL,
    feature_version INTEGER NOT NULL,
    features TEXT NOT NULL,
    label INTEGER NOT NULL,
    source TEXT NOT NULL,
    prior REAL NOT NULL DEFAULT -1,
    UNIQUE (chat_type, chat_id, message_id)
);

CREATE TABLE IF NOT EXISTS reply_effects (
    id INTEGER PRIMARY KEY,
    chat_type TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    her_message_id INTEGER NOT NULL,
    target_message_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    sticker_id INTEGER NOT NULL DEFAULT 0,
    to_her TEXT NOT NULL DEFAULT '',
    for_her TEXT NOT NULL,
    answered_by_other TEXT NOT NULL,
    stance TEXT NOT NULL,
    contribution TEXT NOT NULL,
    reward REAL NOT NULL,
    time INTEGER NOT NULL,
    UNIQUE (chat_type, chat_id, her_message_id)
);

CREATE TABLE IF NOT EXISTS addressee_models (
    chat_type TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    feature_version INTEGER NOT NULL,
    weights TEXT NOT NULL,
    rows INTEGER NOT NULL,
    positives INTEGER NOT NULL,
    log_loss REAL NOT NULL,
    fitted_at INTEGER NOT NULL,
    PRIMARY KEY (chat_type, chat_id)
);

CREATE TABLE IF NOT EXISTS feedback_progress (
    chat_type TEXT NOT NULL,
    chat_id INTEGER NOT NULL,
    through_row INTEGER NOT NULL,
    PRIMARY KEY (chat_type, chat_id)
);

CREATE TABLE IF NOT EXISTS embeddings (
    profile TEXT NOT NULL,
    hash TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (profile, hash)
) WITHOUT ROWID;
)sql";

/// Columns that held vectors before schema 12, when embeddings moved to their
/// own table.
constexpr std::array<std::pair<std::string_view, std::string_view>, 9> k_vector_columns{{
    {"memory_chunks", "model"},
    {"memory_chunks", "vector"},
    {"person_facts", "model"},
    {"person_facts", "vector"},
    {"chat_summaries", "model"},
    {"chat_summaries", "vector"},
    {"chat_summaries", "cue_vectors"},
    {"expressions", "model"},
    {"expressions", "vector"},
}};

Result<Store> initialize(Result<Database> opened, bool file_backed) {
    if (opened.has_error()) return lighter::outcome_error(std::move(opened).error());
    Store store{.database = std::move(*opened)};

    if (file_backed) {
        auto wal = store.database.execute("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;");
        if (wal.has_error()) return lighter::outcome_error(std::move(wal).error());
    }

    i64 current = 0;
    {
        // A statement left open blocks the DROP TABLE below.
        auto version = store.database.prepare("PRAGMA user_version");
        if (version.has_error()) return lighter::outcome_error(std::move(version).error());
        auto row = version->step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        current = *row ? version->column_int(0) : 0;
    }
    if (current > k_schema_version) {
        return lighter::outcome_error(Error::storage("database schema version " + std::to_string(current) + " is newer than this build"));
    }

    auto created = store.database.execute(k_schema);
    if (created.has_error()) return lighter::outcome_error(std::move(created).error());
    // Version 9 rebuilt expressions (per chat, merged situations, review
    // flags, embeddings); older rows become shared AI-made patterns. Decided
    // by what the table looks like, not by the version number.
    const auto count_of = [&store](std::string_view sql) -> Result<i64> {
        auto statement = store.database.prepare(sql);
        if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
        auto row = statement->step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        return *row ? statement->column_int(0) : 0;
    };
    auto table = count_of("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'expressions'");
    if (table.has_error()) return lighter::outcome_error(std::move(table).error());
    if (*table == 0) {
        auto fresh = store.database.execute(std::string(k_expressions_table));
        if (fresh.has_error()) return lighter::outcome_error(std::move(fresh).error());
    } else {
        auto rebuilt = count_of("SELECT COUNT(*) FROM pragma_table_info('expressions') WHERE name = 'chat_type'");
        if (rebuilt.has_error()) return lighter::outcome_error(std::move(rebuilt).error());
        if (*rebuilt == 0) {
            auto migrated =
                store.database.execute("BEGIN; ALTER TABLE expressions RENAME TO expressions_old;" + std::string(k_expressions_table) +
                                       "INSERT INTO expressions (id, situation, style, situations, count, first_seen, last_active) "
                                       "SELECT id, situation, style, situation, count, first_seen, last_active FROM expressions_old;"
                                       "DROP TABLE expressions_old; COMMIT;");
            if (migrated.has_error()) {
                (void) store.database.execute("ROLLBACK");
                return lighter::outcome_error(std::move(migrated).error());
            }
        }
    }
    // Left behind by the first schema 9 build, whose DROP failed after the copy.
    auto leftover = store.database.execute("DROP TABLE IF EXISTS expressions_old");
    if (leftover.has_error()) return lighter::outcome_error(std::move(leftover).error());
    auto indexed = store.database.execute("CREATE INDEX IF NOT EXISTS expressions_by_chat ON expressions (chat_type, chat_id)");
    if (indexed.has_error()) return lighter::outcome_error(std::move(indexed).error());
    // Schema 12 moved vectors into `embeddings` (re-embedded by the
    // backfill). Chunks are rebuilt to gain their search text; summaries,
    // facts and expressions keep their rows. Decided by the table's shape.
    auto old_chunks = count_of("SELECT COUNT(*) FROM pragma_table_info('memory_chunks') WHERE name = 'vector'");
    if (old_chunks.has_error()) return lighter::outcome_error(std::move(old_chunks).error());
    if (*old_chunks > 0) {
        std::string sql = "BEGIN; DROP INDEX IF EXISTS memory_chunks_by_chat; DELETE FROM memory_chunks;";
        for (const auto &[table_name, column] : k_vector_columns) {
            auto present = count_of(std::format("SELECT COUNT(*) FROM pragma_table_info('{}') WHERE name = '{}'", table_name, column));
            if (present.has_error()) return lighter::outcome_error(std::move(present).error());
            if (*present > 0) sql += std::format("ALTER TABLE {} DROP COLUMN {};", table_name, column);
        }
        sql += "ALTER TABLE memory_chunks ADD COLUMN search_text TEXT NOT NULL DEFAULT ''; COMMIT;";
        if (auto migrated = store.database.execute(sql); migrated.has_error()) {
            (void) store.database.execute("ROLLBACK");
            return lighter::outcome_error(std::move(migrated).error());
        }
    }
    // Columns added after their table, by shape: absent ones are added.
    const auto ensure_column = [&](std::string_view table_name, std::string_view column, std::string_view declaration) -> Result<void> {
        auto present = count_of(std::format("SELECT COUNT(*) FROM pragma_table_info('{}') WHERE name = '{}'", table_name, column));
        if (present.has_error()) return lighter::outcome_error(std::move(present).error());
        if (*present > 0) return {};
        return store.database.execute(std::format("ALTER TABLE {} ADD COLUMN {} {}", table_name, column, declaration));
    };
    for (const auto &[column, declaration] : std::array<std::pair<std::string_view, std::string_view>, 6>{{
             {"modified_by", "TEXT NOT NULL DEFAULT 'AI'"},
             {"usage", "TEXT NOT NULL DEFAULT ''"},
             {"usage_count", "INTEGER NOT NULL DEFAULT 0"},
             {"usage_inferred_count", "INTEGER NOT NULL DEFAULT 0"},
             {"tags_by", "TEXT NOT NULL DEFAULT 'VISION'"},
             {"reward", "REAL NOT NULL DEFAULT 0"},
         }}) {
        if (auto added = ensure_column("stickers", column, declaration); added.has_error()) {
            return lighter::outcome_error(std::move(added).error());
        }
    }
    // The media table predates its description column in versions 2 and 3.
    if (current == 2 || current == 3) {
        auto added = store.database.execute("ALTER TABLE media ADD COLUMN description TEXT NOT NULL DEFAULT ''");
        if (added.has_error()) return lighter::outcome_error(std::move(added).error());
    }
    auto stamped = store.database.execute("PRAGMA user_version = " + std::to_string(k_schema_version));
    if (stamped.has_error()) return lighter::outcome_error(std::move(stamped).error());
    return store;
}

} // namespace

Result<Store> Store::open(const std::filesystem::path &path) { return initialize(Database::open(path), true); }

Result<Store> Store::open_memory() { return initialize(Database::open_memory(), false); }

Result<Store> Store::open_readonly(const std::filesystem::path &path) {
    auto database = Database::open_readonly(path);
    if (database.has_error()) return lighter::outcome_error(std::move(database).error());
    return Store{.database = *std::move(database)};
}

Result<void> Store::add_message(const StoredMessage &message) {
    auto statement = database.prepare("INSERT OR IGNORE INTO messages (chat_type, chat_id, message_id, sender_id, sender_name, time, "
                                      "segments, text, from_self) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, message.chat.type);
    statement->bind(2, message.chat.id);
    statement->bind(3, message.message_id);
    statement->bind(4, message.sender_id);
    statement->bind(5, message.sender_name);
    statement->bind(6, message.time);
    statement->bind(7, message.segments_json);
    statement->bind(8, message.text);
    statement->bind(9, i64{message.from_self ? 1 : 0});
    return statement->run();
}

Result<std::vector<StoredMessage>> Store::recent_messages(const ChatKey &chat, usize limit) {
    auto statement = database.prepare("SELECT message_id, sender_id, sender_name, time, segments, text, from_self, id FROM messages "
                                      "WHERE chat_type = ?1 AND chat_id = ?2 ORDER BY time DESC, id DESC LIMIT ?3");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, chat.type);
    statement->bind(2, chat.id);
    statement->bind(3, static_cast<i64>(limit));

    std::vector<StoredMessage> messages;
    for (;;) {
        auto row = statement->step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        if (!*row) break;
        messages.push_back({
            .chat = chat,
            .message_id = statement->column_int(0),
            .sender_id = statement->column_int(1),
            .sender_name = statement->column_text(2),
            .time = statement->column_int(3),
            .segments_json = statement->column_text(4),
            .text = statement->column_text(5),
            .from_self = statement->column_int(6) != 0,
            .row_id = statement->column_int(7),
        });
    }
    std::ranges::reverse(messages);
    return messages;
}

Result<std::vector<StoredMessage>> Store::messages_after(const ChatKey &chat, i64 after_row, usize limit) {
    auto statement = database.prepare("SELECT message_id, sender_id, sender_name, time, segments, text, from_self, id FROM messages "
                                      "WHERE chat_type = ?1 AND chat_id = ?2 AND id > ?3 ORDER BY id LIMIT ?4");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, chat.type);
    statement->bind(2, chat.id);
    statement->bind(3, after_row);
    statement->bind(4, static_cast<i64>(limit));
    std::vector<StoredMessage> messages;
    for (;;) {
        auto row = statement->step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        if (!*row) break;
        messages.push_back({
            .chat = chat,
            .message_id = statement->column_int(0),
            .sender_id = statement->column_int(1),
            .sender_name = statement->column_text(2),
            .time = statement->column_int(3),
            .segments_json = statement->column_text(4),
            .text = statement->column_text(5),
            .from_self = statement->column_int(6) != 0,
            .row_id = statement->column_int(7),
        });
    }
    return messages;
}

Result<void> Store::add_chunk(const MemoryChunk &chunk) {
    auto statement = database.prepare("INSERT INTO memory_chunks (chat_type, chat_id, first_row, last_row, start_time, end_time, text, "
                                      "search_text) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, chunk.chat.type);
    statement->bind(2, chunk.chat.id);
    statement->bind(3, chunk.first_row);
    statement->bind(4, chunk.last_row);
    statement->bind(5, chunk.start_time);
    statement->bind(6, chunk.end_time);
    statement->bind(7, chunk.text);
    statement->bind(8, chunk.search_text);
    return statement->run();
}

Result<i64> Store::last_chunk_row(const ChatKey &chat) {
    auto statement = database.prepare("SELECT COALESCE(MAX(last_row), 0) FROM memory_chunks WHERE chat_type = ?1 AND chat_id = ?2");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, chat.type);
    statement->bind(2, chat.id);
    auto row = statement->step();
    if (row.has_error()) return lighter::outcome_error(std::move(row).error());
    return *row ? statement->column_int(0) : 0;
}

Result<std::vector<MemoryChunk>> Store::chunks(const ChatKey &chat) {
    auto statement = database.prepare("SELECT id, first_row, last_row, start_time, end_time, text, search_text FROM memory_chunks "
                                      "WHERE chat_type = ?1 AND chat_id = ?2 ORDER BY last_row");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, chat.type);
    statement->bind(2, chat.id);
    std::vector<MemoryChunk> chunks;
    for (;;) {
        auto row = statement->step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        if (!*row) break;
        chunks.push_back({
            .id = statement->column_int(0),
            .chat = chat,
            .first_row = statement->column_int(1),
            .last_row = statement->column_int(2),
            .start_time = statement->column_int(3),
            .end_time = statement->column_int(4),
            .text = statement->column_text(5),
            .search_text = statement->column_text(6),
        });
    }
    return chunks;
}

Result<void> Store::add_summary(const ChatSummary &summary) {
    auto statement = database.prepare("INSERT INTO chat_summaries (chat_type, chat_id, first_row, last_row, start_time, end_time, summary, "
                                      "cues) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, summary.chat.type);
    statement->bind(2, summary.chat.id);
    statement->bind(3, summary.first_row);
    statement->bind(4, summary.last_row);
    statement->bind(5, summary.start_time);
    statement->bind(6, summary.end_time);
    statement->bind(7, summary.summary);
    statement->bind(8, summary.cues);
    return statement->run();
}

Result<i64> Store::last_summary_row(const ChatKey &chat) {
    auto statement = database.prepare("SELECT COALESCE(MAX(last_row), 0) FROM chat_summaries WHERE chat_type = ?1 AND chat_id = ?2");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, chat.type);
    statement->bind(2, chat.id);
    auto row = statement->step();
    if (row.has_error()) return lighter::outcome_error(std::move(row).error());
    return *row ? statement->column_int(0) : 0;
}

Result<std::vector<ChatSummary>> Store::summaries(const ChatKey &chat, usize limit) {
    auto statement = database.prepare("SELECT * FROM (SELECT id, first_row, last_row, start_time, end_time, summary, cues "
                                      "FROM chat_summaries "
                                      "WHERE chat_type = ?1 AND chat_id = ?2 ORDER BY last_row DESC LIMIT ?3) ORDER BY last_row");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, chat.type);
    statement->bind(2, chat.id);
    statement->bind(3, limit == 0 ? i64{-1} : static_cast<i64>(limit));
    std::vector<ChatSummary> summaries;
    for (;;) {
        auto row = statement->step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        if (!*row) break;
        summaries.push_back({
            .id = statement->column_int(0),
            .chat = chat,
            .first_row = statement->column_int(1),
            .last_row = statement->column_int(2),
            .start_time = statement->column_int(3),
            .end_time = statement->column_int(4),
            .summary = statement->column_text(5),
            .cues = statement->column_text(6),
        });
    }
    return summaries;
}

namespace {

constexpr std::string_view k_sticker_columns = "id, sha1, path, media_type, bytes, description, tags, status, seen, used, first_seen, "
                                               "last_seen, last_used, modified_by, usage, usage_count, "
                                               "usage_inferred_count, tags_by, reward";

Sticker read_sticker(const Statement &statement) {
    return {
        .id = statement.column_int(0),
        .sha1 = statement.column_text(1),
        .path = statement.column_text(2),
        .media_type = statement.column_text(3),
        .bytes = statement.column_int(4),
        .description = statement.column_text(5),
        .tags = statement.column_text(6),
        .status = statement.column_text(7),
        .seen = statement.column_int(8),
        .used = statement.column_int(9),
        .first_seen = statement.column_int(10),
        .last_seen = statement.column_int(11),
        .last_used = statement.column_int(12),
        .modified_by = statement.column_text(13),
        .usage = statement.column_text(14),
        .usage_count = statement.column_int(15),
        .usage_inferred_count = statement.column_int(16),
        .tags_by = statement.column_text(17),
        .reward = statement.column_double(18),
    };
}

} // namespace

Result<std::optional<Sticker>> Store::sticker_by_hash(std::string_view sha1) {
    auto statement = database.prepare(std::format("SELECT {} FROM stickers WHERE sha1 = ?1", k_sticker_columns));
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, sha1);
    auto row = statement->step();
    if (row.has_error()) return lighter::outcome_error(std::move(row).error());
    if (!*row) return std::optional<Sticker>{};
    return std::optional<Sticker>{read_sticker(*statement)};
}

Result<std::optional<Sticker>> Store::sticker_by_id(i64 id) {
    auto statement = database.prepare(std::format("SELECT {} FROM stickers WHERE id = ?1", k_sticker_columns));
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, id);
    auto row = statement->step();
    if (row.has_error()) return lighter::outcome_error(std::move(row).error());
    if (!*row) return std::optional<Sticker>{};
    return std::optional<Sticker>{read_sticker(*statement)};
}

Result<i64> Store::add_sticker(const Sticker &sticker) {
    auto statement =
        database.prepare("INSERT INTO stickers (sha1, path, media_type, bytes, description, tags, status, seen, used, "
                         "first_seen, last_seen, last_used, modified_by) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, sticker.sha1);
    statement->bind(2, sticker.path);
    statement->bind(3, sticker.media_type);
    statement->bind(4, sticker.bytes);
    statement->bind(5, sticker.description);
    statement->bind(6, sticker.tags);
    statement->bind(7, sticker.status);
    statement->bind(8, sticker.seen);
    statement->bind(9, sticker.used);
    statement->bind(10, sticker.first_seen);
    statement->bind(11, sticker.last_seen);
    statement->bind(12, sticker.last_used);
    statement->bind(13, sticker.modified_by);
    if (auto ran = statement->run(); ran.has_error()) return lighter::outcome_error(std::move(ran).error());
    return sqlite3_last_insert_rowid(database.value);
}

Result<void> Store::sticker_seen(i64 id, i64 time) {
    auto statement = database.prepare("UPDATE stickers SET seen = seen + 1, last_seen = ?2 WHERE id = ?1");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, id);
    statement->bind(2, time);
    return statement->run();
}

Result<void> Store::describe_sticker(i64 id, std::string_view description, std::string_view tags, std::string_view status) {
    // A person's edit outlasts machine tagging; the status still moves.
    auto statement = database.prepare("UPDATE stickers SET description = iif(modified_by = 'USER', description, ?2), "
                                      "tags = iif(modified_by = 'USER', tags, ?3), status = ?4 WHERE id = ?1");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, id);
    statement->bind(2, description);
    statement->bind(3, tags);
    statement->bind(4, status);
    if (auto ran = statement->run(); ran.has_error()) return ran;
    // A rejected sticker is never sent: its usage snippets go too.
    if (status != "rejected") return {};
    auto evidence = database.prepare("DELETE FROM sticker_evidence WHERE sticker_id = ?1");
    if (evidence.has_error()) return lighter::outcome_error(std::move(evidence).error());
    evidence->bind(1, id);
    return evidence->run();
}

Result<void> Store::edit_sticker(i64 id, std::string_view tags, std::string_view description) {
    auto statement = database.prepare("UPDATE stickers SET tags = ?2, description = ?3, modified_by = 'USER' WHERE id = ?1");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, id);
    statement->bind(2, tags);
    statement->bind(3, description);
    return statement->run();
}

Result<Sticker> Store::add_sticker_evidence(i64 id, const ChatKey &chat, std::string_view snippet, bool own, i64 time, usize keep) {
    auto statement = database.prepare("INSERT INTO sticker_evidence (sticker_id, chat_type, chat_id, snippet, own, time) "
                                      "VALUES (?1, ?2, ?3, ?4, ?5, ?6)");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, id);
    statement->bind(2, chat.type);
    statement->bind(3, chat.id);
    statement->bind(4, snippet);
    statement->bind(5, i64{own ? 1 : 0});
    statement->bind(6, time);
    if (auto ran = statement->run(); ran.has_error()) return lighter::outcome_error(std::move(ran).error());
    auto pruned = database.prepare("DELETE FROM sticker_evidence WHERE sticker_id = ?1 AND id NOT IN "
                                   "(SELECT id FROM sticker_evidence WHERE sticker_id = ?1 ORDER BY time DESC, id DESC LIMIT ?2)");
    if (pruned.has_error()) return lighter::outcome_error(std::move(pruned).error());
    pruned->bind(1, id);
    pruned->bind(2, static_cast<i64>(std::max<usize>(keep, 1)));
    if (auto ran = pruned->run(); ran.has_error()) return lighter::outcome_error(std::move(ran).error());
    auto counted = database.prepare("UPDATE stickers SET usage_count = usage_count + 1 WHERE id = ?1");
    if (counted.has_error()) return lighter::outcome_error(std::move(counted).error());
    counted->bind(1, id);
    if (auto ran = counted->run(); ran.has_error()) return lighter::outcome_error(std::move(ran).error());
    auto sticker = sticker_by_id(id);
    if (sticker.has_error()) return lighter::outcome_error(std::move(sticker).error());
    if (!*sticker) return lighter::outcome_error(Error::storage("no sticker " + std::to_string(id)));
    return **sticker;
}

Result<std::vector<StickerEvidence>> Store::sticker_evidence(i64 id, usize limit) {
    auto statement = database.prepare("SELECT id, sticker_id, chat_type, chat_id, snippet, own, time FROM (SELECT * FROM sticker_evidence "
                                      "WHERE sticker_id = ?1 ORDER BY time DESC, id DESC LIMIT ?2) ORDER BY time, id");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, id);
    statement->bind(2, static_cast<i64>(limit));
    std::vector<StickerEvidence> out;
    for (;;) {
        auto row = statement->step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        if (!*row) break;
        out.push_back({.id = statement->column_int(0),
                       .sticker_id = statement->column_int(1),
                       .chat = {.type = statement->column_text(2), .id = statement->column_int(3)},
                       .snippet = statement->column_text(4),
                       .own = statement->column_int(5) != 0,
                       .time = statement->column_int(6)});
    }
    return out;
}

Result<void> Store::set_sticker_usage(i64 id, std::optional<std::string_view> usage, std::optional<std::string_view> tags, i64 at_count) {
    auto statement = database.prepare("UPDATE stickers SET usage = coalesce(?2, usage), usage_inferred_count = ?4, "
                                      "tags = iif(?3 IS NULL OR modified_by = 'USER', tags, ?3), "
                                      "tags_by = iif(?3 IS NULL OR modified_by = 'USER', tags_by, 'USAGE') WHERE id = ?1");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, id);
    if (usage) statement->bind(2, *usage);
    if (tags) statement->bind(3, *tags);
    statement->bind(4, at_count);
    return statement->run();
}

Result<i64> Store::delete_sticker_evidence_of_group(i64 chat_id) {
    auto statement = database.prepare("DELETE FROM sticker_evidence WHERE chat_type = 'group' AND chat_id = ?1");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, chat_id);
    if (auto ran = statement->run(); ran.has_error()) return lighter::outcome_error(std::move(ran).error());
    return static_cast<i64>(sqlite3_changes(database.value));
}

Result<void> Store::set_sticker_status(i64 id, std::string_view status) {
    auto statement = database.prepare("UPDATE stickers SET status = ?2 WHERE id = ?1");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, id);
    statement->bind(2, status);
    return statement->run();
}

Result<void> Store::sticker_used(i64 id, i64 time) {
    auto statement = database.prepare("UPDATE stickers SET used = used + 1, last_used = ?2 WHERE id = ?1");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, id);
    statement->bind(2, time);
    return statement->run();
}

Result<std::vector<Sticker>> Store::stickers(std::string_view status) {
    auto statement =
        database.prepare(std::format("SELECT {} FROM stickers WHERE status = ?1 ORDER BY seen DESC, last_seen DESC", k_sticker_columns));
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, status);
    std::vector<Sticker> stickers;
    for (;;) {
        auto row = statement->step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        if (!*row) break;
        stickers.push_back(read_sticker(*statement));
    }
    return stickers;
}

Result<void> Store::add_fact(const PersonFact &fact) {
    auto statement = database.prepare("INSERT INTO person_facts (person_id, fact, quote, chat_type, chat_id, message_id, time) "
                                      "VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, fact.person_id);
    statement->bind(2, fact.fact);
    statement->bind(3, fact.quote);
    statement->bind(4, fact.chat.type);
    statement->bind(5, fact.chat.id);
    statement->bind(6, fact.message_id);
    statement->bind(7, fact.time);
    return statement->run();
}

Result<std::optional<LearnedSpan>> Store::learned_span(i64 group_id) {
    auto statement = database.prepare("SELECT first_time, last_time FROM learned_spans WHERE group_id = ?1");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, group_id);
    auto row = statement->step();
    if (row.has_error()) return lighter::outcome_error(std::move(row).error());
    if (!*row) return std::optional<LearnedSpan>{};
    return std::optional<LearnedSpan>(LearnedSpan{.first_time = statement->column_int(0), .last_time = statement->column_int(1)});
}

Result<void> Store::extend_learned_span(i64 group_id, i64 first_time, i64 last_time) {
    auto statement = database.prepare("INSERT INTO learned_spans (group_id, first_time, last_time) VALUES (?1, ?2, ?3) "
                                      "ON CONFLICT (group_id) DO UPDATE SET first_time = min(first_time, excluded.first_time), "
                                      "last_time = max(last_time, excluded.last_time)");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, group_id);
    statement->bind(2, first_time);
    statement->bind(3, last_time);
    return statement->run();
}

Result<void> Store::delete_fact(i64 id) {
    auto statement = database.prepare("DELETE FROM person_facts WHERE id = ?1");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, id);
    return statement->run();
}

namespace {

Result<std::vector<PersonFact>> read_facts(Statement &statement) {
    std::vector<PersonFact> facts;
    for (;;) {
        auto row = statement.step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        if (!*row) break;
        facts.push_back({
            .id = statement.column_int(0),
            .person_id = statement.column_int(7),
            .fact = statement.column_text(1),
            .quote = statement.column_text(2),
            .chat = {.type = statement.column_text(3), .id = statement.column_int(4)},
            .message_id = statement.column_int(5),
            .time = statement.column_int(6),
        });
    }
    return facts;
}

} // namespace

Result<std::vector<PersonFact>> Store::facts(i64 person_id, usize limit) {
    auto statement = database.prepare("SELECT id, fact, quote, chat_type, chat_id, message_id, time, person_id FROM person_facts "
                                      "WHERE person_id = ?1 ORDER BY time DESC, id DESC LIMIT ?2");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, person_id);
    statement->bind(2, limit == 0 ? i64{-1} : static_cast<i64>(limit));
    return read_facts(*statement);
}

Result<std::vector<PersonFact>> Store::all_facts() {
    auto statement =
        database.prepare("SELECT id, fact, quote, chat_type, chat_id, message_id, time, person_id FROM person_facts ORDER BY id");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    return read_facts(*statement);
}

Result<std::vector<std::string>> Store::search_texts() {
    auto statement = database.prepare("SELECT search_text FROM memory_chunks ORDER BY id");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    std::vector<std::string> texts;
    for (;;) {
        auto row = statement->step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        if (!*row) break;
        texts.push_back(statement->column_text(0));
    }
    return texts;
}

Result<std::vector<ChatSummary>> Store::all_summaries() {
    auto statement = database.prepare("SELECT id, first_row, last_row, start_time, end_time, summary, cues, chat_type, chat_id "
                                      "FROM chat_summaries ORDER BY id");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    std::vector<ChatSummary> summaries;
    for (;;) {
        auto row = statement->step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        if (!*row) break;
        summaries.push_back({
            .id = statement->column_int(0),
            .chat = {.type = statement->column_text(7), .id = statement->column_int(8)},
            .first_row = statement->column_int(1),
            .last_row = statement->column_int(2),
            .start_time = statement->column_int(3),
            .end_time = statement->column_int(4),
            .summary = statement->column_text(5),
            .cues = statement->column_text(6),
        });
    }
    return summaries;
}

Result<std::vector<std::pair<std::string, std::string>>> Store::embeddings(std::string_view profile,
                                                                           const std::vector<std::string> &hashes) {
    auto statement = database.prepare("SELECT vector FROM embeddings WHERE profile = ?1 AND hash = ?2");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    std::vector<std::pair<std::string, std::string>> found;
    for (const auto &hash : hashes) {
        statement->reset();
        statement->bind(1, profile);
        statement->bind(2, hash);
        auto row = statement->step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        if (*row) found.emplace_back(hash, statement->column_blob(0));
    }
    return found;
}

Result<void> Store::add_embeddings(std::string_view profile, const std::vector<std::pair<std::string, std::string>> &rows) {
    if (auto begun = database.execute("BEGIN"); begun.has_error()) return begun;
    auto statement = database.prepare("INSERT OR REPLACE INTO embeddings (profile, hash, vector) VALUES (?1, ?2, ?3)");
    if (statement.has_error()) {
        (void) database.execute("ROLLBACK");
        return lighter::outcome_error(std::move(statement).error());
    }
    for (const auto &[hash, vector] : rows) {
        statement->reset();
        statement->bind(1, profile);
        statement->bind(2, hash);
        statement->bind_blob(3, vector);
        if (auto ran = statement->run(); ran.has_error()) {
            (void) database.execute("ROLLBACK");
            return ran;
        }
    }
    return database.execute("COMMIT");
}

Result<i64> Store::delete_embeddings_except(std::string_view profile) {
    auto statement = database.prepare("DELETE FROM embeddings WHERE profile != ?1");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, profile);
    if (auto ran = statement->run(); ran.has_error()) return lighter::outcome_error(std::move(ran).error());
    return database.changes();
}

Result<std::optional<StoredMessage>> Store::message(const ChatKey &chat, i64 message_id) {
    auto statement = database.prepare("SELECT message_id, sender_id, sender_name, time, segments, text, from_self, id FROM messages "
                                      "WHERE chat_type = ?1 AND chat_id = ?2 AND message_id = ?3");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, chat.type);
    statement->bind(2, chat.id);
    statement->bind(3, message_id);
    auto row = statement->step();
    if (row.has_error()) return lighter::outcome_error(std::move(row).error());
    if (!*row) return std::optional<StoredMessage>{};
    return std::optional<StoredMessage>(StoredMessage{
        .chat = chat,
        .message_id = statement->column_int(0),
        .sender_id = statement->column_int(1),
        .sender_name = statement->column_text(2),
        .time = statement->column_int(3),
        .segments_json = statement->column_text(4),
        .text = statement->column_text(5),
        .from_self = statement->column_int(6) != 0,
        .row_id = statement->column_int(7),
    });
}

Result<std::string> Store::sender_name(const ChatKey &chat, i64 user_id) {
    auto statement = database.prepare("SELECT sender_name FROM messages WHERE chat_type = ?1 AND chat_id = ?2 AND sender_id = ?3 "
                                      "ORDER BY id DESC LIMIT 1");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, chat.type);
    statement->bind(2, chat.id);
    statement->bind(3, user_id);
    auto row = statement->step();
    if (row.has_error()) return lighter::outcome_error(std::move(row).error());
    return *row ? statement->column_text(0) : std::string();
}

Result<i64> Store::add_planner_turn(const PlannerTurn &turn) {
    auto statement = database.prepare(
        "INSERT INTO planner_turns (time, chat_type, chat_id, cycle, round, first_new_row, shown_row, tool, argument, target_message_id, "
        "to_her, sent_message_ids, analysis, reason, hint) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, turn.time);
    statement->bind(2, turn.chat.type);
    statement->bind(3, turn.chat.id);
    statement->bind(4, static_cast<i64>(turn.cycle));
    statement->bind(5, turn.round);
    statement->bind(6, turn.first_new_row);
    statement->bind(7, turn.shown_row);
    statement->bind(8, turn.tool);
    statement->bind(9, turn.argument);
    statement->bind(10, turn.target_message_id);
    statement->bind(11, turn.to_her);
    statement->bind(12, turn.sent_message_ids);
    statement->bind(13, turn.analysis);
    statement->bind(14, turn.reason);
    statement->bind(15, turn.hint);
    if (auto ran = statement->run(); ran.has_error()) return lighter::outcome_error(std::move(ran).error());
    return sqlite3_last_insert_rowid(database.value);
}

Result<void> Store::set_planner_turn_sent(i64 id, std::string_view message_ids) {
    auto statement = database.prepare("UPDATE planner_turns SET sent_message_ids = ?2 WHERE id = ?1");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, id);
    statement->bind(2, message_ids);
    return statement->run();
}

Result<std::vector<PlannerTurn>> Store::planner_turns(const ChatKey &chat, i64 since) {
    auto statement = database.prepare("SELECT id, time, cycle, round, first_new_row, shown_row, tool, argument, target_message_id, to_her, "
                                      "sent_message_ids, analysis, reason, hint FROM planner_turns WHERE chat_type = ?1 AND chat_id = ?2 "
                                      "AND time >= ?3 ORDER BY id");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, chat.type);
    statement->bind(2, chat.id);
    statement->bind(3, since);
    std::vector<PlannerTurn> out;
    for (;;) {
        auto row = statement->step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        if (!*row) break;
        out.push_back({.id = statement->column_int(0),
                       .time = statement->column_int(1),
                       .chat = chat,
                       .cycle = static_cast<u64>(statement->column_int(2)),
                       .round = statement->column_int(3),
                       .first_new_row = statement->column_int(4),
                       .shown_row = statement->column_int(5),
                       .tool = statement->column_text(6),
                       .argument = statement->column_text(7),
                       .target_message_id = statement->column_int(8),
                       .to_her = statement->column_text(9),
                       .sent_message_ids = statement->column_text(10),
                       .analysis = statement->column_text(11),
                       .reason = statement->column_text(12),
                       .hint = statement->column_text(13)});
    }
    return out;
}

Result<i64> Store::prune_planner_turns(i64 before) {
    auto statement = database.prepare("DELETE FROM planner_turns WHERE time < ?1");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, before);
    if (auto ran = statement->run(); ran.has_error()) return lighter::outcome_error(std::move(ran).error());
    return static_cast<i64>(sqlite3_changes(database.value));
}

Result<std::vector<StoredMessage>> Store::messages_before(const ChatKey &chat, i64 before_row, usize limit) {
    auto statement = database.prepare("SELECT message_id, sender_id, sender_name, time, segments, text, from_self, id FROM (SELECT * FROM "
                                      "messages WHERE chat_type = ?1 AND chat_id = ?2 AND id < ?3 ORDER BY id DESC LIMIT ?4) ORDER BY id");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, chat.type);
    statement->bind(2, chat.id);
    statement->bind(3, before_row);
    statement->bind(4, static_cast<i64>(limit));
    std::vector<StoredMessage> messages;
    for (;;) {
        auto row = statement->step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        if (!*row) break;
        messages.push_back({
            .chat = chat,
            .message_id = statement->column_int(0),
            .sender_id = statement->column_int(1),
            .sender_name = statement->column_text(2),
            .time = statement->column_int(3),
            .segments_json = statement->column_text(4),
            .text = statement->column_text(5),
            .from_self = statement->column_int(6) != 0,
            .row_id = statement->column_int(7),
        });
    }
    return messages;
}

Result<i64> Store::feedback_progress(const ChatKey &chat) {
    auto statement = database.prepare("SELECT through_row FROM feedback_progress WHERE chat_type = ?1 AND chat_id = ?2");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, chat.type);
    statement->bind(2, chat.id);
    auto row = statement->step();
    if (row.has_error()) return lighter::outcome_error(std::move(row).error());
    return *row ? statement->column_int(0) : i64{0};
}

Result<void> Store::set_feedback_progress(const ChatKey &chat, i64 through_row) {
    auto statement = database.prepare("INSERT INTO feedback_progress (chat_type, chat_id, through_row) VALUES (?1, ?2, ?3) "
                                      "ON CONFLICT (chat_type, chat_id) DO UPDATE SET through_row = excluded.through_row");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, chat.type);
    statement->bind(2, chat.id);
    statement->bind(3, through_row);
    return statement->run();
}

Result<void> Store::add_addressee_label(const AddresseeLabel &label) {
    // A person's label replaces the judge's; the judge never replaces anything.
    auto statement = database.prepare(
        "INSERT INTO addressee_labels (chat_type, chat_id, message_id, row_id, time, feature_version, features, label, source, prior) "
        "VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10) ON CONFLICT (chat_type, chat_id, message_id) DO UPDATE SET "
        "label = excluded.label, source = excluded.source WHERE excluded.source = 'manual'");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, label.chat.type);
    statement->bind(2, label.chat.id);
    statement->bind(3, label.message_id);
    statement->bind(4, label.row_id);
    statement->bind(5, label.time);
    statement->bind(6, label.feature_version);
    statement->bind(7, label.features);
    statement->bind(8, i64{label.label ? 1 : 0});
    statement->bind(9, label.source);
    statement->bind_double(10, label.prior);
    return statement->run();
}

Result<std::vector<AddresseeLabel>> Store::addressee_labels(i64 feature_version, std::string_view source) {
    auto statement =
        database.prepare("SELECT chat_type, chat_id, message_id, row_id, time, feature_version, features, label, source, prior "
                         "FROM addressee_labels WHERE feature_version = ?1 AND (?2 = '' OR source = ?2) ORDER BY time, id");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, feature_version);
    statement->bind(2, source);
    std::vector<AddresseeLabel> out;
    for (;;) {
        auto row = statement->step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        if (!*row) break;
        out.push_back({.chat = {.type = statement->column_text(0), .id = statement->column_int(1)},
                       .message_id = statement->column_int(2),
                       .row_id = statement->column_int(3),
                       .time = statement->column_int(4),
                       .feature_version = statement->column_int(5),
                       .features = statement->column_text(6),
                       .label = statement->column_int(7) != 0,
                       .source = statement->column_text(8),
                       .prior = statement->column_double(9)});
    }
    return out;
}

Result<void> Store::add_reply_effect(const ReplyEffect &effect) {
    auto statement = database.prepare(
        "INSERT OR REPLACE INTO reply_effects (chat_type, chat_id, her_message_id, target_message_id, kind, sticker_id, to_her, for_her, "
        "answered_by_other, stance, contribution, reward, time) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, effect.chat.type);
    statement->bind(2, effect.chat.id);
    statement->bind(3, effect.her_message_id);
    statement->bind(4, effect.target_message_id);
    statement->bind(5, effect.kind);
    statement->bind(6, effect.sticker_id);
    statement->bind(7, effect.to_her);
    statement->bind(8, effect.for_her);
    statement->bind(9, effect.answered_by_other);
    statement->bind(10, effect.stance);
    statement->bind(11, effect.contribution);
    statement->bind_double(12, effect.reward);
    statement->bind(13, effect.time);
    return statement->run();
}

Result<std::vector<ReplyEffect>> Store::reply_effects(const ChatKey &chat, i64 since) {
    auto statement = database.prepare("SELECT her_message_id, target_message_id, kind, sticker_id, to_her, for_her, answered_by_other, "
                                      "stance, contribution, reward, time FROM reply_effects WHERE chat_type = ?1 AND chat_id = ?2 AND "
                                      "time >= ?3 ORDER BY time, id");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, chat.type);
    statement->bind(2, chat.id);
    statement->bind(3, since);
    std::vector<ReplyEffect> out;
    for (;;) {
        auto row = statement->step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        if (!*row) break;
        out.push_back({.chat = chat,
                       .her_message_id = statement->column_int(0),
                       .target_message_id = statement->column_int(1),
                       .kind = statement->column_text(2),
                       .sticker_id = statement->column_int(3),
                       .to_her = statement->column_text(4),
                       .for_her = statement->column_text(5),
                       .answered_by_other = statement->column_text(6),
                       .stance = statement->column_text(7),
                       .contribution = statement->column_text(8),
                       .reward = statement->column_double(9),
                       .time = statement->column_int(10)});
    }
    return out;
}

Result<void> Store::save_addressee_model(const AddresseeModel &model) {
    auto statement = database.prepare(
        "INSERT OR REPLACE INTO addressee_models (chat_type, chat_id, feature_version, weights, rows, positives, log_loss, fitted_at) "
        "VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, model.chat.type);
    statement->bind(2, model.chat.id);
    statement->bind(3, model.feature_version);
    statement->bind(4, model.weights);
    statement->bind(5, model.rows);
    statement->bind(6, model.positives);
    statement->bind_double(7, model.log_loss);
    statement->bind(8, model.fitted_at);
    return statement->run();
}

Result<std::vector<AddresseeModel>> Store::addressee_models() {
    auto statement =
        database.prepare("SELECT chat_type, chat_id, feature_version, weights, rows, positives, log_loss, fitted_at FROM addressee_models");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    std::vector<AddresseeModel> out;
    for (;;) {
        auto row = statement->step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        if (!*row) break;
        out.push_back({.chat = {.type = statement->column_text(0), .id = statement->column_int(1)},
                       .feature_version = statement->column_int(2),
                       .weights = statement->column_text(3),
                       .rows = statement->column_int(4),
                       .positives = statement->column_int(5),
                       .log_loss = statement->column_double(6),
                       .fitted_at = statement->column_int(7)});
    }
    return out;
}

Result<void> Store::add_sticker_reward(i64 id, double reward) {
    auto statement = database.prepare("UPDATE stickers SET reward = reward + ?2 WHERE id = ?1");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, id);
    statement->bind_double(2, reward);
    return statement->run();
}

Result<void> Store::add_llm_call(const LlmCall &call) {
    auto statement = database.prepare("INSERT INTO llm_calls (time, chat_type, chat_id, model, input_tokens, output_tokens, "
                                      "cache_read_tokens, cache_write_tokens, stop, request_id, error) "
                                      "VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, call.time);
    statement->bind(2, call.chat.type);
    statement->bind(3, call.chat.id);
    statement->bind(4, call.model);
    statement->bind(5, static_cast<i64>(call.input_tokens));
    statement->bind(6, static_cast<i64>(call.output_tokens));
    statement->bind(7, static_cast<i64>(call.cache_read_tokens));
    statement->bind(8, static_cast<i64>(call.cache_write_tokens));
    statement->bind(9, call.stop);
    statement->bind(10, call.request_id);
    statement->bind(11, call.error);
    return statement->run();
}

Result<u64> Store::tokens_since(i64 since, TokenScope scope) {
    const std::string_view filter = scope == TokenScope::CHATS      ? " AND chat_type IN ('group', 'private')" :
                                    scope == TokenScope::BACKGROUND ? " AND chat_type NOT IN ('group', 'private')" :
                                                                      "";
    auto statement =
        database.prepare(std::format("SELECT COALESCE(SUM(input_tokens + output_tokens), 0) FROM llm_calls WHERE time >= ?1{}", filter));
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, since);
    auto row = statement->step();
    if (row.has_error()) return lighter::outcome_error(std::move(row).error());
    return static_cast<u64>(*row ? statement->column_int(0) : 0);
}

Result<void> Store::add_media(const MediaRecord &media) {
    // A retry after a failure replaces the earlier row for the same segment.
    auto statement = database.prepare("INSERT OR REPLACE INTO media (chat_type, chat_id, message_id, segment_index, kind, status, sha1, "
                                      "path, bytes, name, source, error, time, description) "
                                      "VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, media.chat.type);
    statement->bind(2, media.chat.id);
    statement->bind(3, media.message_id);
    statement->bind(4, media.segment_index);
    statement->bind(5, media.kind);
    statement->bind(6, media.status);
    statement->bind(7, media.sha1);
    statement->bind(8, media.path);
    statement->bind(9, media.bytes);
    statement->bind(10, media.name);
    statement->bind(11, media.source);
    statement->bind(12, media.error);
    statement->bind(13, media.time);
    statement->bind(14, media.description);
    return statement->run();
}

Result<std::string> Store::media_description(std::string_view sha1) {
    auto statement = database.prepare("SELECT description FROM media WHERE sha1 = ?1 AND description != '' LIMIT 1");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, sha1);
    auto row = statement->step();
    if (row.has_error()) return lighter::outcome_error(std::move(row).error());
    return *row ? statement->column_text(0) : std::string();
}

Result<void> Store::update_message_text(const ChatKey &chat, i64 message_id, std::string_view text) {
    auto statement = database.prepare("UPDATE messages SET text = ?4 WHERE chat_type = ?1 AND chat_id = ?2 AND message_id = ?3");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, chat.type);
    statement->bind(2, chat.id);
    statement->bind(3, message_id);
    statement->bind(4, text);
    return statement->run();
}

namespace {

/// SQLite treats a negative LIMIT as no limit.
i64 sql_limit(usize limit) { return limit == 0 ? -1 : static_cast<i64>(limit); }

constexpr std::string_view k_expression_columns =
    "id, chat_type, chat_id, situation, style, situations, count, first_seen, last_active, checked, modified_by";

Result<std::vector<Expression>> read_expressions(Statement &statement) {
    std::vector<Expression> rows;
    for (;;) {
        auto row = statement.step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        if (!*row) break;
        rows.push_back({
            .id = statement.column_int(0),
            .chat = {.type = statement.column_text(1), .id = statement.column_int(2)},
            .situation = statement.column_text(3),
            .style = statement.column_text(4),
            .situations = statement.column_text(5),
            .count = statement.column_int(6),
            .first_seen = statement.column_int(7),
            .last_active = statement.column_int(8),
            .checked = statement.column_int(9) != 0,
            .modified_by = statement.column_text(10),
        });
    }
    return rows;
}

} // namespace

Result<i64> Store::learn_expression(const ChatKey &chat, std::string_view situation, std::string_view style, i64 now) {
    auto statement =
        database.prepare("INSERT INTO expressions (chat_type, chat_id, situation, style, situations, count, first_seen, last_active) "
                         "VALUES (?1, ?2, ?3, ?4, ?3, 1, ?5, ?5) "
                         "ON CONFLICT (chat_type, chat_id, style) DO UPDATE SET situation = ?3, count = count + 1, last_active = ?5, "
                         "situations = CASE WHEN instr(char(10) || situations || char(10), char(10) || ?3 || char(10)) > 0 THEN situations "
                         "ELSE situations || char(10) || ?3 END "
                         "RETURNING id");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, chat.type);
    statement->bind(2, chat.id);
    statement->bind(3, situation);
    statement->bind(4, style);
    statement->bind(5, now);
    auto row = statement->step();
    if (row.has_error()) return lighter::outcome_error(std::move(row).error());
    const i64 id = *row ? statement->column_int(0) : 0;
    while (true) { // drain RETURNING so the statement completes
        auto more = statement->step();
        if (more.has_error() || !*more) break;
    }
    return id;
}

Result<void> Store::touch_expressions(const std::vector<i64> &ids, i64 now) {
    for (const auto id : ids) {
        auto statement = database.prepare("UPDATE expressions SET last_active = ?2 WHERE id = ?1");
        if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
        statement->bind(1, id);
        statement->bind(2, now);
        if (auto ran = statement->run(); ran.has_error()) return ran;
    }
    return {};
}

Result<void> Store::set_expression_checked(i64 id, bool checked) {
    auto statement = database.prepare("UPDATE expressions SET checked = ?2, modified_by = 'USER' WHERE id = ?1");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, id);
    statement->bind(2, i64{checked ? 1 : 0});
    return statement->run();
}

Result<std::vector<std::string>> Store::expression_styles() {
    auto statement = database.prepare("SELECT style FROM expressions");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    std::vector<std::string> styles;
    for (;;) {
        auto row = statement->step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        if (!*row) break;
        styles.push_back(statement->column_text(0));
    }
    return styles;
}

Result<std::vector<Expression>> Store::expressions(usize limit) {
    auto statement =
        database.prepare(std::format("SELECT {} FROM expressions ORDER BY count DESC, last_active DESC LIMIT ?1", k_expression_columns));
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, sql_limit(limit));
    return read_expressions(*statement);
}

Result<void> Store::add_expression_review(const ExpressionReview &review) {
    auto statement = database.prepare("INSERT INTO expression_reviews (time, chat_type, chat_id, situation, style, verdict, reason) "
                                      "VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, review.time);
    statement->bind(2, review.chat.type);
    statement->bind(3, review.chat.id);
    statement->bind(4, review.situation);
    statement->bind(5, review.style);
    statement->bind(6, review.verdict);
    statement->bind(7, review.reason);
    return statement->run();
}

namespace {

Result<std::vector<ExpressionReview>> read_reviews(Statement &statement) {
    std::vector<ExpressionReview> reviews;
    for (;;) {
        auto row = statement.step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        if (!*row) break;
        reviews.push_back({
            .id = statement.column_int(0),
            .time = statement.column_int(1),
            .chat = {.type = statement.column_text(2), .id = statement.column_int(3)},
            .situation = statement.column_text(4),
            .style = statement.column_text(5),
            .verdict = statement.column_text(6),
            .reason = statement.column_text(7),
        });
    }
    return reviews;
}

} // namespace

Result<std::vector<ExpressionReview>> Store::expression_reviews(std::string_view verdict, usize limit) {
    auto statement = database.prepare("SELECT id, time, chat_type, chat_id, situation, style, verdict, reason FROM expression_reviews "
                                      "WHERE ?1 = '' OR verdict = ?1 ORDER BY id DESC LIMIT ?2");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, verdict);
    statement->bind(2, sql_limit(limit));
    return read_reviews(*statement);
}

Result<std::optional<ExpressionReview>> Store::expression_review(i64 id) {
    auto statement =
        database.prepare("SELECT id, time, chat_type, chat_id, situation, style, verdict, reason FROM expression_reviews WHERE id = ?1");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, id);
    auto reviews = read_reviews(*statement);
    if (reviews.has_error()) return lighter::outcome_error(std::move(reviews).error());
    if (reviews->empty()) return std::optional<ExpressionReview>{};
    return std::optional<ExpressionReview>{std::move(reviews->front())};
}

namespace {

constexpr std::string_view k_jargon_columns =
    "id, content, meaning, is_jargon, count, last_inference_count, complete, created_by, first_seen, last_seen";

Jargon read_jargon(const Statement &statement) {
    return {
        .id = statement.column_int(0),
        .content = statement.column_text(1),
        .meaning = statement.column_text(2),
        .is_jargon = statement.column_int(3),
        .count = statement.column_int(4),
        .last_inference_count = statement.column_int(5),
        .complete = statement.column_int(6) != 0,
        .created_by = statement.column_text(7),
        .first_seen = statement.column_int(8),
        .last_seen = statement.column_int(9),
    };
}

} // namespace

Result<std::vector<Jargon>> Store::jargon_terms() {
    auto statement = database.prepare(std::format("SELECT {} FROM jargon ORDER BY count DESC, last_seen DESC", k_jargon_columns));
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    std::vector<Jargon> terms;
    for (;;) {
        auto row = statement->step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        if (!*row) break;
        terms.push_back(read_jargon(*statement));
    }
    return terms;
}

Result<std::optional<Jargon>> Store::jargon_by_content(std::string_view content) {
    auto statement = database.prepare(std::format("SELECT {} FROM jargon WHERE content = ?1", k_jargon_columns));
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, content);
    auto row = statement->step();
    if (row.has_error()) return lighter::outcome_error(std::move(row).error());
    if (!*row) return std::optional<Jargon>{};
    return std::optional<Jargon>{read_jargon(*statement)};
}

Result<Jargon> Store::add_jargon_hit(std::string_view content, const ChatKey &chat, std::string_view snippet, i64 now,
                                     usize keep_evidence) {
    const auto run = [this](std::string_view sql, const auto &bind) -> Result<void> {
        auto statement = database.prepare(sql);
        if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
        bind(*statement);
        return statement->run();
    };
    auto upserted = run("INSERT INTO jargon (content, count, first_seen, last_seen) VALUES (?1, 1, ?2, ?2) "
                        "ON CONFLICT (content) DO UPDATE SET count = count + 1, last_seen = ?2",
                        [&](Statement &s) {
                            s.bind(1, content);
                            s.bind(2, now);
                        });
    if (upserted.has_error()) return lighter::outcome_error(std::move(upserted).error());
    auto term = jargon_by_content(content);
    if (term.has_error()) return lighter::outcome_error(std::move(term).error());
    if (!*term) return lighter::outcome_error(Error::storage("jargon term vanished"));
    const i64 id = (*term)->id;
    auto counted = run("INSERT INTO jargon_chats (jargon_id, chat_type, chat_id, count) VALUES (?1, ?2, ?3, 1) "
                       "ON CONFLICT (jargon_id, chat_type, chat_id) DO UPDATE SET count = count + 1",
                       [&](Statement &s) {
                           s.bind(1, id);
                           s.bind(2, chat.type);
                           s.bind(3, chat.id);
                       });
    if (counted.has_error()) return lighter::outcome_error(std::move(counted).error());
    if (!snippet.empty() && keep_evidence > 0) {
        auto added = run("INSERT INTO jargon_evidence (jargon_id, chat_type, chat_id, snippet, time) VALUES (?1, ?2, ?3, ?4, ?5)",
                         [&](Statement &s) {
                             s.bind(1, id);
                             s.bind(2, chat.type);
                             s.bind(3, chat.id);
                             s.bind(4, snippet);
                             s.bind(5, now);
                         });
        if (added.has_error()) return lighter::outcome_error(std::move(added).error());
        auto trimmed = run("DELETE FROM jargon_evidence WHERE jargon_id = ?1 AND id NOT IN "
                           "(SELECT id FROM jargon_evidence WHERE jargon_id = ?1 ORDER BY time DESC, id DESC LIMIT ?2)",
                           [&](Statement &s) {
                               s.bind(1, id);
                               s.bind(2, static_cast<i64>(keep_evidence));
                           });
        if (trimmed.has_error()) return lighter::outcome_error(std::move(trimmed).error());
    }
    return **term;
}

Result<std::vector<JargonEvidence>> Store::jargon_evidence(i64 jargon_id, usize limit) {
    auto statement = database.prepare("SELECT id, chat_type, chat_id, snippet, time FROM jargon_evidence WHERE jargon_id = ?1 "
                                      "ORDER BY time DESC, id DESC LIMIT ?2");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, jargon_id);
    statement->bind(2, sql_limit(limit));
    std::vector<JargonEvidence> evidence;
    for (;;) {
        auto row = statement->step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        if (!*row) break;
        evidence.push_back({
            .id = statement->column_int(0),
            .chat = {.type = statement->column_text(1), .id = statement->column_int(2)},
            .snippet = statement->column_text(3),
            .time = statement->column_int(4),
        });
    }
    return evidence;
}

Result<std::vector<i64>> Store::jargon_chats(i64 jargon_id) {
    auto statement = database.prepare("SELECT chat_id FROM jargon_chats WHERE jargon_id = ?1");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, jargon_id);
    std::vector<i64> chats;
    for (;;) {
        auto row = statement->step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        if (!*row) break;
        chats.push_back(statement->column_int(0));
    }
    return chats;
}

Result<void> Store::set_jargon_inference(i64 id, std::string_view meaning, i64 is_jargon, i64 at_count, bool complete) {
    auto statement = database.prepare("UPDATE jargon SET meaning = ?2, is_jargon = ?3, last_inference_count = ?4, complete = ?5 "
                                      "WHERE id = ?1 AND created_by != 'MANUAL'");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, id);
    statement->bind(2, meaning);
    statement->bind(3, is_jargon);
    statement->bind(4, at_count);
    statement->bind(5, i64{complete ? 1 : 0});
    return statement->run();
}

Result<void> Store::set_jargon_manual(std::string_view content, std::string_view meaning, i64 now) {
    auto statement =
        database.prepare("INSERT INTO jargon (content, meaning, is_jargon, complete, created_by, first_seen, last_seen) VALUES (?1, ?2, 1, "
                         "1, 'MANUAL', ?3, ?3) "
                         "ON CONFLICT (content) DO UPDATE SET meaning = ?2, is_jargon = 1, complete = 1, created_by = 'MANUAL'");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, content);
    statement->bind(2, meaning);
    statement->bind(3, now);
    return statement->run();
}

Result<void> Store::delete_jargon(std::string_view content) {
    auto term = jargon_by_content(content);
    if (term.has_error()) return lighter::outcome_error(std::move(term).error());
    if (!*term) return {};
    for (const std::string_view sql : {"DELETE FROM jargon_evidence WHERE jargon_id = ?1", "DELETE FROM jargon_chats WHERE jargon_id = ?1",
                                       "DELETE FROM jargon WHERE id = ?1"}) {
        auto statement = database.prepare(sql);
        if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
        statement->bind(1, (*term)->id);
        if (auto ran = statement->run(); ran.has_error()) return ran;
    }
    return {};
}

Result<void> Store::merge_frequent_terms(const ChatKey &chat, const std::vector<TermTally> &tallies, i64 now, usize max_terms) {
    if (tallies.empty()) return {};
    // MaiBot's order, also for which terms survive the cap.
    constexpr std::string_view k_order = "occurrence_count DESC, message_count DESC, length(term) DESC, term";
    const auto run = [this, &chat](std::string_view sql, const auto &bind) -> Result<void> {
        auto statement = database.prepare(sql);
        if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
        statement->bind(1, chat.type);
        statement->bind(2, chat.id);
        bind(*statement);
        return statement->run();
    };
    if (auto begun = database.execute("BEGIN"); begun.has_error()) return begun;
    const auto merged = [&]() -> Result<void> {
        for (const auto &tally : tallies) {
            if (tally.term.empty()) continue;
            auto added =
                run("INSERT INTO frequent_terms (chat_type, chat_id, term, occurrence_count, message_count, created_at, updated_at) "
                    "VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6) ON CONFLICT (chat_type, chat_id, term) DO UPDATE SET "
                    "occurrence_count = occurrence_count + excluded.occurrence_count, "
                    "message_count = message_count + excluded.message_count, updated_at = excluded.updated_at",
                    [&](Statement &s) {
                        s.bind(3, tally.term);
                        s.bind(4, tally.count);
                        s.bind(5, tally.message_count);
                        s.bind(6, now);
                    });
            if (added.has_error()) return added;
        }
        auto capped = run(std::format("DELETE FROM frequent_terms WHERE chat_type = ?1 AND chat_id = ?2 AND id NOT IN "
                                      "(SELECT id FROM frequent_terms WHERE chat_type = ?1 AND chat_id = ?2 ORDER BY {} LIMIT ?3)",
                                      k_order),
                          [&](Statement &s) { s.bind(3, static_cast<i64>(max_terms)); });
        if (capped.has_error()) return capped;
        return run(
            std::format("UPDATE frequent_terms SET rank = ranked.rank, "
                        "frequency = CASE WHEN ranked.occurrences > 0 THEN CAST(frequent_terms.occurrence_count AS REAL) / "
                        "ranked.occurrences ELSE 0 END, "
                        "message_frequency = CASE WHEN ranked.messages > 0 THEN CAST(frequent_terms.message_count AS REAL) / "
                        "ranked.messages ELSE 0 END "
                        "FROM (SELECT id, ROW_NUMBER() OVER (ORDER BY {}) AS rank, SUM(occurrence_count) OVER () AS occurrences, "
                        "SUM(message_count) OVER () AS messages FROM frequent_terms WHERE chat_type = ?1 AND chat_id = ?2) AS ranked "
                        "WHERE frequent_terms.id = ranked.id",
                        k_order),
            [](Statement &) {});
    }();
    if (merged.has_error()) {
        (void) database.execute("ROLLBACK");
        return merged;
    }
    return database.execute("COMMIT");
}

Result<std::vector<FrequentTerm>> Store::frequent_terms(const ChatKey &chat) {
    auto statement = database.prepare("SELECT term, occurrence_count, message_count, rank, frequency, message_frequency, created_at, "
                                      "updated_at FROM frequent_terms WHERE chat_type = ?1 AND chat_id = ?2 ORDER BY rank");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, chat.type);
    statement->bind(2, chat.id);
    std::vector<FrequentTerm> terms;
    for (;;) {
        auto row = statement->step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        if (!*row) break;
        terms.push_back({
            .term = statement->column_text(0),
            .occurrence_count = statement->column_int(1),
            .message_count = statement->column_int(2),
            .rank = statement->column_int(3),
            .frequency = statement->column_double(4),
            .message_frequency = statement->column_double(5),
            .created_at = statement->column_int(6),
            .updated_at = statement->column_int(7),
        });
    }
    return terms;
}

Result<void> Store::delete_expressions(const std::vector<i64> &ids) {
    for (const auto id : ids) {
        auto statement = database.prepare("DELETE FROM expressions WHERE id = ?1");
        if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
        statement->bind(1, id);
        if (auto ran = statement->run(); ran.has_error()) return ran;
    }
    return {};
}

Result<void> Store::prune_expressions(usize keep) {
    auto statement = database.prepare(
        "DELETE FROM expressions WHERE id NOT IN (SELECT id FROM expressions ORDER BY count DESC, last_active DESC LIMIT ?1)");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, static_cast<i64>(keep));
    return statement->run();
}

Result<std::vector<MediaRecord>> Store::message_media(const ChatKey &chat, i64 message_id) {
    auto statement =
        database.prepare("SELECT segment_index, kind, status, sha1, path, bytes, name, source, error, time, description FROM media "
                         "WHERE chat_type = ?1 AND chat_id = ?2 AND message_id = ?3 ORDER BY segment_index");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, chat.type);
    statement->bind(2, chat.id);
    statement->bind(3, message_id);

    std::vector<MediaRecord> media;
    for (;;) {
        auto row = statement->step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        if (!*row) break;
        media.push_back({
            .chat = chat,
            .message_id = message_id,
            .segment_index = statement->column_int(0),
            .kind = statement->column_text(1),
            .status = statement->column_text(2),
            .sha1 = statement->column_text(3),
            .path = statement->column_text(4),
            .bytes = statement->column_int(5),
            .name = statement->column_text(6),
            .source = statement->column_text(7),
            .error = statement->column_text(8),
            .time = statement->column_int(9),
            .description = statement->column_text(10),
        });
    }
    return media;
}

} // namespace lilia::store
