#include "store.h"

#include <algorithm>
#include <utility>

namespace lilia::store {

namespace {

// 2: media table. 3: expressions table. 4: media.description.
constexpr int k_schema_version = 4;

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

CREATE TABLE IF NOT EXISTS expressions (
    id INTEGER PRIMARY KEY,
    situation TEXT NOT NULL,
    style TEXT NOT NULL UNIQUE,
    count INTEGER NOT NULL DEFAULT 1,
    first_seen INTEGER NOT NULL,
    last_active INTEGER NOT NULL
);
)sql";

Result<Store> initialize(Result<Database> opened, bool file_backed) {
    if (opened.has_error()) return lighter::outcome_error(std::move(opened).error());
    Store store{.database = std::move(*opened)};

    if (file_backed) {
        auto wal = store.database.execute("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;");
        if (wal.has_error()) return lighter::outcome_error(std::move(wal).error());
    }

    auto version = store.database.prepare("PRAGMA user_version");
    if (version.has_error()) return lighter::outcome_error(std::move(version).error());
    auto row = version->step();
    if (row.has_error()) return lighter::outcome_error(std::move(row).error());
    const auto current = *row ? version->column_int(0) : 0;
    if (current > k_schema_version) {
        return lighter::outcome_error(Error::storage("database schema version " + std::to_string(current) + " is newer than this build"));
    }

    auto created = store.database.execute(k_schema);
    if (created.has_error()) return lighter::outcome_error(std::move(created).error());
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
    auto statement = database.prepare("SELECT message_id, sender_id, sender_name, time, segments, text, from_self FROM messages "
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
        });
    }
    std::ranges::reverse(messages);
    return messages;
}

Result<bool> Store::is_own_message(const ChatKey &chat, i64 message_id) {
    auto statement =
        database.prepare("SELECT 1 FROM messages WHERE chat_type = ?1 AND chat_id = ?2 AND message_id = ?3 AND from_self = 1 LIMIT 1");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, chat.type);
    statement->bind(2, chat.id);
    statement->bind(3, message_id);
    return statement->step();
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

Result<u64> Store::tokens_since(i64 since) {
    auto statement = database.prepare("SELECT COALESCE(SUM(input_tokens + output_tokens), 0) FROM llm_calls WHERE time >= ?1");
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

Result<void> Store::learn_expression(std::string_view situation, std::string_view style, i64 now) {
    auto statement =
        database.prepare("INSERT INTO expressions (situation, style, count, first_seen, last_active) VALUES (?1, ?2, 1, ?3, ?3) "
                         "ON CONFLICT (style) DO UPDATE SET situation = ?1, count = count + 1, last_active = ?3");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, situation);
    statement->bind(2, style);
    statement->bind(3, now);
    return statement->run();
}

namespace {

/// SQLite treats a negative LIMIT as no limit.
i64 sql_limit(usize limit) { return limit == 0 ? -1 : static_cast<i64>(limit); }

Result<std::vector<Expression>> read_expressions(Statement &statement) {
    std::vector<Expression> rows;
    for (;;) {
        auto row = statement.step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        if (!*row) break;
        rows.push_back({
            .id = statement.column_int(0),
            .situation = statement.column_text(1),
            .style = statement.column_text(2),
            .count = statement.column_int(3),
            .last_active = statement.column_int(4),
        });
    }
    return rows;
}

} // namespace

Result<std::vector<Expression>> Store::random_expressions(usize skip_top, usize limit) {
    auto statement =
        database.prepare("SELECT id, situation, style, count, last_active FROM expressions WHERE id NOT IN "
                         "(SELECT id FROM expressions ORDER BY count DESC, last_active DESC LIMIT ?1) ORDER BY RANDOM() LIMIT ?2");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, static_cast<i64>(skip_top));
    statement->bind(2, sql_limit(limit));
    return read_expressions(*statement);
}

Result<std::vector<Expression>> Store::expressions(usize limit) {
    auto statement =
        database.prepare("SELECT id, situation, style, count, last_active FROM expressions ORDER BY count DESC, last_active DESC LIMIT ?1");
    if (statement.has_error()) return lighter::outcome_error(std::move(statement).error());
    statement->bind(1, sql_limit(limit));
    return read_expressions(*statement);
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
