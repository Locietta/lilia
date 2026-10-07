#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

#include <lighter/types.hpp>

#include <lilia/error.h>

struct sqlite3;
struct sqlite3_stmt;

namespace lilia::store {

using namespace lighter::types;

// Thin RAII layer over the SQLite C API, extracted from the file-local helpers
// in liminal's session store. All access happens on the event-loop thread.

struct Statement {
    Statement() = default;
    explicit Statement(sqlite3_stmt *value) noexcept : value(value) {}
    Statement(Statement &&other) noexcept;
    Statement &operator=(Statement &&other) noexcept;
    Statement(const Statement &) = delete;
    Statement &operator=(const Statement &) = delete;
    ~Statement();

    /// Bind indexes are 1-based, as in SQLite.
    void bind(int index, i64 value);
    void bind(int index, std::string_view value);
    void bind_optional(int index, const std::optional<std::string> &value);

    /// Advances one row. True when a row is available, false when done.
    Result<bool> step();

    /// Runs to completion, ignoring result rows.
    Result<void> run();

    i64 column_int(int index) const;
    std::string column_text(int index) const;

    sqlite3_stmt *value = nullptr;
};

struct Database {
    Database() = default;
    Database(Database &&other) noexcept;
    Database &operator=(Database &&other) noexcept;
    Database(const Database &) = delete;
    Database &operator=(const Database &) = delete;
    ~Database();

    /// Opens (creating if needed) the database file and its parent directory.
    static Result<Database> open(const std::filesystem::path &path);

    /// In-memory database, for tests.
    static Result<Database> open_memory();

    /// Existing database, read-only (safe next to a running writer in WAL mode).
    static Result<Database> open_readonly(const std::filesystem::path &path);

    Result<Statement> prepare(std::string_view sql);
    Result<void> execute(std::string_view sql);

    Error error(std::string_view action, int code) const;

    sqlite3 *value = nullptr;
};

} // namespace lilia::store
