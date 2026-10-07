#include "sqlite.h"

#include <system_error>
#include <utility>

#include <sqlite3.h>

namespace lilia::store {

Statement::Statement(Statement &&other) noexcept : value(std::exchange(other.value, nullptr)) {}

Statement &Statement::operator=(Statement &&other) noexcept {
    if (this == &other) return *this;
    if (value) sqlite3_finalize(value);
    value = std::exchange(other.value, nullptr);
    return *this;
}

Statement::~Statement() {
    if (value) sqlite3_finalize(value);
}

void Statement::bind(int index, i64 integer) { sqlite3_bind_int64(value, index, integer); }

void Statement::bind(int index, std::string_view text) {
    sqlite3_bind_text(value, index, text.data(), static_cast<int>(text.size()), SQLITE_TRANSIENT);
}

void Statement::bind_optional(int index, const std::optional<std::string> &text) {
    if (text)
        bind(index, std::string_view(*text));
    else
        sqlite3_bind_null(value, index);
}

Result<bool> Statement::step() {
    const auto code = sqlite3_step(value);
    if (code == SQLITE_ROW) return true;
    if (code == SQLITE_DONE) return false;
    sqlite3 *database = sqlite3_db_handle(value);
    return lighter::outcome_error(Error::storage(std::string("query failed: ") + sqlite3_errmsg(database)));
}

Result<void> Statement::run() {
    for (;;) {
        auto row = step();
        if (row.has_error()) return lighter::outcome_error(std::move(row).error());
        if (!*row) return {};
    }
}

i64 Statement::column_int(int index) const { return sqlite3_column_int64(value, index); }

std::string Statement::column_text(int index) const {
    const auto *text = reinterpret_cast<const char *>(sqlite3_column_text(value, index));
    if (!text) return {};
    return std::string(text, static_cast<usize>(sqlite3_column_bytes(value, index)));
}

Database::Database(Database &&other) noexcept : value(std::exchange(other.value, nullptr)) {}

Database &Database::operator=(Database &&other) noexcept {
    if (this == &other) return *this;
    if (value) sqlite3_close_v2(value);
    value = std::exchange(other.value, nullptr);
    return *this;
}

Database::~Database() {
    if (value) sqlite3_close_v2(value);
}

namespace {

Result<Database> open_path(const std::string &path) {
    Database database;
    const auto code = sqlite3_open_v2(path.c_str(), &database.value, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_URI, nullptr);
    if (code != SQLITE_OK) return lighter::outcome_error(database.error("cannot open database " + path, code));
    sqlite3_busy_timeout(database.value, 5000);
    return database;
}

} // namespace

Result<Database> Database::open(const std::filesystem::path &path) {
    std::error_code ec;
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) return lighter::outcome_error(Error::storage("cannot create " + path.parent_path().string() + ": " + ec.message()));
    const auto utf8 = path.generic_u8string();
    return open_path(std::string(reinterpret_cast<const char *>(utf8.data()), utf8.size()));
}

Result<Database> Database::open_memory() { return open_path(":memory:"); }

Result<Database> Database::open_readonly(const std::filesystem::path &path) {
    const auto utf8 = path.generic_u8string();
    const std::string name(reinterpret_cast<const char *>(utf8.data()), utf8.size());
    Database database;
    const auto code = sqlite3_open_v2(name.c_str(), &database.value, SQLITE_OPEN_READONLY, nullptr);
    if (code != SQLITE_OK) return lighter::outcome_error(database.error("cannot open database " + name, code));
    sqlite3_busy_timeout(database.value, 5000);
    return database;
}

Result<Statement> Database::prepare(std::string_view sql) {
    Statement statement;
    const auto code = sqlite3_prepare_v2(value, sql.data(), static_cast<int>(sql.size()), &statement.value, nullptr);
    if (code != SQLITE_OK) return lighter::outcome_error(error("cannot prepare query", code));
    return statement;
}

Result<void> Database::execute(std::string_view sql) {
    const std::string owned(sql);
    char *message = nullptr;
    const auto code = sqlite3_exec(value, owned.c_str(), nullptr, nullptr, &message);
    if (code == SQLITE_OK) return {};
    auto detail = std::string("cannot update database: ") + (message ? message : sqlite3_errmsg(value));
    sqlite3_free(message);
    return lighter::outcome_error(Error::storage(std::move(detail)));
}

Error Database::error(std::string_view action, int code) const {
    return Error::storage(std::string(action) + ": " + (value ? sqlite3_errmsg(value) : sqlite3_errstr(code)));
}

} // namespace lilia::store
