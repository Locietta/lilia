#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <lilia/config.h>
#include <lilia/store/store.h>
#include <lilia/tests/support/fake_ws_peer.h>

namespace {

using namespace lilia;
using lilia::test::require;

store::StoredMessage message(i64 id, i64 time, std::string text, bool from_self = false) {
    return {
        .chat = {.type = "group", .id = 42},
        .message_id = id,
        .sender_id = from_self ? 1 : 7,
        .sender_name = from_self ? "lilia" : "魔王",
        .time = time,
        .segments_json = "[]",
        .text = std::move(text),
        .from_self = from_self,
    };
}

void test_messages() {
    auto opened = store::Store::open_memory();
    require(opened.has_value(), opened ? std::string() : opened.error().message());
    auto &db = *opened;

    for (i64 i = 1; i <= 5; ++i) require(db.add_message(message(i, 100 + i, "m" + std::to_string(i))).has_value(), "insert");
    require(db.add_message(message(3, 999, "duplicate")).has_value(), "duplicates are ignored, not errors");
    require(db.add_message(message(-77, 200, "mine", true)).has_value(), "own message");
    store::StoredMessage other_chat = message(1, 300, "other");
    other_chat.chat.id = 43;
    require(db.add_message(other_chat).has_value(), "other chat");

    auto recent = db.recent_messages({.type = "group", .id = 42}, 3);
    require(recent.has_value() && recent->size() == 3, "limit applies");
    require((*recent)[0].text == "m4" && (*recent)[1].text == "m5" && (*recent)[2].text == "mine", "newest messages, oldest first");
    require((*recent)[2].from_self && (*recent)[2].message_id == -77, "own message fields round-trip");

    auto all = db.recent_messages({.type = "group", .id = 42}, 100);
    require(all.has_value() && all->size() == 6, "duplicate was not inserted");
    require((*all)[2].text == "m3", "the original row is kept on conflict");

    auto own = db.message({.type = "group", .id = 42}, -77);
    require(own.has_value() && *own && (*own)->from_self && (*own)->row_id > 0, "one message by its QQ id, with its row");
    require(!*db.message({.type = "group", .id = 43}, -77), "per chat");
    require(*db.sender_name({.type = "group", .id = 42}, 7) == "魔王" && db.sender_name({.type = "group", .id = 42}, 8)->empty(),
            "the name on someone's newest message");
}

void test_llm_calls() {
    auto db = store::Store::open_memory();
    require(db.has_value(), "open");
    require(db->add_llm_call({.time = 10, .chat = {.type = "group", .id = 1}, .model = "m", .input_tokens = 100, .output_tokens = 20})
                .has_value(),
            "call 1");
    require(db->add_llm_call(
                  {.time = 20, .chat = {.type = "group", .id = 1}, .model = "m", .input_tokens = 7, .output_tokens = 3, .error = "boom"})
                .has_value(),
            "call 2");
    require(*db->tokens_since(0) == 130, "all tokens");
    require(*db->tokens_since(15) == 10, "tokens since a time");
    require(*db->tokens_since(30) == 0, "no calls");
    require(
        db->add_llm_call({.time = 40, .chat = {.type = "learn", .id = 9}, .model = "m", .input_tokens = 7, .output_tokens = 1}).has_value(),
        "a learning call");
    require(*db->tokens_since(0, store::TokenScope::CHATS) == 130 && *db->tokens_since(0, store::TokenScope::BACKGROUND) == 8 &&
                *db->tokens_since(0) == 138,
            "chats and background are counted apart");

    require(db->learn_expression({.type = "group", .id = 1}, "觉得离谱", "用\"笑死\"", 1).has_value(), "an expression");
    require(db->set_expression_vector(1, "m", std::string(16, 'v')).has_value(), "with a vector");
    require(*db->expression_styles() == std::vector<std::string>{"用\"笑死\""}, "styles alone");
}

void test_file_store() {
    const auto dir = std::filesystem::temp_directory_path() / ("lilia-test-store-" + std::to_string(std::rand()));
    {
        auto db = store::Store::open(dir / "nested" / "lilia.db");
        require(db.has_value(), db ? std::string() : db.error().message());
        require(db->add_message(message(1, 1, "persisted")).has_value(), "insert");
    }
    {
        auto db = store::Store::open(dir / "nested" / "lilia.db");
        require(db.has_value(), "reopen");
        auto recent = db->recent_messages({.type = "group", .id = 42}, 10);
        require(recent.has_value() && recent->size() == 1 && (*recent)[0].text == "persisted", "data survives reopening");
    }
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

void set_env(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

void test_config() {
    auto parsed = parse_config(R"jsonc({
        // comments are allowed
        "onebot": {"port": 3002, "access_token": "$LILIA_TEST_TOKEN"},
        "model": {"api_key": "$LILIA_TEST_KEY", "model": "deepseek-v4-pro"},
        "chat": {"groups": [123456789012], "aliases": ["莉莉娅"]}
    })jsonc");
    require(parsed.has_value(), parsed ? std::string() : parsed.error().message());
    require(parsed->onebot.port == 3002 && parsed->onebot.host == "127.0.0.1", "defaults merge with given fields");
    require(parsed->chat.groups.size() == 1 && parsed->chat.groups[0] == 123456789012, "large group ids are exact");
    require(parsed->model.base_url == "https://api.deepseek.com/anthropic", "default base url");

    auto typo = parse_config(R"({"chat": {"grups": []}})");
    require(typo.has_error(), "unknown keys must be rejected");

    set_env("LILIA_TEST_KEY", "sk-test");
    auto resolved = resolve_secret("$LILIA_TEST_KEY", "model.api_key");
    require(resolved.has_value() && *resolved == "sk-test", "secret reference resolves");
    require(*resolve_secret("literal", "x") == "literal", "literal values pass through");
    auto missing = resolve_secret("$LILIA_TEST_UNSET_VARIABLE", "x");
    require(missing.has_error(), "an unset reference is an error");

    const auto dir = std::filesystem::temp_directory_path() / ("lilia-test-config-" + std::to_string(std::rand()));
    std::filesystem::create_directories(dir);
    const auto path = dir / "config.json";
    std::ofstream(path) << R"({"onebot": {"access_token": "$LILIA_TEST_TOKEN"}, "model": {"api_key": "$LILIA_TEST_KEY"},
                              "data_dir": "~/lilia-data"})";
    set_env("LILIA_TEST_TOKEN", "tok");
    auto loaded = load_config(path);
    require(loaded.has_value(), loaded ? std::string() : loaded.error().message());
    require(loaded->onebot.access_token == "tok" && loaded->model.api_key == "sk-test", "load_config resolves secrets");
    require(!loaded->data_dir.starts_with("~"), "data_dir expands ~");

#ifndef _WIN32
    std::ofstream(path) << R"({"model": {"api_key": "sk-literal"}})";
    std::filesystem::permissions(path, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write |
                                           std::filesystem::perms::group_read | std::filesystem::perms::others_read);
    require(load_config(path).has_error(), "a world-readable literal secret is refused");
    std::filesystem::permissions(path, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write);
    require(load_config(path).has_value(), "a private literal secret is accepted");
#endif

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

} // namespace

int main() {
    try {
        test_messages();
        test_llm_calls();
        test_file_store();
        test_config();
        return 0;
    } catch (const std::exception &e) {
        std::fputs(e.what(), stderr);
        std::fputc('\n', stderr);
        return 1;
    }
}
