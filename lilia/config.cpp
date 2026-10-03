#include "config.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <system_error>
#include <utility>

#include <glaze/json.hpp>

#include <lighter/codec/json/json.h>

namespace lilia {

namespace {

constexpr lighter::codec::json::Opts k_config_opts{{.null_terminated = false, .comments = true}};

std::optional<std::string> environment(const char *name) {
    const char *value = std::getenv(name);
    if (!value || !*value) return std::nullopt;
    return std::string(value);
}

std::filesystem::path home_directory() {
#ifdef _WIN32
    if (auto profile = environment("USERPROFILE")) return *profile;
#endif
    if (auto home = environment("HOME")) return *home;
    return std::filesystem::current_path();
}

bool is_reference(std::string_view value) { return value.starts_with('$'); }

/// On POSIX, refuses literal secrets in a file other users can read.
Result<void> check_private(const std::filesystem::path &path) {
#ifdef _WIN32
    (void) path;
    return {};
#else
    std::error_code ec;
    const auto status = std::filesystem::status(path, ec);
    if (ec) return lighter::outcome_error(Error::config("cannot stat " + path.string() + ": " + ec.message()));
    using std::filesystem::perms;
    const auto shared = perms::group_all | perms::others_all;
    if ((status.permissions() & shared) != perms::none) {
        return lighter::outcome_error(Error::config(path.string() + " contains a literal secret but is readable by other users; "
                                                                    "chmod 600 it or use \"$VAR\" references"));
    }
    return {};
#endif
}

} // namespace

Result<std::string> resolve_secret(std::string_view value, std::string_view field) {
    if (!is_reference(value)) return std::string(value);
    const std::string name(value.substr(1));
    if (name.empty()) return lighter::outcome_error(Error::config(std::string(field) + ": empty environment variable name"));
    auto resolved = environment(name.c_str());
    if (!resolved) {
        return lighter::outcome_error(
            Error::config(std::string(field) + ": environment variable " + name + " is not set", ErrorCode::AUTH_NOT_CONFIGURED));
    }
    return *std::move(resolved);
}

Result<Config> parse_config(std::string_view text) {
    Config config;
    if (auto ec = glz::read<k_config_opts>(config, text)) {
        return lighter::outcome_error(Error::config("invalid config: " + glz::format_error(ec, text)));
    }
    return config;
}

Result<Config> load_config(const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return lighter::outcome_error(Error::config("cannot read " + path.string(), ErrorCode::NOT_FOUND));
    std::ostringstream buffer;
    buffer << file.rdbuf();

    auto config = parse_config(buffer.str());
    if (config.has_error()) return config;

    if (!config->onebot.access_token.empty() && !is_reference(config->onebot.access_token) ||
        !config->model.api_key.empty() && !is_reference(config->model.api_key)) {
        auto checked = check_private(path);
        if (checked.has_error()) return lighter::outcome_error(std::move(checked).error());
    }

    auto token = resolve_secret(config->onebot.access_token, "onebot.access_token");
    if (token.has_error()) return lighter::outcome_error(std::move(token).error());
    config->onebot.access_token = *std::move(token);

    auto key = resolve_secret(config->model.api_key, "model.api_key");
    if (key.has_error()) return lighter::outcome_error(std::move(key).error());
    config->model.api_key = *std::move(key);
    if (config->model.api_key.empty()) {
        return lighter::outcome_error(Error::config("model.api_key is required", ErrorCode::AUTH_NOT_CONFIGURED));
    }

    if (config->data_dir.empty()) {
        config->data_dir = default_data_dir().string();
    } else if (config->data_dir.starts_with("~/")) {
        config->data_dir = (home_directory() / config->data_dir.substr(2)).string();
    }
    if (config->archive.host_qq_dir.starts_with("~/")) {
        config->archive.host_qq_dir = (home_directory() / config->archive.host_qq_dir.substr(2)).string();
    }
    return config;
}

std::filesystem::path default_data_dir() {
#ifdef _WIN32
    if (auto local = environment("LOCALAPPDATA")) return std::filesystem::path(*local) / "lilia";
#endif
    if (auto xdg = environment("XDG_DATA_HOME")) return std::filesystem::path(*xdg) / "lilia";
    return home_directory() / ".local" / "share" / "lilia";
}

std::filesystem::path default_config_path() {
#ifdef _WIN32
    if (auto roaming = environment("APPDATA")) return std::filesystem::path(*roaming) / "lilia" / "config.json";
#endif
    if (auto xdg = environment("XDG_CONFIG_HOME")) return std::filesystem::path(*xdg) / "lilia" / "config.json";
    return home_directory() / ".config" / "lilia" / "config.json";
}

} // namespace lilia
