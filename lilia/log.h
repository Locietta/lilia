#pragma once

#include <cstdio>
#include <format>
#include <string_view>
#include <utility>

namespace lilia::log {

// Line-oriented stderr logging. Under systemd every line becomes one journal
// entry, so the journal supplies timestamps; the "<n>" prefix is the sd-daemon
// priority (journald strips it and records the level).

enum struct Level {
    ERROR = 3,
    WARN = 4,
    INFO = 6,
    DEBUG = 7,
};

inline Level g_threshold = Level::INFO;

inline void write(Level level, std::string_view line) {
    if (static_cast<int>(level) > static_cast<int>(g_threshold)) return;
    std::fprintf(stderr, "<%d>%.*s\n", static_cast<int>(level), static_cast<int>(line.size()), line.data());
    std::fflush(stderr);
}

template <typename... Args>
void error(std::format_string<Args...> format, Args &&...args) {
    write(Level::ERROR, std::format(format, std::forward<Args>(args)...));
}

template <typename... Args>
void warn(std::format_string<Args...> format, Args &&...args) {
    write(Level::WARN, std::format(format, std::forward<Args>(args)...));
}

template <typename... Args>
void info(std::format_string<Args...> format, Args &&...args) {
    write(Level::INFO, std::format(format, std::forward<Args>(args)...));
}

template <typename... Args>
void debug(std::format_string<Args...> format, Args &&...args) {
    write(Level::DEBUG, std::format(format, std::forward<Args>(args)...));
}

} // namespace lilia::log
