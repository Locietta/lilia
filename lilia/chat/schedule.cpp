#include "schedule.h"

#include <algorithm>
#include <charconv>
#include <random>

#include <lilia/chat/context.h>

namespace lilia::chat {

namespace {

constexpr i32 k_busy_earliest = 9 * 60;
constexpr i32 k_busy_latest = 22 * 60;

} // namespace

std::optional<i32> parse_clock(std::string_view text) {
    const auto colon = text.find(':');
    if (colon == std::string_view::npos) return std::nullopt;
    i32 hours = 0;
    i32 minutes = 0;
    const auto h = text.substr(0, colon);
    const auto m = text.substr(colon + 1);
    if (std::from_chars(h.data(), h.data() + h.size(), hours).ec != std::errc{} ||
        std::from_chars(m.data(), m.data() + m.size(), minutes).ec != std::errc{})
        return std::nullopt;
    if (hours < 0 || hours > 23 || minutes < 0 || minutes > 59) return std::nullopt;
    return hours * 60 + minutes;
}

std::vector<std::pair<i64, i64>> busy_stretches(const ScheduleConfig &schedule, i64 now, i32 utc_offset_minutes) {
    const i64 day = local_day_start(now, utc_offset_minutes);
    // The same draw for the whole local day.
    std::mt19937 rng(static_cast<std::mt19937::result_type>(day / 86'400 * 2'654'435'761u));
    const auto low = std::min(schedule.busy_min_minutes, schedule.busy_max_minutes);
    const auto high = std::max(schedule.busy_min_minutes, schedule.busy_max_minutes);
    std::vector<std::pair<i64, i64>> stretches;
    for (u32 i = 0; i < schedule.busy_blocks; ++i) {
        const i32 start = std::uniform_int_distribution<i32>(k_busy_earliest, k_busy_latest - 1)(rng);
        const auto length = std::uniform_int_distribution<u32>(low, high)(rng);
        stretches.emplace_back(day + i64{start} * 60, day + i64{start} * 60 + i64{length} * 60);
    }
    std::ranges::sort(stretches);
    return stretches;
}

PresenceState presence_at(const ScheduleConfig &schedule, i64 now, i32 utc_offset_minutes) {
    if (!schedule.enabled) return {};
    const auto sleep_start = parse_clock(schedule.sleep_start);
    const auto sleep_end = parse_clock(schedule.sleep_end);
    const i64 day = local_day_start(now, utc_offset_minutes);
    const auto minute = static_cast<i32>((now - day) / 60);
    if (sleep_start && sleep_end && *sleep_start != *sleep_end) {
        const bool asleep =
            *sleep_start < *sleep_end ? minute >= *sleep_start && minute < *sleep_end : minute >= *sleep_start || minute < *sleep_end;
        if (asleep) {
            // Wakes at the next sleep_end: today's, or tomorrow's when bedtime came before midnight.
            i64 wake = day + i64{*sleep_end} * 60;
            if (wake <= now) wake += 86'400;
            return {.presence = Presence::ASLEEP, .until = wake};
        }
    }
    for (const auto &[start, end] : busy_stretches(schedule, now, utc_offset_minutes)) {
        if (now >= start && now < end) return {.presence = Presence::BUSY, .until = end};
    }
    return {};
}

} // namespace lilia::chat
