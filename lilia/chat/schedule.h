#pragma once

#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <lilia/config.h>

namespace lilia::chat {

// Her day: when she is asleep or busy. Pure functions of the time, so the
// same moment always gives the same answer (busy stretches are drawn from
// the local date).

enum struct Presence {
    AWAKE,
    ASLEEP,
    BUSY,
};

struct PresenceState {
    Presence presence = Presence::AWAKE;
    /// Unix seconds when asleep or busy ends (0 when awake).
    i64 until = 0;
};

/// "HH:MM" -> minutes after midnight; nullopt when malformed.
std::optional<i32> parse_clock(std::string_view text);

/// The busy stretches of the local day containing `now`, as unix-second
/// [start, end) pairs, ordered.
std::vector<std::pair<i64, i64>> busy_stretches(const ScheduleConfig &schedule, i64 now, i32 utc_offset_minutes);

/// Asleep, busy or awake at `now` (always awake when the schedule is off).
PresenceState presence_at(const ScheduleConfig &schedule, i64 now, i32 utc_offset_minutes);

} // namespace lilia::chat
