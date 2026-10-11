#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <lilia/store/store.h>

namespace lilia::chat {

// When to ask the planner whether lilia speaks. Deterministic and cheap: the
// planner (an LLM call) only runs when this says so.

struct FlowSettings {
    /// 0..1. Outside a conversation the planner runs every ceil(1/talk_value)
    /// new messages; 0 means lilia only reacts when addressed.
    double talk_value = 0.3;
    /// After lilia speaks, every new message goes to the planner for this long.
    i64 window_seconds = 300;
    /// Whom she last answered counts as still talking with her for this long.
    i64 partner_seconds = 900;
    /// This many pending messages bypass the idle backoff.
    usize burst_bypass = 6;
};

/// Per-chat state between planner runs.
struct Flow {
    /// Others' messages since the planner last ran, oldest first.
    std::vector<store::StoredMessage> unseen;
    /// The newest unseen message that calls lilia (an @, a reply to her, a
    /// private message): she answers it.
    std::optional<i64> addressed;
    /// The newest unseen message that only names her: the planner looks, but
    /// it may be about her rather than to her.
    std::optional<i64> named;
    i64 last_arrival = 0; ///< unix seconds the newest message arrived
    /// Seconds of silence before the first unseen message.
    i64 quiet_before = 0;
    /// Smoothed seconds between messages here (0 until two have arrived).
    double interval = 0;
    i64 last_spoke = 0;
    /// Whom she last answered; their messages within `partner_seconds` always
    /// reach the planner, backoff or not (they are probably talking to her).
    i64 partner = 0;
    u32 silent_streak = 0;
    i64 backoff_until = 0;
};

/// MaiBot's silence credit: outside a conversation the planner looks every
/// ceil(1/talk_value) messages, but the silence before them counts too, one
/// virtual message per average interval (at least 30 s), at most one short
/// of the threshold.
bool should_plan(const Flow &flow, const FlowSettings &settings, i64 now);

/// True when one of `messages` comes from whom she last answered, within
/// `partner_seconds` of her last line: they are probably still talking to her.
bool partner_in(const Flow &flow, const FlowSettings &settings, i64 now, const std::vector<store::StoredMessage> &messages);

/// A message arrived (before it joins `unseen`): keeps the silence before the
/// first unseen one and the average interval.
void note_arrival(Flow &flow, i64 now);

/// The planner chose silence while nobody addressed lilia: back off
/// (15s, doubling to 5 min) after two such decisions in a row.
void note_silent(Flow &flow, i64 now);

void note_spoke(Flow &flow, i64 now, i64 partner = 0);

/// Text QQ users would repeat (复读): short, plain text (no media, mention or
/// command).
bool repeatable(std::string_view text);

/// A repeat chain at the end of `log` (oldest first): the newest message's
/// text and how many people in a row sent it. Nothing when it is hers, is not
/// repeatable, or she already joined this chain.
struct RepeatChain {
    std::string text;
    usize people = 0;
};
std::optional<RepeatChain> repeat_chain(const std::vector<store::StoredMessage> &log);

} // namespace lilia::chat
