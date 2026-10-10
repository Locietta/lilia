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
    /// This many pending messages bypass the idle backoff.
    usize burst_bypass = 6;
};

/// Per-chat state between planner runs.
struct Flow {
    /// Others' messages since the planner last ran, oldest first.
    std::vector<store::StoredMessage> unseen;
    /// The newest unseen message that addresses lilia (mention, reply, name).
    std::optional<i64> addressed;
    i64 last_arrival = 0; ///< unix seconds the newest unseen message arrived
    i64 last_spoke = 0;
    /// Whom she last answered; their messages inside the window always reach
    /// the planner, backoff or not (they are probably talking to her).
    i64 partner = 0;
    u32 silent_streak = 0;
    i64 backoff_until = 0;
};

bool should_plan(const Flow &flow, const FlowSettings &settings, i64 now);

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
