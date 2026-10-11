#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <lilia/store/store.h>

namespace lilia::chat {

// When to run the planner, MaiBot's way: this decides only when it looks,
// never what it sees (every pending message reaches it at the next cycle).
// Deterministic and cheap: the planner (an LLM call) only runs when this says so.

struct FlowSettings {
    /// 0..1. The planner looks every ceil(1/talk_value) new messages (1:
    /// every batch); 0 means only when someone calls her.
    double talk_value = 1;
    /// This many pending messages bypass the idle backoff.
    usize burst_bypass = 6;
};

/// Per-chat state between planner runs.
struct Flow {
    /// Others' messages since the planner last ran, oldest first.
    std::vector<store::StoredMessage> unseen;
    /// The newest unseen message that @s her, replies to her or is private:
    /// the planner looks at once (it still decides).
    std::optional<i64> addressed;
    /// The newest unseen message that names her: the planner looks at once.
    std::optional<i64> named;
    i64 last_arrival = 0; ///< unix seconds the newest message arrived
    /// Seconds of silence before the first unseen message.
    i64 quiet_before = 0;
    /// Smoothed seconds between messages here (0 until two have arrived).
    double interval = 0;
    u32 silent_streak = 0;
    i64 backoff_until = 0;
};

/// Pending messages and either someone calling or naming her, or, outside
/// the idle backoff, ceil(1/talk_value) of them, the silence before them
/// counting too (MaiBot's credit: one virtual message per average interval,
/// at least 30 s, at most one short of the threshold).
bool should_plan(const Flow &flow, const FlowSettings &settings, i64 now);

/// When pending messages held back only by the idle backoff become due (its
/// end); nullopt when nothing waits for that.
std::optional<i64> next_check(const Flow &flow, const FlowSettings &settings, i64 now);

/// A message arrived (before it joins `unseen`): keeps the silence before the
/// first unseen one and the average interval.
void note_arrival(Flow &flow, i64 now);

/// A cycle ended without her saying anything: after two in a row the planner
/// backs off (15 s, doubling to 5 min) unless someone calls her or a burst
/// arrives.
void note_silent(Flow &flow, i64 now);

/// She said something: the backoff ends.
void note_spoke(Flow &flow);

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
