#include "flow.h"

#include <algorithm>
#include <cmath>
#include <set>

#include <lilia/chat/delivery.h>

namespace lilia::chat {

namespace {

constexpr usize k_repeat_max_code_points = 20;
/// Silence credit: the interval never counts as less than this, and a gap
/// longer than this cap enters the average as the cap.
constexpr double k_min_interval = 30;
constexpr i64 k_max_interval_sample = 600;

} // namespace

bool repeatable(std::string_view text) {
    if (text.empty() || text.starts_with('/') || count_code_points(text) > k_repeat_max_code_points) return false;
    // Media ("[图片]"), mentions and quoted replies are not repeated as text.
    return text.find('[') == std::string_view::npos && text.find('@') == std::string_view::npos &&
           text.find('\n') == std::string_view::npos;
}

std::optional<RepeatChain> repeat_chain(const std::vector<store::StoredMessage> &log) {
    if (log.empty() || log.back().from_self || !repeatable(log.back().text)) return std::nullopt;
    const auto &text = log.back().text;
    std::set<i64> people;
    for (auto it = log.rbegin(); it != log.rend() && it->text == text; ++it) {
        if (it->from_self) return std::nullopt; // she is already in this chain
        people.insert(it->sender_id);
    }
    return RepeatChain{.text = text, .people = people.size()};
}

void note_arrival(Flow &flow, i64 now) {
    if (flow.last_arrival > 0) {
        const i64 gap = std::max<i64>(now - flow.last_arrival, 0);
        if (flow.unseen.empty()) flow.quiet_before = gap;
        // A night's silence says nothing about how fast people talk.
        const double sample = static_cast<double>(std::min(gap, k_max_interval_sample));
        flow.interval = flow.interval == 0 ? sample : 0.8 * flow.interval + 0.2 * sample;
    } else if (flow.unseen.empty()) {
        flow.quiet_before = 0;
    }
    flow.last_arrival = now;
}

bool should_plan(const Flow &flow, const FlowSettings &settings, i64 now) {
    if (flow.unseen.empty()) return false;
    if (flow.addressed || flow.named) return true;
    const bool in_window = flow.last_spoke > 0 && now - flow.last_spoke <= settings.window_seconds;
    if (in_window && flow.partner != 0 &&
        std::ranges::any_of(flow.unseen, [&](const store::StoredMessage &m) { return m.sender_id == flow.partner; })) {
        return true;
    }
    if (now < flow.backoff_until && flow.unseen.size() < settings.burst_bypass) return false;
    if (in_window) return true;
    if (settings.talk_value <= 0) return false;
    const auto threshold = static_cast<usize>(std::ceil(1.0 / std::min(settings.talk_value, 1.0)));
    const double pace = std::max(flow.interval, k_min_interval);
    const auto credit = std::min(static_cast<usize>(static_cast<double>(flow.quiet_before) / pace), threshold - 1);
    return flow.unseen.size() + credit >= threshold;
}

void note_silent(Flow &flow, i64 now) {
    flow.silent_streak += 1;
    if (flow.silent_streak < 2) return;
    const u32 doublings = std::min<u32>(flow.silent_streak - 2, 5);
    flow.backoff_until = now + std::min<i64>(i64{15} << doublings, 300);
}

void note_spoke(Flow &flow, i64 now, i64 partner) {
    flow.last_spoke = now;
    flow.partner = partner;
    flow.silent_streak = 0;
    flow.backoff_until = 0;
}

} // namespace lilia::chat
