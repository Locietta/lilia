#include <lilia/chat/detail/responder.h>

#include <chrono>
#include <format>

#include <glaze/json.hpp>

#include <lighter/async/io/watcher.h>
#include <lighter/async/runtime/timeout.h>

#include <lilia/log.h>

// Model calls (deadlines, usage records, planner fallbacks) and sending.

namespace lilia::chat {

using lighter::Task;
using namespace detail;

namespace {

/// A failing planner model is skipped (fallbacks answer) for this long.
constexpr i64 k_planner_rest_seconds = 600;

/// with_timeout takes lighter's error type; this hands the provider's
/// outcome through as a plain value instead.
Task<Result<provider::ProviderCallCompletion>> settle(lighter::Task<provider::ProviderCallCompletion, Error> call) {
    auto outcome = co_await std::move(call);
    co_return outcome;
}

/// Per-purpose deadline for one model call (including the client's retries).
std::chrono::seconds call_deadline(const Responder *r, std::string_view purpose) {
    const auto &model = r->config.model;
    return std::chrono::seconds(purpose == "search" ? model.search_timeout_seconds :
                                purpose == "writer" ? model.writer_timeout_seconds :
                                                      model.timeout_seconds);
}

/// Claude's cache entry lives 5 minutes from the start of the call that last
/// wrote or read it; a keep-alive goes out a minute before it would expire.
constexpr u32 k_warm_minutes = 4;

Task<> warm(Responder *r, store::ChatKey chat, u64 generation) {
    for (u32 left = r->config.model.cache_keepalive_minutes; left >= k_warm_minutes; left -= k_warm_minutes) {
        co_await lighter::sleep(std::chrono::minutes(k_warm_minutes));
        const auto &state = r->chats[chat];
        // A newer prompt took over, or the fallbacks answer: nothing to keep.
        if (state.warm_generation != generation || r->now() < r->planner_down_until) co_return;
        const auto prompt = state.warm_prompt;
        const auto tools = state.warm_tools;
        std::ignore = co_await call_model(r, *r->warmer, r->config.model.planner_model, "keep-warm", chat, prompt, tools);
    }
}

} // namespace

namespace detail {

std::string describe(const store::ChatKey &chat) { return chat.type + ":" + std::to_string(chat.id); }

std::string situated(std::string_view prompt, const store::ChatKey &chat, std::vector<std::pair<std::string_view, std::string_view>> values,
                     std::string_view group, std::string_view private_chat) {
    // The variant may use the same placeholders; fill_template() does one pass.
    auto situation = fill_template(chat.type == "group" ? group : private_chat, values);
    while (!situation.empty() && (situation.back() == '\n' || situation.back() == '\r')) situation.pop_back();
    values.emplace_back("situation", situation);
    return fill_template(prompt, values);
}

Task<std::optional<i64>> send_message(Responder *r, const store::ChatKey &chat, onebot::Message message) {
    if (r->outbox) co_return co_await r->outbox(chat, std::move(message));
    auto session = r->session;
    if (!session || !session->is_open()) {
        log::warn("no live OneBot session; cannot send to {}", describe(chat));
        co_return std::nullopt;
    }
    auto sent = co_await (chat.type == "group" ? session->send_group_msg(chat.id, std::move(message)) :
                                                 session->send_private_msg(chat.id, std::move(message)));
    if (sent.has_error()) {
        log::error("send to {} failed: {}", describe(chat), sent.error().message());
        co_return std::nullopt;
    }
    co_return sent->message_id;
}

Task<std::optional<i64>> send_and_record(Responder *r, const store::ChatKey &chat, onebot::Message message, std::string text) {
    auto segments_json = glz::write_json(message).value_or("[]");
    if (text.empty()) text = onebot::render_plain(message);
    auto plain = onebot::text_only(message);
    const auto id = co_await send_message(r, chat, std::move(message));
    if (!id) co_return std::nullopt;
    auto stored = r->store.add_message({
        .chat = chat,
        .message_id = *id,
        .sender_id = r->self.user_id,
        .sender_name = r->self_name(chat),
        .time = r->now(),
        .segments_json = std::move(segments_json),
        .text = std::move(text),
        .from_self = true,
    });
    if (stored.has_error()) log::error("cannot record sent message: {}", stored.error().message());
    // Her own lines join the next learning batch as [SELF] context, and the
    // lines around stickers.
    if (r->stickers && chat.type == "group") r->stickers->observe_own_text(chat.id, plain);
    if (r->learner && chat.type == "group") r->learner->observe_own(chat.id, std::move(plain));
    co_return id;
}

Task<std::optional<std::vector<provider::OutputItem>>> call_model(Responder *r, provider::Provider &provider, std::string model,
                                                                  std::string_view purpose, const store::ChatKey &chat,
                                                                  const provider::History &history,
                                                                  const std::vector<provider::ToolDefinition> &tools, Error *failure) {
    std::vector<provider::OutputItem> items;
    const provider::StreamCallbacks callbacks{
        .on_item_completed = [&items](const provider::OutputItem &item) { items.push_back(item); },
    };
    const i64 started = r->now();
    const auto clock = std::chrono::steady_clock::now();
    // A request that never finishes must not leave her silent: past the
    // deadline it fails like any other error (and the writer falls back).
    auto settled = co_await lighter::with_timeout(settle(provider->complete(history, tools, callbacks)), call_deadline(r, purpose));
    auto completion = settled.has_error() ?
                          Result<provider::ProviderCallCompletion>(lighter::outcome_error(Error::api(
                              "timeout", std::format("{} call exceeded {}s", purpose, call_deadline(r, purpose).count()), "", true))) :
                          *std::move(settled);
    const auto took = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - clock).count();

    store::LlmCall call{.time = started, .chat = chat, .model = std::move(model)};
    if (completion.has_error()) {
        call.error = completion.error().message();
        call.request_id = completion.error().request_id;
        log::error("{} call for {} failed after {} ms: {}", purpose, describe(chat), took, call.error);
    } else {
        if (!completion->model.empty()) call.model = completion->model;
        call.input_tokens = completion->usage.input_tokens;
        call.output_tokens = completion->usage.output_tokens;
        call.cache_read_tokens = completion->usage.cache_read_tokens;
        call.cache_write_tokens = completion->usage.cache_write_tokens;
        call.stop = completion->stop_detail;
        call.request_id = completion->request_id;
        log::info("{} call for {} in {} ms: {} in / {} out / {} cache read / {} cache write tokens", purpose, describe(chat), took,
                  call.input_tokens, call.output_tokens, call.cache_read_tokens, call.cache_write_tokens);
    }
    if (auto recorded = r->store.add_llm_call(call); recorded.has_error()) {
        log::error("cannot record model call: {}", recorded.error().message());
    }
    if (completion.has_error()) {
        if (failure) *failure = completion.error();
        co_return std::nullopt;
    }
    co_return std::move(items);
}

Task<std::optional<std::vector<provider::OutputItem>>> call_planner(Responder *r, std::string_view purpose, const store::ChatKey &chat,
                                                                    const provider::History &history,
                                                                    const std::vector<provider::ToolDefinition> &tools) {
    std::optional<std::vector<provider::OutputItem>> items;
    if (r->planner_fallbacks.empty() || r->now() >= r->planner_down_until) {
        Error failure;
        items = co_await call_model(r, r->planner, r->config.model.planner_model, purpose, chat, history, tools, &failure);
        if (items) {
            r->planner_failures = 0;
        } else if (!r->planner_fallbacks.empty() && (failure.status == 429 || ++r->planner_failures >= 2)) {
            r->planner_down_until = r->now() + k_planner_rest_seconds;
            r->planner_failures = 1; // after the rest, one more failure is enough
            log::warn("planner {} is resting for {} min; the fallbacks answer meanwhile", r->config.model.planner_model,
                      k_planner_rest_seconds / 60);
        }
    }
    for (auto &fallback : r->planner_fallbacks) {
        if (items) break;
        log::warn("planner fallback for {}: {}", describe(chat), fallback.model);
        items = co_await call_model(r, fallback.provider, fallback.model, purpose, chat, history, tools);
    }
    co_return items;
}

Task<std::optional<std::vector<provider::OutputItem>>> call_selector(Responder *r, const store::ChatKey &chat,
                                                                     const provider::History &history,
                                                                     const std::vector<provider::ToolDefinition> &tools) {
    if (r->selector && r->now() >= r->planner_down_until) {
        if (auto items = co_await call_model(r, *r->selector, r->config.model.planner_model, "select", chat, history, tools))
            co_return items;
    }
    co_return co_await call_planner(r, "select", chat, history, tools);
}

std::optional<provider::History> warm_prefix(provider::History prompt) {
    if (prompt.empty() || prompt.back().parts.size() < 2) return std::nullopt;
    auto &parts = prompt.back().parts;
    const auto *stable = std::get_if<provider::TextPart>(&parts[parts.size() - 2]);
    if (!stable || !stable->cache) return std::nullopt;
    parts.pop_back();
    return prompt;
}

void keep_warm(Responder *r, const store::ChatKey &chat, const provider::History &prompt,
               const std::vector<provider::ToolDefinition> &tools) {
    if (!r->warmer || !r->realtime) return;
    auto part = warm_prefix(prompt);
    if (!part) return;
    auto &state = r->chats[chat];
    state.warm_prompt = *std::move(part);
    state.warm_tools = tools;
    r->spawn(warm(r, chat, ++state.warm_generation));
}

std::string final_text(const std::vector<provider::OutputItem> &items) {
    std::string answer;
    for (const auto &item : items) {
        if (const auto *message = std::get_if<provider::AssistantMessageItem>(&item)) {
            for (const auto &part : message->parts) answer += part.text;
        } else {
            answer.clear();
        }
    }
    return answer;
}

Task<bool> call_action(Responder *r, std::string action, std::string params) {
    auto session = r->session;
    if (!session || !session->is_open()) co_return false;
    auto result = co_await session->call(action, std::move(params));
    if (result.has_error()) {
        log::warn("{} failed: {}", action, result.error().message());
        co_return false;
    }
    co_return true;
}

} // namespace detail

} // namespace lilia::chat
