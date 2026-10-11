#include <lilia/chat/detail/responder.h>

#include <chrono>
#include <format>

#include <glaze/json.hpp>

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

} // namespace

namespace detail {

std::string describe(const store::ChatKey &chat) { return chat.type + ":" + std::to_string(chat.id); }

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
    // Her own lines join the next learning batch as [SELF] context.
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
    // A request that never finishes must not leave her silent: past the
    // deadline it fails like any other error (and the writer falls back).
    auto settled = co_await lighter::with_timeout(settle(provider->complete(history, tools, callbacks)), call_deadline(r, purpose));
    auto completion = settled.has_error() ?
                          Result<provider::ProviderCallCompletion>(lighter::outcome_error(Error::api(
                              "timeout", std::format("{} call exceeded {}s", purpose, call_deadline(r, purpose).count()), "", true))) :
                          *std::move(settled);

    store::LlmCall call{.time = started, .chat = chat, .model = std::move(model)};
    if (completion.has_error()) {
        call.error = completion.error().message();
        call.request_id = completion.error().request_id;
        log::error("{} call for {} failed: {}", purpose, describe(chat), call.error);
    } else {
        if (!completion->model.empty()) call.model = completion->model;
        call.input_tokens = completion->usage.input_tokens;
        call.output_tokens = completion->usage.output_tokens;
        call.cache_read_tokens = completion->usage.cache_read_tokens;
        call.cache_write_tokens = completion->usage.cache_write_tokens;
        call.stop = completion->stop_detail;
        call.request_id = completion->request_id;
        log::info("{} call for {}: {} in / {} out / {} cache read / {} cache write tokens", purpose, describe(chat), call.input_tokens,
                  call.output_tokens, call.cache_read_tokens, call.cache_write_tokens);
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
