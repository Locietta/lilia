#include "calls.h"

#include <format>
#include <utility>
#include <variant>

#include <lighter/async/runtime/timeout.h>

#include <lilia/log.h>

namespace lilia {

using lighter::Task;

namespace {

/// with_timeout takes lighter's error type; hand the outcome through as a value.
Task<Result<provider::ProviderCallCompletion>> settle(lighter::Task<provider::ProviderCallCompletion, Error> call) {
    auto outcome = co_await std::move(call);
    co_return outcome;
}

} // namespace

Task<Result<std::vector<provider::OutputItem>>> call_recorded(provider::Provider &provider, const provider::History &history,
                                                              const std::vector<provider::ToolDefinition> &tools,
                                                              std::chrono::seconds deadline, store::Store &store, store::LlmCall &call) {
    std::vector<provider::OutputItem> items;
    const provider::StreamCallbacks callbacks{.on_item_completed = [&items](const provider::OutputItem &item) { items.push_back(item); }};
    auto settled = co_await lighter::with_timeout(settle(provider->complete(history, tools, callbacks)), deadline);
    auto completion = settled.has_error() ?
                          Result<provider::ProviderCallCompletion>(lighter::outcome_error(
                              Error::api("timeout", std::format("the call exceeded its {}s deadline", deadline.count()), "", true))) :
                          *std::move(settled);
    if (completion.has_error()) {
        call.error = completion.error().message();
        call.request_id = completion.error().request_id;
    } else {
        if (!completion->model.empty()) call.model = completion->model;
        call.input_tokens = completion->usage.input_tokens;
        call.output_tokens = completion->usage.output_tokens;
        call.cache_read_tokens = completion->usage.cache_read_tokens;
        call.cache_write_tokens = completion->usage.cache_write_tokens;
        call.stop = completion->stop_detail;
        call.request_id = completion->request_id;
    }
    if (auto recorded = store.add_llm_call(call); recorded.has_error())
        log::error("cannot record model call: {}", recorded.error().message());
    if (completion.has_error()) co_return lighter::outcome_error(std::move(completion).error());
    co_return std::move(items);
}

std::string answer_text(const std::vector<provider::OutputItem> &items) {
    std::string answer;
    for (const auto &item : items) {
        if (const auto *message = std::get_if<provider::AssistantMessageItem>(&item)) {
            for (const auto &part : message->parts) answer += part.text;
        }
    }
    return answer;
}

} // namespace lilia
