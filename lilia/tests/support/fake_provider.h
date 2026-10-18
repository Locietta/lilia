#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <glaze/json.hpp>

#include <lighter/async/runtime/task.h>

#include <lilia/provider/provider.h>

namespace lilia::test {

/// A model that answers from a script, for tests that drive the Responder.
struct FakeProvider {
    struct State {
        /// Decides each answer from the request; nullopt fails the call.
        std::move_only_function<std::optional<std::vector<provider::OutputItem>>(const provider::History &,
                                                                                 const std::vector<provider::ToolDefinition> &)>
            answer;
        /// Every request seen, in order.
        std::vector<provider::History> requests;
        std::vector<std::vector<provider::ToolDefinition>> tools;
    };
    std::shared_ptr<State> state;

    lighter::Task<provider::ProviderCallCompletion, Error> complete(const provider::History &history,
                                                                    const std::vector<provider::ToolDefinition> &tools,
                                                                    const provider::StreamCallbacks &callbacks) {
        state->requests.push_back(history);
        state->tools.push_back(tools);
        auto items = state->answer ? state->answer(history, tools) : std::nullopt;
        if (!items) co_await lighter::fail(Error::api("overloaded_error", "fake failure", "", false));
        for (const auto &item : *items) {
            if (callbacks.on_item_completed) callbacks.on_item_completed(item);
        }
        co_return provider::ProviderCallCompletion{.stop = provider::StopKind::DONE};
    }
};

inline provider::Provider fake_provider(std::shared_ptr<FakeProvider::State> state) {
    return pro::make_proxy<provider::ProviderFacade, FakeProvider>(FakeProvider{.state = std::move(state)});
}

/// One tool call as a model output item.
inline provider::OutputItem tool_call(std::string name, std::string_view input_json, std::string id = "toolu_1") {
    provider::ToolCallItem item{.call = {.id = std::move(id), .name = std::move(name)}};
    std::ignore = glz::read_json(item.call.input, input_json);
    return item;
}

/// Plain assistant text as a model output item.
inline provider::OutputItem text_answer(std::string text) {
    return provider::AssistantMessageItem{.parts = {provider::TextPart{.text = std::move(text)}}};
}

} // namespace lilia::test
