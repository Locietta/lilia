#pragma once

#include <chrono>
#include <vector>

#include <lighter/async/runtime/task.h>

#include <lilia/error.h>
#include <lilia/provider/provider.h>
#include <lilia/store/store.h>

namespace lilia {

/// One model call with a deadline (the client's retries included), recorded
/// in llm_calls. `call` names the row (time, chat, model) and comes back
/// filled in: the model that answered, usage, stop and request id, or the
/// error. Past the deadline the call fails like any other error, so a hung
/// request cannot hold up whoever waits on it.
lighter::Task<Result<std::vector<provider::OutputItem>>> call_recorded(provider::Provider &provider, const provider::History &history,
                                                                       const std::vector<provider::ToolDefinition> &tools,
                                                                       std::chrono::seconds deadline, store::Store &store,
                                                                       store::LlmCall &call);

/// The text of the assistant messages among `items`, joined.
std::string answer_text(const std::vector<provider::OutputItem> &items);

} // namespace lilia
