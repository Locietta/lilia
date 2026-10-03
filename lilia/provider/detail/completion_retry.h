#pragma once

#include <chrono>
#include <functional>
#include <string>
#include <string_view>

#include <lighter/async/runtime/task.h>
#include <lighter/types.hpp>

#include <lilia/error.h>
#include <lilia/provider/common.h>
#include <lilia/provider/history.h>

namespace lilia::provider::detail {

/// Injectable boundary around one provider streaming attempt and retry delay.
struct CompletionAttempts {
    std::copyable_function<lighter::Task<ProviderCallCompletion, Error>(const std::string &body, const StreamCallbacks &callbacks,
                                                                        bool &output_emitted) const>
        stream;
    std::copyable_function<lighter::Task<>(std::chrono::milliseconds delay) const> sleep;
};

lighter::Task<ProviderCallCompletion, Error> complete_with_retry(const CompletionAttempts &attempts, std::string body,
                                                                 const StreamCallbacks &callbacks, usize max_retries,
                                                                 std::chrono::milliseconds initial_retry_delay);

} // namespace lilia::provider::detail
