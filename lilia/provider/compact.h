#pragma once

#include <lighter/async/runtime/task.h>

#include <lilia/error.h>
#include <lilia/provider/provider.h>

namespace lilia::provider {

/// Provider-agnostic compaction: asks the model (through `provider->complete()`)
/// to summarize the supplied history, including progress on an active task, then
/// replaces the conversation with the most recent user prompts (bounded) and
/// the summary as the final item. A summarization request the provider
/// rejects as too large is retried without its oldest items. Works against
/// any conforming provider. The caller excludes pending new input.
/// Providers with a native compaction endpoint should
/// try that first and use this as the fallback.
lighter::Task<void, Error> local_compact(ProviderView provider, History &history, CompactRequest request);

} // namespace lilia::provider
