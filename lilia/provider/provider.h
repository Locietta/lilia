#pragma once

#include <optional>
#include <string>
#include <vector>

#include <proxy/proxy.h>

#include <lighter/async/runtime/task.h>

#include <lilia/error.h>
#include <lilia/provider/common.h>
#include <lilia/provider/history.h>

namespace lilia::provider {

// Behavior-first provider abstraction: the agent talks to "a source that
// receives history and streams back tokens", never to a concrete vendor.
// Every capability here must be satisfiable by every provider, so callers
// never probe for support.

PRO_DEF_MEM_DISPATCH(CompleteDispatch, complete);

struct ProviderFacade
    : pro::facade_builder //
      ::add_convention<CompleteDispatch,
                       lighter::Task<ProviderCallCompletion, Error>(const History &history, const std::vector<ToolDefinition> &tools,
                                                                    const StreamCallbacks &callbacks)>::build {};

using Provider = pro::proxy<ProviderFacade>;

} // namespace lilia::provider
