#pragma once

#include <chrono>
#include <optional>
#include <string>
#include <vector>

#include <lighter/async/runtime/task.h>
#include <lighter/http/http.h>
#include <lighter/types.hpp>

#include <lilia/error.h>
#include <lilia/provider/auth.h>
#include <lilia/provider/common.h>
#include <lilia/provider/history.h>
#include <lilia/provider/provider.h>

namespace lilia::anthropic {

using namespace lighter::types;

/// Tag identifying this provider's OpaquePart payloads (thinking blocks and
/// their signatures, serialized in wire form).
inline constexpr std::string_view k_provider_tag = "anthropic";

struct ClientOptions {
    provider::AuthResolver auth;
    std::string base_url = "https://api.anthropic.com";
    std::string model;
    std::optional<std::string> reasoning_effort;
    /// Send `thinking: {"type": "disabled"}` when no reasoning effort is set.
    /// DeepSeek's Anthropic endpoint thinks by default otherwise.
    bool disable_thinking = false;
    /// The "off" value: "disabled" for most models; newer Claude models
    /// (claude-sonnet-5-5) reject it and want "between_tools".
    std::string thinking_off = "disabled";
    /// Send `tool_choice: {"type": "any"}` whenever tools are offered, so
    /// the model must call one of them.
    bool require_tool = false;
    /// Sampling temperature; unset leaves the provider default.
    std::optional<double> temperature;
    u32 max_tokens = 8192;
    usize max_retries = 2;
    std::chrono::milliseconds initial_retry_delay{500};
    /// Ask for Anthropic prompt caching: on the whole request (automatic
    /// caching, so each tool round reuses the one before), on the last tool
    /// and after the SYSTEM instructions, which `system` then sends as text
    /// blocks (the next reply reuses tools and system). Claude only caches
    /// when asked; DeepSeek and MiniMax cache by themselves and need not see
    /// the field.
    bool prompt_cache = false;
};

/// Anthropic Messages API client, conforming to provider::ProviderFacade.
/// Wire types and the SSE accumulator are implementation details; the public
/// surface speaks the neutral transcript only.
struct Client {
    explicit Client(ClientOptions options);

    /// Streams one Messages API call to completion, invoking callbacks live.
    /// Retries transport/429/5xx failures with backoff, but only while no
    /// text has been surfaced through the callbacks.
    lighter::Task<provider::ProviderCallCompletion, Error> complete(const provider::History &history,
                                                                    const std::vector<provider::ToolDefinition> &tools,
                                                                    const provider::StreamCallbacks &callbacks);

    ClientOptions options;
    lighter::http::Client http_client;
};

} // namespace lilia::anthropic
