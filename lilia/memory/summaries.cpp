#include "memory.h"

#include <algorithm>
#include <chrono>
#include <format>

#include <glaze/json.hpp>

#include <xmake/bin2obj/summary_md.hpp>
#include <xmake/bin2obj/window_summary_md.hpp>

#include <lilia/calls.h>
#include <lilia/chat/context.h>
#include <lilia/log.h>

namespace lilia::memory {

using lighter::Task;

namespace {

constexpr usize k_max_cues = 5;

const glz::generic *field(const glz::generic &object, std::string_view key) {
    if (!object.is_object()) return nullptr;
    const auto &map = object.get_object();
    const auto it = map.find(key);
    return it == map.end() ? nullptr : &it->second;
}

provider::ToolDefinition record_summary_tool() {
    glz::generic cue_schema;
    std::ignore = glz::read_json(cue_schema, R"({"type": "string"})");
    provider::SchemaProperty cues{.type = "array", .description = "3 到 5 句回想线索；没有新内容时为空"};
    cues.items.emplace(std::move(cue_schema));
    return {
        .name = "record_summary",
        .description = "提交这段聊天的增量摘要和回想线索",
        .input_schema = {.properties = {{"summary", {.type = "string", .description = "新内容的摘要；没有就写空字符串"}},
                                        {"recall_cues", std::move(cues)}},
                         .required = {"summary", "recall_cues"}},
    };
}

constexpr std::chrono::seconds k_summary_deadline{90};

std::string local_time(i64 unix_seconds, i32 utc_offset_minutes) {
    return std::format("{:%m-%d %H:%M}", std::chrono::sys_seconds(std::chrono::seconds(unix_seconds + i64{utc_offset_minutes} * 60)));
}

} // namespace

FactRequest build_summary_request(std::string_view prompt, const std::vector<store::ChatSummary> &previous, std::string_view lines) {
    FactRequest request;
    provider::append_system(request.history, std::string(prompt));
    std::string user;
    std::string earlier;
    for (const auto &summary : previous) {
        if (!summary.summary.empty()) earlier += "- " + summary.summary + "\n";
    }
    if (!earlier.empty()) user += "以前的摘要（只用来理解上下文，别照抄）：\n" + earlier + "\n";
    user += "新的聊天记录：\n";
    user += lines;
    provider::append_user(request.history, std::move(user));
    request.tools.push_back(record_summary_tool());
    return request;
}

std::optional<SummaryResult> parse_summary(const std::vector<provider::OutputItem> &items) {
    for (const auto &item : items) {
        const auto *call = std::get_if<provider::ToolCallItem>(&item);
        if (!call || call->call.name != "record_summary") continue;
        SummaryResult result;
        if (const auto *summary = field(call->call.input, "summary"); summary && summary->is_string())
            result.summary = summary->get_string();
        if (const auto *cues = field(call->call.input, "recall_cues"); cues && cues->is_array()) {
            for (const auto &cue : cues->get_array()) {
                if (cue.is_string() && !cue.get_string().empty() && result.cues.size() < k_max_cues)
                    result.cues.push_back(cue.get_string());
            }
        }
        // Cues without a summary have nothing to bring back.
        if (result.summary.empty()) result.cues.clear();
        return result;
    }
    return std::nullopt;
}

store::MemoryChunk summary_chunk(const store::ChatSummary &summary, i32 utc_offset_minutes) {
    return {
        .id = -summary.id,
        .chat = summary.chat,
        .first_row = summary.first_row,
        .last_row = summary.last_row,
        .start_time = summary.start_time,
        .end_time = summary.end_time,
        .text = std::format("[{} 至 {} 的聊天摘要] {}\n", local_time(summary.start_time, utc_offset_minutes),
                            local_time(summary.end_time, utc_offset_minutes), summary.summary),
    };
}

std::vector<std::string> summary_texts(const store::ChatSummary &summary) {
    if (summary.summary.empty()) return {};
    std::vector<std::string> texts{summary.summary};
    for (usize start = 0; start < summary.cues.size();) {
        const auto end = std::min(summary.cues.find('\n', start), summary.cues.size());
        if (end > start) texts.push_back(summary.cues.substr(start, end - start));
        start = end + 1;
    }
    return texts;
}

Task<bool> Memory::write_summary(store::ChatKey chat, std::string_view kind, std::string_view prompt,
                                 std::vector<store::ChatSummary> previous, std::vector<store::StoredMessage> messages) {
    const auto request = build_summary_request(prompt, previous, display_text(messages, self_name, utc_offset_minutes));
    store::LlmCall call{.time = now(), .chat = chat, .model = summarizer_model};
    // Past the deadline the call counts as failed; a hang must not block indexing.
    const auto items = co_await call_recorded(*summarizer, request.history, request.tools, k_summary_deadline, store, call);
    if (items.has_error()) {
        log::warn("memory: {} summary call failed: {}", kind, call.error);
        co_return false;
    }
    const auto result = parse_summary(*items);
    if (!result) {
        log::warn("memory: the {} summary call did not use record_summary", kind);
        co_return false;
    }

    store::ChatSummary summary{
        .chat = chat,
        .first_row = messages.front().row_id,
        .last_row = messages.back().row_id,
        .start_time = messages.front().time,
        .end_time = messages.back().time,
        .summary = result->summary,
        .kind = std::string(kind),
    };
    for (usize i = 0; i < result->cues.size(); ++i) summary.cues += (i ? "\n" : "") + result->cues[i];
    if (auto added = store.add_summary(summary); added.has_error()) {
        log::error("memory: {}", added.error().message());
        co_return false;
    }
    // Embedded now; one that fails is embedded by the next recall or backfill.
    std::ignore = co_await retriever.backfill(summary_texts(summary));
    log::info("memory: {} summary of {}:{} ({} messages): {}", kind, chat.type, chat.id, messages.size(),
              result->summary.empty() ? "(nothing new)" : result->summary);
    co_return true;
}

Task<> Memory::summarize(store::ChatKey chat) {
    if (!summarizer || config.summary_messages == 0) co_return;
    for (;;) {
        auto last = store.last_summary_row(chat);
        if (last.has_error()) co_return;
        auto messages = store.messages_after(chat, *last, config.summary_messages);
        if (messages.has_error() || messages->size() < config.summary_messages) co_return;
        auto previous = store.summaries(chat, 2);
        const auto prompt = chat::fill_template({reinterpret_cast<const char *>(xmake::summary_md.data()), xmake::summary_md.size()},
                                                {{"name", self_name}});
        if (!co_await write_summary(chat, store::k_rolling_summary, prompt,
                                    previous.has_value() ? *std::move(previous) : std::vector<store::ChatSummary>{}, *std::move(messages)))
            co_return;
    }
}

Task<> Memory::summarize_window(store::ChatKey chat, i64 through_row, usize span) {
    if (!summarizer || std::ranges::contains(windowing, chat)) co_return;
    auto last = store.last_summary_row(chat, store::k_window_summary);
    // The newest `span` lines up to the anchor; older ones are not picked up.
    auto messages = store.messages_before(chat, through_row + 1, span);
    if (last.has_error() || messages.has_error()) co_return;
    std::erase_if(*messages, [&](const store::StoredMessage &message) { return message.row_id <= *last; });
    if (messages->empty()) co_return;
    windowing.push_back(chat);
    const auto prompt = chat::fill_template(
        {reinterpret_cast<const char *>(xmake::window_summary_md.data()), xmake::window_summary_md.size()}, {{"name", self_name}});
    std::ignore = co_await write_summary(chat, store::k_window_summary, prompt, {}, *std::move(messages));
    std::erase(windowing, chat);
}

std::vector<store::ChatSummary> Memory::window_summaries(const store::ChatKey &chat) {
    auto summaries = store.summaries(chat, 0, store::k_window_summary);
    if (summaries.has_error()) return {};
    std::erase_if(*summaries, [](const store::ChatSummary &summary) { return summary.summary.empty(); });
    return *std::move(summaries);
}

std::string Memory::earlier(const store::ChatKey &chat) {
    const auto summaries = window_summaries(chat);
    std::string text;
    for (usize i = summaries.size() > k_earlier_summaries ? summaries.size() - k_earlier_summaries : 0; i < summaries.size(); ++i) {
        const auto &summary = summaries[i];
        text += std::format("[{} 至 {}] {}\n", local_time(summary.start_time, utc_offset_minutes),
                            local_time(summary.end_time, utc_offset_minutes), summary.summary);
    }
    return text;
}

Task<std::string> Memory::recall(store::ChatKey chat, std::vector<store::StoredMessage> log) {
    if (log.empty() || config.recall_messages == 0) co_return std::string();
    // MaiBot's mid-term recall: among the window summaries that scrolled out
    // of view, except the newest few the planner sees anyway (earlier()).
    auto summaries = window_summaries(chat);
    const i64 first_visible = log.front().row_id;
    std::erase_if(summaries, [&](const store::ChatSummary &summary) { return first_visible > 0 && summary.last_row >= first_visible; });
    summaries.resize(summaries.size() > k_earlier_summaries ? summaries.size() - k_earlier_summaries : 0);
    if (summaries.empty()) co_return std::string();

    const std::vector<store::StoredMessage> recent(log.end() - static_cast<std::ptrdiff_t>(std::min(log.size(), config.recall_messages)),
                                                   log.end());
    std::vector<retrieval::Document> documents;
    for (const auto &summary : summaries) documents.push_back({.id = summary.id, .texts = summary_texts(summary)});
    // The summary whose best cue matches the chat.
    const auto matches = co_await retriever.search(retrieval::k_recall_task, embed_text(recent, self_name), documents,
                                                   {.top_k = 1, .min_similarity = config.recall_score});
    if (matches.empty()) co_return std::string();
    const auto &best = summaries[matches.front().index];
    log::info("memory: recalled a summary of {}:{} ({:.2f})", chat.type, chat.id, matches.front().similarity.value_or(0));
    co_return summary_chunk(best, utc_offset_minutes).text;
}

} // namespace lilia::memory
