#include "feedback.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <format>
#include <variant>

#include <glaze/json.hpp>

#include <xmake/bin2obj/hindsight_md.hpp>

#include <lilia/calls.h>
#include <lilia/chat/context.h>
#include <lilia/chat/session.h>
#include <lilia/log.h>

namespace lilia::feedback {

using lighter::Task;

namespace {

constexpr std::chrono::seconds k_call_deadline{90};
/// Lines before a window: shown to the judge, and behind the features.
constexpr usize k_shown_context = 5;
constexpr usize k_feature_context = 40;
/// A quote of her message within the window adds to its reward.
constexpr double k_quoted_reward = 0.3;
constexpr double k_global_l2 = 1;
constexpr double k_group_l2 = 3;

constexpr std::array<std::string_view, 5> k_addressees{"her", "other", "group", "self", "unclear"};
/// The reward of a judged reaction; reward_of is their only reader. Hand-set:
/// nobody labels how well her messages went down, so there is nothing to fit
/// them to.
constexpr std::array<std::pair<std::string_view, double>, 8> k_stances{{{"appreciation", 1.0},
                                                                        {"playful", 0.7},
                                                                        {"neutral", 0.2},
                                                                        {"ignored", -0.2},
                                                                        {"confusion", -0.5},
                                                                        {"correction", -0.5},
                                                                        {"rejection", -1.0},
                                                                        {"attack", -1.0}}};
constexpr std::array<std::pair<std::string_view, double>, 6> k_contributions{
    {{"advance", 0.5}, {"maintain", 0.3}, {"acknowledge", 0.0}, {"close", 0.0}, {"wrong_push", -0.7}, {"intrusion", -1.0}}};

/// "#12", "12" or 12 -> 12; 0 otherwise.
i64 row_number(const glz::generic *value) {
    if (!value) return 0;
    if (value->is_number()) return static_cast<i64>(value->get_number());
    if (!value->is_string()) return 0;
    std::string_view text = value->get_string();
    if (text.starts_with('#')) text.remove_prefix(1);
    i64 row = 0;
    for (const char c : text) {
        if (c < '0' || c > '9') break;
        row = row * 10 + (c - '0');
    }
    return row;
}

std::string one_of(std::string value, std::span<const std::string_view> allowed) {
    return std::ranges::contains(allowed, value) ? value : std::string("unclear");
}

/// The sticker one of her messages sent ("sticker:<sha1>" in its segments).
std::string sticker_hash(const store::StoredMessage &message) {
    constexpr std::string_view k_marker = "sticker:";
    const auto at = message.segments_json.find(k_marker);
    if (at == std::string::npos) return {};
    const auto start = at + k_marker.size();
    const auto end = message.segments_json.find('"', start);
    return message.segments_json.substr(start, end == std::string::npos ? std::string::npos : end - start);
}

/// Whether a planner round's comma-separated message ids include `id`.
bool sent_in(std::string_view sent, i64 id) {
    const auto wanted = std::to_string(id);
    while (!sent.empty()) {
        const auto comma = sent.find(',');
        if (sent.substr(0, comma) == wanted) return true;
        sent = comma == std::string_view::npos ? std::string_view{} : sent.substr(comma + 1);
    }
    return false;
}

std::vector<chat::LineEdges> edges_for(store::Store &store, const store::ChatKey &chat, const std::vector<store::StoredMessage> &rows,
                                       i64 self_id, std::string_view self_name, const std::vector<std::string> &aliases) {
    const auto lookup = [&](i64 message_id) -> std::optional<store::StoredMessage> {
        if (const auto it = std::ranges::find(rows, message_id, &store::StoredMessage::message_id); it != rows.end()) return *it;
        auto stored = store.message(chat, message_id);
        return stored.has_value() && *stored ? std::optional<store::StoredMessage>(**stored) : std::nullopt;
    };
    const auto name_of = [&](i64 user) {
        auto name = store.sender_name(chat, user);
        return name.has_value() && !name->empty() ? *name : std::to_string(user);
    };
    std::vector<chat::LineEdges> edges;
    for (const auto &row : rows) edges.push_back(chat::line_edges(row, self_id, self_name, aliases, lookup, name_of));
    return edges;
}

} // namespace

learn::LearnerRequest build_hindsight_request(std::string_view prompt, std::string_view lines, std::string_view targets,
                                              std::string_view own) {
    learn::LearnerRequest request;
    provider::append_user(request.history,
                          chat::fill_template(prompt, {{"lines", lines}, {"targets", targets}, {"own", own.empty() ? "（没有）" : own}}));
    glz::generic line_schema;
    std::ignore = glz::read_json(line_schema, R"({"type": "object", "properties": {"row": {"type": "string"},
        "to": {"type": "string", "enum": ["her", "other", "group", "self", "unclear"]}}, "required": ["row", "to"]})");
    glz::generic reply_schema;
    std::ignore = glz::read_json(reply_schema, R"({"type": "object", "properties": {"row": {"type": "string"},
        "for_her": {"type": "string"}, "answered_by_other": {"type": "string"}, "stance": {"type": "string"},
        "contribution": {"type": "string"}}, "required": ["row", "for_her", "answered_by_other", "stance", "contribution"]})");
    provider::SchemaProperty lines_property{.type = "array", .description = "每条要判断的消息是对谁说的"};
    lines_property.items.emplace(std::move(line_schema));
    provider::SchemaProperty replies_property{.type = "array", .description = "她的每条发言的效果；她没发言时为空"};
    replies_property.items.emplace(std::move(reply_schema));
    request.tools.push_back({
        .name = "record_hindsight",
        .description = "提交事后判断",
        .input_schema = {.properties = {{"lines", std::move(lines_property)}, {"replies", std::move(replies_property)}},
                         .required = {"lines", "replies"}},
    });
    return request;
}

std::optional<Hindsight> parse_hindsight(const std::vector<provider::OutputItem> &items) {
    const auto *input = learn::tool_input(items, "record_hindsight");
    if (!input) return std::nullopt;
    Hindsight out;
    static constexpr std::array<std::string_view, 3> k_answers{"yes", "no", "unclear"};
    std::vector<std::string_view> stances;
    for (const auto &[name, score] : k_stances) stances.push_back(name);
    std::vector<std::string_view> contributions;
    for (const auto &[name, score] : k_contributions) contributions.push_back(name);
    if (const auto *lines = learn::field(*input, "lines"); lines && lines->is_array()) {
        for (const auto &entry : lines->get_array()) {
            const i64 row = row_number(learn::field(entry, "row"));
            if (row > 0) out.lines.push_back({.row = row, .to = one_of(learn::text_field(entry, "to"), k_addressees)});
        }
    }
    if (const auto *replies = learn::field(*input, "replies"); replies && replies->is_array()) {
        for (const auto &entry : replies->get_array()) {
            const i64 row = row_number(learn::field(entry, "row"));
            if (row <= 0) continue;
            out.replies.push_back({.row = row,
                                   .for_her = one_of(learn::text_field(entry, "for_her"), k_answers),
                                   .answered_by_other = one_of(learn::text_field(entry, "answered_by_other"), k_answers),
                                   .stance = one_of(learn::text_field(entry, "stance"), stances),
                                   .contribution = one_of(learn::text_field(entry, "contribution"), contributions)});
        }
    }
    return out;
}

double reward_of(std::string_view stance, std::string_view contribution) {
    double reward = 0;
    for (const auto &[name, score] : k_stances) {
        if (name == stance) reward += score;
    }
    for (const auto &[name, score] : k_contributions) {
        if (name == contribution) reward += score;
    }
    return reward;
}

std::string_view level(double probability) { return probability < 0.2 ? "低" : probability <= 0.6 ? "中" : "高"; }

std::vector<LabelledSample> samples_of(const std::vector<store::AddresseeLabel> &labels) {
    std::vector<LabelledSample> samples;
    for (const auto &label : labels) {
        std::vector<double> values;
        if (glz::read_json(values, label.features) || values.size() != chat::k_features) continue;
        Sample sample{.label = label.label ? 1.0 : 0.0};
        std::ranges::copy(values, sample.x.begin());
        samples.emplace_back(label.chat, sample);
    }
    return samples;
}

const Weights &FittedModels::model_for(const store::ChatKey &chat_key) const {
    static const Weights k_bootstrap = bootstrap_weights();
    if (const auto it = groups.find(chat_key); it != groups.end() && it->second.fitted) return it->second.weights;
    return global.fitted ? global.weights : k_bootstrap;
}

FittedModels fit_models(std::span<const LabelledSample> samples, const FeedbackConfig &config) {
    const auto positives = [](const std::vector<Sample> &rows) {
        return static_cast<usize>(std::ranges::count_if(rows, [](const Sample &sample) { return sample.label > 0.5; }));
    };
    std::vector<Sample> all;
    std::map<store::ChatKey, std::vector<Sample>> per_chat;
    for (const auto &[chat_key, sample] : samples) {
        all.push_back(sample);
        per_chat[chat_key].push_back(sample);
    }
    FittedModels out;
    out.global = {.weights = bootstrap_weights(), .rows = all.size(), .positives = positives(all)};
    if (all.size() >= config.min_rows_global) {
        out.global.weights = fit_logistic(all, out.global.weights, k_global_l2);
        out.global.fitted = true;
    }
    out.global.log_loss = log_loss(out.global.weights, all);
    for (const auto &[chat_key, rows] : per_chat) {
        FittedModel model{.weights = out.global.weights, .rows = rows.size(), .positives = positives(rows)};
        if (model.rows >= config.min_rows_group && model.positives >= config.min_positives) {
            model.weights = fit_logistic(rows, out.global.weights, k_group_l2);
            model.fitted = true;
        }
        model.log_loss = log_loss(model.weights, rows);
        out.groups[chat_key] = model;
    }
    return out;
}

Feedback::Feedback(const FeedbackConfig &config, const ChatConfig &chat, store::Store &store, provider::Provider judge, Spawn spawn)
    : now([] { return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count(); }),
      config(config), chat(chat), store(store), judge_model(std::move(judge)), spawn(std::move(spawn)),
      prompt(chat::fill_template({reinterpret_cast<const char *>(xmake::hindsight_md.data()), xmake::hindsight_md.size()},
                                 {{"name", chat.name}})) {}

void Feedback::load() {
    auto stored = store.addressee_models();
    if (stored.has_error()) return;
    for (const auto &model : *stored) {
        if (model.feature_version != chat::k_feature_version) continue;
        const auto weights = read_weights(model.weights);
        if (!weights) continue;
        if (model.chat.type.empty()) {
            global = *weights;
            global_fitted = true;
        } else {
            models[model.chat] = *weights;
        }
    }
}

const Weights &Feedback::model_for(const store::ChatKey &chat_key) const {
    if (const auto it = models.find(chat_key); it != models.end()) return it->second;
    return global;
}

bool Feedback::fitted(const store::ChatKey &chat_key) const { return models.contains(chat_key) || global_fitted; }

void Feedback::observe(const store::ChatKey &chat_key, i64 self_id) {
    if (!config.enabled || judging.contains(chat_key)) return;
    auto through = store.feedback_progress(chat_key);
    if (through.has_error()) return;
    if (*through == 0) {
        // Live judging starts from now; the backfill tool covers what came before.
        auto latest = store.recent_messages(chat_key, 1);
        if (latest.has_value() && !latest->empty()) std::ignore = store.set_feedback_progress(chat_key, latest->back().row_id);
        return;
    }
    auto pending = store.messages_after(chat_key, *through, config.judge_window_lines);
    if (pending.has_error() || pending->empty()) return;
    const auto last_hers = std::ranges::find_if(pending->rbegin(), pending->rend(), &store::StoredMessage::from_self);
    const bool settled = last_hers != pending->rend() && now() - last_hers->time >= config.judge_after_seconds;
    if (!settled && pending->size() < config.judge_window_lines) return;
    const i64 current_hour = now() / 3600;
    if (current_hour != hour) {
        hour = current_hour;
        judges_this_hour = 0;
    }
    if (judges_this_hour >= config.judges_per_hour) return;
    if (background_budget > 0) {
        auto used = store.tokens_since(now() - 86'400, store::TokenScope::BACKGROUND);
        if (used.has_value() && *used >= background_budget) return;
    }
    ++judges_this_hour;
    judging.insert(chat_key);
    spawn([](Feedback *self, store::ChatKey key, i64 self_id, usize limit) -> Task<> {
        std::ignore = co_await self->judge(key, self_id, limit);
        self->judging.erase(key);
    }(this, chat_key, self_id, std::max<usize>(config.judge_window_lines, 1) * 3 / 2));
}

Task<bool> Feedback::judge(store::ChatKey chat_key, i64 self_id, usize limit) {
    auto through = store.feedback_progress(chat_key);
    if (through.has_error()) co_return false;
    auto window = store.messages_after(chat_key, *through, limit);
    if (window.has_error() || window->empty()) co_return false;
    auto context = store.messages_before(chat_key, window->front().row_id, k_feature_context);
    std::vector<store::StoredMessage> rows = context.has_value() ? *std::move(context) : std::vector<store::StoredMessage>{};
    const usize first = rows.size();
    rows.insert(rows.end(), window->begin(), window->end());
    const auto edges = edges_for(store, chat_key, rows, self_id, chat.name, chat.aliases);

    // What the judge reads: a little context, then the window.
    std::string lines;
    for (usize i = first > k_shown_context ? first - k_shown_context : 0; i < rows.size(); ++i) {
        lines += chat::session_line(rows[i], chat.utc_offset_minutes, &edges[i]);
    }
    std::string targets;
    for (usize i = first; i < rows.size(); ++i) {
        if (!rows[i].from_self && !chat::is_notice(rows[i])) targets += std::format("{}#{}", targets.empty() ? "" : "、", rows[i].row_id);
    }
    // Her messages, with the line each answered (from the planner's log).
    auto turns = store.planner_turns(chat_key, rows[first].time - 3600);
    const auto answered = [&](i64 message_id) -> i64 {
        if (!turns.has_value()) return 0;
        for (const auto &turn : *turns) {
            if (sent_in(turn.sent_message_ids, message_id)) return turn.target_message_id;
        }
        return 0;
    };
    std::string own;
    std::map<i64, i64> targets_of; // her row -> the message it answered
    for (usize i = first; i < rows.size(); ++i) {
        if (!rows[i].from_self) continue;
        const i64 target = answered(rows[i].message_id);
        targets_of[rows[i].row_id] = target;
        const auto it = std::ranges::find(rows, target, &store::StoredMessage::message_id);
        own += std::format("{}#{}（回应 {}）", own.empty() ? "" : "、", rows[i].row_id,
                           target != 0 && it != rows.end() ? std::format("#{}", it->row_id) : std::string("不知道"));
    }
    if (targets.empty() && own.empty()) {
        std::ignore = store.set_feedback_progress(chat_key, rows.back().row_id);
        co_return true;
    }

    const auto request = build_hindsight_request(prompt, lines, targets, own);
    store::LlmCall call{.time = now(), .chat = {.type = "feedback", .id = chat_key.id}, .model = judge_model_name};
    const auto items = co_await call_recorded(judge_model, request.history, request.tools, k_call_deadline, store, call);
    if (items.has_error()) {
        log::warn("hindsight judging of {}:{} failed: {}", chat_key.type, chat_key.id, call.error);
        co_return false;
    }
    const auto hindsight = parse_hindsight(*items);
    if (!hindsight) {
        log::warn("hindsight judging of {}:{}: no answer", chat_key.type, chat_key.id);
        co_return false;
    }

    const auto &model = model_for(chat_key);
    usize labelled = 0;
    for (const auto &verdict : hindsight->lines) {
        const auto it =
            std::ranges::find(rows.begin() + static_cast<std::ptrdiff_t>(first), rows.end(), verdict.row, &store::StoredMessage::row_id);
        if (it == rows.end() || it->from_self || chat::is_notice(*it) || verdict.to == "unclear") continue;
        const auto index = static_cast<usize>(it - rows.begin());
        const auto x = chat::line_features(rows, edges, index);
        std::vector<double> values(x.begin(), x.end());
        auto added = store.add_addressee_label({.chat = chat_key,
                                                .message_id = it->message_id,
                                                .row_id = it->row_id,
                                                .time = it->time,
                                                .feature_version = chat::k_feature_version,
                                                .features = glz::write_json(values).value_or("[]"),
                                                .label = verdict.to == "her",
                                                .source = "judge",
                                                .prior = predict(model, x),
                                                .addressee = verdict.to});
        if (added.has_error()) {
            log::error("cannot store an addressee label: {}", added.error().message());
            continue;
        }
        ++labelled;
    }
    for (const auto &verdict : hindsight->replies) {
        const auto it =
            std::ranges::find(rows.begin() + static_cast<std::ptrdiff_t>(first), rows.end(), verdict.row, &store::StoredMessage::row_id);
        if (it == rows.end() || !it->from_self) continue;
        double reward = reward_of(verdict.stance, verdict.contribution);
        // Someone quoting her message is a reaction too.
        for (usize i = static_cast<usize>(it - rows.begin()) + 1; i < rows.size(); ++i) {
            if (edges[i].quoted_row == it->row_id && !rows[i].from_self) {
                reward += k_quoted_reward;
                break;
            }
        }
        const auto hash = sticker_hash(*it);
        auto sticker = hash.empty() ? Result<std::optional<store::Sticker>>(std::optional<store::Sticker>{}) : store.sticker_by_hash(hash);
        const i64 sticker_id = sticker.has_value() && *sticker ? (*sticker)->id : 0;
        std::string to_her;
        if (turns.has_value()) {
            for (const auto &turn : *turns) {
                if (sent_in(turn.sent_message_ids, it->message_id)) to_her = turn.to_her;
            }
        }
        std::ignore = store.add_reply_effect({.chat = chat_key,
                                              .her_message_id = it->message_id,
                                              .target_message_id = targets_of[it->row_id],
                                              .kind = sticker_id != 0 ? "sticker" : "text",
                                              .sticker_id = sticker_id,
                                              .to_her = to_her,
                                              .for_her = verdict.for_her,
                                              .answered_by_other = verdict.answered_by_other,
                                              .stance = verdict.stance,
                                              .contribution = verdict.contribution,
                                              .reward = reward,
                                              .time = it->time});
    }
    std::ignore = store.set_feedback_progress(chat_key, rows.back().row_id);
    log::info("hindsight {}:{}: {} lines labelled, {} of her messages judged", chat_key.type, chat_key.id, labelled,
              hindsight->replies.size());
    labels_since_refit += labelled;
    if (labels_since_refit >= std::max<usize>(config.refit_every, 1)) refit();
    co_return true;
}

void Feedback::refit() {
    labels_since_refit = 0;
    auto labels = store.addressee_labels(chat::k_feature_version);
    if (labels.has_error()) return;
    const auto fitted_models = fit_models(samples_of(*labels), config);
    const i64 fitted_at = now();
    const auto save = [&](const store::ChatKey &chat_key, const FittedModel &model) {
        std::ignore = store.save_addressee_model({.chat = chat_key,
                                                  .feature_version = chat::k_feature_version,
                                                  .weights = write_weights(model.weights),
                                                  .rows = static_cast<i64>(model.rows),
                                                  .positives = static_cast<i64>(model.positives),
                                                  .log_loss = model.log_loss,
                                                  .fitted_at = fitted_at});
    };
    if (fitted_models.global.fitted) {
        global = fitted_models.global.weights;
        global_fitted = true;
        save({}, fitted_models.global);
    }
    for (const auto &[chat_key, model] : fitted_models.groups) {
        if (!model.fitted) continue;
        models[chat_key] = model.weights;
        save(chat_key, model);
    }
    log::info("addressee models refitted from {} labels ({} chats with their own)", fitted_models.global.rows, models.size());
}

Feedback::Hint Feedback::hints(const store::ChatKey &chat_key, const std::vector<store::StoredMessage> &rows,
                               const std::vector<chat::LineEdges> &edges, i64 first_new_row) const {
    Hint hint;
    if (first_new_row <= 0) return hint;
    const auto &model = model_for(chat_key);
    std::string shown;
    for (usize i = 0; i < rows.size() && i < edges.size(); ++i) {
        const auto &row = rows[i];
        if (row.row_id < first_new_row || row.from_self || chat::is_notice(row)) continue;
        // Lines that say whom they are for need no estimate.
        if (edges[i].to_her || !edges[i].to.empty() || edges[i].to_all) continue;
        const double p = predict(model, chat::line_features(rows, edges, i));
        hint.logged += std::format("{}#{} {:.2f}", hint.logged.empty() ? "" : ", ", row.row_id, p);
        shown += std::format("{}#{} {}", shown.empty() ? "" : "，", row.row_id, level(p));
    }
    if (config.hint && !shown.empty()) {
        hint.shown = "根据这个群以往的情况估计的「像是在对她说」的可能（只是参考，可能不准，以聊天内容为准）：" + shown + "。";
    }
    return hint;
}

} // namespace lilia::feedback
