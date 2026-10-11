#include <lilia/chat/detail/responder.h>

#include <algorithm>
#include <charconv>
#include <format>

#include <glaze/json.hpp>

#include <lilia/log.h>
#include <lilia/text.h>

// Notices of allowed groups as lines in the chat log; recalls and mutes.

namespace lilia::chat {

using lighter::Task;
using namespace detail;

namespace {

/// "小明的「前十五个字…」", the message a notice is about; "一条消息" when
/// it is not stored.
std::string message_ref(Responder *r, const store::ChatKey &chat, i64 message_id) {
    auto found = r->store.message(chat, message_id);
    if (!found.has_value() || !*found) return "一条消息";
    const auto &message = **found;
    auto text = bounded_utf8(message.text, 45);
    if (text.size() < message.text.size()) text += "…";
    return std::format("{}的「{}」", message.from_self ? r->self_name(chat) : person(r, chat, message.sender_id), text);
}

/// Stores a notice line (who did it, when known, in its segments: a poke can
/// be returned); returns it.
store::StoredMessage record_notice(Responder *r, const store::ChatKey &chat, i64 time, std::string text, i64 actor = 0) {
    store::StoredMessage line{
        .chat = chat,
        .message_id = r->next_notice_id--,
        .sender_id = 0,
        .sender_name = std::string(k_notice_sender),
        .time = time > 0 ? time : r->now(),
        .segments_json = actor ? std::format(R"([{{"type":"notice","data":{{"user_id":{}}}}}])", actor) : std::string("[]"),
        .text = std::move(text),
    };
    if (auto added = r->store.add_message(line); added.has_error()) log::error("cannot store a notice: {}", added.error().message());
    return line;
}

/// "15分钟" / "2小时" / "3天" for a mute duration in seconds.
std::string duration_text(i64 seconds) {
    if (seconds >= 86'400) return std::format("{}天", seconds / 86'400);
    if (seconds >= 3'600) return std::format("{}小时", seconds / 3'600);
    return std::format("{}分钟", std::max<i64>(seconds / 60, 1));
}

/// "[赞]" etc. for the faces of an emoji reaction; "表情" when none is known.
std::string reaction_faces(const std::optional<glz::generic_i64> &likes) {
    std::string out;
    if (likes && likes->is_array()) {
        for (const auto &like : likes->get_array()) {
            if (!like.is_object()) continue;
            const auto &object = like.get_object();
            const auto it = object.find("emoji_id");
            if (it == object.end()) continue;
            i64 id = -1;
            if (it->second.is_int64()) {
                id = it->second.get<i64>();
            } else if (it->second.is_string()) {
                const auto &text = it->second.get_string();
                std::from_chars(text.data(), text.data() + text.size(), id);
            }
            if (const auto name = onebot::face_name(id); !name.empty()) out += std::format("[{}]", name);
        }
    }
    return out.empty() ? std::string("表情") : out;
}

} // namespace

namespace detail {

std::string person(Responder *r, const store::ChatKey &chat, i64 user) {
    if (user == r->self.user_id) return r->self_name(chat);
    auto name = r->store.sender_name(chat, user);
    return name.has_value() && !name->empty() ? *name : std::to_string(user);
}

i64 notice_actor(const store::StoredMessage &line) {
    onebot::Message segments;
    if (glz::read<onebot::k_read_opts>(segments, line.segments_json) || segments.empty()) return 0;
    return onebot::data_integer(segments.front(), "user_id").value_or(0);
}

} // namespace detail

void Responder::recall(const store::ChatKey &chat, i64 message_id) {
    if (auto updated = store.update_message_text(chat, message_id, onebot::k_recalled_text); updated.has_error()) {
        log::error("cannot mark a recalled message: {}", updated.error().message());
    }
    auto &flow = chats[chat].flow;
    std::erase_if(flow.unseen, [&](const store::StoredMessage &message) { return message.message_id == message_id; });
    if (flow.addressed == message_id) flow.addressed.reset();
    if (flow.named == message_id) flow.named.reset();
    log::info("{}: message {} was recalled", describe(chat), message_id);
}

void Responder::notice(const onebot::NoticeEvent &notice) {
    if (!notice.group_id || !std::ranges::contains(config.chat.groups, *notice.group_id)) return;
    const store::ChatKey chat{.type = "group", .id = *notice.group_id};
    auto &state = chats[chat];
    const auto &type = notice.notice_type;
    const auto &sub = notice.sub_type;
    const auto operator_name = [&] { return person(this, chat, notice.operator_id.value_or(0)); };
    if (type == "notify" && sub == "poke" && notice.target_id) {
        auto line = record_notice(this, chat, notice.time,
                                  std::format("{}戳了戳{}", person(this, chat, notice.user_id), person(this, chat, *notice.target_id)),
                                  notice.user_id);
        // Poked: like an @, the planner looks at once (and may poke back).
        if (notice.target_id == self.user_id && notice.user_id != self.user_id && state.muted_until <= now()) {
            state.flow.addressed = line.message_id;
            state.flow.unseen.push_back(std::move(line));
            wake(chat);
        }
    } else if (type == "group_recall" && notice.message_id) {
        recall(chat, *notice.message_id);
    } else if (type == "group_ban") {
        const bool everyone = notice.user_id == 0;
        const bool lifted = sub == "lift_ban";
        if (everyone || notice.user_id == self.user_id) {
            // A whole-group mute has no duration; a day bounds a missed lift.
            const i64 seconds = everyone ? 86'400 : notice.duration.value_or(0);
            state.muted_until = lifted ? 0 : now() + seconds;
            log::info("{}: {}", describe(chat), lifted ? "she can speak again" : "she is muted");
        }
        record_notice(this, chat, notice.time,
                      everyone ? std::format("{}{}了全员禁言", operator_name(), lifted ? "解除" : "开启") :
                      lifted   ? std::format("{}解除了{}的禁言", operator_name(), person(this, chat, notice.user_id)) :
                                 std::format("{}把{}禁言了{}", operator_name(), person(this, chat, notice.user_id),
                                             duration_text(notice.duration.value_or(0))));
    } else if (type == "group_msg_emoji_like" && notice.message_id && notice.is_add.value_or(true)) {
        record_notice(this, chat, notice.time,
                      std::format("{}给{}贴了{}", person(this, chat, notice.user_id), message_ref(this, chat, *notice.message_id),
                                  reaction_faces(notice.likes)));
    } else if (type == "essence" && sub == "add" && notice.message_id) {
        record_notice(this, chat, notice.time,
                      std::format("{}把{}设为了精华消息", operator_name(), message_ref(this, chat, *notice.message_id)));
    } else if (type == "group_increase") {
        record_notice(this, chat, notice.time, std::format("{}加入了群聊", person(this, chat, notice.user_id)));
    } else if (type == "group_decrease" && sub != "kick_me") {
        record_notice(this, chat, notice.time,
                      std::format("{}{}了群聊", person(this, chat, notice.user_id), sub == "kick" ? "被移出" : "退出"));
    }
}

} // namespace lilia::chat
