#include "message.h"

#include <charconv>
#include <cmath>
#include <utility>

namespace lilia::onebot {

namespace {

const glz::generic_i64 *find_field(const Segment &segment, std::string_view key) {
    if (!segment.data.is_object()) return nullptr;
    const auto &object = segment.data.get_object();
    const auto it = object.find(key);
    return it == object.end() ? nullptr : &it->second;
}

std::string bracketed(std::string_view label, std::optional<std::string> summary) {
    std::string out = "[";
    out += label;
    // NapCat wraps sticker summaries in brackets already, e.g. "[动画表情]".
    if (summary && !summary->empty() && *summary != "[" + std::string(label) + "]") {
        std::string_view inner = *summary;
        if (inner.starts_with('[') && inner.ends_with(']')) inner = inner.substr(1, inner.size() - 2);
        out += ": ";
        out += inner;
    }
    out += "]";
    return out;
}

} // namespace

std::optional<std::string> data_string(const Segment &segment, std::string_view key) {
    const auto *field = find_field(segment, key);
    if (!field) return std::nullopt;
    if (field->is_string()) return field->get_string();
    if (field->is_int64()) return std::to_string(field->get<i64>());
    if (field->is_number()) {
        const double value = field->as_number();
        if (std::isfinite(value) && value == std::trunc(value)) return std::to_string(static_cast<i64>(value));
        return std::to_string(value);
    }
    return std::nullopt;
}

std::optional<i64> data_integer(const Segment &segment, std::string_view key) {
    const auto text = data_string(segment, key);
    if (!text) return std::nullopt;
    i64 value = 0;
    const auto [end, ec] = std::from_chars(text->data(), text->data() + text->size(), value);
    if (ec != std::errc{} || end != text->data() + text->size()) return std::nullopt;
    return value;
}

Segment text_segment(std::string text) {
    Segment segment{.type = "text"};
    segment.data["text"] = std::move(text);
    return segment;
}

Segment at_segment(i64 user_id) {
    Segment segment{.type = "at"};
    segment.data["qq"] = std::to_string(user_id);
    return segment;
}

Segment reply_segment(i64 message_id) {
    Segment segment{.type = "reply"};
    segment.data["id"] = std::to_string(message_id);
    return segment;
}

bool mentions(const Message &message, i64 user_id, bool include_all) {
    for (const auto &segment : message) {
        if (segment.type != "at") continue;
        const auto target = data_string(segment, "qq");
        if (!target) continue;
        if (*target == std::to_string(user_id)) return true;
        if (include_all && *target == "all") return true;
    }
    return false;
}

std::optional<i64> reply_target(const Message &message) {
    for (const auto &segment : message) {
        if (segment.type == "reply") return data_integer(segment, "id");
    }
    return std::nullopt;
}

std::string text_only(const Message &message) {
    std::string out;
    for (const auto &segment : message) {
        if (segment.type == "text") out += data_string(segment, "text").value_or("");
    }
    return out;
}

std::string render_plain(const Message &message) { return render_plain(message, {}); }

std::string render_plain(const Message &message, const std::map<usize, std::string> &descriptions) {
    std::string out;
    for (usize index = 0; index < message.size(); ++index) {
        const auto &segment = message[index];
        const auto &type = segment.type;
        const auto described = descriptions.find(index);
        if (described != descriptions.end() && (type == "image" || type == "mface")) {
            const bool sticker = type == "mface" || data_integer(segment, "sub_type").value_or(0) == 1;
            out += std::string(sticker ? "[表情: " : "[图片: ") + described->second + "]";
            continue;
        }
        if (type == "text") {
            out += data_string(segment, "text").value_or("");
        } else if (type == "at") {
            const auto target = data_string(segment, "qq").value_or("?");
            const auto name = data_string(segment, "name");
            out += "@";
            out += target == "all" ? std::string("全体成员") : name && !name->empty() ? *name : target;
        } else if (type == "reply") {
            continue;
        } else if (type == "image") {
            // sub_type 1 marks a sticker sent from the emoji panel.
            const bool sticker = data_integer(segment, "sub_type").value_or(0) == 1;
            out += bracketed(sticker ? "表情" : "图片", data_string(segment, "summary"));
        } else if (type == "mface") {
            out += bracketed("表情", data_string(segment, "summary"));
        } else if (type == "face") {
            out += "[表情]";
        } else if (type == "record") {
            out += "[语音]";
        } else if (type == "video") {
            out += "[视频]";
        } else if (type == "file") {
            out += bracketed("文件", data_string(segment, "file"));
        } else if (type == "forward") {
            out += "[聊天记录]";
        } else if (type == "json") {
            out += "[卡片]";
        } else {
            out += "[" + type + "]";
        }
    }
    return out;
}

} // namespace lilia::onebot
