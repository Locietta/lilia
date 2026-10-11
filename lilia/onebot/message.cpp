#include "message.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <format>
#include <utility>
#include <vector>

#include <lilia/net/codec.h>
#include <lilia/text.h>

namespace lilia::onebot {

namespace {

/// Forward previews: this many nodes, each line and the whole bounded (bytes).
constexpr usize k_forward_nodes = 6;
constexpr usize k_forward_line_bytes = 90;
constexpr usize k_forward_bytes = 600;
constexpr usize k_card_bytes = 240;

/// QQ system faces people actually send, by id. Ids missing here render as
/// a bare "[表情]".
constexpr auto k_faces = std::to_array<std::pair<i64, std::string_view>>({
    {0, "惊讶"},     {1, "撇嘴"},       {2, "色"},       {3, "发呆"},       {4, "得意"},       {5, "流泪"},       {6, "害羞"},
    {7, "闭嘴"},     {8, "睡"},         {9, "大哭"},     {10, "尴尬"},      {11, "发怒"},      {12, "调皮"},      {13, "呲牙"},
    {14, "微笑"},    {15, "难过"},      {16, "酷"},      {18, "抓狂"},      {19, "吐"},        {20, "偷笑"},      {21, "可爱"},
    {22, "白眼"},    {23, "傲慢"},      {24, "饥饿"},    {25, "困"},        {26, "惊恐"},      {27, "流汗"},      {28, "憨笑"},
    {29, "悠闲"},    {30, "奋斗"},      {31, "咒骂"},    {32, "疑问"},      {33, "嘘"},        {34, "晕"},        {35, "折磨"},
    {36, "衰"},      {37, "骷髅"},      {38, "敲打"},    {39, "再见"},      {41, "发抖"},      {42, "爱情"},      {43, "跳跳"},
    {46, "猪头"},    {49, "拥抱"},      {53, "蛋糕"},    {59, "便便"},      {60, "咖啡"},      {63, "玫瑰"},      {64, "凋谢"},
    {66, "爱心"},    {67, "心碎"},      {74, "太阳"},    {75, "月亮"},      {76, "赞"},        {77, "踩"},        {78, "握手"},
    {79, "胜利"},    {85, "飞吻"},      {86, "怄火"},    {89, "西瓜"},      {96, "冷汗"},      {97, "擦汗"},      {98, "抠鼻"},
    {99, "鼓掌"},    {100, "糗大了"},   {101, "坏笑"},   {102, "左哼哼"},   {103, "右哼哼"},   {104, "哈欠"},     {105, "鄙视"},
    {106, "委屈"},   {107, "快哭了"},   {108, "阴险"},   {109, "左亲亲"},   {110, "吓"},       {111, "可怜"},     {112, "菜刀"},
    {114, "篮球"},   {116, "示爱"},     {118, "抱拳"},   {119, "勾引"},     {120, "拳头"},     {121, "差劲"},     {123, "NO"},
    {124, "OK"},     {125, "转圈"},     {129, "挥手"},   {144, "喝彩"},     {147, "棒棒糖"},   {171, "茶"},       {173, "泪奔"},
    {174, "无奈"},   {175, "卖萌"},     {176, "小纠结"}, {177, "喷血"},     {178, "斜眼笑"},   {179, "doge"},     {181, "戳一戳"},
    {182, "笑哭"},   {183, "我最美"},   {201, "点赞"},   {212, "托腮"},     {262, "脑阔疼"},   {264, "捂脸"},     {265, "辣眼睛"},
    {266, "哦哟"},   {267, "头秃"},     {268, "问号脸"}, {269, "暗中观察"}, {270, "emm"},      {271, "吃瓜"},     {272, "呵呵哒"},
    {273, "我酸了"}, {277, "汪汪"},     {281, "无眼笑"}, {282, "敬礼"},     {284, "面无表情"}, {285, "摸鱼"},     {287, "哦"},
    {289, "睁眼"},   {293, "摸锦鲤"},   {294, "期待"},   {297, "拜谢"},     {299, "牛啊"},     {305, "右亲亲"},   {306, "牛气冲天"},
    {307, "喵喵"},   {314, "仔细分析"}, {318, "崇拜"},   {319, "比心"},     {320, "庆祝"},     {323, "嫌弃"},     {324, "吃糖"},
    {325, "惊吓"},   {326, "生气"},     {332, "举牌牌"}, {333, "烟花"},     {337, "花朵脸"},   {338, "我想开了"}, {339, "舔屏"},
    {341, "打招呼"}, {342, "酸Q"},      {343, "我方了"}, {344, "大怨种"},   {345, "红包多多"}, {346, "你真棒棒"}, {349, "坚强"},
    {350, "贴贴"},   {351, "敲敲"},     {352, "咦"},     {353, "拜托"},     {354, "尊嘟假嘟"}, {355, "耶"},       {356, "666"},
    {357, "裂开"},
});
static_assert(std::ranges::is_sorted(k_faces, {}, &std::pair<i64, std::string_view>::first), "face_name binary-searches by id");

const glz::generic_i64 *find_field(const glz::generic_i64 &value, std::string_view key) {
    if (!value.is_object()) return nullptr;
    const auto &object = value.get_object();
    const auto it = object.find(key);
    return it == object.end() ? nullptr : &it->second;
}

const glz::generic_i64 *find_field(const Segment &segment, std::string_view key) { return find_field(segment.data, key); }

std::string string_of(const glz::generic_i64 *value) {
    if (!value) return {};
    if (value->is_string()) return value->get_string();
    if (value->is_int64()) return std::to_string(value->get<i64>());
    return {};
}

/// A field of a parsed card, as text; base64 is undone when the card says so
/// (group announcements set "encode": 1).
std::string card_text(const glz::generic &object, std::string_view key, bool encoded) {
    if (!object.is_object()) return {};
    const auto &map = object.get_object();
    const auto it = map.find(key);
    if (it == map.end() || !it->second.is_string()) return {};
    const auto &text = it->second.get_string();
    if (!encoded) return text;
    auto decoded = net::base64_decode(text);
    return decoded ? *std::move(decoded) : text;
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

std::string_view face_name(i64 id) {
    const auto it = std::ranges::lower_bound(k_faces, id, {}, &std::pair<i64, std::string_view>::first);
    return it != k_faces.end() && it->first == id ? it->second : std::string_view{};
}

std::string render_card(std::string_view card_json) {
    glz::generic card;
    if (glz::read<k_read_opts>(card, card_json) || !card.is_object()) return "[卡片]";
    std::vector<std::string> parts;
    const auto add = [&parts](std::string text) {
        while (!text.empty() && (text.back() == ' ' || text.back() == '\n')) text.pop_back();
        if (text.empty()) return;
        // "[QQ小程序]哔哩哔哩" and a title "哔哩哔哩" say the same thing once.
        if (std::ranges::any_of(parts, [&](const std::string &part) { return part.contains(text); })) return;
        std::erase_if(parts, [&](const std::string &part) { return text.contains(part); });
        parts.push_back(std::move(text));
    };
    add(card_text(card, "prompt", false));
    const auto &object = card.get_object();
    if (const auto meta = object.find("meta"); meta != object.end() && meta->second.is_object()) {
        // One entry named after the card kind: detail_1, news, music, mannounce...
        for (const auto &[kind, detail] : meta->second.get_object()) {
            if (!detail.is_object()) continue;
            const auto &fields = detail.get_object();
            const auto encode = fields.find("encode");
            const bool encoded = encode != fields.end() && encode->second.is_number() && encode->second.get_number() == 1;
            add(card_text(detail, "title", encoded));
            auto description = card_text(detail, "desc", encoded);
            add(description.empty() ? card_text(detail, "text", encoded) : std::move(description));
            break;
        }
    }
    if (parts.empty()) return "[卡片]";
    std::string joined;
    for (const auto &part : parts) joined += (joined.empty() ? "" : "｜") + part;
    std::ranges::replace(joined, '\n', ' ');
    return "[卡片: " + bounded_utf8(joined, k_card_bytes) + "]";
}

const glz::generic_i64 *forward_nodes(const Segment &segment) {
    const auto *content = find_field(segment, "content");
    return content && content->is_array() && !content->get_array().empty() ? content : nullptr;
}

std::string render_forward(const glz::generic_i64 &nodes) {
    const auto *list = &nodes;
    if (const auto *messages = find_field(nodes, "messages")) list = messages;
    if (!list->is_array()) return {};
    std::string out;
    usize shown = 0;
    for (const auto &node : list->get_array()) {
        if (shown == k_forward_nodes || out.size() >= k_forward_bytes) break;
        if (!node.is_object()) continue;
        // NapCat nodes are messages (sender + message); go-cqhttp-style ones
        // carry nickname + content.
        std::string name;
        if (const auto *sender = find_field(node, "sender")) {
            name = string_of(find_field(*sender, "card"));
            if (name.empty()) name = string_of(find_field(*sender, "nickname"));
        }
        if (name.empty()) name = string_of(find_field(node, "nickname"));
        if (name.empty()) name = string_of(find_field(node, "name"));
        const auto *body = find_field(node, "message");
        if (!body) body = find_field(node, "content");
        std::string text;
        if (body && body->is_array()) {
            Message message;
            for (const auto &part : body->get_array()) {
                const auto *type = find_field(part, "type");
                const auto *data = find_field(part, "data");
                if (!type || !type->is_string()) continue;
                message.push_back({.type = type->get_string(), .data = data ? *data : glz::generic_i64{}});
            }
            text = render_plain(message);
        } else {
            text = string_of(body);
        }
        std::ranges::replace(text, '\n', ' ');
        if (text.empty()) continue;
        if (!out.empty()) out += " / ";
        out += (name.empty() ? std::string("某人") : name) + ": " + bounded_utf8(text, k_forward_line_bytes);
        ++shown;
    }
    if (out.empty()) return {};
    return std::format("{}（共{}条）", bounded_utf8(out, k_forward_bytes), list->get_array().size());
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
        if (described != descriptions.end() && type == "forward") {
            out += "[聊天记录: " + described->second + "]";
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
            const auto name = face_name(data_integer(segment, "id").value_or(-1));
            out += name.empty() ? std::string("[表情]") : "[表情: " + std::string(name) + "]";
        } else if (type == "record") {
            out += "[语音]";
        } else if (type == "video") {
            out += "[视频]";
        } else if (type == "file") {
            out += bracketed("文件", data_string(segment, "file"));
        } else if (type == "forward") {
            const auto *nodes = forward_nodes(segment);
            const auto preview = nodes ? render_forward(*nodes) : std::string();
            out += preview.empty() ? std::string("[聊天记录]") : "[聊天记录: " + preview + "]";
        } else if (type == "json") {
            out += render_card(data_string(segment, "data").value_or(""));
        } else {
            out += "[" + type + "]";
        }
    }
    return out;
}

} // namespace lilia::onebot
