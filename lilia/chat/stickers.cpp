#include <lilia/chat/detail/responder.h>

#include <algorithm>
#include <format>

#include <lilia/log.h>
#include <lilia/net/codec.h>

// Stickers the planner sends itself: find_sticker shows it the closest few
// from her pool, send_sticker sends the one it picks.

namespace lilia::chat {

using lighter::Task;
using namespace detail;

namespace {

/// Stickers embedded per request when the pool is first searched.
constexpr usize k_embed_batch = 32;

/// How the query is put to an instruction-tuned embedding model.
constexpr std::string_view k_sticker_instruction = "根据想表达的情绪或反应，找出合适的表情包";

} // namespace

namespace detail {

bool stickers_ready(Responder *r) {
    if (!r->stickers) return false;
    auto pool = r->store.stickers("registered");
    return pool.has_value() && !pool->empty();
}

Task<std::vector<store::Sticker>> find_stickers(Responder *r, std::string feeling) {
    auto pool = r->stickers ? r->store.stickers("registered") : Result<std::vector<store::Sticker>>(std::vector<store::Sticker>{});
    if (pool.has_error() || pool->empty()) co_return std::vector<store::Sticker>{};
    const usize count = std::max<usize>(r->config.stickers.candidates, 1);
    if (r->memory) {
        // Stickers are embedded once per process (the pool changes slowly).
        auto &vectors = r->stickers->vectors;
        std::vector<const store::Sticker *> missing;
        for (const auto &sticker : *pool) {
            if (!vectors.contains(sticker.sha1)) missing.push_back(&sticker);
        }
        for (usize start = 0; start < missing.size(); start += k_embed_batch) {
            const usize end = std::min(missing.size(), start + k_embed_batch);
            std::vector<std::string> texts;
            for (usize i = start; i < end; ++i) texts.push_back(sticker::embedding_text(*missing[i]));
            auto embedded = co_await r->memory->embed(std::move(texts));
            if (embedded.has_error() || embedded->size() != end - start) {
                log::warn("stickers: embedding failed: {}", embedded.has_error() ? embedded.error().message() : "wrong number of vectors");
                break;
            }
            for (usize i = start; i < end; ++i) vectors[missing[i]->sha1] = std::move((*embedded)[i - start]);
        }
        const auto &instruction = r->memory->config.query_instruction;
        auto query = co_await r->memory->embed(
            {instruction.empty() ? feeling : std::format("Instruct: {}\nQuery: {}", k_sticker_instruction, feeling)});
        if (query.has_value() && !query->empty()) {
            if (auto ranked = sticker::rank(*pool, query->front(), vectors, count); !ranked.empty()) co_return ranked;
        }
    }
    co_return sticker::match(*std::move(pool), feeling, count, r->rng);
}

Task<bool> send_sticker(Responder *r, store::ChatKey chat, store::Sticker sticker) {
    const auto bytes = r->stickers ? r->stickers->read(sticker) : std::nullopt;
    if (!bytes) co_return false;
    onebot::Segment image{.type = "image"};
    image.data["file"] = "base64://" + net::base64_encode(*bytes);
    image.data["sub_type"] = i64{1};
    image.data["summary"] = "[动画表情]";
    const auto id = co_await send_message(r, chat, {std::move(image)});
    if (!id) co_return false;
    // The stored copy names the sticker instead of carrying its bytes.
    auto stored = r->store.add_message({
        .chat = chat,
        .message_id = *id,
        .sender_id = r->self.user_id,
        .sender_name = r->self_name(chat),
        .time = r->now(),
        .segments_json = std::format(R"([{{"type":"image","data":{{"file":"sticker:{}","sub_type":1}}}}])", sticker.sha1),
        .text = std::format("[表情: {}]", sticker.description),
        .from_self = true,
    });
    if (stored.has_error()) log::error("cannot record sent sticker: {}", stored.error().message());
    std::ignore = r->store.sticker_used(sticker.id, r->now());
    r->chats[chat].last_sticker = r->now();
    log::info("{} sent sticker {}: [{}] {}", describe(chat), sticker::handle(sticker), sticker.tags, sticker.description);
    co_return true;
}

} // namespace detail

} // namespace lilia::chat
