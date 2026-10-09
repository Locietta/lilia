"""End-to-end test: the real lilia binary against a fake NapCat (OneBot 11
forward websocket) and a fake Anthropic-compatible model endpoint.

Scenario:
  1. lilia connects, authenticates with the bearer token, calls get_login_info.
  2. An @-mention in a group that is not allow-listed is ignored entirely.
  3. An @-mention in the allowed group produces exactly one model call and one
     quoted send_group_msg reply.
  4. The fake NapCat drops the connection; lilia reconnects and identifies again.
Afterwards the SQLite database must hold the trigger, the bot's own reply and
one llm_calls row, and nothing from the ignored group.
"""

import asyncio
import base64
import hashlib
import json
import os
import sqlite3
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from websockets.asyncio.server import serve

ROOT = Path(__file__).resolve().parents[2]
SELF_ID = 10001
GROUP = 555
IGNORED_GROUP = 999
TRIGGER_ID = -12345
MEDIA_ID = 4242
TOKEN = "onebot-test-token"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + bytes(range(64))
VOICE_BYTES = b"#!AMR\n" + bytes(range(32))
API_KEY = "test-key"


def find_binary() -> Path:
    name = "lilia.exe" if os.name == "nt" else "lilia"
    candidates = [p for p in (ROOT / "build").glob(f"*/*/*/{name}") if p.is_file()]
    assert candidates, "build lilia first (pixi run build)"
    return max(candidates, key=lambda p: p.stat().st_mtime)


def sse(events):
    return "".join(
        f"event: {name}\ndata: {json.dumps(payload)}\n\n" for name, payload in events
    ).encode()


def text_block(index, text):
    return [
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": index}),
    ]


BROKEN_WRITER = "broken-writer"
WRITER_TIMEOUT = 2


def start_model_server():
    state = {"bodies": [], "errors": [], "gets": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            # Stands in for Tencent's media CDN.
            state["gets"].append(self.path)
            self.send_response(200)
            self.send_header("content-type", "image/png")
            self.send_header("content-length", str(len(PNG_BYTES)))
            self.end_headers()
            self.wfile.write(PNG_BYTES)

        def do_POST(self):
            try:
                assert self.path == "/v1/messages", f"unexpected path {self.path}"
                assert self.headers["x-api-key"] == API_KEY, "missing/wrong x-api-key"
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                state["bodies"].append(body)
            except AssertionError as error:
                state["errors"].append(str(error))
                self.send_error(500, str(error))
                return
            if body.get("model") == BROKEN_WRITER:
                # The primary writer is down; lilia must fall back.
                error = json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "type": "invalid_request_error",
                            "message": "writer is down",
                        },
                    }
                )
                self.send_response(400)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(error.encode())
                return
            tools = [tool.get("name") for tool in body.get("tools", [])]
            if "reply" in tools:
                data = planner_answer(body)
            elif "record_expressions" in tools:
                data = learner_answer()
            elif "review_expressions" in tools:
                data = review_answer()
            elif "record_jargon" in tools:
                data = tool_answer(
                    "record_jargon", {"items": [{"content": "这也行", "source": "2"}]}
                )
            elif has_image(body):
                data = sse(
                    [
                        message_start(6),
                        *text_block(0, VISION_TEXT),
                        *message_end("end_turn", 9),
                    ]
                )
            else:
                data = writer_answer()
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("request-id", "req_test")
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state


def message_start(input_tokens):
    return (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "model": "fake-model",
                "usage": {"input_tokens": input_tokens},
            },
        },
    )


def message_end(stop_reason, output_tokens):
    return [
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason},
                "usage": {"output_tokens": output_tokens},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]


def planner_answer(body):
    """Replies to the line that says 你好呀, by its #number."""
    log = body["messages"][-1]["content"][0]["text"]
    target = next(
        line.split(" ", 1)[0] for line in log.splitlines() if "你好呀" in line
    )
    arguments = {
        "target": target,
        "quote": False,
        "style": "short",
        "search": False,
        "intent": "打个招呼",
    }
    return sse(
        [
            message_start(5),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "reply",
                        "input": {},
                    },
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(arguments, ensure_ascii=False),
                    },
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            *message_end("tool_use", 3),
        ]
    )


VISION_TEXT = "一只橘猫趴在键盘上\n睡着了"


def has_image(body):
    return any(
        isinstance(message["content"], list)
        and any(block.get("type") == "image" for block in message["content"])
        for message in body["messages"]
    )


def learner_answer():
    """Two patterns the review sees, plus one citing a line that does not exist."""
    return tool_answer(
        "record_expressions",
        {
            "items": [
                {"situation": "觉得离谱", "style": '用"笑死 这也行"', "source": "2"},
                {"situation": "不想理人", "style": '用"别理我"', "source": "1"},
                {"situation": "编造的", "style": "不存在的行", "source": "7"},
            ]
        },
    )


def review_answer():
    """The review drops the second pattern."""
    return tool_answer("review_expressions", {"reject": ["2"]})


def tool_answer(name, arguments):
    return sse(
        [
            message_start(8),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": name,
                        "input": {},
                    },
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(arguments, ensure_ascii=False),
                    },
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            *message_end("tool_use", 4),
        ]
    )


def writer_answer():
    """Narration, a server-side search, then the answer."""
    return sse(
        [
            message_start(10),
            *text_block(0, "我搜一下。"),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "server_tool_use",
                        "id": "srvtoolu_1",
                        "name": "web_search",
                        "input": {},
                    },
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": '{"query": "minecraft"}',
                    },
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 1}),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 2,
                    "content_block": {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_1",
                        "content": [
                            {
                                "type": "web_search_result",
                                "url": "https://minecraft.wiki/",
                                "title": "Minecraft Wiki",
                                "encrypted_content": "x",
                            }
                        ],
                    },
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 2}),
            *text_block(3, "莉莉娅：**fake** reply。"),
            *message_end("end_turn", 12),
        ]
    )


def group_event(group_id, message_id, text, segments=None):
    return json.dumps(
        {
            "self_id": SELF_ID,
            "user_id": 20002,
            "time": int(time.time()),
            "message_id": message_id,
            "message_type": "group",
            "sub_type": "normal",
            "sender": {
                "user_id": 20002,
                "nickname": "魔王",
                "card": "",
                "role": "member",
            },
            "message": segments
            or [
                {"type": "at", "data": {"qq": str(SELF_ID), "name": "lilia"}},
                {"type": "text", "data": {"text": text}},
            ],
            "raw_message": text,
            "message_format": "array",
            "post_type": "message",
            "group_id": group_id,
            "group_name": "测试群",
        },
        ensure_ascii=False,
    )


class FakeNapCat:
    def __init__(self, media_port):
        self.media_port = media_port
        self.port = None
        self.file_requests = []
        self.connections = 0
        self.errors = []
        self.sent = []
        self.ready = threading.Event()
        self.reconnected = threading.Event()
        self.loop = None
        self.stop = None

    async def answer_login(self, ws):
        request = json.loads(await asyncio.wait_for(ws.recv(), 20))
        assert request["action"] == "get_login_info", request
        await ws.send(
            json.dumps(
                {
                    "status": "ok",
                    "retcode": 0,
                    "data": {"user_id": SELF_ID, "nickname": "lilia"},
                    "echo": request["echo"],
                }
            )
        )

    async def handler(self, ws):
        try:
            if ws.request.headers.get("Authorization") != f"Bearer {TOKEN}":
                self.errors.append("missing bearer token")
                await ws.close(1008)
                return
            self.connections += 1
            await self.answer_login(ws)
            if self.connections > 1:
                self.reconnected.set()
                await ws.wait_closed()
                return

            # lilia only handles messages once it knows its own id; give the
            # login response a moment to be processed.
            await asyncio.sleep(0.3)
            media = f"http://127.0.0.1:{self.media_port}"
            ignored = [
                {"type": "at", "data": {"qq": str(SELF_ID)}},
                {"type": "text", "data": {"text": " 别理我"}},
                {
                    "type": "image",
                    "data": {"file": "s.png", "url": f"{media}/secret.png"},
                },
            ]
            await ws.send(group_event(IGNORED_GROUP, 1, "", ignored))
            # A second line from the ignored group completes a learning batch
            # (batch_messages is 2): lilia learns from it but stays read-only.
            await ws.send(
                group_event(
                    IGNORED_GROUP,
                    2,
                    "",
                    [{"type": "text", "data": {"text": "笑死 这也行"}}],
                )
            )
            # Media without a trigger: archived, not answered. The picture has a
            # direct url; the voice clip must be resolved through get_file.
            await ws.send(
                group_event(
                    GROUP,
                    MEDIA_ID,
                    "",
                    [
                        {
                            "type": "image",
                            "data": {
                                "file": "p.png",
                                "url": f"{media}/img.png",
                                "sub_type": 0,
                            },
                        },
                        {"type": "record", "data": {"file": "voice.amr"}},
                    ],
                )
            )
            await ws.send(group_event(GROUP, TRIGGER_ID, " 你好呀"))

            # Answer requests in whatever order lilia sends them.
            while not (self.sent and self.file_requests):
                request = json.loads(await asyncio.wait_for(ws.recv(), 20))
                if request["action"] == "send_group_msg":
                    self.sent.append(request)
                    data = {"message_id": 777}
                elif request["action"] == "get_file":
                    self.file_requests.append(request)
                    data = {
                        "file": "/app/.config/QQ/nt_qq_t/nt_data/Ptt/voice.amr",
                        "file_name": "voice.amr",
                    }
                else:
                    self.errors.append(f"unexpected action {request['action']}")
                    data = None
                await ws.send(
                    json.dumps(
                        {
                            "status": "ok",
                            "retcode": 0,
                            "data": data,
                            "echo": request["echo"],
                        }
                    )
                )  # No second reply may follow (the ignored group must stay silent).
            try:
                extra = await asyncio.wait_for(ws.recv(), 2)
                self.errors.append(f"unexpected extra request: {extra}")
            except asyncio.TimeoutError:
                pass
            await ws.close(1001)
        except Exception as error:  # noqa: BLE001 - recorded for the assertion below
            self.errors.append(repr(error))

    async def main(self):
        self.loop = asyncio.get_running_loop()
        self.stop = self.loop.create_future()
        async with serve(self.handler, "127.0.0.1", 0) as server:
            self.port = server.sockets[0].getsockname()[1]
            self.ready.set()
            await self.stop

    def start(self):
        threading.Thread(target=lambda: asyncio.run(self.main()), daemon=True).start()
        assert self.ready.wait(10), "fake NapCat did not start"

    def shutdown(self):
        if self.loop and self.stop:
            self.loop.call_soon_threadsafe(
                lambda: self.stop.done() or self.stop.set_result(None)
            )


def test_reply_and_reconnect(tmp_path):
    model, model_state = start_model_server()
    napcat = FakeNapCat(model.server_port)
    napcat.start()

    data_dir = tmp_path / "data"
    qq_dir = tmp_path / "qq"
    voice_path = qq_dir / "nt_qq_t" / "nt_data" / "Ptt" / "voice.amr"
    voice_path.parent.mkdir(parents=True)
    voice_path.write_bytes(VOICE_BYTES)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "onebot": {
                    "port": napcat.port,
                    "access_token": "$LILIA_TEST_ONEBOT_TOKEN",
                },
                "model": {
                    "base_url": f"http://127.0.0.1:{model.server_port}",
                    "model": BROKEN_WRITER,
                    "writer_fallbacks": [{"model": "fake-model"}],
                    "writer_timeout_seconds": WRITER_TIMEOUT,
                    "api_key": "$LILIA_TEST_KEY",
                },
                "chat": {"groups": [GROUP], "debounce_seconds": 0, "typing_speed": 0},
                "learning": {"batch_messages": 2},
                "archive": {
                    "container_qq_dir": "/app/.config/QQ",
                    "host_qq_dir": str(qq_dir),
                },
                "data_dir": str(data_dir),
                "log_level": "debug",
            }
        ),
        encoding="utf-8",
    )
    env = dict(os.environ, LILIA_TEST_ONEBOT_TOKEN=TOKEN, LILIA_TEST_KEY=API_KEY)
    log_path = tmp_path / "lilia.log"
    with open(log_path, "wb") as log:
        process = subprocess.Popen(
            [str(find_binary()), "--config", str(config)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            reconnected = napcat.reconnected.wait(45)
        finally:
            process.terminate()
            process.wait(20)
            napcat.shutdown()
            model.shutdown()

    output = log_path.read_text(encoding="utf-8", errors="replace")
    assert reconnected, f"lilia did not reconnect\n{output}"
    assert not napcat.errors, f"{napcat.errors}\n{output}"
    assert not model_state["errors"], model_state["errors"]

    def offered(body):
        return [tool.get("name") for tool in body.get("tools", [])]

    learners = [b for b in model_state["bodies"] if "record_expressions" in offered(b)]
    reviews = [b for b in model_state["bodies"] if "review_expressions" in offered(b)]
    jargons = [b for b in model_state["bodies"] if "record_jargon" in offered(b)]
    visions = [b for b in model_state["bodies"] if has_image(b)]
    turns = [
        b
        for b in model_state["bodies"]
        if b not in learners
        and b not in reviews
        and b not in jargons
        and b not in visions
    ]
    assert len(visions) == 1, "the picture is described once; voice is not an image"
    image = visions[0]["messages"][0]["content"][0]
    assert image["source"]["media_type"] == "image/png"
    assert base64.b64decode(image["source"]["data"]) == PNG_BYTES
    assert len(learners) == 1, "one learning batch from the ignored group"
    assert len(reviews) == 1, "the extracted patterns are reviewed before storing"
    assert len(jargons) == 1, "the same batch is mined for slang"
    reviewed = reviews[0]["messages"][-1]["content"][0]["text"]
    assert (
        reviewed
        == '1. 当觉得离谱时，可以用"笑死 这也行"\n2. 当不想理人时，可以用"别理我"\n'
    ), reviewed
    learned_from = learners[0]["messages"][-1]["content"][0]["text"]
    assert learned_from == "1. 群友1: 别理我\n2. 群友1: 笑死 这也行\n", learned_from
    assert len(turns) == 3, "one planner call, the broken writer, then its fallback"
    planner, broken, writer = turns
    assert broken["model"] == BROKEN_WRITER and writer["model"] == "fake-model"
    assert broken["messages"] == writer["messages"], (
        "the fallback gets the same request"
    )
    assert [tool["name"] for tool in planner["tools"]] == ["reply", "stay_silent"]
    assert (
        "莉莉娅" in planner["system"]
        and "测试群" in planner["messages"][-1]["content"][0]["text"]
    )
    assert planner.get("thinking") == {"type": "disabled"}, "thinking is off by default"
    assert planner.get("tool_choice") == {"type": "any"}, "the planner must call a tool"
    log = planner["messages"][-1]["content"][0]["text"]
    assert "你好呀" in log and "[图片: 一只橘猫趴在键盘上 睡着了][语音]" in log, (
        "the turn waits for the description and sees it"
    )
    assert "别理我" not in log
    assert "tools" not in writer, "no search unless the planner asks for it"
    assert "莉莉娅" in writer["system"] and "测试群" in writer["system"]
    last = writer["messages"][-1]
    texts = "".join(b.get("text", "") for b in last["content"] if b["type"] == "text")
    assert "你好呀" in texts and "打个招呼" in texts and "别理我" not in texts
    assert len(napcat.sent) == 1
    params = napcat.sent[0]["params"]
    assert params["group_id"] == GROUP
    assert params["message"] == [{"type": "text", "data": {"text": "fake reply"}}], (
        "no quote; narration, speaker label, Markdown and the final period are dropped"
    )

    db = sqlite3.connect(data_dir / "lilia.db")
    rows = db.execute(
        "SELECT chat_id, message_id, from_self, text FROM messages ORDER BY id"
    ).fetchall()
    assert rows == [
        (GROUP, MEDIA_ID, 0, "[图片: 一只橘猫趴在键盘上 睡着了][语音]"),
        (GROUP, TRIGGER_ID, 0, "@lilia 你好呀"),
        (GROUP, 777, 1, "fake reply"),
    ], rows
    learned = db.execute(
        "SELECT chat_id, situation, style, count FROM expressions"
    ).fetchall()
    assert learned == [(IGNORED_GROUP, "觉得离谱", '用"笑死 这也行"', 1)], learned
    verdicts = db.execute(
        "SELECT style, verdict FROM expression_reviews ORDER BY id"
    ).fetchall()
    assert verdicts == [('用"笑死 这也行"', "kept"), ('用"别理我"', "rejected")], (
        verdicts
    )
    slang = db.execute("SELECT content, count, is_jargon FROM jargon").fetchall()
    assert slang == [("这也行", 1, -1)], slang
    evidence = db.execute("SELECT snippet FROM jargon_evidence").fetchall()
    assert evidence == [("群友1: 别理我\n群友1: 笑死 这也行\n",)], evidence
    assert db.execute(
        "SELECT count(*) FROM llm_calls WHERE chat_type = 'learn'"
    ).fetchone() == (3,), "learning, review and slang calls"
    calls = db.execute(
        "SELECT chat_id, model, input_tokens, output_tokens, error != '' FROM llm_calls "
        "WHERE chat_type = 'group' ORDER BY id"
    ).fetchall()
    assert calls == [
        # The model column is what the response reports; the fake says fake-model.
        (GROUP, "fake-model", 6, 9, 0),
        (GROUP, "fake-model", 5, 3, 0),
        (GROUP, BROKEN_WRITER, 0, 0, 1),
        (GROUP, "fake-model", 10, 12, 0),
    ], ("vision, planner, failed writer, fallback writer", calls)

    media = db.execute(
        "SELECT chat_id, message_id, segment_index, kind, status, sha1, path, bytes, error, description "
        "FROM media ORDER BY segment_index"
    ).fetchall()
    db.close()
    png_sha1 = hashlib.sha1(PNG_BYTES).hexdigest()
    voice_sha1 = hashlib.sha1(VOICE_BYTES).hexdigest()
    assert media == [
        (
            GROUP,
            MEDIA_ID,
            0,
            "image",
            "stored",
            png_sha1,
            f"media/{png_sha1[:2]}/{png_sha1}.png",
            len(PNG_BYTES),
            "",
            "一只橘猫趴在键盘上 睡着了",
        ),
        (
            GROUP,
            MEDIA_ID,
            1,
            "voice",
            "stored",
            voice_sha1,
            f"media/{voice_sha1[:2]}/{voice_sha1}.amr",
            len(VOICE_BYTES),
            "",
            "",
        ),
    ], media
    assert (data_dir / media[0][6]).read_bytes() == PNG_BYTES
    assert (data_dir / media[1][6]).read_bytes() == VOICE_BYTES
    assert napcat.file_requests[0]["params"] == {"file": "voice.amr"}
    assert model_state["gets"] == ["/img.png"], "only allowed-group media is fetched"
