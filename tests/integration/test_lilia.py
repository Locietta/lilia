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
TOKEN = "onebot-test-token"
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


def start_model_server():
    state = {"bodies": [], "errors": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

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
            data = sse(
                [
                    (
                        "message_start",
                        {
                            "type": "message_start",
                            "message": {
                                "id": "msg_1",
                                "model": "fake-model",
                                "usage": {"input_tokens": 10},
                            },
                        },
                    ),
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
                    *text_block(3, "lilia：**fake** reply"),
                    (
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {"stop_reason": "end_turn"},
                            "usage": {"output_tokens": 12},
                        },
                    ),
                    ("message_stop", {"type": "message_stop"}),
                ]
            )
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("request-id", "req_test")
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state


def group_event(group_id, message_id, text):
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
            "message": [
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
    def __init__(self):
        self.port = None
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
            await ws.send(group_event(IGNORED_GROUP, 1, " 别理我"))
            await ws.send(group_event(GROUP, TRIGGER_ID, " 你好呀"))

            request = json.loads(await asyncio.wait_for(ws.recv(), 20))
            self.sent.append(request)
            assert request["action"] == "send_group_msg", request
            await ws.send(
                json.dumps(
                    {
                        "status": "ok",
                        "retcode": 0,
                        "data": {"message_id": 777},
                        "echo": request["echo"],
                    }
                )
            )
            # No second reply may follow (the ignored group must stay silent).
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
    napcat = FakeNapCat()
    napcat.start()

    data_dir = tmp_path / "data"
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
                    "model": "fake-model",
                    "api_key": "$LILIA_TEST_KEY",
                },
                "chat": {"groups": [GROUP], "min_reply_interval_seconds": 0},
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

    assert len(model_state["bodies"]) == 1, "exactly one model call"
    body = model_state["bodies"][0]
    assert "lilia" in body["system"], "persona and runtime facts go to the system field"
    assert "测试群" in body["system"]
    assert {"type": "web_search_20250305", "name": "web_search"} in body.get(
        "tools", []
    ), "the hosted web_search tool must be offered"
    last = body["messages"][-1]
    texts = "".join(b.get("text", "") for b in last["content"] if b["type"] == "text")
    assert "你好呀" in texts and "别理我" not in texts

    assert len(napcat.sent) == 1
    params = napcat.sent[0]["params"]
    assert params["group_id"] == GROUP
    assert params["message"][0] == {"type": "reply", "data": {"id": str(TRIGGER_ID)}}
    assert params["message"][1] == {
        "type": "text",
        "data": {"text": "fake reply"},
    }, "narration before the search, the speaker label and Markdown must be dropped"

    db = sqlite3.connect(data_dir / "lilia.db")
    rows = db.execute(
        "SELECT chat_id, message_id, from_self, text FROM messages ORDER BY id"
    ).fetchall()
    assert rows == [
        (GROUP, TRIGGER_ID, 0, "@lilia 你好呀"),
        (GROUP, 777, 1, "fake reply"),
    ], rows
    calls = db.execute(
        "SELECT chat_id, input_tokens, output_tokens, request_id, error FROM llm_calls"
    ).fetchall()
    assert calls == [(GROUP, 10, 12, "req_test", "")], calls
    db.close()
