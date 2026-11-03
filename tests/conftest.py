from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional
from urllib.parse import parse_qs

import httpx
import pytest

from app.config import Settings
from app.main import create_app

TEST_SETTINGS = Settings(
    stt_api_key="stt-key",
    tts_api_key="tts-key",
    anthropic_api_key="sk-ant-test",
)


class FakeYandex:
    """Records requests to SpeechKit/Anthropic and returns scripted responses."""

    def __init__(self):
        self.requests: List[httpx.Request] = []
        self.stt_result = "привет"
        self.stt_status = 200
        self.tts_status = 200
        self.tts_audio = b"ID3-fake-mp3"
        self.llm_replies: List[str] = ["Здравствуйте!"]
        self.llm_status = 200
        self.llm_stop_reason = "end_turn"
        self.llm_bodies: List[Dict] = []
        self.on_request: Optional[Callable[[httpx.Request], None]] = None

    def by_host(self, host: str) -> List[httpx.Request]:
        return [r for r in self.requests if r.url.host == host]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        host = request.url.host
        if host == "stt.api.cloud.yandex.net":
            if self.stt_status != 200:
                return httpx.Response(self.stt_status, json={"error_message": "bad audio"})
            return httpx.Response(200, json={"result": self.stt_result})
        if host == "tts.api.cloud.yandex.net":
            if self.tts_status != 200:
                return httpx.Response(self.tts_status, json={"error_message": "tts failed"})
            return httpx.Response(200, content=self.tts_audio, headers={"Content-Type": "audio/mpeg"})
        if host == "api.anthropic.com" and request.url.path == "/v1/messages":
            self.llm_bodies.append(json.loads(request.content))
            if self.llm_status != 200:
                return httpx.Response(
                    self.llm_status,
                    json={"type": "error", "error": {"type": "api_error", "message": "overloaded"}},
                )
            text = self.llm_replies.pop(0) if len(self.llm_replies) > 1 else self.llm_replies[0]
            return httpx.Response(
                200,
                json={
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-opus-5-5",
                    "content": [{"type": "text", "text": text}],
                    "stop_reason": self.llm_stop_reason,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            )
        return httpx.Response(404)


def form_fields(request: httpx.Request) -> Dict[str, str]:
    return {k: v[0] for k, v in parse_qs(request.content.decode()).items()}


@pytest.fixture
def fake() -> FakeYandex:
    return FakeYandex()


@pytest.fixture
async def client(fake: FakeYandex):
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    app = create_app(TEST_SETTINGS, upstream)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    await upstream.aclose()
