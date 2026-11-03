import base64
import json

from tests.conftest import form_fields

STT_HOST = "stt.api.cloud.yandex.net"
TTS_HOST = "tts.api.cloud.yandex.net"
LLM_HOST = "api.anthropic.com"


async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "stt_configured": True,
        "tts_configured": True,
        "llm_configured": True,
        "llm_model": "claude-opus-5-5",
    }


async def test_languages(client):
    r = await client.get("/api/languages")
    assert [l["code"] for l in r.json()] == ["ru", "kk", "en"]


# --- STT ---------------------------------------------------------------

async def test_stt_uses_stt_key_and_language(client, fake):
    fake.stt_result = "сәлем әлем"
    r = await client.post(
        "/api/stt",
        files={"audio": ("a.ogg", b"OggS-audio", "audio/ogg")},
        data={"lang": "kz"},
    )
    assert r.status_code == 200
    assert r.json() == {"text": "сәлем әлем", "lang": "kk", "profanity": False}

    req = fake.by_host(STT_HOST)[0]
    assert req.headers["Authorization"] == "Api-Key stt-key"
    assert req.url.params["lang"] == "kk-KZ"
    assert req.url.params["format"] == "oggopus"
    assert req.url.params["profanityFilter"] == "true"
    assert req.content == b"OggS-audio"


async def test_stt_lpcm_sends_sample_rate(client, fake):
    await client.post(
        "/api/stt",
        files={"audio": ("a.pcm", b"\x00\x01" * 100)},
        data={"lang": "en", "format": "lpcm", "sample_rate": "16000"},
    )
    params = fake.by_host(STT_HOST)[0].url.params
    assert params["lang"] == "en-US"
    assert params["format"] == "lpcm"
    assert params["sampleRateHertz"] == "16000"


async def test_stt_masks_profanity(client, fake):
    fake.stt_result = "ну блять опять"
    r = await client.post("/api/stt", files={"audio": ("a.ogg", b"x")}, data={"lang": "ru"})
    body = r.json()
    assert body["profanity"] is True
    assert "блять" not in body["text"]


async def test_stt_rejects_empty_and_large_audio(client, fake):
    r = await client.post("/api/stt", files={"audio": ("a.ogg", b"")}, data={"lang": "ru"})
    assert r.status_code == 400
    r = await client.post(
        "/api/stt", files={"audio": ("a.ogg", b"0" * (1024 * 1024 + 1))}, data={"lang": "ru"}
    )
    assert r.status_code == 413
    assert fake.requests == []


async def test_stt_unsupported_language(client, fake):
    r = await client.post("/api/stt", files={"audio": ("a.ogg", b"x")}, data={"lang": "de"})
    assert r.status_code == 422
    assert fake.requests == []


async def test_stt_upstream_error(client, fake):
    fake.stt_status = 400
    r = await client.post("/api/stt", files={"audio": ("a.ogg", b"x")}, data={"lang": "ru"})
    assert r.status_code == 502
    assert r.json()["service"] == "stt"
    assert "bad audio" in r.json()["detail"]


# --- TTS ---------------------------------------------------------------

async def test_tts_uses_tts_key_voice_and_language(client, fake):
    r = await client.post("/api/tts", json={"text": "Сәлеметсіз бе", "lang": "kk"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/mpeg"
    assert r.content == fake.tts_audio

    req = fake.by_host(TTS_HOST)[0]
    assert req.headers["Authorization"] == "Api-Key tts-key"
    fields = form_fields(req)
    assert fields["lang"] == "kk-KK"
    assert fields["voice"] == "madi"
    assert fields["format"] == "mp3"


async def test_tts_filters_profanity_before_synthesis(client, fake):
    await client.post("/api/tts", json={"text": "what the fuck", "lang": "en"})
    fields = form_fields(fake.by_host(TTS_HOST)[0])
    assert fields["text"] == "what the f***"
    assert fields["voice"] == "john"


async def test_tts_upstream_error(client, fake):
    fake.tts_status = 401
    r = await client.post("/api/tts", json={"text": "hi", "lang": "en"})
    assert r.status_code == 502
    assert r.json()["service"] == "tts"


# --- Chat / language enforcement ------------------------------------------

async def test_chat_sends_language_system_prompt(client, fake):
    fake.llm_replies = ["Сәлем! Қалай көмектесе аламын?"]
    r = await client.post("/api/chat", json={"text": "Сәлем", "lang": "kk"})
    body = r.json()
    assert r.status_code == 200
    assert body["reply"] == "Сәлем! Қалай көмектесе аламын?"
    assert body["language_retries"] == 0
    assert body["fallback_used"] is False

    req = fake.by_host(LLM_HOST)[0]
    assert req.headers["x-api-key"] == "sk-ant-test"
    assert "server-side-fallback-2026-07-01" in req.headers["anthropic-beta"]
    sent = fake.llm_bodies[0]
    assert sent["model"] == "claude-opus-5-5"
    assert sent["output_config"] == {"effort": "low"}
    assert sent["fallbacks"] == "default"
    assert "қазақ тілінде" in sent["system"]
    assert sent["messages"] == [{"role": "user", "content": "Сәлем"}]


async def test_chat_retries_when_reply_in_wrong_language(client, fake):
    fake.llm_replies = [
        "Конечно, я с радостью помогу вам с этим вопросом прямо сейчас.",
        "Sure, I'd be happy to help.",
    ]
    r = await client.post("/api/chat", json={"text": "Can you help me?", "lang": "en"})
    body = r.json()
    assert body["reply"] == "Sure, I'd be happy to help."
    assert body["language_retries"] == 1
    assert body["fallback_used"] is False
    assert len(fake.llm_bodies) == 2
    assert "Rewrite it strictly in English" in fake.llm_bodies[1]["messages"][-1]["content"]


async def test_chat_falls_back_when_language_cannot_be_fixed(client, fake):
    fake.llm_replies = ["Hello! I can only speak English, sorry about that, my friend."]
    r = await client.post("/api/chat", json={"text": "Привет", "lang": "ru"})
    body = r.json()
    assert body["fallback_used"] is True
    assert body["reply"].startswith("Извините")


async def test_chat_flags_input_language_mismatch(client, fake):
    fake.llm_replies = ["Привет! Чем могу помочь?"]
    r = await client.post("/api/chat", json={"text": "Hello, how are you doing?", "lang": "ru"})
    body = r.json()
    assert body["input_language"] == "en"
    assert body["input_language_mismatch"] is True
    assert body["reply"] == "Привет! Чем могу помочь?"


async def test_chat_filters_profanity_both_ways(client, fake):
    fake.llm_replies = ["Не надо говорить сука, давайте вежливо."]
    r = await client.post("/api/chat", json={"text": "ты сука", "lang": "ru"})
    body = r.json()
    assert body["input_profanity"] is True
    assert body["output_profanity"] is True
    assert "сука" not in fake.llm_bodies[0]["messages"][-1]["content"]
    assert "сука" not in body["reply"]


async def test_chat_passes_history(client, fake):
    history = [
        {"role": "user", "text": "Меня зовут Дария"},
        {"role": "assistant", "text": "Приятно познакомиться!"},
    ]
    fake.llm_replies = ["Вас зовут Дария."]
    await client.post("/api/chat", json={"text": "Как меня зовут?", "lang": "ru", "history": history})
    roles = [m["role"] for m in fake.llm_bodies[0]["messages"]]
    assert roles == ["user", "assistant", "user"]


async def test_chat_llm_error(client, fake):
    fake.llm_status = 400
    r = await client.post("/api/chat", json={"text": "hi", "lang": "en"})
    assert r.status_code == 502
    assert r.json()["service"] == "llm"


async def test_chat_invalid_anthropic_key(client, fake):
    fake.llm_status = 401
    r = await client.post("/api/chat", json={"text": "hi", "lang": "en"})
    assert r.status_code == 502
    assert "ANTHROPIC_API_KEY" in r.json()["detail"]


async def test_chat_refusal_uses_fallback_reply(client, fake):
    fake.llm_stop_reason = "refusal"
    fake.llm_replies = [""]
    r = await client.post("/api/chat", json={"text": "Сәлем", "lang": "kk"})
    body = r.json()
    assert body["fallback_used"] is True
    assert body["reply"].startswith("Кешіріңіз")


async def test_chat_validation(client):
    r = await client.post("/api/chat", json={"text": "", "lang": "ru"})
    assert r.status_code == 422


# --- Full voice flow -----------------------------------------------------

async def test_voice_flow_kazakh(client, fake):
    fake.stt_result = "Ауа райы қандай?"
    fake.llm_replies = ["Бүгін күн ашық, жылы."]
    r = await client.post(
        "/api/voice",
        files={"audio": ("a.pcm", b"\x00" * 3200)},
        data={"lang": "kk", "format": "lpcm", "sample_rate": "16000"},
    )
    body = r.json()
    assert r.status_code == 200
    assert body["transcript"] == "Ауа райы қандай?"
    assert body["reply"] == "Бүгін күн ашық, жылы."
    assert base64.b64decode(body["audio_base64"]) == fake.tts_audio
    assert body["audio_mime"] == "audio/mpeg"

    hosts = [req.url.host for req in fake.requests]
    assert hosts == [STT_HOST, LLM_HOST, TTS_HOST]
    tts_fields = form_fields(fake.by_host(TTS_HOST)[0])
    assert tts_fields["text"] == "Бүгін күн ашық, жылы."
    assert tts_fields["lang"] == "kk-KK"


async def test_voice_flow_no_speech_skips_llm(client, fake):
    fake.stt_result = ""
    r = await client.post("/api/voice", files={"audio": ("a.ogg", b"x")}, data={"lang": "en"})
    body = r.json()
    assert body["fallback_used"] is True
    assert body["reply"].startswith("Sorry")
    assert [req.url.host for req in fake.requests] == [STT_HOST, TTS_HOST]


async def test_voice_flow_with_history(client, fake):
    fake.llm_replies = ["Хорошо."]
    history = json.dumps([{"role": "user", "text": "Привет"}, {"role": "assistant", "text": "Привет!"}])
    r = await client.post(
        "/api/voice", files={"audio": ("a.ogg", b"x")}, data={"lang": "ru", "history": history}
    )
    assert r.status_code == 200
    assert len(fake.llm_bodies[0]["messages"]) == 3


async def test_voice_bad_history(client, fake):
    r = await client.post(
        "/api/voice", files={"audio": ("a.ogg", b"x")}, data={"lang": "ru", "history": "not json"}
    )
    assert r.status_code == 422
    assert fake.requests == []


async def test_missing_keys_return_503(fake):
    import httpx

    from app.config import Settings
    from app.main import create_app

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(fake.handler))
    app = create_app(Settings(), upstream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        r = await c.post("/api/tts", json={"text": "hi", "lang": "en"})
        assert r.status_code == 503
        r = await c.post("/api/chat", json={"text": "hi", "lang": "en"})
        assert r.status_code == 503
    assert fake.requests == []
    await upstream.aclose()
