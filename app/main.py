from __future__ import annotations

import base64
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Literal, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .assistant import Assistant
from .config import Settings
from .languages import LANGUAGES, Language, UnsupportedLanguage, get_language
from .profanity import filter_profanity
from .llm import ClaudeLLM
from .yandex import STT_MAX_BYTES, ServiceError, SpeechKit

STATIC_DIR = Path(__file__).parent / "static"
AUDIO_MIME = {"mp3": "audio/mpeg", "oggopus": "audio/ogg", "lpcm": "audio/L16"}


class Message(BaseModel):
    role: Literal["user", "assistant"]
    text: str


class ChatRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    lang: str = "ru"
    history: List[Message] = []


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    lang: str = "ru"
    voice: Optional[str] = None
    format: Literal["mp3", "oggopus", "lpcm"] = "mp3"
    speed: float = Field(1.0, ge=0.1, le=3.0)


def _lang(code: str) -> Language:
    try:
        return get_language(code)
    except UnsupportedLanguage as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _reply_payload(reply) -> Dict:
    return {
        "reply": reply.text,
        "input_text": reply.input_text,
        "input_language": reply.input_language,
        "input_language_mismatch": reply.input_language_mismatch,
        "input_profanity": reply.input_profanity,
        "output_profanity": reply.output_profanity,
        "language_retries": reply.language_retries,
        "fallback_used": reply.fallback_used,
    }


async def _read_audio(audio: UploadFile) -> bytes:
    data = await audio.read(STT_MAX_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Audio file is empty")
    if len(data) > STT_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Audio is larger than 1 MB (~30 s)")
    return data


def create_app(
    settings: Optional[Settings] = None, http_client: Optional[httpx.AsyncClient] = None
) -> FastAPI:
    settings = settings or Settings.from_env()
    client = http_client or httpx.AsyncClient(timeout=settings.request_timeout)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await client.aclose()

    app = FastAPI(title="RU/KZ/EN Voice Assistant", lifespan=lifespan)
    speechkit = SpeechKit(settings, client)
    # Tests inject a mock transport via http_client; in production the SDK manages its own.
    assistant = Assistant(ClaudeLLM(settings, http_client))

    @app.exception_handler(ServiceError)
    async def service_error_handler(_: Request, exc: ServiceError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.message, "service": exc.service},
        )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "stt_configured": bool(settings.stt_api_key),
            "tts_configured": bool(settings.tts_api_key),
            "llm_configured": bool(settings.anthropic_api_key),
            "llm_model": settings.claude_model,
        }

    @app.get("/api/languages")
    async def languages():
        return [
            {"code": l.code, "name": l.name, "stt": l.stt_lang, "tts": l.tts_lang, "voice": l.default_voice}
            for l in LANGUAGES.values()
        ]

    @app.post("/api/stt")
    async def stt(
        audio: UploadFile = File(...),
        lang: str = Form("ru"),
        format: Literal["oggopus", "lpcm"] = Form("oggopus"),
        sample_rate: int = Form(16000),
    ):
        language = _lang(lang)
        text = await speechkit.recognize(await _read_audio(audio), language, format, sample_rate)
        filtered = filter_profanity(text)
        return {"text": filtered.text, "lang": language.code, "profanity": filtered.has_profanity}

    @app.post("/api/tts")
    async def tts(req: TTSRequest):
        language = _lang(req.lang)
        text = filter_profanity(req.text).text
        audio = await speechkit.synthesize(text, language, req.voice, req.format, req.speed)
        return Response(content=audio, media_type=AUDIO_MIME[req.format])

    @app.post("/api/chat")
    async def chat(req: ChatRequest):
        language = _lang(req.lang)
        history = [m.model_dump() for m in req.history]
        reply = await assistant.reply(req.text, language, history)
        return {"lang": language.code, **_reply_payload(reply)}

    @app.post("/api/voice")
    async def voice(
        audio: UploadFile = File(...),
        lang: str = Form("ru"),
        format: Literal["oggopus", "lpcm"] = Form("oggopus"),
        sample_rate: int = Form(16000),
        history: str = Form("[]"),
        voice: Optional[str] = Form(None),
        tts_format: Literal["mp3", "oggopus"] = Form("mp3"),
    ):
        language = _lang(lang)
        try:
            history_items = [Message(**m).model_dump() for m in json.loads(history)]
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail="history must be a JSON list of messages") from exc

        transcript = await speechkit.recognize(await _read_audio(audio), language, format, sample_rate)
        if transcript:
            reply = await assistant.reply(transcript, language, history_items)
            payload = _reply_payload(reply)
            reply_text = reply.text
        else:
            reply_text = language.fallback_reply
            payload = {"reply": reply_text, "input_text": "", "fallback_used": True}

        speech = await speechkit.synthesize(reply_text, language, voice, tts_format)
        return {
            "lang": language.code,
            "transcript": filter_profanity(transcript).text,
            **payload,
            "audio_format": tts_format,
            "audio_mime": AUDIO_MIME[tts_format],
            "audio_base64": base64.b64encode(speech).decode("ascii"),
        }

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
