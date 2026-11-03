"""Async client for Yandex SpeechKit (STT/TTS, API v1)."""
from __future__ import annotations

from typing import Dict, Optional

import httpx

from .config import Settings
from .languages import Language

STT_MAX_BYTES = 1024 * 1024  # SpeechKit v1 sync recognition limit
TTS_MAX_CHARS = 5000


class ServiceError(Exception):
    def __init__(self, service: str, message: str, status_code: int = 502):
        super().__init__(f"{service}: {message}")
        self.service = service
        self.message = message
        self.status_code = status_code


def _error_text(resp: httpx.Response) -> str:
    try:
        data = resp.json()
        return data.get("error_message") or data.get("message") or resp.text
    except ValueError:
        return resp.text[:300]


class SpeechKit:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.settings = settings
        self.client = client

    def _headers(self, api_key: str, service: str) -> Dict[str, str]:
        if not api_key:
            raise ServiceError(service, "API key is not configured", status_code=503)
        return {"Authorization": f"Api-Key {api_key}"}

    async def recognize(
        self,
        audio: bytes,
        lang: Language,
        audio_format: str = "oggopus",
        sample_rate: Optional[int] = None,
    ) -> str:
        headers = self._headers(self.settings.stt_api_key, "stt")
        if len(audio) > STT_MAX_BYTES:
            raise ServiceError("stt", "audio is larger than 1 MB", status_code=413)
        params = {
            "lang": lang.stt_lang,
            "format": audio_format,
            "profanityFilter": "true",
        }
        if audio_format == "lpcm":
            params["sampleRateHertz"] = str(sample_rate or 16000)
        if self.settings.folder_id:
            params["folderId"] = self.settings.folder_id
        try:
            resp = await self.client.post(
                self.settings.stt_url, params=params, content=audio, headers=headers
            )
        except httpx.HTTPError as exc:
            raise ServiceError("stt", f"request failed: {exc}") from exc
        if resp.status_code != 200:
            raise ServiceError("stt", f"HTTP {resp.status_code}: {_error_text(resp)}")
        return (resp.json().get("result") or "").strip()

    async def synthesize(
        self,
        text: str,
        lang: Language,
        voice: Optional[str] = None,
        audio_format: str = "mp3",
        speed: float = 1.0,
    ) -> bytes:
        headers = self._headers(self.settings.tts_api_key, "tts")
        data = {
            "text": text[:TTS_MAX_CHARS],
            "lang": lang.tts_lang,
            "voice": voice or lang.default_voice,
            "format": audio_format,
            "speed": str(speed),
        }
        if self.settings.folder_id:
            data["folderId"] = self.settings.folder_id
        try:
            resp = await self.client.post(self.settings.tts_url, data=data, headers=headers)
        except httpx.HTTPError as exc:
            raise ServiceError("tts", f"request failed: {exc}") from exc
        if resp.status_code != 200:
            raise ServiceError("tts", f"HTTP {resp.status_code}: {_error_text(resp)}")
        return resp.content

