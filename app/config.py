from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    stt_api_key: str = ""
    tts_api_key: str = ""
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-5-5"
    folder_id: str = ""
    stt_url: str = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
    tts_url: str = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"
    request_timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            stt_api_key=os.getenv("YANDEX_STT_API_KEY", ""),
            tts_api_key=os.getenv("YANDEX_TTS_API_KEY", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            claude_model=os.getenv("CLAUDE_MODEL", cls.claude_model),
            folder_id=os.getenv("YANDEX_FOLDER_ID", ""),
        )
