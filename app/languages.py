"""Supported languages and a lightweight script-based language detector."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    stt_lang: str
    tts_lang: str
    default_voice: str
    system_prompt: str
    fallback_reply: str


LANGUAGES: Dict[str, Language] = {
    "ru": Language(
        code="ru",
        name="Русский",
        stt_lang="ru-RU",
        tts_lang="ru-RU",
        default_voice="alena",
        system_prompt=(
            "Ты дружелюбный голосовой ассистент. Отвечай ТОЛЬКО на русском языке, "
            "даже если пользователь пишет на другом языке. Отвечай кратко (1-3 предложения), "
            "без markdown, списков и эмодзи — ответ будет озвучен. Не используй ненормативную лексику."
        ),
        fallback_reply="Извините, я не смог сформулировать ответ. Повторите, пожалуйста, вопрос.",
    ),
    "kk": Language(
        code="kk",
        name="Қазақша",
        stt_lang="kk-KZ",
        tts_lang="kk-KK",
        default_voice="madi",
        system_prompt=(
            "Сен мейірімді дауыстық көмекшісің. ТЕК қазақ тілінде жауап бер, тіпті пайдаланушы "
            "басқа тілде жазса да. Қысқа жауап бер (1-3 сөйлем), markdown, тізім және эмодзи "
            "қолданба — жауап дауыспен айтылады. Балағат сөздерді қолданба."
        ),
        fallback_reply="Кешіріңіз, жауап бере алмадым. Сұрағыңызды қайталаңызшы.",
    ),
    "en": Language(
        code="en",
        name="English",
        stt_lang="en-US",
        tts_lang="en-US",
        default_voice="john",
        system_prompt=(
            "You are a friendly voice assistant. Reply ONLY in English, even if the user writes "
            "in another language. Keep replies short (1-3 sentences), no markdown, lists or emoji — "
            "the reply will be spoken aloud. Never use profanity."
        ),
        fallback_reply="Sorry, I couldn't come up with an answer. Could you repeat the question?",
    ),
}

_ALIASES = {
    "ru": "ru", "rus": "ru", "ru-ru": "ru",
    "kk": "kk", "kz": "kk", "kaz": "kk", "kk-kz": "kk", "kk-kk": "kk",
    "en": "en", "eng": "en", "en-us": "en", "en-gb": "en",
}

# Letters that exist in Kazakh Cyrillic but not in Russian.
KAZAKH_LETTERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")


class UnsupportedLanguage(ValueError):
    pass


def get_language(code: str) -> Language:
    key = _ALIASES.get((code or "").strip().lower())
    if key is None:
        raise UnsupportedLanguage(
            f"Unsupported language '{code}'. Use one of: {', '.join(LANGUAGES)}"
        )
    return LANGUAGES[key]


def detect_language(text: str) -> Optional[str]:
    """Guess ru/kk/en from the script. Returns None when there are no letters."""
    cyrillic = latin = kazakh = 0
    for ch in text:
        if not ch.isalpha():
            continue
        if ch in KAZAKH_LETTERS:
            kazakh += 1
            cyrillic += 1
        elif "Ѐ" <= ch <= "ӿ":
            cyrillic += 1
        elif ch.isascii():
            latin += 1
    if cyrillic + latin == 0:
        return None
    if latin > cyrillic:
        return "en"
    return "kk" if kazakh > 0 else "ru"


def matches_language(text: str, lang: Language) -> bool:
    """True when the text is plausibly written in `lang`.

    Short Kazakh phrases may contain no Kazakh-only letters, so Cyrillic
    text without them is accepted for Kazakh as long as it is short.
    """
    detected = detect_language(text)
    if detected is None or detected == lang.code:
        return True
    if lang.code == "kk" and detected == "ru":
        letters = sum(1 for ch in text if ch.isalpha())
        return letters <= 25
    return False
