"""Reply generation with language enforcement and profanity filtering."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .languages import Language, detect_language, matches_language
from .llm import ClaudeLLM
from .profanity import filter_profanity

MAX_HISTORY = 10

_RETRY_HINT = {
    "ru": "Твой предыдущий ответ был не на русском. Перепиши его строго на русском языке.",
    "kk": "Алдыңғы жауабың қазақ тілінде емес. Оны тек қазақ тілінде қайта жаз.",
    "en": "Your previous reply was not in English. Rewrite it strictly in English.",
}


@dataclass
class Reply:
    text: str
    input_text: str
    input_language: Optional[str]
    input_language_mismatch: bool
    input_profanity: bool
    output_profanity: bool
    language_retries: int
    fallback_used: bool


class Assistant:
    def __init__(self, llm: ClaudeLLM, max_retries: int = 1):
        self.llm = llm
        self.max_retries = max_retries

    async def reply(
        self, user_text: str, lang: Language, history: Optional[List[Dict[str, str]]] = None
    ) -> Reply:
        cleaned_input = filter_profanity(user_text)
        detected = detect_language(user_text)

        messages: List[Dict[str, str]] = [{"role": "system", "text": lang.system_prompt}]
        for msg in (history or [])[-MAX_HISTORY:]:
            if msg.get("role") in ("user", "assistant") and msg.get("text"):
                messages.append({"role": msg["role"], "text": filter_profanity(msg["text"]).text})
        messages.append({"role": "user", "text": cleaned_input.text})

        answer = await self.llm.complete(messages)
        retries = 0
        while not matches_language(answer, lang) and retries < self.max_retries:
            retries += 1
            answer = await self.llm.complete(
                messages
                + [
                    {"role": "assistant", "text": answer},
                    {"role": "user", "text": _RETRY_HINT[lang.code]},
                ]
            )

        fallback = not answer or not matches_language(answer, lang)
        if fallback:
            answer = lang.fallback_reply

        cleaned_output = filter_profanity(answer)
        return Reply(
            text=cleaned_output.text,
            input_text=cleaned_input.text,
            input_language=detected,
            input_language_mismatch=not matches_language(user_text, lang),
            input_profanity=cleaned_input.has_profanity,
            output_profanity=cleaned_output.has_profanity,
            language_retries=retries,
            fallback_used=fallback,
        )
