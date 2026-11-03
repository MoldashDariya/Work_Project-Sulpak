"""Reply generation via the Anthropic Messages API (Claude)."""
from __future__ import annotations

from typing import Dict, List, Optional

import anthropic
import httpx

from .config import Settings
from .yandex import ServiceError

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeLLM:
    def __init__(self, settings: Settings, http_client: Optional[httpx.AsyncClient] = None):
        self.settings = settings
        self.client = (
            anthropic.AsyncAnthropic(
                api_key=settings.anthropic_api_key,
                http_client=http_client,
                timeout=settings.request_timeout,
                max_retries=1,
            )
            if settings.anthropic_api_key
            else None
        )

    async def complete(self, messages: List[Dict[str, str]], max_tokens: int = 2000) -> str:
        """`messages` use the app format: [{"role": "system"|"user"|"assistant", "text": ...}]."""
        if self.client is None:
            raise ServiceError("llm", "ANTHROPIC_API_KEY is not configured", status_code=503)

        system = "\n\n".join(m["text"] for m in messages if m["role"] == "system")
        turns = [
            {"role": m["role"], "content": m["text"]}
            for m in messages
            if m["role"] in ("user", "assistant")
        ]
        try:
            response = await self.client.beta.messages.create(
                model=self.settings.claude_model,
                max_tokens=max_tokens,
                system=system,
                messages=turns,
                # Voice replies are short; low effort keeps latency down.
                output_config={"effort": "low"},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as exc:
            raise ServiceError("llm", "invalid ANTHROPIC_API_KEY", status_code=502) from exc
        except anthropic.RateLimitError as exc:
            raise ServiceError("llm", "rate limited by Anthropic, try again later", status_code=429) from exc
        except anthropic.APIStatusError as exc:
            raise ServiceError("llm", f"HTTP {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise ServiceError("llm", f"request failed: {exc}") from exc

        if response.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in response.content if b.type == "text").strip()
