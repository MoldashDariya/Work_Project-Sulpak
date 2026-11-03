"""Profanity filter for Russian, Kazakh and English.

Words are matched individually against root patterns (anchored at the word
start where needed to avoid false positives like "употреблять" or "страхую")
and masked as "х***".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

_RU_PREFIX = r"(?:за|на|по|от|до|вы|у|раз|рас|об|под|при|про|пере|ни|о|въ|съ|отъ|подъ|разъ)?ъ?"

_PATTERNS = [
    # Russian
    rf"^{_RU_PREFIX}ху[йяеёию]",
    r"пизд",
    r"^(?:вы|за|по)?бля(?:$|д|т|ха)",
    rf"^{_RU_PREFIX}[её]б(?:а|ан|ал|ат|ен|ну|ло|ли|л|у|ыв|ись|ище|уч)",
    r"^долбо[её]б",
    r"^(?:за|от|по)?муд[ао]к",
    r"^мудил",
    r"^сук(?:а|и|е|у|ой|ин|ам)?$",
    r"^сучк",
    r"^гандон",
    r"^мраз[ьи]",
    r"^шлюх",
    r"^пид[оа]р",
    r"^залуп",
    # Kazakh
    r"^сік",
    r"^сиг[еі]й",
    r"^қотақ",
    r"^ам(?:ың|ын|ыңды|ынды)$",
    r"^шешеңді",
    r"^боқ(?:$|қа|тан|ы)",
    # English
    r"fuck",
    r"^(?:bull)?shit",
    r"^bitch",
    r"^asshole",
    r"^cunts?$",
    r"^bastard",
    r"^motherf",
    r"^dickhead",
    r"^whore",
]

_COMPILED = [re.compile(p) for p in _PATTERNS]
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass
class FilterResult:
    text: str
    found: List[str] = field(default_factory=list)

    @property
    def has_profanity(self) -> bool:
        return bool(self.found)


def is_profane(word: str) -> bool:
    w = word.lower().replace("ё", "е")
    return any(p.search(w) for p in _COMPILED)


def _mask(word: str) -> str:
    return word[0] + "*" * (len(word) - 1)


def filter_profanity(text: str) -> FilterResult:
    found: List[str] = []

    def repl(m: "re.Match[str]") -> str:
        word = m.group(0)
        if is_profane(word):
            found.append(word)
            return _mask(word)
        return word

    return FilterResult(text=_WORD.sub(repl, text), found=found)
