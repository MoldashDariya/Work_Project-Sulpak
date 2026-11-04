# SULPAK Voice Assistant (RU / KZ / EN)

Voice assistant module developed as part of a project at **SULPAK**.

Built at **2Code LLP** for **SULPAK**, team of 8. My part: the NLP setup and the NLP work in this module: speech recognition and synthesis (Yandex SpeechKit), LLM replies, language enforcement and profanity filtering.

> **Note:** this repository contains only the parts of the code that I was allowed to publish.
> The rest of the SULPAK project (business logic, integrations, data and infrastructure) is
> confidential and is not included here.

## About

- Built RU/KZ/EN voice flows with speech-to-text, text-to-speech (Yandex SpeechKit) and
  LLM-generated replies
- Added language enforcement, profanity filtering and async API tests

Pipeline: speech → text (Yandex SpeechKit STT) → LLM reply → speech (Yandex SpeechKit TTS),
served by a FastAPI backend with a simple web page for recording voice and chatting.

### Features

- **Three languages:** Russian (`ru`), Kazakh (`kk`, alias `kz`) and English (`en`), each with its
  own default voice (`alena`, `madi`, `john`).
- **Language enforcement:** the system prompt is written in the selected language; if the LLM still
  answers in another language, the reply is regenerated once, and if that fails a fallback phrase
  in the right language is returned. Input in a different language is flagged with
  `input_language_mismatch`.
- **Profanity filtering (RU/KZ/EN):** offensive words are masked (e.g. `б****`) in user input
  before it reaches the LLM, in the LLM reply before TTS, and in `/api/tts`. SpeechKit's own
  `profanityFilter` is also enabled for STT.
- **Async API tests:** `pytest-asyncio` + `httpx.ASGITransport`, with all external APIs replaced by
  `httpx.MockTransport`, so tests run without any real keys.

## Project structure

```
app/
  main.py        FastAPI app and endpoints
  yandex.py      SpeechKit STT / TTS client
  llm.py         LLM reply generation
  assistant.py   dialogue flow and language enforcement
  languages.py   supported languages, voices, prompts
  profanity.py   RU/KZ/EN profanity filter
  config.py      settings
  static/        web page (voice recording + text chat)
tests/           async API and filter tests
```

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | service status |
| GET | `/api/languages` | supported languages and voices |
| POST | `/api/stt` | multipart `audio`, `lang`, `format` (`oggopus`/`lpcm`), `sample_rate` → `{text, profanity}` |
| POST | `/api/tts` | JSON `{text, lang, voice?, format?, speed?}` → audio |
| POST | `/api/chat` | JSON `{text, lang, history?}` → LLM reply + flags |
| POST | `/api/voice` | multipart `audio`, `lang`, … → transcript, reply and audio (base64) |

SpeechKit v1 limits: audio up to 1 MB / 30 s, TTS text up to 5000 characters.

## Running locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload
```

API credentials are not part of this repository and must be configured locally.
Then open http://localhost:8000.

## Tests

```bash
.venv/bin/python -m pytest
```

## Tech stack

Python, FastAPI, httpx, Yandex SpeechKit, pytest / pytest-asyncio.
