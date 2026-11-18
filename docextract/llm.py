"""LLM-бэкенды. Все реализуют один метод complete(messages, schema) -> str (сырой ответ).

- OllamaClient      — локальная модель через Ollama /api/chat, structured output (format=JSON Schema).
- OpenAICompatClient — любой OpenAI-совместимый API (OpenAI, vLLM, LM Studio, OpenRouter...).
- RulesMockLLM      — офлайн-мок: эвристический экстрактор вместо модели (см. rules.py).
- ScriptedLLM       — отдаёт заранее заданные ответы; для тестов ретраев/фолбэка.

Зависимостей нет: HTTP через urllib из стандартной библиотеки.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Protocol


class LLMClient(Protocol):
    name: str

    def complete(self, messages: list[dict], schema: dict) -> str: ...


def _post(url: str, payload: dict, headers: dict | None = None, timeout: float = 300) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class OllamaClient:
    def __init__(self, model: str | None = None, host: str | None = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.name = f"ollama:{self.model}"

    def complete(self, messages: list[dict], schema: dict) -> str:
        resp = _post(
            f"{self.host}/api/chat",
            {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "format": schema,  # constrained decoding по JSON Schema
                "options": {"temperature": 0, "num_ctx": 8192},
            },
        )
        return resp["message"]["content"]


class OpenAICompatClient:
    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.name = f"openai:{self.model}"

    def complete(self, messages: list[dict], schema: dict) -> str:
        resp = _post(
            f"{self.base_url}/chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else None,
        )
        return resp["choices"][0]["message"]["content"]


def document_from_messages(messages: list[dict]) -> str:
    """Достаёт текст документа из user-сообщения (см. prompt.USER_TEMPLATE)."""
    for m in messages:
        if m["role"] == "user":
            g = re.search(r"<<<\n(.*)\n>>>", m["content"], re.S)
            if g:
                return g.group(1)
    raise ValueError("document not found in messages")


class RulesMockLLM:
    """Мок LLM: вместо модели — детерминированный эвристический экстрактор.
    Возвращает такую же сырую JSON-строку, как вернула бы модель, и дальше
    проходит тот же путь: парсинг -> валидация -> гардрейлы."""

    name = "mock:rules"

    def complete(self, messages: list[dict], schema: dict) -> str:
        from .rules import extract_rules

        return json.dumps(extract_rules(document_from_messages(messages)), ensure_ascii=False)


class ScriptedLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []
        self.name = "scripted"

    def complete(self, messages: list[dict], schema: dict) -> str:
        self.calls.append(list(messages))
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def make_client(backend: str) -> LLMClient:
    if backend == "rules":
        return RulesMockLLM()
    if backend == "ollama":
        return OllamaClient()
    if backend == "openai":
        return OpenAICompatClient()
    raise ValueError(f"unknown backend {backend!r}; expected rules | ollama | openai")
