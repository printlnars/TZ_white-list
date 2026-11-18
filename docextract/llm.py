"""LLM-бэкенды. Все реализуют один метод complete(messages, schema) -> str (сырой ответ).

- OllamaClient      — локальная модель через Ollama /api/chat, structured output (format=JSON Schema).
- OpenAICompatClient — любой OpenAI-совместимый API (OpenAI, vLLM, LM Studio, OpenRouter...).
- RulesMockLLM      — офлайн-мок: эвристический экстрактор вместо модели (см. rules.py).
- GreedyMockLLM     — мок «самоуверенной» модели с типичными галлюцинациями (для проверки гардрейлов).
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


class GreedyMockLLM:
    """Мок «самоуверенной» модели: берёт ответ rules и заполняет пустые поля ровно так,
    как типично ошибаются LLM (и как запрещает ТЗ). Нужен, чтобы eval показал,
    сколько галлюцинаций ловят гардрейлы: метрики считаются до и после них."""

    name = "mock:greedy"

    def complete(self, messages: list[dict], schema: dict) -> str:
        import datetime as dt
        import zlib

        from .rules import extract_rules

        text = document_from_messages(messages)
        d = extract_rules(text)
        if d["doc_date"] is None:
            d["doc_date"] = dt.date.today().isoformat()  # подставляет сегодняшнюю дату
        if d["currency"] is None:
            d["currency"] = "KZT"  # «документ казахстанский — значит тенге»
        if d["supplier_bin"] is None and d["supplier_name"]:
            # «выводит» БИН из названия
            d["supplier_bin"] = f"{zlib.crc32(d['supplier_name'].encode()) % 10**12:012d}"
        if d["doc_number"] is None:
            d["doc_number"] = "1"
        items = d["line_items"]
        if d["total_amount"] is None and items and all(i["qty"] and i["price"] for i in items):
            d["total_amount"] = sum(i["qty"] * i["price"] for i in items)  # складывает позиции сам
        if d["vat_amount"] is None and d["total_amount"]:
            d["vat_amount"] = round(d["total_amount"] * 16 / 116, 2)  # считает НДС 16% сам
        for i in items:
            if i["qty"] is None:
                i["qty"] = 1  # «раз не указано — значит одна штука»
        return json.dumps(d, ensure_ascii=False)


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
    if backend == "greedy":
        return GreedyMockLLM()
    if backend == "ollama":
        return OllamaClient()
    if backend == "openai":
        return OpenAICompatClient()
    raise ValueError(f"unknown backend {backend!r}; expected rules | greedy | ollama | openai")
