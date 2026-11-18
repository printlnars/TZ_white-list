"""Пайплайн: LLM -> парсинг JSON -> строгая валидация (pydantic) -> ретрай с ошибками -> гардрейлы.

Что происходит при невалидном ответе модели:
  1. Ошибки валидации (поле, причина) отправляются модели обратно, просим исправить.
  2. Так до max_attempts попыток (по умолчанию 3).
  3. Если валидного ответа так и нет — возвращаем «пустой» результат (doc_type=unknown,
     все поля null) с предупреждением VALIDATION_FAILED и ok=False. CLI в этом случае
     завершается с кодом 2 — документ уходит на ручную разметку. Невалидные данные
     дальше по системе не идут никогда.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from . import guardrails
from .llm import LLMClient
from .prompt import REPAIR_TEMPLATE, SYSTEM_PROMPT, USER_TEMPLATE
from .schema import Extraction, LLMOutput, empty_extraction

LLM_JSON_SCHEMA = LLMOutput.model_json_schema()


@dataclass
class PipelineResult:
    extraction: Extraction
    raw: LLMOutput | None  # валидный ответ модели ДО гардрейлов (для eval: сколько поймали гардрейлы)
    ok: bool
    attempts: int
    errors: list[str] = field(default_factory=list)


def _parse(raw: str) -> LLMOutput:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)  # модели любят оборачивать JSON в ```
    try:
        data = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"не JSON: {e.msg} (позиция {e.pos})") from None
    return LLMOutput.model_validate(data)


def _format_errors(e: Exception) -> str:
    if isinstance(e, ValidationError):
        return "\n".join(f"- {'.'.join(map(str, err['loc'])) or '<root>'}: {err['msg']}" for err in e.errors())
    return f"- {e}"


def extract(text: str, llm: LLMClient, max_attempts: int = 3) -> PipelineResult:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=text)},
    ]
    errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        raw = llm.complete(messages, LLM_JSON_SCHEMA)
        try:
            parsed = _parse(raw)
        except (ValueError, ValidationError) as e:
            err = _format_errors(e)
            errors.append(f"attempt {attempt}:\n{err}")
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": REPAIR_TEMPLATE.format(errors=err)},
            ]
            continue
        final = guardrails.apply(parsed, text)
        return PipelineResult(final, parsed, ok=True, attempts=attempt, errors=errors)

    fallback = empty_extraction(
        [f"VALIDATION_FAILED: модель {max_attempts} раз вернула ответ, не прошедший валидацию схемы; "
         f"документ требует ручной обработки. Последние ошибки: {errors[-1] if errors else ''}"]
    )
    return PipelineResult(fallback, None, ok=False, attempts=max_attempts, errors=errors)
