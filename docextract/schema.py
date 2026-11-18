"""Pydantic-схемы.

LLMOutput   — то, что мы требуем от модели (без field_confidence: уверенность
              считаем сами, самооценке модели не доверяем).
Extraction  — финальный ответ сервиса по схеме из ТЗ.

Обе схемы строгие: лишние поля запрещены, строки не превращаются в числа,
БИН — ровно 12 цифр, дата — реальная дата в формате YYYY-MM-DD.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, field_validator

DocType = Literal["invoice", "commercial_offer", "act", "unknown"]
Currency = Literal["KZT", "USD", "EUR"]
# StrictInt | StrictFloat: "160 500" или "160500" строкой — ошибка валидации, а не тихое приведение.
Number = Union[StrictInt, StrictFloat]

SCALAR_FIELDS = (
    "doc_type",
    "supplier_name",
    "supplier_bin",
    "doc_number",
    "doc_date",
    "currency",
    "total_amount",
    "vat_amount",
)


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1)
    qty: Optional[Number] = Field(default=None, ge=0)
    price: Optional[Number] = Field(default=None, ge=0)


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    doc_type: DocType
    supplier_name: Optional[str] = Field(default=None, min_length=1)
    supplier_bin: Optional[str] = Field(default=None, pattern=r"^\d{12}$")
    doc_number: Optional[str] = Field(default=None, min_length=1)
    doc_date: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    currency: Optional[Currency] = None
    total_amount: Optional[Number] = Field(default=None, ge=0)
    vat_amount: Optional[Number] = Field(default=None, ge=0)
    line_items: list[LineItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("doc_date")
    @classmethod
    def _real_date(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            dt.date.fromisoformat(v)  # 2026-02-30 -> ValueError -> ValidationError
        return v


class LLMOutput(_Base):
    """Схема ответа модели. Её JSON Schema передаётся в Ollama как `format`."""


class Extraction(_Base):
    """Финальный ответ сервиса (схема из ТЗ)."""

    field_confidence: dict[str, float] = Field(default_factory=dict)

    @field_validator("field_confidence")
    @classmethod
    def _conf_keys(cls, v: dict[str, float]) -> dict[str, float]:
        allowed = set(SCALAR_FIELDS) | {"line_items"}
        for k, x in v.items():
            if k not in allowed:
                raise ValueError(f"unknown field in field_confidence: {k}")
            if not 0.0 <= x <= 1.0:
                raise ValueError(f"confidence out of range for {k}: {x}")
        return v

    def to_json_dict(self) -> dict:
        # Порядок ключей как в ТЗ.
        d = self.model_dump()
        order = list(SCALAR_FIELDS) + ["line_items", "field_confidence", "warnings"]
        return {k: d[k] for k in order}


def empty_extraction(warnings: list[str]) -> Extraction:
    """Безопасный ответ, когда модель так и не выдала валидный JSON: всё null."""
    return Extraction(doc_type="unknown", warnings=warnings, field_confidence={})
