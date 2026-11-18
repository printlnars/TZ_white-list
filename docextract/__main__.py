"""CLI: python -m docextract FILE [--backend rules|greedy|ollama|openai] [--debug]

Печатает JSON по схеме из ТЗ в stdout.
Код выхода: 0 — успех; 2 — модель не выдала валидный JSON (возвращён пустой результат
с предупреждением VALIDATION_FAILED, документ нужно разобрать вручную).
"""

from __future__ import annotations

import argparse
import json
import sys

from .llm import make_client
from .pipeline import extract


def main() -> int:
    ap = argparse.ArgumentParser(prog="python -m docextract", description="Извлечение полей из текста документа")
    ap.add_argument("file", help="путь к .txt с текстом документа или '-' для stdin")
    ap.add_argument("--backend", default="rules", help="rules | greedy | ollama | openai (default: rules)")
    ap.add_argument("--debug", action="store_true", help="в stderr: сырой ответ модели и ошибки валидации")
    a = ap.parse_args()

    text = sys.stdin.read() if a.file == "-" else open(a.file, encoding="utf-8").read()
    r = extract(text, make_client(a.backend))
    if a.debug:
        print(f"attempts={r.attempts} ok={r.ok}", file=sys.stderr)
        for e in r.errors:
            print(e, file=sys.stderr)
        if r.raw:
            print("raw (до гардрейлов):", r.raw.model_dump_json(), file=sys.stderr)
    print(json.dumps(r.extraction.to_json_dict(), ensure_ascii=False, indent=2))
    return 0 if r.ok else 2


if __name__ == "__main__":
    sys.exit(main())
