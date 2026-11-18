"""Эвристический экстрактор — мок вместо LLM (бэкенд `rules`).

Нужен, чтобы пайплайн и eval запускались без GPU и без ключей. Он возвращает
сырой словарь «как ответ модели», который дальше проходит тот же путь:
JSON -> pydantic -> гардрейлы. Это честный бейзлайн, а не подгонка под эталон:
его ошибки видны в eval.
"""

from __future__ import annotations

import re

from . import patterns as P
from .textnorm import amounts_in_line, amounts_in_words, issue_date_candidates, ocr_fix, parse_amount

_NUM_TOKEN = re.compile(r"^\d+(?:[.,]\d{1,2})?$")
_GROUP_TAIL = re.compile(r"^\d{3}(?:[.,]\d{1,2})?$")
_UNITS = {"шт", "шт.", "мес", "ед", "ед.", "кг", "т", "м", "мешок", "усл", "дана", "pcs", "упак", "упак.", "л", "ай"}
_BULLET = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s+")
_DASH_SPLIT = re.compile(r"\s+[—–-]\s+")
_QTY_PRICE = re.compile(
    r"^(\d+(?:[.,]\d+)?)\s*(?:шт|дана|pcs|ед|units?)?\.?\s*(?:[x×х*]|по|[—–-])\s*(?:EUR|USD|KZT|€|\$)?\s*(\d[\d  .,]*\d|\d)",
    re.I,
)
_PRICE_ONLY = re.compile(r"^(?:EUR|USD|€|\$)?\s*(\d[\d  .,]*\d)\s*(?:тг|тенге|теңге|₸|kzt|usd|eur|€|\$)?\.?$", re.I)


def extract_rules(text: str) -> dict:
    fixed = ocr_fix(text)
    lines = fixed.split("\n")
    low_lines = [l.lower() for l in lines]
    roles = P.line_roles(fixed)
    nonempty = [l.lower() for l in lines if l.strip()]
    head = "\n".join(nonempty[:4])

    # тип: самое раннее ключевое слово в шапке
    pos = {t: m.start() for t, rx in P.DOC_TYPE_KW.items() if (m := rx.search(head))}
    doc_type = min(pos, key=pos.get) if pos else "unknown"

    return {
        "doc_type": doc_type,
        "supplier_name": _supplier_name(lines, roles),
        "supplier_bin": _supplier_bin(lines, roles),
        "doc_number": _doc_number(lines),
        "doc_date": (c[0].date.isoformat() if (c := issue_date_candidates(fixed)) else None),
        "currency": _currency(fixed.lower()),
        "total_amount": _total(low_lines),
        "vat_amount": _vat(low_lines),
        "line_items": _line_items(lines),
        "warnings": [],
    }


def _supplier_name(lines: list[str], roles: list) -> str | None:
    for l in lines:
        low = l.lower()
        m = P.SUPPLIER_LABEL.search(low)
        if m and not P.BUYER_LABEL.search(low):
            rest = l[m.end():].lstrip(" :")
            rest = re.split(r",\s*(?:БИН|БСН|BIN|ИИН)|\s+(?:БИН|БСН)\b", rest)[0]
            rest = rest.strip(" ,.")
            if rest:
                return rest
    for i, l in enumerate(lines[:4]):
        if P.LEGAL_FORM.search(l) and roles[i] != "buyer":
            return l.split(" / ")[0].strip()
    return None


def _supplier_bin(lines: list[str], roles: list) -> str | None:
    for i, l in enumerate(lines):
        m = re.search(r"(?<!\d)\d{12}(?!\d)", l)
        if m and roles[i] != "buyer":
            return m.group(0)
    return None


def _doc_number(lines: list[str]) -> str | None:
    for l in lines:
        if re.search(r"договор|келісім|contract", l, re.I):
            continue
        for m in re.finditer(r"(?:№|\bNo\b\.?|N°)\s*([^\s|]+)", l):
            tok = m.group(1).strip(".,;:")
            if re.search(r"\d", tok):
                return tok
    return None


def _currency(low: str) -> str | None:
    counts = {c: len(rx.findall(low)) for c, rx in P.CURRENCY_KW.items()}
    best = max(counts, key=counts.get)
    return best if counts[best] else None


def _total(low_lines: list[str]) -> float | None:
    total = None
    for l in low_lines:
        if not P.TOTAL_KW.search(l) or P.SUBTOTAL_KW.search(l):
            continue
        part = P.split_total_vat(l)[0]
        words = amounts_in_words(l)
        digits = amounts_in_line(part)
        if words:
            total = float(words[-1])
        elif digits:
            total = digits[-1]
    return total


def _vat(low_lines: list[str]) -> float | None:
    vals = P.vat_amounts([l for l in low_lines if not P.VAT_EXEMPT.search(l)])
    return vals[0] if vals else None


def _num(v: float):
    return int(v) if v == int(v) else v


def _segment(tokens: list[str]) -> tuple[int, list[float]] | None:
    """Разбивает хвост числовых токенов на числа с учётом пробелов-разделителей тысяч:
    ['50','2','100','105','000'] -> возможные [50, 2100, 105000]. Возвращает тройку
    (кол-во, цена, сумма), где кол-во × цена = сумма, и индекс токена, с которого она начинается
    (всё левее — часть названия)."""

    def splits(toks: list[str], k: int):
        if k == 0:
            if not toks:
                yield []
            return
        for n in range(1, len(toks) + 1):
            head, rest = toks[:n], toks[n:]
            if any(not _GROUP_TAIL.match(t) for t in head[1:]) or any(re.search(r"[.,]", t) for t in head[:-1]):
                continue
            v = parse_amount(" ".join(head))
            if v is None:
                continue
            for tail in splits(rest, k - 1):
                yield [v, *tail]

    for start in range(len(tokens) - 2):
        for q, p, s in splits(tokens[start:], 3):
            if q > 0 and abs(q * p - s) < 0.01:
                return start, [q, p, s]
    return None


def _line_items(lines: list[str]) -> list[dict]:
    items = []
    has_unit_col = any(re.search(r"\bед\.?(\s|$)|\bunit\b", l.lower()) for l in lines)
    for raw in lines:
        low = raw.lower()
        if not raw.strip() or P.TOTAL_KW.search(low) or P.VAT_KW.search(low) or ":" in raw:
            continue
        line = raw.replace("|", " ")

        # 1) строка-список: «- Разработка сайта — 450 000 тг», «Ключи — 12 шт. по 1 500»
        body = _BULLET.sub("", line, count=1)
        parts = _DASH_SPLIT.split(body, maxsplit=1)
        if len(parts) == 2:
            name, right = parts[0].strip(), parts[1].strip()
            if m := _QTY_PRICE.match(right):
                items.append({"name": name, "qty": _num(float(m.group(1).replace(",", "."))),
                              "price": _num(parse_amount(m.group(2).strip()) or 0) or None})
                continue
            if m := _PRICE_ONLY.match(right):
                items.append({"name": name, "qty": None, "price": _num(parse_amount(m.group(1)))})
                continue

        # 2) строка таблицы: «Бумага А4 80г/м2 50 2 100 105 000»
        tokens = line.split()
        if tokens and tokens[0].isdigit() and len(tokens) > 1 and not _NUM_TOKEN.match(tokens[1]):
            tokens = tokens[1:]  # номер строки таблицы
        i = len(tokens)
        while i > 0 and _NUM_TOKEN.match(tokens[i - 1]):
            i -= 1
        name_toks, num_toks = tokens[:i], tokens[i:]
        if not name_toks:
            continue
        found = _segment(num_toks) if len(num_toks) >= 3 else None
        if found:
            start, seg = found
            name_toks = tokens[: i + start]
            if has_unit_col and name_toks[-1].lower() in _UNITS:
                name_toks = name_toks[:-1]
            items.append({"name": " ".join(name_toks).strip(" ,"), "qty": _num(seg[0]), "price": _num(seg[1])})
        elif len(num_toks) == 1 and has_unit_col and name_toks[-1].lower() in _UNITS and len(name_toks) > 1:
            # таблица без цен (накладная): «Цемент М400  мешок  40»
            items.append({"name": " ".join(name_toks[:-1]), "qty": _num(float(num_toks[0])), "price": None})
    return items

