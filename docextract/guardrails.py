"""Гардрейлы: детерминированная проверка ответа модели по исходному тексту.

Принцип: гардрейлы НИКОГДА не придумывают значения. Они могут только
  (а) обнулить поле, которого нет в тексте (галлюцинация модели),
  (б) выбрать между значениями, буквально присутствующими в тексте, по
      правилу из README (противоречия: две даты, две итоговые суммы),
  (в) добавить предупреждение.
Уверенность по полям считается здесь же, из того, насколько надёжно
значение подтверждено текстом (самооценке LLM не доверяем).
"""

from __future__ import annotations

import re

from . import patterns as P
from .schema import SCALAR_FIELDS, Extraction, LLMOutput
from .textnorm import (
    amounts_in_line,
    amounts_in_words,
    bin_checksum_ok,
    find_dates,
    issue_date_candidates,
    norm_name,
    number_in_text,
    ocr_fix,
)


class _Doc:
    def __init__(self, text: str):
        self.text = text
        self.fixed = ocr_fix(text)
        self.lines = self.fixed.split("\n")
        self.low_lines = [l.lower() for l in self.lines]
        self.low = self.fixed.lower()
        self.norm = norm_name(text)
        self.roles = P.line_roles(self.fixed)


def _close(a: float | None, b: float | None, tol: float = 0.01) -> bool:
    return a is not None and b is not None and abs(a - b) <= tol


def _fmt(v: float) -> str:
    return f"{v:,.2f}".replace(",", " ").replace(".00", "")


def apply(raw: LLMOutput, text: str) -> Extraction:
    doc = _Doc(text)
    out = raw.model_dump()
    conf: dict[str, float] = {}
    warns: list[str] = [f"LLM: {w}" for w in raw.warnings]

    def drop(field: str, why: str) -> None:
        warns.append(f"UNGROUNDED: поле {field}={out[field]!r} не найдено в документе ({why}) -> null")
        out[field] = None

    # ------------------------------------------------------------ doc_type
    if out["doc_type"] != "unknown":
        rx = P.DOC_TYPE_KW[out["doc_type"]]
        if rx.search(doc.low):
            conf["doc_type"] = 0.9
        else:
            conf["doc_type"] = 0.5
            warns.append(f"DOC_TYPE_WEAK: в тексте нет ключевых слов для типа {out['doc_type']}")
    else:
        conf["doc_type"] = 0.6

    # ------------------------------------------------------------ supplier_name
    if out["supplier_name"] is not None:
        n = norm_name(out["supplier_name"])
        hit_lines = [i for i, l in enumerate(doc.lines) if n and n in norm_name(l)]
        if not n or n not in doc.norm:
            drop("supplier_name", "названия нет в тексте")
        elif hit_lines and all(doc.roles[i] == "buyer" for i in hit_lines):
            warns.append(f"SUPPLIER_IS_BUYER: {out['supplier_name']!r} указан как покупатель/заказчик -> null")
            out["supplier_name"] = None
        else:
            conf["supplier_name"] = 0.9

    # ------------------------------------------------------------ supplier_bin
    if out["supplier_bin"] is not None:
        b = out["supplier_bin"]
        found = [(i, m) for i, l in enumerate(doc.lines) for m in re.finditer(r"(?<!\d)\d{12}(?!\d)", l) if m.group(0) == b]
        if not found:
            drop("supplier_bin", "такого 12-значного номера в тексте нет")
        elif all(doc.roles[i] == "buyer" for i, _ in found):
            warns.append(f"BIN_IS_BUYER: БИН {b} относится к покупателю/заказчику -> null")
            out["supplier_bin"] = None
        else:
            c = 0.95
            if not re.search(rf"(?<!\d){b}(?!\d)", text):
                orig_line = text.split("\n")[found[0][0]].strip()
                warns.append(f"BIN_OCR_CORRECTED: БИН восстановлен из OCR-искажённой строки {orig_line!r}")
                c = 0.7
            if not bin_checksum_ok(b):
                warns.append(f"BIN_CHECKSUM_INVALID: БИН {b} не проходит контрольную сумму РК — проверьте вручную")
                c = min(c, 0.5)
            conf["supplier_bin"] = c

    # ------------------------------------------------------------ doc_number
    if out["doc_number"] is not None:
        num = out["doc_number"].lstrip("№").strip()
        out["doc_number"] = num
        key = norm_name(num)
        rx = re.compile(r"(?<!\w)" + re.escape(key) + r"(?!\w)") if key else None
        hits = [i for i, l in enumerate(doc.lines) if rx and rx.search(norm_name(l))]
        if not hits:
            drop("doc_number", "номера нет в тексте")
        elif all(re.search(r"договор|келісім|contract", doc.low_lines[i]) for i in hits):
            warns.append(f"DOC_NUMBER_IS_CONTRACT: {num!r} — номер договора, не документа -> null")
            out["doc_number"] = None
        else:
            conf["doc_number"] = 0.9 if num in text else 0.7

    # ------------------------------------------------------------ doc_date
    all_dates = {f.date.isoformat() for f in find_dates(doc.fixed)}
    cands: list[str] = []
    for f in issue_date_candidates(doc.fixed):
        if f.date.isoformat() not in cands:
            cands.append(f.date.isoformat())
    if len(cands) > 1:
        warns.append(f"DATE_CONFLICT: в документе несколько дат документа {cands}; по правилу берём первую (из шапки)")
    if out["doc_date"] is not None:
        d = out["doc_date"]
        if d not in all_dates:
            drop("doc_date", "такой даты в тексте нет")
        elif d not in cands:
            if cands:
                warns.append(f"DOC_DATE_OVERRIDDEN: {d} — срок оплаты/действия или дата договора; дата документа {cands[0]}")
                out["doc_date"] = cands[0]
                conf["doc_date"] = 0.6
            else:
                warns.append(f"DOC_DATE_NOT_ISSUE: {d} — срок оплаты/действия или дата договора, а не дата документа -> null")
                out["doc_date"] = None
        else:
            if len(cands) > 1 and d != cands[0]:
                out["doc_date"] = cands[0]
            conf["doc_date"] = 0.9 if len(cands) == 1 else 0.6

    # ------------------------------------------------------------ currency
    evidence = [c for c, rx in P.CURRENCY_KW.items() if rx.search(doc.low)]
    if len(evidence) > 1:
        warns.append(f"CURRENCY_CONFLICT: в документе упомянуто несколько валют {evidence}")
    if out["currency"] is not None:
        if out["currency"] not in evidence:
            drop("currency", "валюта явно не указана")
        else:
            conf["currency"] = 0.9 if len(evidence) == 1 else 0.6

    # ------------------------------------------------------------ line_items
    items_ok = []
    for it in out["line_items"]:
        n = norm_name(it["name"])
        if not n or (n not in doc.norm and _token_overlap(n, doc.norm) < 0.8):
            warns.append(f"UNGROUNDED: позиция {it['name']!r} не найдена в документе -> удалена")
            continue
        for k in ("qty", "price"):
            if it[k] is not None and not number_in_text(it[k], doc.fixed):
                warns.append(f"UNGROUNDED: {k}={it[k]} позиции {it['name']!r} нет в тексте -> null")
                it[k] = None
        items_ok.append(it)
    out["line_items"] = items_ok
    if items_ok:
        conf["line_items"] = 0.9 if len(items_ok) == len(raw.line_items) else 0.5

    # ------------------------------------------------------------ vat_amount
    vat_lines = [l for l in doc.low_lines if P.VAT_KW.search(l)]
    text_vats = [v for l in vat_lines for v in amounts_in_line(l[P.VAT_KW.search(l).start():])]
    exempt = bool(P.VAT_EXEMPT.search(doc.low))
    if out["vat_amount"] is not None:
        if not any(_close(out["vat_amount"], v) for v in text_vats):
            drop("vat_amount", "суммы нет в строке с НДС/ҚҚС/VAT — НДС не вычисляем сами")
        else:
            conf["vat_amount"] = 0.9
    if exempt and out["vat_amount"] is None:
        warns.append("VAT_NOT_APPLICABLE: в документе указано, что НДС не облагается/без НДС -> vat_amount = null")

    # ------------------------------------------------------------ total_amount
    _resolve_total(doc, out, conf, warns, text_vats, drop)

    if out["total_amount"] is not None and out["vat_amount"] is not None and out["vat_amount"] >= out["total_amount"]:
        warns.append("VAT_GE_TOTAL: НДС не может быть больше итога -> vat_amount = null")
        out["vat_amount"] = None
        conf.pop("vat_amount", None)

    # ------------------------------------------------------------ арифметика
    s = _items_sum(out["line_items"])
    t, v = out["total_amount"], out["vat_amount"]
    if s is not None and t is not None and not (_close(s, t, 1) or (v is not None and _close(s + v, t, 1))):
        warns.append(f"ARITHMETIC_MISMATCH: сумма позиций {_fmt(s)} не сходится с итогом {_fmt(t)}")
        conf["total_amount"] = min(conf.get("total_amount", 1), 0.6)

    # null -> 0.0: «значения нет», ничего не утверждаем.
    out["field_confidence"] = {
        f: (0.0 if out[f] is None else conf.get(f, 0.5)) for f in SCALAR_FIELDS
    } | {"line_items": conf.get("line_items", 0.0) if out["line_items"] else 0.0}
    out["warnings"] = warns
    # Проверяем и собственный выход: если гардрейлы где-то ошиблись, это исключение, а не тихий мусор.
    return Extraction.model_validate(out)


def _token_overlap(a: str, b: str) -> float:
    ta, tb = a.split(), set(b.split())
    return sum(t in tb for t in ta) / len(ta) if ta else 0.0


def _items_sum(items: list[dict]) -> float | None:
    if not items or any(i["qty"] is None or i["price"] is None for i in items):
        return None
    return sum(i["qty"] * i["price"] for i in items)


def _resolve_total(doc: _Doc, out: dict, conf: dict, warns: list, text_vats: list[float], drop) -> None:
    """Итоговая сумма. Правило (см. README):
    1. Итог должен быть в тексте (цифрами или прописью) в строке с «итого/всего/к оплате/жалпы/total».
    2. Промежуточные итоги (без НДС; или X, где X + НДС = другой итог) — не итог.
    3. Если осталось несколько разных итогов — TOTAL_CONFLICT и выбор:
       сходящийся с суммой позиций (с НДС или без) > записанный прописью > последний по тексту."""
    finals: list[tuple[float, str]] = []  # (значение, 'digits'|'words') в порядке текста
    subtotals: set[float] = set()
    for l in doc.low_lines:
        if not P.TOTAL_KW.search(l):
            continue
        part = l
        vm = P.VAT_KW.search(l)
        if vm and vm.start() > P.TOTAL_KW.search(l).start():
            part = l[: vm.start()]  # «Итого: 475 000, в т.ч. НДС: 65 517» -> до НДС
        vals = [(v, "digits") for v in amounts_in_line(part)] + [(float(v), "words") for v in amounts_in_words(l)]
        if P.SUBTOTAL_KW.search(l):
            subtotals.update(v for v, _ in vals)
            continue
        finals.extend(vals)
    if finals:  # счётчики вида «всего наименований 2» — не суммы
        mx = max(v for v, _ in finals)
        finals = [(v, k) for v, k in finals if v >= mx * 0.01]
    for v, _ in finals:
        if any(_close(v + tv, w, 1) for tv in text_vats for w, _ in finals):
            subtotals.add(v)
    finals = [(v, k) for v, k in finals if not any(_close(v, s) for s in subtotals)]
    distinct: list[float] = []
    for v, _ in finals:
        if not any(_close(v, d) for d in distinct):
            distinct.append(v)

    choice = None
    if distinct:
        choice = distinct[-1]
        if len(distinct) > 1:
            s = _items_sum(out["line_items"])
            consistent = [d for d in distinct if s is not None and any(_close(d, s + tv, 1) for tv in [0.0, *text_vats])]
            words = [v for v, k in finals if k == "words"]
            choice = consistent[0] if consistent else (words[0] if words else distinct[-1])
            warns.append(
                f"TOTAL_CONFLICT: в документе разные итоговые суммы {[_fmt(d) for d in distinct]}; "
                f"по правилу выбрано {_fmt(choice)}"
            )

    t = out["total_amount"]
    if t is None:
        return
    grounded = number_in_text(t, doc.fixed) or any(_close(t, w) for w in amounts_in_words(doc.fixed))
    if not grounded:
        drop("total_amount", "суммы нет в тексте ни цифрами, ни прописью — итог не вычисляем сами")
        return
    if choice is None:
        drop("total_amount", "в документе нет строки с итогом")
        return
    if not _close(t, choice):
        reason = "промежуточный итог/без НДС" if any(_close(t, s) for s in subtotals) else "не итоговая сумма"
        warns.append(f"TOTAL_OVERRIDDEN: {_fmt(t)} — {reason}; итог по документу {_fmt(choice)}")
        out["total_amount"] = choice
    conf["total_amount"] = 0.9 if len(distinct) == 1 else 0.6
