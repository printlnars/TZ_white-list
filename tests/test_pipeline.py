"""Тесты пайплайна: валидация, ретрай, фолбэк, гардрейлы против типичных галлюцинаций."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from docextract.llm import ScriptedLLM
from docextract.pipeline import extract
from docextract.schema import LLMOutput

DATA = Path(__file__).parent.parent / "dataset"
DOC1 = (DATA / "01_given_invoice.txt").read_text(encoding="utf-8")


def resp(**kw) -> str:
    base = dict(doc_type="invoice", supplier_name=None, supplier_bin=None, doc_number=None, doc_date=None,
                currency=None, total_amount=None, vat_amount=None, line_items=[], warnings=[])
    base.update(kw)
    return json.dumps(base, ensure_ascii=False)


# ---------------------------------------------------------------- схема

@pytest.mark.parametrize("bad", [
    {"supplier_bin": "17034002155"},          # 11 цифр
    {"supplier_bin": "17034002155A"},
    {"doc_date": "12.11.2026"},               # не ISO
    {"doc_date": "2026-02-30"},               # несуществующая дата
    {"total_amount": "186 180"},              # строка вместо числа
    {"currency": "RUB"},
    {"doc_type": "bill"},
    {"total_amount": -5},
    {"comment": "лишнее поле"},
])
def test_schema_rejects(bad):
    with pytest.raises(ValidationError):
        LLMOutput.model_validate_json(resp(**bad))


# ---------------------------------------------------------------- ретрай / фолбэк

def test_retry_after_invalid_json_then_success():
    llm = ScriptedLLM(["Вот ваш JSON: {doc_type: invoice", resp(doc_number="1245")])
    r = extract(DOC1, llm)
    assert r.ok and r.attempts == 2
    assert r.extraction.doc_number == "1245"
    # во второй запрос модели ушли ошибки первой попытки
    assert "не прошёл валидацию" in llm.calls[1][-1]["content"]


def test_retry_sends_schema_errors_to_model():
    llm = ScriptedLLM([resp(supplier_bin="17034002155"), resp(supplier_bin="170340021554")])
    r = extract(DOC1, llm)
    assert r.ok and r.attempts == 2
    assert "supplier_bin" in llm.calls[1][-1]["content"]


def test_fallback_when_model_never_returns_valid_json():
    r = extract(DOC1, ScriptedLLM(['{"doc_type": "bill"}']))
    assert not r.ok and r.attempts == 3
    e = r.extraction
    assert e.doc_type == "unknown" and e.total_amount is None and e.supplier_bin is None
    assert e.warnings[0].startswith("VALIDATION_FAILED")


def test_code_fences_are_tolerated():
    r = extract(DOC1, ScriptedLLM(["```json\n" + resp(doc_number="1245") + "\n```"]))
    assert r.ok and r.attempts == 1


# ---------------------------------------------------------------- гардрейлы

def run(text: str, **kw):
    return extract(text, ScriptedLLM([resp(**kw)])).extraction


def codes(e):
    return {w.split(":")[0] for w in e.warnings}


def test_bin_not_in_text_is_nulled():
    e = run(DOC1, supplier_bin="170340021555")
    assert e.supplier_bin is None and "UNGROUNDED" in codes(e)


def test_todays_date_is_nulled():
    e = run(DOC1, doc_date="2026-11-18")
    assert e.doc_date is None


def test_computed_vat_is_nulled():
    text = (DATA / "13_hard_offer_mixed_words.txt").read_text(encoding="utf-8")
    e = run(text, total_amount=1650000, vat_amount=227586.21)  # 1 650 000 * 16/116 — в тексте нет
    assert e.total_amount == 1650000 and e.vat_amount is None


def test_sum_of_items_is_not_a_total():
    text = (DATA / "02_given_offer.txt").read_text(encoding="utf-8")
    e = run(text, doc_type="commercial_offer", total_amount=510000)
    assert e.total_amount is None


def test_buyer_bin_is_not_supplier_bin():
    text = (DATA / "08_hard_missing_fields.txt").read_text(encoding="utf-8")
    e = run(text, supplier_bin="230140055413")
    assert e.supplier_bin is None and "BIN_IS_BUYER" in codes(e)


def test_due_date_is_not_doc_date():
    text = (DATA / "08_hard_missing_fields.txt").read_text(encoding="utf-8")
    e = run(text, doc_date="2026-11-30")
    assert e.doc_date is None


def test_validity_date_replaced_by_issue_date():
    text = (DATA / "02_given_offer.txt").read_text(encoding="utf-8")
    e = run(text, doc_type="commercial_offer", doc_date="2026-11-30")
    assert e.doc_date == "2026-11-05" and "DOC_DATE_OVERRIDDEN" in codes(e)


def test_ocr_bin_accepted_with_warning_and_lower_confidence():
    text = (DATA / "03_given_kz_ocr.txt").read_text(encoding="utf-8")
    e = run(text, supplier_bin="210640900180")
    assert e.supplier_bin == "210640900180"
    assert "BIN_OCR_CORRECTED" in codes(e)
    assert e.field_confidence["supplier_bin"] < 0.9


def test_total_in_words_is_grounded():
    text = (DATA / "03_given_kz_ocr.txt").read_text(encoding="utf-8")
    e = run(text, total_amount=340000)
    assert e.total_amount == 340000


def test_subtotal_replaced_by_final_total():
    e = run(DOC1, total_amount=160500, vat_amount=25680)
    assert e.total_amount == 186180 and "TOTAL_OVERRIDDEN" in codes(e)


def test_total_conflict_resolved_by_items():
    text = (DATA / "06_hard_total_conflict.txt").read_text(encoding="utf-8")
    items = [{"name": "Пакет ПНД 30х60, упак.", "qty": 100, "price": 1200},
             {"name": "Стретч-плёнка 500 мм", "qty": 20, "price": 6500}]
    e = run(text, total_amount=255000, line_items=items)
    assert e.total_amount == 250000 and "TOTAL_CONFLICT" in codes(e)


def test_date_conflict_warns_and_takes_header_date():
    text = (DATA / "09_hard_date_conflict.txt").read_text(encoding="utf-8")
    e = run(text, doc_date="2026-10-03")
    assert e.doc_date == "2026-10-01" and "DATE_CONFLICT" in codes(e)


def test_invented_line_item_removed_and_price_nulled():
    items = [{"name": "Бумага А4 80г/м2", "qty": 50, "price": 2099},
             {"name": "Доставка до офиса", "qty": 1, "price": 5000}]
    e = run(DOC1, line_items=items)
    assert [i.name for i in e.line_items] == ["Бумага А4 80г/м2"]
    assert e.line_items[0].price is None and e.line_items[0].qty == 50


def test_currency_not_in_text_is_nulled():
    text = (DATA / "08_hard_missing_fields.txt").read_text(encoding="utf-8")
    assert run(text, currency="KZT").currency is None


def test_null_fields_have_zero_confidence():
    e = run(DOC1, doc_number="1245")
    assert e.field_confidence["doc_number"] > 0.5
    assert e.field_confidence["supplier_bin"] == 0.0
