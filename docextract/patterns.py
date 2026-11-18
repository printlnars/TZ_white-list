"""Ключевые слова (рус/каз/англ) и разметка строк документа по ролям.
Общие для гардрейлов и мок-бэкенда. Регэкспы применяются к тексту после ocr_fix(), в нижнем регистре."""

from __future__ import annotations

import re

DOC_TYPE_KW = {
    "invoice": re.compile(r"сч[её]т|\bшот\b|шот-фактура|invoice"),
    "commercial_offer": re.compile(r"коммерческ\w*\s+предложени|коммерциялық\s+ұсыныс|commercial\s+offer|quotation|\bкп\b"),
    "act": re.compile(r"\bакт\b|актісі|выполненных\s+работ|оказанных\s+услуг|орындалған\s+жұмыс"),
}

CURRENCY_KW = {
    "KZT": re.compile(r"тенге|теңге|\bтг\b|₸|\bkzt\b|тиын"),
    "USD": re.compile(r"\busd\b|\$|доллар|dollar"),
    "EUR": re.compile(r"\beur\b|€|евро|\beuro\b"),
}

VAT_KW = re.compile(r"ндс|ққс|\bvat\b")
VAT_EXEMPT = re.compile(
    r"без\s+ндс|ндс\W*не\s+облага|не\s+облагается\s+ндс|не\s+является\s+плательщиком\s+ндс|"
    r"ққс\W*сыз|ққс\s+салынбайды|vat\s+exempt|without\s+vat"
)

TOTAL_KW = re.compile(
    r"итог|всего|к\s+оплате|общая\s+стоимость|общая\s+сумма|на\s+сумму|жалпы|барлығы|төлеуге|"
    r"\btotal\b|amount\s+due"
)
SUBTOTAL_KW = re.compile(r"без\s+ндс|ққс\W*сыз|subtotal|excl\w*\s+vat|before\s+vat")

SUPPLIER_LABEL = re.compile(
    r"поставщик|исполнитель|продавец|отправитель|от\s+компании|орындаушы|жеткізуші|сатушы|\bfrom\b|supplier|seller"
)
BUYER_LABEL = re.compile(
    r"покупател|заказчик|получател|плательщик|тапсырыс\s+беруші|сатып\s+алушы|\bалушы|\bto\s*:|buyer|customer|bill\s+to"
)

LEGAL_FORM = re.compile(r"\b(ТОО|ЖШС|ИП|ЖК|АО|АҚ|КХ|LLP|LLC|GmbH|Ltd)\b")


def line_roles(text: str) -> list[str | None]:
    """Для каждой строки: 'supplier' / 'buyer' / None.
    Метка действует на свою строку и на следующие строки того же блока
    (до пустой строки или до другой метки) — так «Покупатель: ...\\nБИН: ...»
    правильно относит БИН к покупателю."""
    roles: list[str | None] = []
    cur: str | None = None
    for line in text.split("\n"):
        low = line.lower()
        if not line.strip():
            cur = None
        elif BUYER_LABEL.search(low):
            cur = "buyer"
        elif SUPPLIER_LABEL.search(low):
            cur = "supplier"
        roles.append(cur)
    return roles


VAT_INCL = re.compile(r"в\s*т\.\s*ч\.?|в\s+том\s+числе|оның\s+ішінде|\bincl")
DOC_NUMBER_MARK = re.compile(r"№|\bno\b|n°|номер|нөмір|\bnumber\b|#")


def split_total_vat(low_line: str) -> tuple[str, str]:
    """Делит строку на часть с итогом и часть с суммой НДС.
    «итого: 475 000, в т.ч. ндс 16%: 65 517»  -> ('итого: 475 000, ', 'ндс 16%: 65 517')
    «итого с ндс: 580 000»                     -> ('итого с ндс: 580 000', '')   (это итог, не НДС)
    «ндс 16%: 25 680»                          -> ('', 'ндс 16%: 25 680')"""
    v = VAT_KW.search(low_line)
    t = TOTAL_KW.search(low_line)
    if not v:
        return (low_line if t else ""), ""
    if not t:
        return "", low_line[v.start():]
    inc = VAT_INCL.search(low_line[: v.start()])
    if inc and inc.start() > t.start():
        return low_line[: inc.start()], low_line[v.start():]
    return low_line, ""


def vat_amounts(low_lines: list[str]) -> list[float]:
    from .textnorm import amounts_in_line

    return [a for l in low_lines for a in amounts_in_line(split_total_vat(l)[1])]
