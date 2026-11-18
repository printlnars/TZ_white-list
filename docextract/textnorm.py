"""Детерминированные утилиты для работы с текстом документа.

Здесь нет ничего «умного»: только то, что можно проверить глазами.
Используется гардрейлами (проверка, что значение реально есть в тексте)
и мок-бэкендом.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

# ---------------------------------------------------------------- OCR

# Буквы, которые OCR путает с цифрами (латиница и кириллица).
_TO_DIGIT = str.maketrans({"O": "0", "o": "0", "О": "0", "о": "0", "l": "1", "I": "1", "|": "1", "З": "3"})
# Латиница, похожая на кириллицу.
_LAT_TO_CYR = str.maketrans("AaBCcEeHKMOoPpTXxy", "АаВСсЕеНКМОоРрТХху")
_CONFUSABLE = "OoОоlI|З"
_TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі|]+")


def _fix_token(tok: str) -> str:
    digits = sum(c.isdigit() for c in tok)
    confusable = sum(c in _CONFUSABLE for c in tok)
    letters = sum(c.isalpha() for c in tok)
    # «2l064090Ol80», «2O26», «50O»: в основном цифры + похожие на цифры буквы.
    if digits and confusable and digits + confusable == len(tok) and digits >= confusable:
        return tok.translate(_TO_DIGIT)
    # «Ит0го», «Cчет», «Бecт»: слово с кириллицей, испорченное цифрой 0 или латиницей.
    has_cyr = any("а" <= c.lower() <= "я" or c in "ёЁ" for c in tok)
    if letters and has_cyr and not any(c.isdigit() and c != "0" for c in tok):
        return tok.replace("0", "о").translate(_LAT_TO_CYR)
    # «TOO», «HДC»: целиком латиница-двойник рядом с кириллицей — правим только
    # известные аббревиатуры, иначе испортим нормальные латинские названия.
    if tok in {"TOO", "HДC", "БИH", "HДС", "БCН"}:
        return tok.translate(_LAT_TO_CYR)
    return tok


def ocr_fix(text: str) -> str:
    return _TOKEN_RE.sub(lambda m: _fix_token(m.group(0)), text)


def digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)


# ---------------------------------------------------------------- БИН/ИИН

def bin_checksum_ok(b: str) -> bool:
    """Контрольный разряд БИН/ИИН Республики Казахстан (12-я цифра)."""
    if not re.fullmatch(r"\d{12}", b):
        return False
    d = [int(c) for c in b]
    s = sum(d[i] * (i + 1) for i in range(11)) % 11
    if s == 10:
        w = [3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2]
        s = sum(d[i] * w[i] for i in range(11)) % 11
        if s == 10:
            return False
    return s == d[11]


# ---------------------------------------------------------------- числа цифрами

_SEP = r"[   ,.']?"


def number_pattern(value: float) -> re.Pattern:
    """Регэксп, который находит число в тексте в любом привычном написании:
    186180 / 186 180 / 186,180 / 186 180,00 / 186180.00."""
    cents = round(abs(value) * 100)
    integer, frac = divmod(cents, 100)
    s = str(integer)
    groups = []
    while s:
        groups.insert(0, s[-3:])
        s = s[:-3]
    int_re = _SEP.join(groups)
    if frac:
        f = f"{frac:02d}"
        frac_re = r"[.,]" + (f[0] + "0?" if f[1] == "0" else f)
    else:
        frac_re = r"(?:[.,]00?)?"
    return re.compile(r"(?<![\d])" + int_re + frac_re + r"(?![\d])")


def number_in_text(value: float, text: str) -> bool:
    return bool(number_pattern(value).search(text))


_AMOUNT_RE = re.compile(r"(?<![\d.,])\d{1,3}(?:[   ]\d{3})+(?:[.,]\d{1,2})?(?![\d])|(?<![\d.,])\d+(?:[.,]\d{3})*(?:[.,]\d{1,2})?(?![\d])")


def parse_amount(s: str) -> float | None:
    """'186 180' -> 186180, '65 517,24' -> 65517.24, '12,500.00' -> 12500.0."""
    s = s.strip().replace(" ", " ").replace(" ", " ").replace(" ", "")
    if not s:
        return None
    m = re.fullmatch(r"(\d+(?:[.,]\d{3})*)(?:[.,](\d{1,2}))?", s)
    if not m:
        return None
    return float(re.sub(r"[.,]", "", m.group(1)) + ("." + m.group(2) if m.group(2) else ""))


def amounts_in_line(line: str) -> list[float]:
    """Суммы в строке, без процентов (НДС 16% -> 16 не сумма) и без дат
    («valid until December 15, 2026» -> 2026 не сумма)."""
    for f in reversed(find_dates(line)):
        line = line[: f.start] + " " + line[f.end:]
    line = re.sub(r"\d+(?:[.,]\d+)?\s*%", " ", line)
    out = []
    for m in _AMOUNT_RE.finditer(line):
        v = parse_amount(m.group(0))
        if v:  # 0 — это «00 тиын», а не сумма
            out.append(v)
    return out


# ---------------------------------------------------------------- числа прописью

_RU_ADD = {
    "ноль": 0, "один": 1, "одна": 1, "одно": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5,
    "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10, "одиннадцать": 11, "двенадцать": 12,
    "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17,
    "восемнадцать": 18, "девятнадцать": 19, "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
    "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90, "сто": 100, "двести": 200,
    "триста": 300, "четыреста": 400, "пятьсот": 500, "шестьсот": 600, "семьсот": 700, "восемьсот": 800,
    "девятьсот": 900,
}
_KZ_ADD = {
    "бір": 1, "екі": 2, "үш": 3, "төрт": 4, "бес": 5, "алты": 6, "жеті": 7, "сегіз": 8, "тоғыз": 9,
    "он": 10, "жиырма": 20, "отыз": 30, "қырық": 40, "елу": 50, "алпыс": 60, "жетпіс": 70, "сексен": 80,
    "тоқсан": 90,
}
_HUNDRED_MUL = {"жүз": 100}  # казахское «үш жүз» = 3 × 100
_SCALE = {
    "тысяча": 1000, "тысячи": 1000, "тысяч": 1000, "мың": 1000,
    "миллион": 10**6, "миллиона": 10**6, "миллионов": 10**6,
}
_NUM_WORDS = set(_RU_ADD) | set(_KZ_ADD) | set(_HUNDRED_MUL) | set(_SCALE)


def words_to_number(words: list[str]) -> int:
    total, cur = 0, 0
    for w in words:
        if w in _RU_ADD:
            cur += _RU_ADD[w]
        elif w in _KZ_ADD:
            cur += _KZ_ADD[w]
        elif w in _HUNDRED_MUL:
            cur = (cur or 1) * _HUNDRED_MUL[w]
        elif w in _SCALE:
            total += (cur or 1) * _SCALE[w]
            cur = 0
    return total + cur


def amounts_in_words(text: str) -> list[int]:
    """Все суммы прописью в тексте (рус/каз). Одиночные слова («он», «три»)
    не считаем: слишком часто это не сумма."""
    tokens = re.findall(r"[^\W\d_]+", text.lower().replace("ё", "е"))
    out, run = [], []
    for t in tokens + [""]:
        if t in _NUM_WORDS:
            run.append(t)
            continue
        if len(run) >= 2:
            out.append(words_to_number(run))
        run = []
    return out


# ---------------------------------------------------------------- даты

_RU_MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6, "июл": 7, "август": 8,
    "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}
_KZ_MONTHS = {
    "қаңтар": 1, "ақпан": 2, "наурыз": 3, "сәуір": 4, "мамыр": 5, "маусым": 6, "шілде": 7,
    "тамыз": 8, "қыркүйек": 9, "қазан": 10, "қараша": 11, "желтоқсан": 12,
}
_EN_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"])}


def _month(word: str) -> int | None:
    w = word.lower()
    if w in _EN_MONTHS:
        return _EN_MONTHS[w]
    for stem, n in _KZ_MONTHS.items():
        if w.startswith(stem):
            return n
    if w in ("мая", "май"):
        return 5
    for stem, n in _RU_MONTHS.items():
        if len(stem) > 2 and w.startswith(stem):
            return n
    return None


@dataclass
class FoundDate:
    date: dt.date
    start: int
    end: int


_W = r"([^\W\d_]+)"
_DATE_RES = [
    ("dmy", re.compile(r"(?<!\d)(\d{1,2})[./-](\d{1,2})[./-](\d{4})(?!\d)")),
    ("ymd", re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")),
    ("d_mon_y", re.compile(r"(?<!\d)(\d{1,2})\s+" + _W + r"\s+(\d{4})(?!\d)")),
    ("y_d_mon", re.compile(r"(?<!\d)(\d{4})\s*(?:жылғы|ж\.)?\s+(\d{1,2})\s+" + _W)),
    ("mon_d_y", re.compile(_W + r"\s+(\d{1,2}),?\s+(\d{4})(?!\d)")),
]


def find_dates(text: str) -> list[FoundDate]:
    """Все даты в тексте (после OCR-нормализации), по порядку появления."""
    out: list[FoundDate] = []
    for kind, rx in _DATE_RES:
        for m in rx.finditer(text):
            g = m.groups()
            try:
                if kind == "dmy":
                    d = dt.date(int(g[2]), int(g[1]), int(g[0]))
                elif kind == "ymd":
                    d = dt.date(int(g[0]), int(g[1]), int(g[2]))
                elif kind == "d_mon_y":
                    mo = _month(g[1])
                    if not mo:
                        continue
                    d = dt.date(int(g[2]), mo, int(g[0]))
                elif kind == "y_d_mon":
                    mo = _month(g[2])
                    if not mo:
                        continue
                    d = dt.date(int(g[0]), mo, int(g[1]))
                else:
                    mo = _month(g[0])
                    if not mo:
                        continue
                    d = dt.date(int(g[2]), mo, int(g[1]))
            except ValueError:
                continue
            out.append(FoundDate(d, m.start(), m.end()))
    out.sort(key=lambda f: f.start)
    # одна и та же дата могла попасть под два шаблона
    uniq: list[FoundDate] = []
    for f in out:
        if not any(u.start <= f.start < u.end for u in uniq):
            uniq.append(f)
    return uniq


# Контекст перед датой, означающий «это не дата документа».
_NOT_ISSUE_CTX = re.compile(
    r"(до|действител\w*|срок\w*|оплатить|оплаты|дейін|valid|until|due|договор\w*|келісім\w*|contract)\W*$",
    re.IGNORECASE,
)


def issue_date_candidates(text: str) -> list[FoundDate]:
    """Даты, которые могут быть датой документа: всё, кроме сроков действия/оплаты
    и дат договоров-оснований. Порядок — как в документе."""
    res = []
    for f in find_dates(text):
        line_start = text.rfind("\n", 0, f.start) + 1
        before = text[line_start:f.start]
        # «Договор № 12/2026 от 15.01.2026» — смотрим всю строку до даты.
        if _NOT_ISSUE_CTX.search(before[-25:]) or re.search(r"договор|келісім|contract", before, re.I):
            continue
        res.append(f)
    return res


# ---------------------------------------------------------------- названия

_KZ_TO_RU = str.maketrans("әғқңөұүһі", "агкноуухи")
_LEGAL_FORMS = r"\b(тоо|жшс|ип|жк|ао|ақ|ак|оао|зао|кх|llp|llc|ltd|gmbh|inc)\b"


def norm_name(s: str) -> str:
    """Нормализация названия для сравнения: регистр, казахские буквы, латиница-двойники,
    кавычки, организационно-правовая форма."""
    s = ocr_fix(s).lower().replace("ё", "е").translate(_KZ_TO_RU)
    s = re.sub(_LEGAL_FORMS, " ", s)
    s = re.sub(r"[^\w]+", " ", s)
    return " ".join(s.split())


def norm_text(s: str) -> str:
    return norm_name(s)
