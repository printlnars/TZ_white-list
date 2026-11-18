import datetime as dt

from docextract.textnorm import (
    amounts_in_line,
    amounts_in_words,
    bin_checksum_ok,
    find_dates,
    number_in_text,
    ocr_fix,
    parse_amount,
)


def test_words_to_numbers_ru_kz():
    assert amounts_in_words("Үш жүз қырық мың теңге 00 тиын") == [340000]
    assert amounts_in_words("Двести пятьдесят пять тысяч тенге") == [255000]
    assert amounts_in_words("бір миллион алты жүз елу мың теңге") == [1650000]
    assert amounts_in_words("Пятьсот восемьдесят тысяч") == [580000]


def test_ocr_fix():
    assert ocr_fix("БИН: 2l064090Ol80") == "БИН: 210640900180"
    assert ocr_fix("Ит0го") == "Итого"
    assert ocr_fix('TOO "Бecт Принт"') == 'ТОО "Бест Принт"'
    assert ocr_fix("Industrial IoT gateway") == "Industrial IoT gateway"  # латиницу не трогаем


def test_bin_checksum():
    assert bin_checksum_ok("210640900180")
    assert not bin_checksum_ok("170340021554")  # БИН из документа 1 ТЗ
    assert not bin_checksum_ok("12345")


def test_dates():
    d = lambda s: [f.date for f in find_dates(s)]
    assert d("09 қазан 2026 ж.") == [dt.date(2026, 10, 9)]
    assert d("2026 жылғы 25 қыркүйек") == [dt.date(2026, 9, 25)]
    assert d("от 15 октября 2026 г.") == [dt.date(2026, 10, 15)]
    assert d("Date: November 4, 2026") == [dt.date(2026, 11, 4)]
    assert d("31.02.2026") == []


def test_amounts():
    assert parse_amount("65 517,24") == 65517.24
    assert parse_amount("12,500.00") == 12500
    assert amounts_in_line("НДС 16%: 25 680") == [25680]
    assert amounts_in_line("Offer valid until December 15, 2026") == []
    assert number_in_text(2100, "50 2 100 105 000")
    assert not number_in_text(186180, "1186 180")
