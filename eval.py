"""Мини-eval: прогоняет весь dataset/ через пайплайн и печатает отчёт.

    python eval.py                      # бэкенд rules (офлайн-мок)
    python eval.py --backend greedy     # мок с типичными галлюцинациями LLM
    python eval.py --backend ollama     # локальная модель (OLLAMA_MODEL, OLLAMA_HOST)

Для каждого поля считаются раздельно:
  correct       — совпало с эталоном (null == null тоже верно);
  hallucination — поле заполнено, а в эталоне null  (самая дорогая ошибка);
  omission      — null, а в эталоне есть значение;
  wrong         — оба заполнены, но значение другое.
doc_type="unknown" считается аналогом null.
Метрики считаются дважды: по сырому ответу модели (после валидации схемы, ДО гардрейлов)
и по финальному ответу — видно, что именно дают гардрейлы.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from docextract.llm import make_client
from docextract.pipeline import extract
from docextract.schema import SCALAR_FIELDS
from docextract.textnorm import norm_name

FIELDS = list(SCALAR_FIELDS) + ["line_items"]
KINDS = ("correct", "hallucination", "omission", "wrong")


def _is_null(field: str, v) -> bool:
    return v is None or (field == "doc_type" and v == "unknown")


def _name_eq(a: str, b: str) -> bool:
    na, nb = norm_name(a), norm_name(b)
    return na == nb or SequenceMatcher(None, na, nb).ratio() >= 0.85


def _num_eq(a, b) -> bool:
    return a is not None and b is not None and abs(a - b) < 0.01


def values_equal(field: str, p, g) -> bool:
    if p is None or g is None:
        return p is None and g is None
    if field in ("total_amount", "vat_amount"):
        return _num_eq(p, g)
    if field == "supplier_name":
        return _name_eq(p, g)
    if field == "doc_number":
        return norm_name(p).replace(" ", "") == norm_name(g).replace(" ", "")
    return p == g


def classify(field: str, p, g) -> str:
    if _is_null(field, p) and _is_null(field, g):
        return "correct"
    if _is_null(field, g):
        return "hallucination"
    if _is_null(field, p):
        return "omission"
    return "correct" if values_equal(field, p, g) else "wrong"


def compare_items(pred: list[dict], gold: list[dict]) -> dict:
    """Позиции: сопоставляем по названию; лишние = галлюцинации, ненайденные = пропуски;
    внутри сопоставленных qty/price классифицируются как обычные поля."""
    c = defaultdict(int)
    unused = list(pred)
    for g in gold:
        best = max(unused, key=lambda p: SequenceMatcher(None, norm_name(p["name"]), norm_name(g["name"])).ratio(), default=None)
        if best is None or SequenceMatcher(None, norm_name(best["name"]), norm_name(g["name"])).ratio() < 0.8:
            c["omission"] += 1
            continue
        unused.remove(best)
        c["matched"] += 1
        for k in ("qty", "price"):
            kind = classify(k, best[k], g[k])
            if kind == "wrong" and _num_eq(best[k], g[k]):
                kind = "correct"
            c[kind if kind != "correct" else "sub_correct"] += 1
    c["hallucination"] += len(unused)
    exact = not unused and c["matched"] == len(gold) and not (c["wrong"] or c["omission"] or c["hallucination"])
    c["exact"] = int(exact)
    return c


def score(pred: dict, gold: dict) -> dict:
    res = {}
    for f in SCALAR_FIELDS:
        res[f] = classify(f, pred.get(f), gold.get(f))
    res["line_items"] = compare_items(pred.get("line_items", []), gold.get("line_items", []))
    return res


def run(backend: str, dataset: Path, verbose: bool) -> dict:
    llm = make_client(backend)
    docs = sorted(dataset.glob("*.txt"))
    stages = {"raw": defaultdict(lambda: defaultdict(int)), "final": defaultdict(lambda: defaultdict(int))}
    warn_hit = warn_total = 0
    failed = []
    per_doc = []
    t0 = time.time()
    for doc in docs:
        gold_file = json.loads(doc.with_suffix(".json").read_text(encoding="utf-8"))
        gold = gold_file["expected"]
        r = extract(doc.read_text(encoding="utf-8"), llm)
        if not r.ok:
            failed.append(doc.stem)
        final = r.extraction.model_dump()
        raw = r.raw.model_dump() if r.raw else final
        doc_report = {"doc": doc.stem, "ok": r.ok, "attempts": r.attempts, "errors": {}}
        for stage, pred in (("raw", raw), ("final", final)):
            sc = score(pred, gold)
            for f in SCALAR_FIELDS:
                stages[stage][f][sc[f]] += 1
                if stage == "final" and sc[f] != "correct":
                    doc_report["errors"][f] = {"kind": sc[f], "pred": pred.get(f), "gold": gold.get(f)}
            li = sc["line_items"]
            stages[stage]["line_items"]["correct" if li["exact"] else "_"] += 1
            for k in ("hallucination", "omission", "wrong"):
                stages[stage]["line_items"][k] += li[k]
            if stage == "final" and not li["exact"]:
                doc_report["errors"]["line_items"] = {k: li[k] for k in ("hallucination", "omission", "wrong")}
        codes = {w.split(":", 1)[0] for w in final["warnings"]}
        for code in gold_file.get("expected_warnings", []):
            warn_total += 1
            warn_hit += code in codes
            if code not in codes:
                doc_report["errors"].setdefault("missing_warnings", []).append(code)
        doc_report["warnings"] = final["warnings"]
        per_doc.append(doc_report)

    n = len(docs)
    report = {
        "backend": llm.name,
        "documents": n,
        "seconds": round(time.time() - t0, 1),
        "validation_failed": failed,
        "fields": {
            stage: {f: {k: stages[stage][f][k] for k in KINDS} | {"accuracy": round(stages[stage][f]["correct"] / n, 3)}
                    for f in FIELDS}
            for stage in stages
        },
        "warnings_recall": f"{warn_hit}/{warn_total}",
        "per_doc": per_doc,
    }
    _print(report, verbose)
    return report


def _print(rep: dict, verbose: bool) -> None:
    fin, raw = rep["fields"]["final"], rep["fields"]["raw"]
    print(f"\nBackend: {rep['backend']}   documents: {rep['documents']}   time: {rep['seconds']}s")
    print(f"Не прошли валидацию схемы (fallback): {len(rep['validation_failed'])} {rep['validation_failed'] or ''}\n")
    print(f"{'field':<15}{'accuracy':>9}{'halluc.':>9}{'omiss.':>8}{'wrong':>7}   | до гардрейлов: acc / halluc / omiss")
    print("-" * 92)
    tot = defaultdict(int)
    for f in FIELDS:
        m, r = fin[f], raw[f]
        print(f"{f:<15}{m['accuracy']:>9.0%}{m['hallucination']:>9}{m['omission']:>8}{m['wrong']:>7}"
              f"   | {r['accuracy']:>5.0%} / {r['hallucination']:>3} / {r['omission']:>3}")
        for k in ("hallucination", "omission", "wrong"):
            tot[k] += m[k]
            tot["raw_" + k] += r[k]
    print("-" * 92)
    scalar_acc = sum(fin[f]["correct"] for f in SCALAR_FIELDS) / (len(SCALAR_FIELDS) * rep["documents"])
    print(f"Точность по скалярным полям (micro): {scalar_acc:.1%}")
    print(f"ГАЛЛЮЦИНАЦИИ (заполнено, в эталоне null): {tot['hallucination']}   (до гардрейлов: {tot['raw_hallucination']})")
    print(f"ПРОПУСКИ     (null, в эталоне значение):  {tot['omission']}   (до гардрейлов: {tot['raw_omission']})")
    print(f"Неверные значения:                        {tot['wrong']}   (до гардрейлов: {tot['raw_wrong']})")
    print(f"Ожидаемые предупреждения найдены:         {rep['warnings_recall']}")
    print("(line_items: accuracy = весь список совпал; halluc/omiss/wrong — по позициям и их qty/price)")
    errs = [d for d in rep["per_doc"] if d["errors"]]
    if errs:
        print("\nОшибки по документам:")
        for d in errs:
            for f, e in d["errors"].items():
                print(f"  {d['doc']:<30} {f:<15} {json.dumps(e, ensure_ascii=False)}")
    if verbose:
        print("\nПредупреждения по документам:")
        for d in rep["per_doc"]:
            for w in d["warnings"]:
                print(f"  {d['doc']:<30} {w}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="rules", help="rules | greedy | ollama | openai (default: rules)")
    ap.add_argument("--dataset", type=Path, default=Path(__file__).parent / "dataset")
    ap.add_argument("--json", type=Path, help="сохранить полный отчёт в JSON")
    ap.add_argument("-v", "--verbose", action="store_true", help="показать предупреждения по каждому документу")
    a = ap.parse_args()
    rep = run(a.backend, a.dataset, a.verbose)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nПолный отчёт: {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
