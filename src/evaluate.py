"""Measure extraction accuracy against hand-labelled ground truth.

Fill data/labels.csv (one row per photo, typed by you from the real receipt):
    file,merchant_name,receipt_date,total
    nota_01.jpg,Indomaret,2026-10-10,45500

Then run:  python -m src.evaluate
The printed numbers are what go in your README and CV.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

FIELDS = ("merchant_name", "receipt_date", "total")


def _norm_text(v) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v or "").lower())


def _norm_amount(v):
    try:
        return Decimal(str(v)).quantize(Decimal("1")) if v not in (None, "") else None
    except InvalidOperation:
        return None


def field_matches(field: str, expected, actual) -> bool:
    if field == "total":
        return _norm_amount(expected) == _norm_amount(actual)
    if field == "merchant_name":
        e, a = _norm_text(expected), _norm_text(actual)
        return bool(e) and (e in a or a in e) if a else False
    return str(expected or "").strip() == str(actual or "").strip()


def evaluate(labels_csv: Path, output_dir: Path) -> dict:
    rows = list(csv.DictReader(labels_csv.open(newline="", encoding="utf-8")))
    correct = {f: 0 for f in FIELDS}
    missing, all_correct, mistakes = 0, 0, []

    for row in rows:
        out = output_dir / f"{Path(row['file']).stem}.json"
        if not out.exists():
            missing += 1
            continue
        pred = json.loads(out.read_text())
        ok_all = True
        for f in FIELDS:
            if field_matches(f, row.get(f), pred.get(f)):
                correct[f] += 1
            else:
                ok_all = False
                mistakes.append((row["file"], f, row.get(f), pred.get(f)))
        all_correct += ok_all

    n = len(rows) - missing
    pct = lambda k: round(100 * k / n, 1) if n else 0.0
    return {
        "evaluated": n,
        "missing_output": missing,
        "field_accuracy_pct": {f: pct(c) for f, c in correct.items()},
        "fully_correct_pct": pct(all_correct),
        "mistakes": mistakes,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("data/labels.csv"))
    ap.add_argument("--output", type=Path, default=Path("data/output"))
    args = ap.parse_args()
    r = evaluate(args.labels, args.output)
    print(f"Evaluated {r['evaluated']} receipts ({r['missing_output']} without output)")
    for f, p in r["field_accuracy_pct"].items():
        print(f"  {f:<14} {p:>5}%")
    print(f"  {'ALL FIELDS':<14} {r['fully_correct_pct']:>5}%")
    if r["mistakes"]:
        print("\nMistakes (file, field, expected, got):")
        for m in r["mistakes"]:
            print("  ", m)


if __name__ == "__main__":
    main()
