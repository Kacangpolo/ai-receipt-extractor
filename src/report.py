"""Turn the extracted JSON files into tables, summaries and an Excel export.

Used by the dashboard, and also runnable on its own:
    python -m src.report            -> data/output/receipts.xlsx
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pandas as pd

from .schema import Receipt

REVIEW_CONFIDENCE = 0.7


def load_results(output_dir: Path = Path("data/output")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read every <receipt>.json into a receipts table and an items table."""
    receipts, items = [], []
    for f in sorted(Path(output_dir).glob("*.json")):
        try:
            r = Receipt.model_validate(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue  # skip files that aren't receipt outputs
        receipts.append({
            "file": f.stem,
            "merchant_name": r.merchant_name,
            "receipt_date": pd.to_datetime(r.receipt_date) if r.receipt_date else pd.NaT,
            "n_items": len(r.items),
            "subtotal": r.subtotal,
            "tax": r.tax,
            "total": r.total,
            "sum_check": r.total_matches_items(),
            "issues": "; ".join(r.issues()),
            "confidence": r.confidence,
            "notes": r.notes,
        })
        for n, it in enumerate(r.items, start=1):
            items.append({
                "file": f.stem, "merchant_name": r.merchant_name, "line_no": n,
                "item": it.name, "quantity": it.quantity,
                "unit_price": it.unit_price, "line_total": it.line_total,
            })

    rec_cols = ["file", "merchant_name", "receipt_date", "n_items", "subtotal", "tax",
                "total", "sum_check", "issues", "confidence", "notes"]
    item_cols = ["file", "merchant_name", "line_no", "item", "quantity", "unit_price", "line_total"]
    rec = pd.DataFrame(receipts, columns=rec_cols)
    itm = pd.DataFrame(items, columns=item_cols)
    for col in ("subtotal", "tax", "total"):
        rec[col] = pd.to_numeric(rec[col], errors="coerce")
    for col in ("quantity", "unit_price", "line_total"):
        itm[col] = pd.to_numeric(itm[col], errors="coerce")
    return rec, itm


def needs_review(rec: pd.DataFrame, min_conf: float = REVIEW_CONFIDENCE) -> pd.DataFrame:
    """Receipts a human should double-check."""
    reasons = []
    for _, r in rec.iterrows():
        why = [r["issues"]] if r["issues"] else []
        if r["confidence"] < min_conf:
            why.append(f"low confidence ({r['confidence']:.2f})")
        reasons.append("; ".join(why))
    out = rec.assign(reason=reasons)
    return out[out["reason"] != ""][["file", "merchant_name", "receipt_date", "total", "reason"]]


def daily_spend(rec: pd.DataFrame) -> pd.DataFrame:
    d = rec.dropna(subset=["receipt_date"])
    return (d.groupby(d["receipt_date"].dt.date)["total"].sum()
             .rename_axis("date").reset_index())


def top_merchants(rec: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return (rec.groupby("merchant_name", dropna=False)
               .agg(receipts=("file", "count"), total=("total", "sum"))
               .sort_values("total", ascending=False).head(n).reset_index())


def top_items(itm: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    key = itm["item"].str.strip().str.lower()
    return (itm.assign(item_key=key)
               .groupby("item_key")
               .agg(item=("item", "first"), quantity=("quantity", "sum"), spend=("line_total", "sum"))
               .sort_values("spend", ascending=False).head(n).reset_index(drop=True))


def to_excel_bytes(rec: pd.DataFrame, itm: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        r = rec.copy()
        r["receipt_date"] = r["receipt_date"].dt.date
        r.to_excel(xw, sheet_name="Receipts", index=False)
        itm.to_excel(xw, sheet_name="Items", index=False)
        for ws in xw.book.worksheets:  # readable column widths
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 10), 40)
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description="Export extracted receipts to Excel")
    ap.add_argument("--output", type=Path, default=Path("data/output"))
    ap.add_argument("--xlsx", type=Path, default=Path("data/output/receipts.xlsx"))
    args = ap.parse_args()
    rec, itm = load_results(args.output)
    args.xlsx.write_bytes(to_excel_bytes(rec, itm))
    print(f"Wrote {len(rec)} receipts / {len(itm)} items to {args.xlsx}")


if __name__ == "__main__":
    main()
