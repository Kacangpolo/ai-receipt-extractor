"""Write extracted receipts to PostgreSQL."""
from __future__ import annotations

import os

from .schema import Receipt


def connect():
    import psycopg  # lazy import
    return psycopg.connect(os.environ["DATABASE_URL"])


def already_processed(conn, file_hash: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM receipts WHERE file_hash = %s", (file_hash,))
        return cur.fetchone() is not None


def save_receipt(conn, receipt: Receipt, *, source_file: str, file_hash: str,
                 model: str, tokens: dict) -> int:
    with conn.transaction(), conn.cursor() as cur:
        cur.execute(
            """INSERT INTO receipts (source_file, file_hash, merchant_name, receipt_date, currency,
                   subtotal, tax, total, total_matches_items, confidence, notes,
                   model, input_tokens, output_tokens, raw_json)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               RETURNING id""",
            (source_file, file_hash, receipt.merchant_name, receipt.receipt_date, receipt.currency,
             receipt.subtotal, receipt.tax, receipt.total, receipt.total_matches_items(),
             receipt.confidence, receipt.notes, model,
             tokens.get("input_tokens"), tokens.get("output_tokens"),
             receipt.model_dump_json()),
        )
        receipt_id = cur.fetchone()[0]
        cur.executemany(
            """INSERT INTO receipt_items (receipt_id, line_no, name, quantity, unit_price, line_total)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            [(receipt_id, n, i.name, i.quantity, i.unit_price, i.line_total)
             for n, i in enumerate(receipt.items, start=1)],
        )
    return receipt_id
