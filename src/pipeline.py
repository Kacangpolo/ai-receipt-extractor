"""Batch-process a folder of receipt photos.

Usage:
    python -m src.pipeline                      # images in data/receipts -> JSON + Postgres
    python -m src.pipeline --no-db              # JSON files only (no Docker needed)
    python -m src.pipeline --input other/folder --limit 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from .extractor import get_extractor, get_fallback_extractor

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
log = logging.getLogger("pipeline")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def run(input_dir: Path, output_dir: Path, use_db: bool, limit: int | None,
        extractor=None, fallback="auto") -> dict:
    """fallback: an extractor for a second opinion on flagged receipts,
    "auto" = build one from .env when first needed, None = never retry."""
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = extractor or get_extractor()
    model = extractor.model
    log.info("Using model %s", model)

    conn = None
    if use_db:
        from . import db
        conn = db.connect()

    stats = {"processed": 0, "skipped": 0, "failed": 0, "flagged": 0, "retried": 0,
             "input_tokens": 0, "output_tokens": 0}
    images = find_images(input_dir)[:limit] if limit else find_images(input_dir)
    log.info("Found %d images in %s", len(images), input_dir)

    for path in images:
        out_file = output_dir / f"{path.stem}.json"
        file_hash = sha256(path)
        if out_file.exists() and (conn is None or db.already_processed(conn, file_hash)):
            stats["skipped"] += 1
            continue
        try:
            t0 = time.perf_counter()
            receipt, tokens = extractor.extract(path)
            used_model, issues = model, receipt.issues()
            if issues and fallback is not None:
                if fallback == "auto":
                    try:
                        fallback = get_fallback_extractor()
                    except Exception as exc:
                        log.warning("No fallback model: %s", exc)
                        fallback = None
                if fallback is not None:
                    log.info("RETRY %s with %s: %s", path.name, fallback.model, "; ".join(issues))
                    stats["retried"] += 1
                    try:
                        second, t2 = fallback.extract(path)
                        tokens = {k: tokens.get(k, 0) + t2.get(k, 0) for k in ("input_tokens", "output_tokens")}
                        if len(second.issues()) < len(issues):
                            receipt, used_model, issues = second, fallback.model, second.issues()
                    except Exception as exc:  # keep the first answer if the retry fails
                        log.warning("Retry failed for %s: %s", path.name, exc)
            out = json.loads(receipt.model_dump_json())
            out["_meta"] = {"model": used_model, "issues": issues}
            out_file.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
            if conn is not None and not db.already_processed(conn, file_hash):
                db.save_receipt(conn, receipt, source_file=path.name, file_hash=file_hash,
                                model=used_model, tokens=tokens)
            stats["processed"] += 1
            stats["flagged"] += bool(issues)
            for k in ("input_tokens", "output_tokens"):
                stats[k] += tokens.get(k, 0)
            log.info("%s %-30s total=%s items=%d %s (%.1fs)", "OK  " if not issues else "FLAG",
                     path.name, receipt.total, len(receipt.items),
                     "; ".join(issues) or "checks passed", time.perf_counter() - t0)
        except Exception as exc:  # keep going; one bad photo shouldn't stop the batch
            stats["failed"] += 1
            log.error("FAIL %s: %s", path.name, exc)

    if conn is not None:
        conn.close()
    log.info("Done: %s", stats)
    return stats


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Extract receipts with an LLM")
    ap.add_argument("--input", type=Path, default=Path("data/receipts"))
    ap.add_argument("--output", type=Path, default=Path("data/output"))
    ap.add_argument("--no-db", action="store_true", help="only write JSON files")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(args.input, args.output, use_db=not args.no_db, limit=args.limit)


if __name__ == "__main__":
    main()
