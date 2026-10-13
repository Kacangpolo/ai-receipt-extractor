"""Google Gemini backend (has a free tier).

Get a key at https://aistudio.google.com/apikey and put it in .env as GEMINI_API_KEY.
"""
from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
import re
import time
from pathlib import Path

from pydantic import ValidationError

from .schema import Receipt, tool_input_schema

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"

PROMPT = """You extract data from a photo of a sales receipt (nota), mostly Indonesian.
Return ONLY JSON matching the given schema.
Rules:
- Copy values exactly as printed; never invent missing values, use null instead.
- Amounts are numbers without separators. Indonesian receipts use "." for thousands:
  "Rp 25.000" -> 25000, "4.000" -> 4000 (never 4).
- merchant_name is the business name printed at the top.
- items: EVERY product line between the header and SUBTOTAL, in order. Count them before answering.
- subtotal, tax and total are copied from their own printed lines. total is the line labelled
  TOTAL; never use TUNAI/CASH/KEMBALI and never copy an item price into total.
- Copy the date into date_printed EXACTLY as printed (do not reformat); leave receipt_date null.
- List each printed line item once; never repeat items or text.
- Put anything unreadable or ambiguous in "notes" and lower "confidence" (0-1)."""

log = logging.getLogger("gemini")
_sleep = time.sleep  # swapped out in tests


def parse_json_text(text: str) -> Receipt:
    """Validate the model's JSON; tolerate ```json fences just in case."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    return Receipt.model_validate(json.loads(cleaned))


def _first_attr(obj, *names, default=0):
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return default


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "resource_exhausted" in msg or "rate" in msg and "limit" in msg


class GeminiExtractor:
    def __init__(self, client=None, model: str | None = None,
                 min_interval: float | None = None, max_retries: int = 4, max_bad_outputs: int = 2):
        if client is None:
            from google import genai  # pip install google-genai
            key = os.getenv("GEMINI_API_KEY", "").strip()
            if not key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Locally: put GEMINI_API_KEY=... in the .env file "
                    "next to dashboard.py. Online: add it under the app's Secrets.")
            client = genai.Client(api_key=key)
        self.client = client
        self.model = model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        # Free tier allows only a few requests per minute: space calls out.
        self.min_interval = float(os.getenv("GEMINI_MIN_INTERVAL", 6) if min_interval is None else min_interval)
        self.max_retries = max_retries
        self.max_bad_outputs = max_bad_outputs
        self._last_call = 0.0

    # --- API calls -----------------------------------------------------------
    def _call_interactions(self, image: bytes, mime: str, schema: dict):
        """Current Gemini API (client.interactions)."""
        it = self.client.interactions.create(
            model=self.model,
            input=[
                {"type": "text", "text": PROMPT},
                {"type": "image", "data": base64.b64encode(image).decode(), "mime_type": mime},
            ],
            response_format={"type": "text", "mime_type": "application/json", "schema": schema},
        )
        usage = getattr(it, "usage", None)
        tokens = {
            "input_tokens": _first_attr(usage, "total_input_tokens", "input_tokens", "prompt_token_count"),
            "output_tokens": _first_attr(usage, "total_output_tokens", "output_tokens", "candidates_token_count"),
        }
        return it.output_text, tokens

    def _call_generate_content(self, image: bytes, mime: str, schema: dict):
        """Older google-genai SDKs without client.interactions."""
        from google.genai import types
        resp = self.client.models.generate_content(
            model=self.model,
            contents=[types.Part.from_bytes(data=image, mime_type=mime), PROMPT],
            config={"response_mime_type": "application/json", "response_json_schema": schema},
        )
        usage = getattr(resp, "usage_metadata", None)
        tokens = {
            "input_tokens": _first_attr(usage, "prompt_token_count"),
            "output_tokens": _first_attr(usage, "candidates_token_count"),
        }
        return resp.text, tokens

    # --- public --------------------------------------------------------------
    def extract(self, path: Path) -> tuple[Receipt, dict]:
        path = Path(path)
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        image, schema = path.read_bytes(), tool_input_schema()
        call = self._call_interactions if hasattr(self.client, "interactions") else self._call_generate_content

        bad_outputs = 0
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                _sleep(wait)
            self._last_call = time.monotonic()
            try:
                text, tokens = call(image, mime, schema)
                return parse_json_text(text), tokens
            except (json.JSONDecodeError, ValidationError) as exc:
                # LLMs occasionally degenerate (repeat text until the JSON breaks).
                # A fresh attempt almost always succeeds.
                bad_outputs += 1
                if bad_outputs <= self.max_bad_outputs:
                    log.warning("Invalid JSON from model (%s...), retrying", str(exc)[:80])
                    continue
                raise
            except Exception as exc:
                if attempt < self.max_retries and _is_rate_limit(exc):
                    backoff = 15 * (attempt + 1)
                    log.warning("Rate limited, waiting %ss (attempt %d)", backoff, attempt + 1)
                    _sleep(backoff)
                    continue
                raise
