"""Send a receipt photo to an LLM and get structured data back."""
from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path

from .schema import Receipt, tool_input_schema

TOOL_NAME = "record_receipt"

SYSTEM_PROMPT = """You extract data from photos of sales receipts (nota), mostly Indonesian.
Rules:
- Copy values exactly as printed; never invent missing values, use null instead.
- Amounts are numbers without separators. Indonesian receipts use "." for thousands:
  "Rp 25.000" -> 25000, "4.000" -> 4000 (never 4).
- merchant_name is the business name printed at the top.
- items: EVERY product line between the header and SUBTOTAL, in order. Count them before answering.
- subtotal, tax and total are copied from their own printed lines. total is the line labelled
  TOTAL; never use TUNAI/CASH/KEMBALI and never copy an item price into total.
- Copy the date into date_printed EXACTLY as printed (do not reformat); leave receipt_date null.
- Put anything unreadable or ambiguous in `notes` and lower `confidence`.
Always answer by calling the record_receipt tool."""

DEFAULT_MODEL = "claude-haiku-5-5"


def _image_block(path: Path) -> dict:
    media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    if media_type not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        raise ValueError(f"Unsupported image type for {path.name}: {media_type}")
    data = base64.standard_b64encode(path.read_bytes()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


def build_request(path: Path, model: str | None = None) -> dict:
    return {
        "model": model or os.getenv("LLM_MODEL", DEFAULT_MODEL),
        "max_tokens": 2048,
        "system": SYSTEM_PROMPT,
        "tools": [{
            "name": TOOL_NAME,
            "description": "Record the structured contents of one receipt.",
            "input_schema": tool_input_schema(),
        }],
        "tool_choice": {"type": "tool", "name": TOOL_NAME},
        "messages": [{
            "role": "user",
            "content": [_image_block(path), {"type": "text", "text": "Extract this receipt."}],
        }],
    }


def parse_response(response) -> Receipt:
    """Pull the tool call out of the API response and validate it."""
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == TOOL_NAME:
            return Receipt.model_validate(block.input)
    raise ValueError("Model did not return a record_receipt tool call")


class AnthropicExtractor:
    def __init__(self, client=None, model: str | None = None):
        if client is None:
            import anthropic  # imported lazily so tests run without the SDK
            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.client = client
        self.model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)

    def extract(self, path: Path) -> tuple[Receipt, dict]:
        """Returns the receipt plus token usage for cost tracking."""
        response = self.client.messages.create(**build_request(Path(path), self.model))
        usage = getattr(response, "usage", None)
        tokens = {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
        }
        return parse_response(response), tokens


# Backwards-compatible name
ReceiptExtractor = AnthropicExtractor


def get_extractor(provider: str | None = None):
    """Pick the LLM backend from LLM_PROVIDER in .env (gemini | anthropic)."""
    provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
    if provider == "gemini":
        from .gemini import GeminiExtractor
        return GeminiExtractor()
    if provider == "anthropic":
        return AnthropicExtractor()
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r} (use 'gemini' or 'anthropic')")


def get_fallback_extractor(provider: str | None = None):
    """A stronger model used only for receipts that fail the checks. None = disabled."""
    provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).lower()
    if provider != "gemini":
        return None
    model = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.8-flash").strip()
    if not model:
        return None
    from .gemini import GeminiExtractor
    return GeminiExtractor(model=model)
