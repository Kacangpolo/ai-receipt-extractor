"""Data model for an extracted receipt (nota)."""
from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


def _clean_amount(v):
    """149850.0 -> 149850 (whole rupiah amounts shouldn't carry '.0')."""
    if v is None or v == "":
        return None
    d = Decimal(str(v))
    return d.quantize(Decimal(1)) if d == d.to_integral_value() else d


def parse_printed_date(text: str | None) -> date | None:
    """Deterministic date parsing. Indonesian receipts print day first: DD/MM/YYYY.
    The LLM only copies the text; code decides what it means."""
    if not text:
        return None
    m = re.search(r"(\d{4})[/.-](\d{1,2})[/.-](\d{1,2})", text)          # 2026-10-03
    if m:
        y, mo, d = map(int, m.groups())
    else:
        m = re.search(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})", text)     # 03/10/2026, 03-10-26
        if not m:
            return None
        d, mo, y = map(int, m.groups())
        if y < 100:
            y += 2000
    try:
        return date(y, mo, d)
    except ValueError:
        return None


class ReceiptItem(BaseModel):
    name: str = Field(description="Item name as printed on the receipt")
    quantity: Optional[Decimal] = Field(default=None, description="Quantity, if printed")
    unit_price: Optional[Decimal] = Field(default=None, description="Price per unit, if printed")
    line_total: Optional[Decimal] = Field(default=None, description="Total for this line")

    @field_validator("quantity", "unit_price", "line_total", mode="before")
    @classmethod
    def clean_amounts(cls, v):
        return _clean_amount(v)

    @model_validator(mode="after")
    def derive_line_total(self):
        """quantity x unit_price is arithmetic, so code fills it in when the model left it out."""
        if self.line_total is None and self.quantity and self.unit_price and self.unit_price > 0:
            self.line_total = (self.quantity * self.unit_price).quantize(Decimal(1))
        return self


class Receipt(BaseModel):
    merchant_name: Optional[str] = Field(default=None, description="Store or business name")
    date_printed: Optional[str] = Field(
        default=None, description="Transaction date copied EXACTLY as printed, e.g. '03/10/2026'")
    receipt_date: Optional[date] = Field(
        default=None, description="Leave null; computed from date_printed by code")
    currency: str = Field(default="IDR", description="ISO currency code")
    items: list[ReceiptItem] = Field(default_factory=list)
    subtotal: Optional[Decimal] = Field(default=None, description="SUBTOTAL as printed, before tax")
    tax: Optional[Decimal] = Field(default=None, description="Tax (PPN/PB1) as printed, null if none")
    total: Optional[Decimal] = Field(default=None, description="Grand TOTAL to pay, as printed (not cash/TUNAI)")
    confidence: float = Field(default=0.0, ge=0, le=1, description="Model's self-rated confidence 0-1")
    notes: Optional[str] = Field(default=None, description="Anything unreadable or ambiguous")

    @field_validator("receipt_date", mode="before")
    @classmethod
    def empty_date_to_none(cls, v):
        return v or None

    @field_validator("subtotal", "tax", "total", mode="before")
    @classmethod
    def clean_amounts(cls, v):
        return _clean_amount(v)

    @model_validator(mode="after")
    def date_from_printed(self):
        parsed = parse_printed_date(self.date_printed)
        if parsed:
            self.receipt_date = parsed
        return self

    def items_sum(self) -> Optional[Decimal]:
        totals = [i.line_total for i in self.items if i.line_total is not None]
        return sum(totals, Decimal(0)) if totals else None

    def issues(self, tolerance: Decimal = Decimal("1")) -> list[str]:
        """Every reason a human should double-check this receipt. Empty list = passed.

        The checks use the redundancy printed on every receipt: items add up to the
        subtotal, and subtotal + tax adds up to the total. A model that skips items
        or reads the wrong line breaks at least one of these equations."""
        found = []
        if not self.merchant_name:
            found.append("merchant missing")
        if self.receipt_date is None:
            found.append("date missing")
        if self.total is None or self.total <= 0:
            found.append("total missing")
        if not self.items:
            found.append("no items")
        if any(i.line_total is None or i.line_total <= 0 for i in self.items):
            found.append("item without a price")
        items = self.items_sum()
        tax = self.tax or Decimal(0)
        if items is not None and self.subtotal is not None and abs(items - self.subtotal) > tolerance:
            found.append(f"items {items:,.0f} != subtotal {self.subtotal:,.0f}")
        if self.subtotal is not None and self.total is not None and abs(self.subtotal + tax - self.total) > tolerance:
            found.append(f"subtotal + tax {self.subtotal + tax:,.0f} != total {self.total:,.0f}")
        if self.subtotal is None and items is not None and self.total is not None and abs(items + tax - self.total) > tolerance:
            found.append(f"items + tax {items + tax:,.0f} != total {self.total:,.0f}")
        return found

    def total_matches_items(self, tolerance: Decimal = Decimal("1")) -> Optional[bool]:
        """Sanity check: do line totals (+tax) add up to the grand total?"""
        s = self.items_sum()
        if s is None or self.total is None:
            return None
        return abs(s + (self.tax or 0) - self.total) <= tolerance


def _simplify(node):
    """Make pydantic's schema easier for an LLM: Decimal -> plain number, drop titles."""
    if isinstance(node, list):
        return [_simplify(n) for n in node]
    if not isinstance(node, dict):
        return node
    node = {k: _simplify(v) for k, v in node.items() if k != "title"}
    variants = node.get("anyOf")
    if variants:
        types = [v.get("type") for v in variants]
        base = next((t for t in ("number", "string", "integer", "boolean") if t in types), None)
        if base and all(t in (base, "string", "null") for t in types):
            node.pop("anyOf")
            node["type"] = [base, "null"] if "null" in types else base
            if base == "string" and any(v.get("format") for v in variants):
                node["format"] = next(v["format"] for v in variants if v.get("format"))
    return node


def tool_input_schema() -> dict:
    """JSON schema handed to the LLM as the tool's input_schema."""
    return _simplify(Receipt.model_json_schema())
