-- Runs automatically the first time the Postgres container starts.

CREATE TABLE IF NOT EXISTS receipts (
    id              SERIAL PRIMARY KEY,
    source_file     TEXT        NOT NULL,
    file_hash       CHAR(64)    NOT NULL UNIQUE,   -- sha256: skip images already processed
    merchant_name   TEXT,
    receipt_date    DATE,
    currency        CHAR(3)     NOT NULL DEFAULT 'IDR',
    subtotal        NUMERIC(14,2),
    tax             NUMERIC(14,2),
    total           NUMERIC(14,2),
    total_matches_items BOOLEAN,                   -- sanity check result
    confidence      REAL,
    notes           TEXT,
    model           TEXT,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    raw_json        JSONB       NOT NULL,
    processed_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS receipt_items (
    id          SERIAL PRIMARY KEY,
    receipt_id  INTEGER NOT NULL REFERENCES receipts(id) ON DELETE CASCADE,
    line_no     INTEGER NOT NULL,
    name        TEXT    NOT NULL,
    quantity    NUMERIC(12,3),
    unit_price  NUMERIC(14,2),
    line_total  NUMERIC(14,2)
);

CREATE INDEX IF NOT EXISTS idx_receipts_date ON receipts (receipt_date);
CREATE INDEX IF NOT EXISTS idx_items_receipt ON receipt_items (receipt_id);

-- Ready-made view for the dashboard step (week 2)
CREATE OR REPLACE VIEW v_daily_spend AS
SELECT receipt_date, merchant_name, COUNT(*) AS n_receipts, SUM(total) AS total_spend
FROM receipts
WHERE receipt_date IS NOT NULL
GROUP BY receipt_date, merchant_name;
