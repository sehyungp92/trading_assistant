-- Event queue with idempotent deduplication
CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    bot_id          TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    payload         TEXT NOT NULL,
    exchange_timestamp TEXT NOT NULL,
    received_at     TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending | processing | acked | failed
    processed_at    TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);
CREATE INDEX IF NOT EXISTS idx_events_bot_id ON events(bot_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);

-- Watermark tracking per bot for relay pull protocol
CREATE TABLE IF NOT EXISTS watermarks (
    bot_id      TEXT PRIMARY KEY,
    last_event_id TEXT NOT NULL,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
