-- Event queue with idempotent deduplication and dead-letter support
CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    bot_id          TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    payload         TEXT NOT NULL,
    exchange_timestamp TEXT NOT NULL,
    received_at     TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',  -- pending | processing | acked | failed | dead_letter
    processed_at    TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    retry_count     INTEGER NOT NULL DEFAULT 0,
    max_retries     INTEGER NOT NULL DEFAULT 3,
    last_error      TEXT
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
