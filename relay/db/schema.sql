CREATE TABLE IF NOT EXISTS events (
    event_id            TEXT PRIMARY KEY,
    bot_id              TEXT NOT NULL,
    event_type          TEXT NOT NULL,
    payload             TEXT NOT NULL,
    exchange_timestamp  TEXT NOT NULL,
    received_at         TEXT NOT NULL DEFAULT (datetime('now')),
    acked               INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_relay_events_acked ON events(acked);
CREATE INDEX IF NOT EXISTS idx_relay_events_bot ON events(bot_id);

CREATE TABLE IF NOT EXISTS watermarks (
    id          INTEGER PRIMARY KEY DEFAULT 1,
    last_event_id TEXT NOT NULL DEFAULT '',
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
