-- alerts.db schema — persistencia de alertas externas (TradingView, AI, internas)
-- Se aplica idempotente por indicatorsForge al arrancar.

PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source_type TEXT NOT NULL,
    symbol      TEXT,
    timeframe   TEXT,
    signal      TEXT,
    direction   TEXT,
    price       REAL,
    raw_json    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_alerts_symbol_ts ON alerts(symbol, ts DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_source_ts ON alerts(source_type, ts DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_ts        ON alerts(ts DESC);
