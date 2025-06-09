-- Stock Prediction ML Database Schema
-- Tables for storing stock data, news data, and features

-- Table for storing stock symbols and metadata
CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol VARCHAR(10) NOT NULL UNIQUE,
    name VARCHAR(255),
    sector VARCHAR(100),
    market_cap VARCHAR(20),
    exchange VARCHAR(10),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing daily stock price data
CREATE TABLE IF NOT EXISTS stock_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_id INTEGER NOT NULL,
    date DATE NOT NULL,
    open_price DECIMAL(10,4),
    high_price DECIMAL(10,4),
    low_price DECIMAL(10,4),
    close_price DECIMAL(10,4),
    adj_close DECIMAL(10,4),
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (symbol_id) REFERENCES symbols(id),
    UNIQUE(symbol_id, date)
);

-- Table for storing news articles
CREATE TABLE IF NOT EXISTS news_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    headline TEXT NOT NULL,
    summary TEXT,
    content TEXT,
    author VARCHAR(255),
    source VARCHAR(100),
    url TEXT,
    published_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sentiment_score DECIMAL(3,2),
    sentiment_label VARCHAR(20)
);

-- Table for linking news articles to symbols (many-to-many)
CREATE TABLE IF NOT EXISTS news_symbols (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    news_id INTEGER NOT NULL,
    symbol_id INTEGER NOT NULL,
    relevance_score DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (news_id) REFERENCES news_articles(id),
    FOREIGN KEY (symbol_id) REFERENCES symbols(id),
    UNIQUE(news_id, symbol_id)
);

-- Table for storing computed features
CREATE TABLE IF NOT EXISTS features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_id INTEGER NOT NULL,
    date DATE NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    feature_value DECIMAL(15,6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (symbol_id) REFERENCES symbols(id),
    UNIQUE(symbol_id, date, feature_name)
);

-- Table for storing model predictions
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol_id INTEGER NOT NULL,
    prediction_date DATE NOT NULL,
    target_date DATE NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    prediction_value DECIMAL(10,4),
    confidence_score DECIMAL(3,2),
    actual_value DECIMAL(10,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (symbol_id) REFERENCES symbols(id)
);

-- Add model tracking table
CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT UNIQUE NOT NULL,
    target_symbol TEXT NOT NULL,
    model_type TEXT NOT NULL,
    train_r2 REAL,
    test_r2 REAL,
    train_mse REAL,
    test_mse REAL,
    n_features INTEGER,
    feature_selection_method TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    config_snapshot TEXT,  -- JSON string of config
    notes TEXT
);

-- Add feature importance tracking
CREATE TABLE IF NOT EXISTS feature_importance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    coefficient REAL,
    abs_coefficient REAL,
    p_value REAL,
    rank_importance INTEGER,
    FOREIGN KEY (run_id) REFERENCES model_runs (run_id)
);

-- Add prediction tracking
CREATE TABLE IF NOT EXISTS model_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    prediction_date DATE NOT NULL,
    actual_return REAL,
    predicted_return REAL,
    residual REAL,
    data_type TEXT CHECK (data_type IN ('train', 'test', 'validation')),
    FOREIGN KEY (run_id) REFERENCES model_runs (run_id)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_date ON stock_prices(symbol_id, date);
CREATE INDEX IF NOT EXISTS idx_news_published_at ON news_articles(published_at);
CREATE INDEX IF NOT EXISTS idx_news_symbols_symbol ON news_symbols(symbol_id);
CREATE INDEX IF NOT EXISTS idx_features_symbol_date ON features(symbol_id, date);
CREATE INDEX IF NOT EXISTS idx_predictions_symbol_date ON predictions(symbol_id, prediction_date);
CREATE INDEX IF NOT EXISTS idx_model_runs_symbol ON model_runs(target_symbol);
CREATE INDEX IF NOT EXISTS idx_model_runs_created ON model_runs(created_at);
CREATE INDEX IF NOT EXISTS idx_feature_importance_run ON feature_importance(run_id);
CREATE INDEX IF NOT EXISTS idx_predictions_run_date ON model_predictions(run_id, prediction_date);