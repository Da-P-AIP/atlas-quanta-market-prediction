-- Atlas Quanta DuckDB Schema Initialization
-- Creates the database schema for multi-dimensional market prediction system

-- Raw data tables (Bronze layer)
CREATE SCHEMA IF NOT EXISTS raw_data;
CREATE SCHEMA IF NOT EXISTS silver_data;
CREATE SCHEMA IF NOT EXISTS gold_data;
CREATE SCHEMA IF NOT EXISTS models;

-- Stock price data (raw)
CREATE TABLE IF NOT EXISTS raw_data.stock_prices (
    symbol VARCHAR,
    timestamp TIMESTAMP,
    open DECIMAL(10,4),
    high DECIMAL(10,4),
    low DECIMAL(10,4),
    close DECIMAL(10,4),
    volume BIGINT,
    source VARCHAR,
    ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Investor flow data (JPX official data)
CREATE TABLE IF NOT EXISTS raw_data.investor_flows (
    date DATE,
    investor_type VARCHAR, -- Individual, Foreign, Trust, Investment Trust, etc.
    market_section VARCHAR, -- TSE Prime, Standard, Growth
    buy_volume BIGINT,
    sell_volume BIGINT,
    buy_value DECIMAL(15,2),
    sell_value DECIMAL(15,2),
    net_volume BIGINT,
    net_value DECIMAL(15,2),
    source VARCHAR,
    ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sentiment data (Twitter, news, etc.)
CREATE TABLE IF NOT EXISTS raw_data.sentiment_data (
    timestamp TIMESTAMP,
    symbol VARCHAR,
    source VARCHAR, -- twitter, news, reddit, etc.
    sentiment_score DECIMAL(5,3), -- -1 to +1
    text_content TEXT,
    confidence_score DECIMAL(4,3),
    engagement_metrics JSON,
    ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Economic indicators
CREATE TABLE IF NOT EXISTS raw_data.economic_indicators (
    date DATE,
    indicator_name VARCHAR,
    value DECIMAL(15,4),
    unit VARCHAR,
    country VARCHAR,
    source VARCHAR,
    ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Crypto data
CREATE TABLE IF NOT EXISTS raw_data.crypto_prices (
    symbol VARCHAR,
    timestamp TIMESTAMP,
    price_usd DECIMAL(15,8),
    volume_24h DECIMAL(20,2),
    market_cap DECIMAL(20,2),
    on_chain_metrics JSON,
    source VARCHAR,
    ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Silver layer (processed data)
CREATE TABLE IF NOT EXISTS silver_data.cleaned_stock_prices (
    symbol VARCHAR,
    timestamp TIMESTAMP,
    open DECIMAL(10,4),
    high DECIMAL(10,4),
    low DECIMAL(10,4),
    close DECIMAL(10,4),
    volume BIGINT,
    adj_close DECIMAL(10,4),
    returns DECIMAL(8,6),
    log_returns DECIMAL(8,6),
    volatility DECIMAL(8,6),
    processed_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, timestamp)
);

-- Investor clustering results
CREATE TABLE IF NOT EXISTS silver_data.investor_clusters (
    date DATE,
    cluster_id INTEGER,
    cluster_name VARCHAR,
    turnover_rate DECIMAL(8,4),
    holding_period DECIMAL(8,2),
    trade_frequency DECIMAL(8,4),
    position_stability DECIMAL(6,4),
    news_sensitivity DECIMAL(6,4),
    member_count INTEGER,
    processed_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- VPIN calculations
CREATE TABLE IF NOT EXISTS silver_data.vpin_indicators (
    symbol VARCHAR,
    timestamp TIMESTAMP,
    vpin_value DECIMAL(6,4),
    volume_bucket INTEGER,
    buy_volume DECIMAL(15,2),
    sell_volume DECIMAL(15,2),
    total_volume DECIMAL(15,2),
    imbalance DECIMAL(15,2),
    volatility_estimate DECIMAL(8,6),
    alert_level VARCHAR,
    processed_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, timestamp)
);

-- Sentiment aggregations
CREATE TABLE IF NOT EXISTS silver_data.sentiment_aggregated (
    symbol VARCHAR,
    date DATE,
    avg_sentiment DECIMAL(5,3),
    sentiment_volatility DECIMAL(6,4),
    positive_ratio DECIMAL(5,4),
    negative_ratio DECIMAL(5,4),
    volume_mentions INTEGER,
    weighted_sentiment DECIMAL(5,3),
    processed_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, date)
);

-- Gold layer (analytics-ready indicators)
CREATE TABLE IF NOT EXISTS gold_data.market_regime_indicators (
    date DATE,
    market_regime VARCHAR, -- Bull, Bear, Sideways, Volatile
    regime_probability DECIMAL(5,4),
    volatility_regime VARCHAR, -- Low, Medium, High
    liquidity_regime VARCHAR, -- Abundant, Normal, Stressed
    confidence_score DECIMAL(5,4),
    processed_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date)
);

-- Investor deviation scores
CREATE TABLE IF NOT EXISTS gold_data.investor_deviation_scores (
    date DATE,
    permanent_holding_deviation DECIMAL(6,3),
    fixed_value_trading_deviation DECIMAL(6,3),
    panic_type_deviation DECIMAL(6,3),
    market_signal VARCHAR, -- BOTTOM_CANDIDATE, TOP_CANDIDATE, NEUTRAL
    signal_confidence DECIMAL(5,4),
    reasoning TEXT,
    processed_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date)
);

-- Seasonal anomaly factors
CREATE TABLE IF NOT EXISTS gold_data.seasonal_factors (
    date DATE,
    month_factor DECIMAL(6,4),
    week_factor DECIMAL(6,4),
    day_of_week_factor DECIMAL(6,4),
    sell_in_may_factor DECIMAL(6,4),
    year_end_factor DECIMAL(6,4),
    earnings_season_factor DECIMAL(6,4),
    processed_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date)
);

-- Model predictions
CREATE TABLE IF NOT EXISTS models.predictions (
    model_name VARCHAR,
    symbol VARCHAR,
    prediction_date DATE,
    target_date DATE,
    horizon_days INTEGER,
    predicted_return DECIMAL(8,6),
    predicted_price DECIMAL(10,4),
    confidence_interval_lower DECIMAL(10,4),
    confidence_interval_upper DECIMAL(10,4),
    model_confidence DECIMAL(5,4),
    feature_importance JSON,
    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (model_name, symbol, prediction_date, target_date)
);

-- Model performance tracking
CREATE TABLE IF NOT EXISTS models.model_performance (
    model_name VARCHAR,
    evaluation_date DATE,
    symbol VARCHAR,
    metric_name VARCHAR, -- RMSE, MAE, Direction_Accuracy, Sharpe_Ratio
    metric_value DECIMAL(10,6),
    time_horizon INTEGER,
    sample_size INTEGER,
    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- TVP-VAR model parameters
CREATE TABLE IF NOT EXISTS models.tvp_var_parameters (
    model_id VARCHAR,
    date DATE,
    parameter_name VARCHAR,
    parameter_value DECIMAL(10,6),
    parameter_std_error DECIMAL(10,6),
    variable_name VARCHAR,
    lag_order INTEGER,
    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_time ON raw_data.stock_prices(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_investor_flows_date_type ON raw_data.investor_flows(date, investor_type);
CREATE INDEX IF NOT EXISTS idx_sentiment_symbol_time ON raw_data.sentiment_data(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_cleaned_prices_symbol_time ON silver_data.cleaned_stock_prices(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_vpin_symbol_time ON silver_data.vpin_indicators(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_model_symbol ON models.predictions(model_name, symbol, prediction_date);

-- Create views for common queries
CREATE OR REPLACE VIEW gold_data.latest_market_signals AS
SELECT 
    ids.date,
    ids.market_signal,
    ids.signal_confidence,
    ids.permanent_holding_deviation,
    ids.fixed_value_trading_deviation,
    ids.panic_type_deviation,
    mri.market_regime,
    mri.regime_probability,
    sf.sell_in_may_factor,
    sf.month_factor
FROM gold_data.investor_deviation_scores ids
LEFT JOIN gold_data.market_regime_indicators mri ON ids.date = mri.date
LEFT JOIN gold_data.seasonal_factors sf ON ids.date = sf.date
WHERE ids.date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY ids.date DESC;

-- Summary statistics view
CREATE OR REPLACE VIEW gold_data.model_performance_summary AS
SELECT 
    model_name,
    symbol,
    metric_name,
    AVG(metric_value) as avg_metric,
    STDDEV(metric_value) as std_metric,
    MIN(metric_value) as min_metric,
    MAX(metric_value) as max_metric,
    COUNT(*) as sample_count,
    MAX(evaluation_date) as latest_evaluation
FROM models.model_performance 
WHERE evaluation_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY model_name, symbol, metric_name
ORDER BY model_name, symbol, metric_name;

-- Insert initial configuration data
INSERT OR IGNORE INTO gold_data.seasonal_factors (date, month_factor, week_factor, day_of_week_factor, sell_in_may_factor, year_end_factor, earnings_season_factor)
VALUES ('2025-01-01', 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);

COMMIT;
