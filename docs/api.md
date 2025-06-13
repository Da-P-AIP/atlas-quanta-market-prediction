# Atlas Quanta API Reference

## Main System Class

### AtlasQuanta

The main class that orchestrates all components of the prediction system.

```python
from atlas_quanta import AtlasQuanta

# Initialize system
predictor = AtlasQuanta(config_path="config/config.yaml")
```

#### Constructor

```python
__init__(config_path: str)
```

**Parameters:**
- `config_path` (str): Path to YAML configuration file

**Raises:**
- `ValueError`: If configuration file cannot be loaded

#### Methods

##### update_data()

```python
update_data(symbols: List[str] = None, days_back: int = 252) -> None
```

Update market data for specified symbols.

**Parameters:**
- `symbols` (List[str], optional): List of symbols to update. Defaults to config symbols.
- `days_back` (int): Number of days of historical data to fetch. Default: 252.

**Example:**
```python
predictor.update_data(['AAPL', 'MSFT', 'GOOGL'], days_back=500)
```

##### generate_features()

```python
generate_features(symbols: List[str] = None) -> Dict[str, pd.DataFrame]
```

Generate all features for prediction models.

**Parameters:**
- `symbols` (List[str], optional): Symbols to generate features for

**Returns:**
- `Dict[str, pd.DataFrame]`: Mapping of symbol to feature DataFrame

**Example:**
```python
features = predictor.generate_features(['AAPL', 'MSFT'])
print(features['AAPL'].columns)
# Output: ['vpin', 'returns', 'volatility', 'rsi', ...]
```

##### predict()

```python
predict(symbols: List[str], 
        horizon_days: int = 100,
        include_sentiment: bool = True,
        include_risk_metrics: bool = True) -> Dict[str, Dict]
```

Generate predictions for specified symbols.

**Parameters:**
- `symbols` (List[str]): List of symbols to predict
- `horizon_days` (int): Prediction horizon in days. Default: 100.
- `include_sentiment` (bool): Include sentiment analysis. Default: True.
- `include_risk_metrics` (bool): Calculate risk metrics. Default: True.

**Returns:**
- `Dict[str, Dict]`: Predictions for each symbol

**Example:**
```python
predictions = predictor.predict(
    symbols=['AAPL', 'TSLA'],
    horizon_days=60,
    include_sentiment=True
)

# Access prediction for AAPL
apple_pred = predictions['AAPL']['composite']
print(f"Direction: {apple_pred['direction']}")
print(f"Confidence: {apple_pred['confidence']:.1%}")
```

##### get_prediction_summary()

```python
get_prediction_summary() -> pd.DataFrame
```

Get summary of all current predictions.

**Returns:**
- `pd.DataFrame`: Summary DataFrame with columns:
  - `symbol`: Stock symbol
  - `direction`: Predicted direction ('up', 'down', 'neutral')
  - `confidence`: Confidence level (0-1)
  - `expected_return`: Expected return percentage
  - `probability_up`: Probability of upward movement
  - `var_95`: 95% Value at Risk
  - `current_sentiment`: Current sentiment score

**Example:**
```python
summary = predictor.get_prediction_summary()
print(summary[['symbol', 'direction', 'confidence']])
```

## Data Sources

### BaseDataSource

Abstract base class for all data sources.

```python
from src.data_sources.base_data_source import BaseDataSource
```

#### Methods

##### get_data()

```python
get_data(symbols: List[str], 
         start_date: datetime, 
         end_date: datetime,
         **kwargs) -> pd.DataFrame
```

Fetch data for specified symbols and date range.

**Parameters:**
- `symbols` (List[str]): List of symbols to fetch
- `start_date` (datetime): Start date for data
- `end_date` (datetime): End date for data
- `**kwargs`: Additional parameters specific to data source

**Returns:**
- `pd.DataFrame`: Fetched data with DatetimeIndex

##### validate_symbols()

```python
validate_symbols(symbols: List[str]) -> Dict[str, bool]
```

Validate if symbols are available from this data source.

**Parameters:**
- `symbols` (List[str]): List of symbols to validate

**Returns:**
- `Dict[str, bool]`: Mapping of symbol to availability status

##### get_data_quality_metrics()

```python
get_data_quality_metrics(data: pd.DataFrame) -> Dict[str, float]
```

Calculate data quality metrics.

**Returns:**
- `Dict[str, float]`: Quality metrics:
  - `completeness`: Percentage of non-null values
  - `consistency`: Percentage passing consistency checks
  - `freshness`: How recent the data is (0-100)

## Indicators

### VPINCalculator

Volume-Synchronized Probability of Informed Trading calculator.

```python
from src.indicators.vpin import VPINCalculator

calculator = VPINCalculator(config)
```

#### Methods

##### calculate_vpin()

```python
calculate_vpin(prices: pd.Series, 
               volumes: pd.Series,
               bucket_size: int = 50,
               window_length: int = 50) -> pd.Series
```

Calculate VPIN indicator.

**Parameters:**
- `prices` (pd.Series): Price series
- `volumes` (pd.Series): Volume series
- `bucket_size` (int): Volume bucket size. Default: 50.
- `window_length` (int): Rolling window length. Default: 50.

**Returns:**
- `pd.Series`: VPIN values (0-1, higher = more informed trading)

**Example:**
```python
vpin_values = calculator.calculate_vpin(
    data['close'], 
    data['volume'],
    bucket_size=100
)

# Alert when VPIN exceeds threshold
high_vpin = vpin_values > 0.8
print(f"High VPIN periods: {high_vpin.sum()}")
```

## Models

### TVPVARModel

Time-Varying Parameter Vector Autoregression model.

```python
from src.models.tvp_var import TVPVARModel

model = TVPVARModel(config)
```

#### Methods

##### fit()

```python
fit(data: pd.DataFrame, 
    lags: int = 1,
    forgetting_factor: float = 0.99) -> None
```

Fit the TVP-VAR model.

**Parameters:**
- `data` (pd.DataFrame): Multivariate time series data
- `lags` (int): Number of lags. Default: 1.
- `forgetting_factor` (float): Forgetting factor for parameter evolution. Default: 0.99.

##### predict()

```python
predict(data: pd.DataFrame, 
        horizon: int = 1) -> Dict[str, np.ndarray]
```

Generate predictions.

**Parameters:**
- `data` (pd.DataFrame): Input data for prediction
- `horizon` (int): Forecast horizon. Default: 1.

**Returns:**
- `Dict[str, np.ndarray]`: Predictions for each variable

### DMAModel

Dynamic Model Averaging.

```python
from src.models.dma import DMAModel

model = DMAModel(config)
```

#### Methods

##### add_model()

```python
add_model(name: str, model: Any) -> None
```

Add a model to the ensemble.

**Parameters:**
- `name` (str): Model name identifier
- `model` (Any): Model object with fit() and predict() methods

##### fit()

```python
fit(data: pd.DataFrame) -> None
```

Fit all models in the ensemble.

##### predict()

```python
predict(data: pd.DataFrame, 
        horizon: int = 1) -> Dict[str, Any]
```

Generate ensemble predictions.

**Returns:**
- `Dict[str, Any]`: Combined predictions and model weights

## Sentiment Analysis

### SentimentAnalyzer

Main sentiment analysis engine.

```python
from src.sentiment.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer(config)
```

#### Methods

##### get_composite_sentiment()

```python
get_composite_sentiment(symbols: List[str],
                        start_date: datetime,
                        end_date: datetime) -> pd.DataFrame
```

Calculate composite sentiment from multiple sources.

**Returns:**
- `pd.DataFrame`: Sentiment scores with columns:
  - `{symbol}_sentiment`: Normalized sentiment (-1 to 1)
  - `{symbol}_sentiment_momentum`: 5-day sentiment momentum
  - `{symbol}_sentiment_zscore`: Z-score relative to 30-day average

##### get_sentiment_extremes()

```python
get_sentiment_extremes(data: pd.DataFrame, 
                      threshold: float = 2.0) -> pd.DataFrame
```

Identify sentiment extremes for contrarian signals.

**Parameters:**
- `threshold` (float): Z-score threshold for extremes. Default: 2.0.

**Returns:**
- `pd.DataFrame`: Boolean indicators for extreme sentiment conditions

## Risk Management

### VaRCalculator

Value at Risk calculator with multiple methodologies.

```python
from src.risk_management.var_calculator import VaRCalculator

calculator = VaRCalculator(config)
```

#### Methods

##### calculate_historical_var()

```python
calculate_historical_var(returns: pd.Series,
                        confidence_level: float = 0.95,
                        window_size: int = 252) -> pd.Series
```

Calculate VaR using historical simulation.

**Parameters:**
- `confidence_level` (float): Confidence level (e.g., 0.95 for 95% VaR)
- `window_size` (int): Rolling window size. Default: 252.

**Returns:**
- `pd.Series`: Rolling VaR values

##### calculate_conditional_var()

```python
calculate_conditional_var(returns: pd.Series,
                         confidence_level: float = 0.95) -> pd.Series
```

Calculate Conditional VaR (Expected Shortfall).

**Returns:**
- `pd.Series`: CVaR values (expected loss beyond VaR threshold)

##### backtest_var_model()

```python
backtest_var_model(returns: pd.Series,
                   var_forecasts: pd.Series,
                   confidence_level: float = 0.95) -> Dict[str, float]
```

Backtest VaR model performance.

**Returns:**
- `Dict[str, float]`: Backtesting metrics including:
  - `violation_rate`: Actual violation rate
  - `kupiec_p_value`: Kupiec test p-value
  - `avg_violation_loss`: Average loss when VaR is exceeded

## Configuration

### Configuration File Format

```yaml
# config/config.yaml
default_symbols:
  - "SPY"
  - "QQQ"
  - "IWM"

data_sources:
  eod_historical:
    api_key: "your_api_key"
    rate_limit: 100
    max_history_days: 3650

models:
  tvp_var:
    forgetting_factor: 0.99
    max_lags: 5
  
  dma:
    forgetting_factor: 0.95
    model_prior_prob: 0.5

indicators:
  vpin:
    bucket_size: 50
    window_length: 50
    alert_threshold: 0.8

sentiment:
  update_frequency: "1H"
  smoothing_window: 24
  sentiment_weights:
    social: 0.3
    news: 0.4
    fear_greed: 0.2
    vix: 0.1

risk:
  confidence_levels: [0.95, 0.99]
  var_window_size: 252
  stress_test_scenarios:
    - "2008_crisis"
    - "2020_covid"
    - "flash_crash"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Error Handling

### Common Exceptions

- `ValueError`: Invalid configuration or parameters
- `ConnectionError`: API connection issues
- `TimeoutError`: Request timeout
- `DataQualityError`: Data quality issues
- `ModelNotFittedError`: Model used before fitting

### Example Error Handling

```python
try:
    predictions = predictor.predict(['AAPL'])
except ConnectionError as e:
    print(f"API connection failed: {e}")
except ValueError as e:
    print(f"Invalid parameters: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Considerations

### Optimization Tips

1. **Data Caching**: Use Redis for frequently accessed data
2. **Batch Processing**: Process multiple symbols together
3. **Parallel Execution**: Use multiprocessing for independent tasks
4. **Memory Management**: Clear large DataFrames when done
5. **API Rate Limits**: Respect API rate limits to avoid blocking

### Memory Usage

- Each symbol's data typically uses 1-5 MB
- Feature matrices scale with history length
- Models maintain state that grows with training data
- Use `del` statements to free memory explicitly

```python
# Good practice
data = predictor.market_data['AAPL']
results = process_data(data)
del data  # Free memory
```
