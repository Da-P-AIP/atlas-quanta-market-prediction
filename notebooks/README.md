# Atlas Quanta Research Notebooks

This directory contains Jupyter notebooks for research, analysis, and experimentation with the Atlas Quanta prediction system.

## Notebook Structure

### 01_Data_Exploration/
- `data_sources_overview.ipynb` - Overview of all data sources and their characteristics
- `jpx_investor_flows.ipynb` - Analysis of JPX investor flow data
- `sentiment_data_exploration.ipynb` - Exploration of sentiment data sources

### 02_Feature_Engineering/
- `investor_clustering_analysis.ipynb` - Development and validation of investor clustering
- `vpin_indicator_development.ipynb` - VPIN indicator implementation and testing
- `seasonal_patterns.ipynb` - Analysis of seasonal market patterns

### 03_Model_Development/
- `tvp_var_implementation.ipynb` - Time-varying parameter VAR model development
- `dma_model_testing.ipynb` - Dynamic model averaging experiments
- `ensemble_methods.ipynb` - Combining multiple prediction models

### 04_Backtesting/
- `model_performance_evaluation.ipynb` - Comprehensive model backtesting
- `risk_adjusted_returns.ipynb` - Risk-adjusted performance metrics
- `regime_analysis.ipynb` - Performance across different market regimes

### 05_Real_Time_Analysis/
- `live_prediction_dashboard.ipynb` - Real-time prediction monitoring
- `alert_system_testing.ipynb` - Testing of prediction alert systems

## Getting Started

1. Ensure you have Jupyter installed:
   ```bash
   pip install jupyter
   ```

2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

3. Navigate to the appropriate notebook based on your research needs.

## Data Requirements

Most notebooks require access to the Atlas Quanta data pipeline. Ensure you have:
- Configured API keys in `config/config.yaml`
- Running Docker containers for data storage
- Executed initial data collection DAGs

## Contributing

When adding new notebooks:
1. Follow the naming convention: `topic_specific_analysis.ipynb`
2. Include clear markdown documentation
3. Add notebook descriptions to this README
4. Ensure notebooks can run with the standard environment
