"""
Atlas Quanta ETL Pipeline - Apache Airflow DAGs

Main data processing pipeline for the Atlas Quanta market prediction system.
Implements the 3-layer architecture: Raw -> Silver -> Gold data zones.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.models import Variable
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import os
import sys

# Add src to path for imports
sys.path.append('/opt/airflow/src')

from clustering.investor_clustering import InvestorClusteringAnalyzer
from indicators.vpin import VPINCalculator
from models.tvp_var import TVPVARModel
from models.dma import DynamicModelAveraging

# Default DAG arguments
default_args = {
    'owner': 'atlas-quanta',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# Configuration
DAG_ID = 'atlas_quanta_main_pipeline'
SCHEDULE_INTERVAL = '0 2 * * *'  # Daily at 2 AM


def extract_stock_data(**context):
    """
    Extract stock price data from multiple sources
    """
    from data_sources.stock_data import StockDataExtractor
    
    extractor = StockDataExtractor()
    
    # Japanese stocks
    symbols_jp = ['6758.T', '7203.T', '8035.T', '9984.T']  # Sony, Toyota, Tokyo Electron, SoftBank
    
    # US stocks
    symbols_us = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    
    # Extract data
    data_jp = extractor.fetch_stock_data(symbols_jp, source='eodhd')
    data_us = extractor.fetch_stock_data(symbols_us, source='iex')
    
    # Save to MinIO raw zone
    extractor.save_to_minio(data_jp, 'raw-zone', 'stock_data/jp/latest.parquet')
    extractor.save_to_minio(data_us, 'raw-zone', 'stock_data/us/latest.parquet')
    
    return {'jp_records': len(data_jp), 'us_records': len(data_us)}


def extract_investor_flows(**context):
    """
    Extract investor flow data (JPX official data)
    """
    from data_sources.jpx_data import JPXDataExtractor
    
    extractor = JPXDataExtractor()
    
    # Download latest investor flow data
    flow_data = extractor.fetch_investor_flows()
    
    # Save to MinIO
    extractor.save_to_minio(flow_data, 'raw-zone', 'investor_flows/latest.parquet')
    
    return {'records': len(flow_data)}


def extract_sentiment_data(**context):
    """
    Extract sentiment data from multiple sources
    """
    from data_sources.sentiment_data import SentimentExtractor
    
    extractor = SentimentExtractor()
    
    # Twitter/X sentiment
    twitter_data = extractor.fetch_twitter_sentiment(['$AAPL', '$TSLA', '$NVDA'])
    
    # News sentiment
    news_data = extractor.fetch_news_sentiment(['Apple', 'Tesla', 'NVIDIA'])
    
    # Save to MinIO
    extractor.save_to_minio(twitter_data, 'raw-zone', 'sentiment/twitter/latest.parquet')
    extractor.save_to_minio(news_data, 'raw-zone', 'sentiment/news/latest.parquet')
    
    return {'twitter_records': len(twitter_data), 'news_records': len(news_data)}


def extract_crypto_data(**context):
    """
    Extract cryptocurrency data and on-chain metrics
    """
    from data_sources.crypto_data import CryptoDataExtractor
    
    extractor = CryptoDataExtractor()
    
    # Major cryptocurrencies
    symbols = ['bitcoin', 'ethereum', 'binancecoin', 'cardano', 'solana']
    
    # Price data
    price_data = extractor.fetch_price_data(symbols)
    
    # On-chain data
    onchain_data = extractor.fetch_onchain_data(['bitcoin', 'ethereum'])
    
    # Save to MinIO
    extractor.save_to_minio(price_data, 'raw-zone', 'crypto/prices/latest.parquet')
    extractor.save_to_minio(onchain_data, 'raw-zone', 'crypto/onchain/latest.parquet')
    
    return {'price_records': len(price_data), 'onchain_records': len(onchain_data)}


def process_to_silver(**context):
    """
    Process raw data to silver layer (cleaned, standardized)
    """
    from data_processing.silver_processor import SilverProcessor
    
    processor = SilverProcessor()
    
    # Process stock data
    processor.process_stock_data()
    
    # Process investor flows
    processor.process_investor_flows()
    
    # Process sentiment data
    processor.process_sentiment_data()
    
    # Process crypto data
    processor.process_crypto_data()
    
    return {'status': 'silver_processing_complete'}


def calculate_investor_clusters(**context):
    """
    Calculate investor behavior clusters and deviation scores
    """
    from data_processing.clustering_processor import ClusteringProcessor
    
    processor = ClusteringProcessor()
    
    # Load investor flow data
    investor_data = processor.load_investor_data()
    
    # Initialize clustering analyzer
    analyzer = InvestorClusteringAnalyzer(window_size=252, n_clusters=3)
    
    # Fit clusters
    analyzer.fit_clusters(investor_data)
    
    # Calculate current deviation scores
    current_data = investor_data.tail(20)  # Last 20 trading days
    deviation_scores = analyzer.calculate_deviation_scores(current_data)
    
    # Detect market extremes
    market_signals = analyzer.detect_market_extremes(deviation_scores)
    
    # Save results to gold layer
    processor.save_clustering_results({
        'deviation_scores': deviation_scores,
        'market_signals': market_signals,
        'cluster_summary': analyzer.get_cluster_summary()
    })
    
    return market_signals


def calculate_vpin_indicators(**context):
    """
    Calculate VPIN indicators for all symbols
    """
    from data_processing.vpin_processor import VPINProcessor
    
    processor = VPINProcessor()
    
    # Load tick data
    symbols = ['AAPL', 'TSLA', 'NVDA', '6758.T', '7203.T']
    
    vpin_results = {}
    
    for symbol in symbols:
        # Load symbol data
        tick_data = processor.load_tick_data(symbol)
        
        if tick_data.empty:
            continue
        
        # Initialize VPIN calculator
        vpin_calc = VPINCalculator(bucket_size=50, sample_window=50)
        
        # Process tick data
        results = []
        for _, row in tick_data.iterrows():
            result = vpin_calc.update_bucket(
                timestamp=row['timestamp'],
                price=row['price'],
                volume=row['volume'],
                bid=row.get('bid'),
                ask=row.get('ask')
            )
            if result:
                results.append(result)
        
        # Flash crash risk assessment
        risk_assessment = vpin_calc.detect_flash_crash_risk()
        
        vpin_results[symbol] = {
            'latest_vpin': vpin_calc.get_current_vpin(),
            'risk_assessment': risk_assessment,
            'history': vpin_calc.get_vpin_history(periods=100)
        }
    
    # Save results
    processor.save_vpin_results(vpin_results)
    
    return {'symbols_processed': len(vpin_results)}


def calculate_seasonal_factors(**context):
    """
    Calculate seasonal anomaly factors
    """
    from data_processing.seasonal_processor import SeasonalProcessor
    
    processor = SeasonalProcessor()
    
    # Calculate various seasonal factors
    current_date = datetime.now().date()
    
    factors = {
        'sell_in_may_factor': processor.calculate_sell_in_may_factor(current_date),
        'month_factor': processor.calculate_month_factor(current_date),
        'week_factor': processor.calculate_week_factor(current_date),
        'day_of_week_factor': processor.calculate_day_of_week_factor(current_date),
        'year_end_factor': processor.calculate_year_end_factor(current_date),
        'earnings_season_factor': processor.calculate_earnings_season_factor(current_date)
    }
    
    # Save to gold layer
    processor.save_seasonal_factors(current_date, factors)
    
    return factors


def train_tvp_var_models(**context):
    """
    Train TVP-VAR models for time-varying relationships
    """
    from data_processing.model_processor import ModelProcessor
    
    processor = ModelProcessor()
    
    # Load preprocessed data
    market_data = processor.load_market_data()
    
    # Initialize TVP-VAR model
    tvp_var = TVPVARModel()
    
    # Fit model
    tvp_var.fit(market_data)
    
    # Generate forecasts
    forecasts = tvp_var.predict(steps_ahead=100)  # 100-day forecasts
    
    # Save model and predictions
    processor.save_model_results('tvp_var', {
        'model': tvp_var,
        'forecasts': forecasts,
        'parameters': tvp_var.get_time_varying_parameters()
    })
    
    return {'forecast_horizon': 100}


def run_dma_ensemble(**context):
    """
    Run Dynamic Model Averaging ensemble
    """
    from data_processing.ensemble_processor import EnsembleProcessor
    
    processor = EnsembleProcessor()
    
    # Initialize DMA
    dma = DynamicModelAveraging()
    
    # Add various models to ensemble
    processor.setup_dma_models(dma)
    
    # Load training data
    X, y = processor.load_training_data()
    
    # Fit ensemble
    dma.fit(X, y)
    
    # Generate predictions
    predictions = dma.predict(X[-1:])  # Latest observation
    
    # Save ensemble results
    processor.save_ensemble_results({
        'predictions': predictions,
        'model_weights': dma.model_weights,
        'performance_summary': dma.get_model_performance_summary()
    })
    
    return predictions


def generate_final_predictions(**context):
    """
    Generate final integrated predictions
    """
    from data_processing.prediction_processor import PredictionProcessor
    
    processor = PredictionProcessor()
    
    # Load all component results
    cluster_signals = processor.load_clustering_results()
    vpin_indicators = processor.load_vpin_results()
    seasonal_factors = processor.load_seasonal_factors()
    tvp_var_forecasts = processor.load_tvp_var_results()
    dma_predictions = processor.load_dma_results()
    
    # Generate integrated predictions
    final_predictions = processor.integrate_predictions({
        'clusters': cluster_signals,
        'vpin': vpin_indicators,
        'seasonal': seasonal_factors,
        'tvp_var': tvp_var_forecasts,
        'dma': dma_predictions
    })
    
    # Save final predictions
    processor.save_final_predictions(final_predictions)
    
    return final_predictions


# Create the DAG
dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='Atlas Quanta Market Prediction Pipeline',
    schedule_interval=SCHEDULE_INTERVAL,
    max_active_runs=1,
    tags=['atlas-quanta', 'market-prediction', 'etl']
)

# Data extraction tasks
extract_stocks = PythonOperator(
    task_id='extract_stock_data',
    python_callable=extract_stock_data,
    dag=dag
)

extract_flows = PythonOperator(
    task_id='extract_investor_flows',
    python_callable=extract_investor_flows,
    dag=dag
)

extract_sentiment = PythonOperator(
    task_id='extract_sentiment_data',
    python_callable=extract_sentiment_data,
    dag=dag
)

extract_crypto = PythonOperator(
    task_id='extract_crypto_data',
    python_callable=extract_crypto_data,
    dag=dag
)

# Silver layer processing
process_silver = PythonOperator(
    task_id='process_to_silver_layer',
    python_callable=process_to_silver,
    dag=dag
)

# Gold layer analytics
calc_clusters = PythonOperator(
    task_id='calculate_investor_clusters',
    python_callable=calculate_investor_clusters,
    dag=dag
)

calc_vpin = PythonOperator(
    task_id='calculate_vpin_indicators',
    python_callable=calculate_vpin_indicators,
    dag=dag
)

calc_seasonal = PythonOperator(
    task_id='calculate_seasonal_factors',
    python_callable=calculate_seasonal_factors,
    dag=dag
)

# Model training and prediction
train_tvp_var = PythonOperator(
    task_id='train_tvp_var_models',
    python_callable=train_tvp_var_models,
    dag=dag
)

run_dma = PythonOperator(
    task_id='run_dma_ensemble',
    python_callable=run_dma_ensemble,
    dag=dag
)

# Final prediction integration
generate_predictions = PythonOperator(
    task_id='generate_final_predictions',
    python_callable=generate_final_predictions,
    dag=dag
)

# Data quality checks
data_quality_check = BashOperator(
    task_id='data_quality_check',
    bash_command='python /opt/airflow/src/data_quality/quality_checks.py',
    dag=dag
)

# Define task dependencies
# Extraction tasks run in parallel
[extract_stocks, extract_flows, extract_sentiment, extract_crypto] >> process_silver

# Silver processing leads to gold layer analytics
process_silver >> [calc_clusters, calc_vpin, calc_seasonal]

# Analytics feed into model training
[calc_clusters, calc_vpin, calc_seasonal] >> train_tvp_var
[calc_clusters, calc_vpin, calc_seasonal] >> run_dma

# Models generate final predictions
[train_tvp_var, run_dma] >> generate_predictions

# Quality check at the end
generate_predictions >> data_quality_check
