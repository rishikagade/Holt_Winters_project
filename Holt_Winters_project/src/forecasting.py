"""
Forecasting Module
==================

This module handles Holt-Winters exponential smoothing for gold price forecasting.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import logging
from typing import Tuple, Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def forecast_gold_price(
    data: pd.DataFrame, 
    target_column: str = 'Price',
    forecast_days: int = 90,
    trend: str = 'add',
    seasonal: Optional[str] = None,
    seasonal_periods: Optional[int] = None
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Forecast gold prices using Holt-Winters exponential smoothing.
    
    Args:
        data (pd.DataFrame): Preprocessed gold price data
        target_column (str): Column to forecast (default: 'Price')
        forecast_days (int): Number of days to forecast (default: 90)
        trend (str): Trend component ('add', 'mul', or None)
        seasonal (str): Seasonal component ('add', 'mul', or None)
        seasonal_periods (int): Number of periods in a season
        
    Returns:
        Tuple[pd.Series, Dict[str, Any]]: Forecast values and model information
    """
    logger.info(f"Starting Holt-Winters forecasting for {forecast_days} days...")
    
    # Extract target series
    y = data[target_column].copy()
    
    # Remove any missing values
    y = y.dropna()
    
    if len(y) == 0:
        raise ValueError("No valid data points for forecasting")
    
    logger.info(f"Fitting model with {len(y)} observations")
    
    try:
        # Fit Holt-Winters model
        if seasonal is not None and seasonal_periods is not None:
            model = ExponentialSmoothing(
                y, 
                trend=trend, 
                seasonal=seasonal, 
                seasonal_periods=seasonal_periods
            )
            logger.info(f"Using Holt-Winters with {trend} trend and {seasonal} seasonality")
        else:
            model = ExponentialSmoothing(y, trend=trend)
            logger.info(f"Using Holt-Winters with {trend} trend (no seasonality)")
        
        fitted_model = model.fit()
        logger.info("Model fitting completed")
        
        # Generate forecast
        forecast_values = fitted_model.forecast(forecast_days)
        
        # Create date index for forecast
        last_date = y.index[-1]
        forecast_index = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )
        forecast_values.index = forecast_index
        
        # Prepare model information
        model_info = {
            'model_type': 'Holt-Winters Exponential Smoothing',
            'trend': trend,
            'seasonal': seasonal,
            'seasonal_periods': seasonal_periods,
            'alpha': getattr(fitted_model.model, 'smoothing_level', None),
            'beta': getattr(fitted_model.model, 'smoothing_trend', None),
            'gamma': getattr(fitted_model.model, 'smoothing_seasonal', None),
            'sum_squared_errors': getattr(fitted_model, 'sse', None),
            'aic': getattr(fitted_model, 'aic', None),
            'bic': getattr(fitted_model, 'bic', None),
            'forecast_dates': forecast_index.tolist(),
            'last_known_value': float(y.iloc[-1]),
            'first_forecast_value': float(forecast_values.iloc[0]),
            'last_forecast_value': float(forecast_values.iloc[-1])
        }
        
        logger.info(f"Forecast generated from {forecast_values.iloc[0]:.2f} to {forecast_values.iloc[-1]:.2f}")
        
        return forecast_values, model_info
        
    except Exception as e:
        logger.error(f"Error in Holt-Winters forecasting: {str(e)}")
        raise


def evaluate_forecast(
    actual: pd.Series, 
    predicted: pd.Series
) -> Dict[str, float]:
    """
    Evaluate forecast accuracy using common metrics.
    
    Args:
        actual (pd.Series): Actual values
        predicted (pd.Series): Predicted values
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics
    """
    # Align the series
    aligned_data = pd.concat([actual, predicted], axis=1, join='inner').dropna()
    if len(aligned_data) == 0:
        raise ValueError("No overlapping data points for evaluation")
    
    actual_vals = aligned_data.iloc[:, 0].values
    pred_vals = aligned_data.iloc[:, 1].values
    
    # Calculate metrics
    mse = np.mean((actual_vals - pred_vals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual_vals - pred_vals))
    mape = np.mean(np.abs((actual_vals - pred_vals) / actual_vals)) * 100
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }


def backtest_model(
    data: pd.DataFrame,
    target_column: str = 'Price',
    train_ratio: float = 0.8,
    forecast_days: int = 30,
    **model_kwargs
) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
    """
    Perform backtesting of the forecasting model.
    
    Args:
        data (pd.DataFrame): Input data
        target_column (str): Target column to forecast
        train_ratio (float): Proportion of data to use for training
        forecast_days (int): Number of days to forecast in each backtest
        **model_kwargs: Additional arguments to pass to forecast_gold_price
        
    Returns:
        Tuple[pd.Series, pd.Series, Dict[str, float]]: 
            Actual values, predicted values, and evaluation metrics
    """
    logger.info("Starting model backtesting...")
    
    # Split data
    train_size = int(len(data) * train_ratio)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:train_size + forecast_days]
    
    if len(test_data) < forecast_days:
        logger.warning(f"Insufficient test data. Using {len(test_data)} days instead of {forecast_days}")
        forecast_days = len(test_data)
    
    # Fit model on training data and forecast
    forecast_values, model_info = forecast_gold_price(
        train_data, 
        target_column=target_column,
        forecast_days=forecast_days,
        **model_kwargs
    )
    
    # Get actual values for the forecast period
    actual_values = test_data[target_column].iloc[:forecast_days]
    
    # Ensure index alignment
    forecast_values.index = actual_values.index
    
    # Evaluate
    metrics = evaluate_forecast(actual_values, forecast_values)
    
    logger.info(f"Backtesting completed. MAPE: {metrics['mape']:.2f}%")
    
    return actual_values, forecast_values, metrics


if __name__ == "__main__":
    # Test the forecasting module
    try:
        # Import required modules
        from data_loader import load_gold_data
        from preprocessing import preprocess_gold_data
        
        # Load and preprocess data
        raw_data = load_gold_data()
        processed_data = preprocess_gold_data(raw_data)
        
        # Test forecasting
        print("\n" + "="*60)
        print("FORECASTING MODULE TEST")
        print("="*60)
        
        forecast_vals, model_info = forecast_gold_price(
            processed_data, 
            target_column='Price',
            forecast_days=30
        )
        
        print(f"Model: {model_info['model_type']}")
        print(f"Trend: {model_info['trend']}")
        print(f"Last known price: ${model_info['last_known_value']:.2f}")
        print(f"Forecast range: ${model_info['first_forecast_value']:.2f} to ${model_info['last_forecast_value']:.2f}")
        print(f"Forecast dates: {model_info['forecast_dates'][0]} to {model_info['forecast_dates'][-1]}")
        
        # Show first 5 forecast values
        print("\nFirst 5 forecast values:")
        for i in range(min(5, len(forecast_vals))):
            print(f"  {forecast_vals.index[i].strftime('%Y-%m-%d')}: ${forecast_vals.iloc[i]:.2f}")
        
        print("="*60)
        
    except Exception as e:
        print(f"Error in forecasting test: {e}")
        import traceback
        traceback.print_exc()