"""
Preprocessing Module
====================

This module handles data cleaning, feature engineering, and preparation 
for gold price forecasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_gold_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess gold price data for forecasting.
    
    Args:
        df (pd.DataFrame): Raw gold price data
        
    Returns:
        pd.DataFrame: Preprocessed data with additional features
    """
    logger.info("Starting gold data preprocessing...")
    
    # Create a copy to avoid modifying original data
    processed_df = df.copy()
    
    # Ensure we have a datetime index
    if not isinstance(processed_df.index, pd.DatetimeIndex):
        if 'Date' in processed_df.columns:
            processed_df['Date'] = pd.to_datetime(processed_df['Date'])
            processed_df = processed_df.set_index('Date')
        else:
            raise ValueError("No Date column or datetime index found")
    
    # Sort by date to ensure chronological order
    processed_df = processed_df.sort_index()
    
    logger.info(f"Data sorted chronologically. Shape: {processed_df.shape}")
    
    # Handle missing values
    initial_missing = processed_df.isnull().sum().sum()
    if initial_missing > 0:
        logger.info(f"Found {initial_missing} missing values. Applying forward fill...")
        processed_df = processed_df.ffill()
        # If any missing values remain (at the beginning), use backward fill
        processed_df = processed_df.bfill()
        logger.info("Missing values handled using forward/backward fill")
    
    # Feature engineering
    logger.info("Adding technical features...")
    
    # Price movements and returns
    processed_df['Price_Return'] = processed_df['Price'].pct_change()
    processed_df['Log_Return'] = np.log(processed_df['Price'] / processed_df['Price'].shift(1))
    
    # Moving averages
    processed_df['MA_7'] = processed_df['Price'].rolling(window=7).mean()
    processed_df['MA_30'] = processed_df['Price'].rolling(window=30).mean()
    
    # Volatility measures
    processed_df['Volatility_7'] = processed_df['Price_Return'].rolling(window=7).std()
    processed_df['Volatility_30'] = processed_df['Price_Return'].rolling(window=30).std()
    
    # Price ratios
    processed_df['High_Low_Ratio'] = processed_df['High'] / processed_df['Low']
    processed_df['Open_Close_Ratio'] = processed_df['Open'] / processed_df['Price']
    
    # Candlestick features (recreating from original script)
    processed_df['PriceMovement'] = ['UP' if close >= open else 'Down' 
                                   for close, open in zip(processed_df['Price'], processed_df['Open'])]
    processed_df['BarHeight'] = [abs(close - open) 
                                for close, open in zip(processed_df['Price'], processed_df['Open'])]
    processed_df['BarStart'] = [min(close, open) 
                               for close, open in zip(processed_df['Price'], processed_df['Open'])]
    
    # Handle missing values created by rolling windows and pct_change
    processed_df = processed_df.bfill().ffill()
    
    logger.info(f"Preprocessing complete. Final shape: {processed_df.shape}")
    logger.info(f"Features added: {[col for col in processed_df.columns if col not in df.columns]}")
    
    return processed_df


def prepare_forecasting_data(df: pd.DataFrame, target_column: str = 'Price') -> Tuple[pd.Series, pd.DataFrame]:
    """
    Prepare data specifically for forecasting models.
    
    Args:
        df (pd.DataFrame): Preprocessed gold price data
        target_column (str): Name of the target column to forecast
        
    Returns:
        Tuple[pd.Series, pd.DataFrame]: Target series and feature matrix
    """
    logger.info("Preparing data for forecasting...")
    
    # Extract target variable
    y = df[target_column].copy()
    
    # Create feature matrix (excluding target and non-numeric columns)
    feature_cols = [col for col in df.columns 
                   if col != target_column and df[col].dtype in ['float64', 'int64']]
    
    X = df[feature_cols].copy()
    
    # Handle any remaining missing values
    X = X.bfill().ffill()
    
    logger.info(f"Target variable: {target_column} ({len(y)} observations)")
    logger.info(f"Feature matrix: {X.shape[1]} features")
    
    return y, X


def get_feature_importance_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract relevant features for analysis and modeling.
    
    Args:
        df (pd.DataFrame): Preprocessed data
        
    Returns:
        pd.DataFrame: Features suitable for modeling
    """
    # Select numeric columns for modeling
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove any columns that might be problematic
    exclude_cols = []  # Add any columns to exclude here
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    return df[feature_cols].copy()


if __name__ == "__main__":
    # Test the preprocessing module
    try:
        # Import data loader to get test data
        from data_loader import load_gold_data
        
        # Load and preprocess data
        raw_data = load_gold_data()
        processed_data = preprocess_gold_data(raw_data)
        
        # Show some info
        print("\n" + "="*50)
        print("PREPROCESSING TEST RESULTS")
        print("="*50)
        print(f"Original shape: {raw_data.shape}")
        print(f"Processed shape: {processed_data.shape}")
        print(f"New features: {[col for col in processed_data.columns if col not in raw_data.columns]}")
        print("\nProcessed data info:")
        print(processed_data.head())
        print("="*50)
        
    except Exception as e:
        print(f"Error in preprocessing test: {e}")
        import traceback
        traceback.print_exc()