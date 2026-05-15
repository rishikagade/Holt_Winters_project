"""
Data Loader Module
=================

This module handles loading and initial inspection of gold price data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_gold_data(file_path: str = "../data/Gold_Daily.xlsx") -> pd.DataFrame:
    """
    Load gold price data from Excel file.
    
    Args:
        file_path (str): Path to the Excel file containing gold price data
        
    Returns:
        pd.DataFrame: Loaded gold price data with Date as index
        
    Raises:
        FileNotFoundError: If the specified file does not exist
        Exception: For any other errors during file loading
    """
    try:
        # Convert to absolute path if relative
        if not Path(file_path).is_absolute():
            base_path = Path(__file__).parent
            file_path = base_path / file_path
            
        logger.info(f"Loading gold price data from: {file_path}")
        
        # Check if file exists
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load data from Excel
        df = pd.read_excel(file_path)
        
        # Basic validation
        if df.empty:
            raise ValueError("Loaded data is empty")
            
        logger.info(f"Successfully loaded {len(df)} records")
        logger.info(f"Columns: {list(df.columns)}")
        
        # Convert Date column to datetime and set as index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            logger.info("Date column converted to datetime and set as index")
        else:
            logger.warning("No 'Date' column found in data")
            
        return df
        
    except Exception as e:
        logger.error(f"Error loading gold price data: {str(e)}")
        raise


def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the loaded data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Dictionary containing data information
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'date_range': {
            'start': df.index.min() if hasattr(df.index, 'min') else None,
            'end': df.index.max() if hasattr(df.index, 'max') else None
        },
        'sample_data': df.head().to_dict() if len(df) > 0 else {}
    }
    
    return info


def print_data_summary(df: pd.DataFrame) -> None:
    """
    Print a summary of the loaded data.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    print("\n" + "="*50)
    print("GOLD PRICE DATA SUMMARY")
    print("="*50)
    print(f"Dataset Shape: {df.shape}")
    print(f"Date Range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {', '.join(df.columns)}")
    print("\nData Types:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")
    print("\nMissing Values:")
    for col, missing in df.isnull().sum().items():
        print(f"  {col}: {missing}")
    print("\nFirst 5 Records:")
    print(df.head())
    print("="*50 + "\n")


if __name__ == "__main__":
    # Test the data loader
    try:
        data = load_gold_data()
        print_data_summary(data)
        info = get_data_info(data)
        print("Data info retrieved successfully")
    except Exception as e:
        print(f"Error in data loader test: {e}")