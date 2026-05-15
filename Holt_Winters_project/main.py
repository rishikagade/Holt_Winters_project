#!/usr/bin/env python
"""
Gold Price Prediction - Main Entry Point
======================================

This script serves as the main entry point for the gold price prediction project.
It orchestrates the data loading, preprocessing, forecasting, and simulation processes.

Author: Bagdai Mishree, Bute Vishwaja, Gade Rishika
"""

import sys
import os
from pathlib import Path
from typing import Tuple

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

def get_user_forecast_days() -> int:
    """
    Get forecast days from user input with validation.
    
    Returns:
        int: Number of days to forecast
    """
    while True:
        try:
            user_input = input("\nEnter number of days to forecast (default: 90): ").strip()
            if not user_input:  # Empty input uses default
                return 90
            forecast_days = int(user_input)
            if forecast_days <= 0:
                print("Please enter a positive number of days.")
                continue
            if forecast_days > 3650:  # Reasonable upper limit (10 years)
                print("Please enter a reasonable number of days (max 3650).")
                continue
            return forecast_days
        except ValueError:
            print("Please enter a valid integer.")

def main():
    """Main function to run the gold price prediction pipeline."""
    print("=" * 60)
    print("GOLD PRICE PREDICTION PROJECT")
    print("=" * 60)
    print("Initializing gold price prediction pipeline...")
    
    try:
        # Import modules
        from data_loader import load_gold_data
        from preprocessing import preprocess_gold_data
        from forecasting import forecast_gold_price
        from simulation import run_monte_carlo_simulation
        
        # Get user input for forecast horizon
        forecast_days = get_user_forecast_days()
        print(f"\nForecast horizon set to: {forecast_days} days")
        
        # Step 1: Load data
        print("\n1. Loading gold price data...")
        raw_data = load_gold_data()
        print(f"   Loaded {len(raw_data)} records")
        
        # Step 2: Preprocess data
        print("\n2. Preprocessing data...")
        processed_data = preprocess_gold_data(raw_data)
        print(f"   Processed data shape: {processed_data.shape}")
        
        # Step 3: Generate forecasts
        print(f"\n3. Generating price forecasts for {forecast_days} days...")
        forecast_results, forecast_info = forecast_gold_price(processed_data, forecast_days=forecast_days)
        print(f"   Generated forecast for {len(forecast_results)} days")
        
        # Step 4: Run Monte Carlo simulation
        print(f"\n4. Running Monte Carlo simulation for {forecast_days} days...")
        simulation_results, simulation_stats = run_monte_carlo_simulation(
            processed_data, forecast_results, simulation_days=forecast_days
        )
        print("   Simulation completed successfully")
        
        # Step 5: Display results summary
        print("\n5. Results Summary:")
        print("   - Data loading: Complete")
        print("   - Preprocessing: Complete") 
        print("   - Forecasting: Complete")
        print("   - Simulation: Complete")
        
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        return 0
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all required modules are installed.")
        return 1
    except Exception as e:
        print(f"Error during pipeline execution: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())