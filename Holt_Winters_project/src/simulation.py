"""
Simulation Module
=================

This module handles Monte Carlo simulation for gold price forecasting.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_monte_carlo_simulation(
    data: pd.DataFrame,
    forecast_values: pd.Series,
    target_column: str = 'Price',
    num_simulations: int = 1000,
    simulation_days: int = 90,
    confidence_levels: list = [0.025, 0.975]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run Monte Carlo simulation for gold price forecasting.
    
    Args:
        data (pd.DataFrame): Preprocessed gold price data
        forecast_values (pd.Series): Forecast values from Holt-Winters model
        target_column (str): Column to simulate (default: 'Price')
        num_simulations (int): Number of Monte Carlo simulations to run
        simulation_days (int): Number of days to simulate
        confidence_levels (list): List of confidence levels for prediction intervals
        
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Simulation results and statistics
    """
    logger.info(f"Starting Monte Carlo simulation with {num_simulations} simulations...")
    
    # Extract historical returns for parameter estimation
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    prices = data[target_column].dropna()
    if len(prices) < 2:
        raise ValueError("Insufficient price data for return calculation")
    
    # Calculate log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    if len(log_returns) == 0:
        raise ValueError("No valid log returns calculated")
    
    # Estimate parameters for geometric Brownian motion
    mu = log_returns.mean()           # drift
    sigma = log_returns.std()         # volatility
    
    logger.info(f"Estimated parameters: mu={mu:.6f}, sigma={sigma:.6f}")
    
    # Get starting price (last known price)
    start_price = prices.iloc[-1]
    logger.info(f"Starting price for simulation: ${start_price:.2f}")
    
    # Initialize simulation array
    simulation_results = np.zeros((simulation_days + 1, num_simulations))
    simulation_results[0, :] = start_price
    
    # Generate random shocks
    np.random.seed(42)  # For reproducibility
    random_shocks = np.random.normal(0, 1, (simulation_days, num_simulations))
    
    # Run simulation using geometric Brownian motion
    for day in range(1, simulation_days + 1):
        # Geometric Brownian motion: S_t = S_{t-1} * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        # Assuming dt = 1 day
        drift = (mu - 0.5 * sigma**2)
        diffusion = sigma * random_shocks[day-1, :]
        simulation_results[day, :] = simulation_results[day-1, :] * np.exp(drift + diffusion)
    
    # Create DataFrame with results
    # Create date index for simulation
    last_date = data.index[-1] if hasattr(data.index, '__getitem__') else pd.Timestamp.now()
    if hasattr(last_date, 'strftime'):
        # If it's already a timestamp
        date_index = pd.date_range(start=last_date, periods=simulation_days+1, freq='D')
    else:
        # Fallback: create a simple date range
        date_index = pd.date_range(start='2023-01-01', periods=simulation_days+1, freq='D')
    
    # Create column names
    column_names = ['Date'] + [f'Simulation_{i+1}' for i in range(num_simulations)]
    
    # Create DataFrame
    sim_df = pd.DataFrame(simulation_results, index=date_index, columns=[f'Simulation_{i+1}' for i in range(num_simulations)])
    sim_df.insert(0, 'Date', sim_df.index)
    
    logger.info(f"Simulation completed. Shape: {sim_df.shape}")
    
    # Calculate statistics
    final_prices = sim_df.iloc[-1, 1:]  # Exclude Date column
    
    stats = {
        'starting_price': float(start_price),
        'mu': float(mu),
        'sigma': float(sigma),
        'num_simulations': num_simulations,
        'simulation_days': simulation_days,
        'mean_final_price': float(final_prices.mean()),
        'std_final_price': float(final_prices.std()),
        'median_final_price': float(final_prices.median()),
        'min_final_price': float(final_prices.min()),
        'max_final_price': float(final_prices.max())
    }
    
    # Add confidence intervals
    for cl in confidence_levels:
        if 0 < cl < 1:
            lower_idx = int((1 - cl) / 2 * num_simulations)
            upper_idx = int((1 + cl) / 2 * num_simulations)
            sorted_prices = np.sort(final_prices)
            stats[f'ci_{int(cl*100)}_lower'] = float(sorted_prices[lower_idx])
            stats[f'ci_{int(cl*100)}_upper'] = float(sorted_prices[upper_idx])
    
    # Compare with forecast if provided
    if forecast_values is not None and len(forecast_values) > 0:
        forecast_final = forecast_values.iloc[-1] if hasattr(forecast_values, 'iloc') else forecast_values[-1]
        stats['forecast_final_price'] = float(forecast_final)
        stats['price_difference_forecast_vs_sim'] = float(stats['mean_final_price'] - forecast_final)
    
    logger.info(f"Simulation statistics calculated:")
    logger.info(f"  Mean final price: ${stats['mean_final_price']:.2f}")
    logger.info(f"  Std final price: ${stats['std_final_price']:.2f}")
    
    return sim_df, stats


def calculate_simulation_metrics(simulation_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate additional metrics from simulation results.
    
    Args:
        simulation_df (pd.DataFrame): Results from Monte Carlo simulation
        
    Returns:
        Dict[str, Any]: Additional simulation metrics
    """
    # Extract simulation columns (exclude Date)
    sim_cols = [col for col in simulation_df.columns if col.startswith('Simulation_')]
    if not sim_cols:
        return {}
    
    sim_data = simulation_df[sim_cols]
    
    # Calculate probability of profit/loss if we have a reference point
    reference_price = simulation_df.iloc[0, sim_cols[0]] if len(sim_cols) > 0 else 0
    
    final_prices = sim_data.iloc[-1]
    profit_probability = (final_prices > reference_price).mean() * 100
    
    # Calculate value at risk (VaR) at different levels
    var_95 = np.percentile(final_prices, 5)  # 5th percentile
    var_99 = np.percentile(final_prices, 1)  # 1st percentile
    
    # Calculate expected shortfall (Conditional VaR)
    cvar_95 = final_prices[final_prices <= var_95].mean()
    cvar_99 = final_prices[final_prices <= var_99].mean()
    
    metrics = {
        'reference_price': float(reference_price),
        'profit_probability': float(profit_probability),
        'value_at_risk_95': float(var_95),
        'value_at_risk_99': float(var_99),
        'conditional_var_95': float(cvar_95),
        'conditional_var_99': float(cvar_99),
        'expected_return': float((final_prices.mean() - reference_price) / reference_price * 100),
        'volatility_annualized': float(final_prices.std() / reference_price * np.sqrt(252) * 100)
    }
    
    return metrics


def plot_simulation_results(
    simulation_df: pd.DataFrame, 
    forecast_values: Optional[pd.Series] = None,
    figsize: tuple = (12, 8)
) -> None:
    """
    Plot simulation results (placeholder for visualization).
    
    Args:
        simulation_df (pd.DataFrame): Simulation results
        forecast_values (pd.Series): Optional forecast values to overlay
        figsize (tuple): Figure size for plotting
    """
    logger.info("Simulation plotting function called (visualization would be implemented here)")
    # In a full implementation, this would create plots using matplotlib/seaborn
    # For now, we'll just log that the function was called
    

if __name__ == "__main__":
    # Test the simulation module
    try:
        # Import required modules
        from data_loader import load_gold_data
        from preprocessing import preprocess_gold_data
        from forecasting import forecast_gold_price
        
        # Load and preprocess data
        raw_data = load_gold_data()
        processed_data = preprocess_gold_data(raw_data)
        
        # Generate forecast
        forecast_vals, _ = forecast_gold_price(
            processed_data, 
            target_column='Price',
            forecast_days=30
        )
        
        # Run Monte Carlo simulation
        print("\n" + "="*60)
        print("MONTE CARLO SIMULATION TEST")
        print("="*60)
        
        sim_results, stats = run_monte_carlo_simulation(
            processed_data,
            forecast_vals,
            target_column='Price',
            num_simulations=1000,  # Reduced for testing
            simulation_days=30
        )
        
        print(f"Starting price: ${stats['starting_price']:.2f}")
        print(f"Mean final price: ${stats['mean_final_price']:.2f}")
        print(f"Price std dev: ${stats['std_final_price']:.2f}")
        print(f"95% CI: [${stats.get('ci_95_lower', 0):.2f}, ${stats.get('ci_95_upper', 0):.2f}]")
        
        if 'forecast_final_price' in stats:
            print(f"Forecast final price: ${stats['forecast_final_price']:.2f}")
            print(f"Difference (sim - forecast): ${stats['price_difference_forecast_vs_sim']:.2f}")
        
        # Calculate additional metrics
        metrics = calculate_simulation_metrics(sim_results)
        print(f"\nAdditional Metrics:")
        print(f"  Profit probability: {metrics.get('profit_probability', 0):.1f}%")
        print(f"  Expected return: {metrics.get('expected_return', 0):.2f}%")
        
        print("="*60)
        
    except Exception as e:
        print(f"Error in simulation test: {e}")
        import traceback
        traceback.print_exc()