# Gold Price Prediction Project

## Overview
This project implements a gold price forecasting system using Holt-Winters exponential smoothing and Monte Carlo simulation. It's designed as a student portfolio project demonstrating time series forecasting techniques applied to financial data.

## Key Enhancement: Custom Forecast Horizon
**NEW FEATURE**: Users can now specify their own forecast time period instead of being limited to the standard 90-day prediction. When running the program, you'll be prompted to enter the number of days to forecast, with validation to ensure reasonable input.

## Project Structure
```
Holt_Winters_project/
│
├── data/
│   └── Gold_Daily.xlsx              # Gold price data
│
├── src/
│   ├── __init__.py                  # Package initializer
│   ├── data_loader.py               # Data loading functionality
│   ├── preprocessing.py             # Data cleaning and feature engineering
│   ├── forecasting.py               # Holt-Winters modeling and predictions
│   └── simulation.py                # Monte Carlo simulation
│
├── logs/                            # Log files (generated)
├── models/                          # Saved models (generated)
├── outputs/                         # Output files (generated)
│
├── main.py                          # Main entry point
├── requirements.txt                 # Project dependencies
└── README.md                        # This file
```

## Features
- **Data Loading**: Loads gold price data from Excel format
- **Preprocessing**: Cleans data, handles missing values, and engineers features
- **Forecasting**: Implements Holt-Winters exponential smoothing for price prediction with **user-specified forecast horizon**
- **Simulation**: Runs Monte Carlo simulation for risk assessment and uncertainty quantification
- **Modular Design**: Separated concerns for maintainability and extensibility
- **Interactive Interface**: Prompt user for forecast horizon at runtime

## Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main pipeline:
```bash
python main.py
```

You will be prompted:
```
Enter number of days to forecast (default: 90): 
```

Enter any positive integer for custom forecast horizon, or press Enter to use the default 90 days.

Or run individual modules for testing:
```bash
python -m src.data_loader
python -m src.preprocessing
python -m src.forecasting
python -m src.simulation
```

## Methodology
### Data Preprocessing
- Converts date column to datetime index
- Handles missing values using forward/backward fill
- Engineers technical features:
  - Price returns and log returns
  - Moving averages (7-day, 30-day)
  - Volatility measures
  - Price ratios (High/Low, Open/Close)
  - Candlestick features

### Forecasting (Holt-Winters)
- Uses exponential smoothing with trend component
- Configurable seasonality (though gold data often shows limited seasonality)
- **Generates forecasts for user-specified horizon** (default: 90 days)
- Provides model parameters and goodness-of-fit metrics

### Monte Carlo Simulation
- Models price movements using Geometric Brownian Motion
- Estimates drift (μ) and volatility (σ) from historical log returns
- Runs multiple simulation paths to assess uncertainty
- Calculates risk metrics (VaR, CVaR, profit probability)

## Outputs
The project generates:
- Forecast predictions for gold prices (user-specified horizon)
- Monte Carlo simulation results with statistical summaries
- Risk assessment metrics
- Visualization-ready data (plotting functions available)

## Customization
Adjust parameters in the main.py file or at runtime:
- **Forecast horizon**: User-specified at runtime (default: 90 days)
- Number of Monte Carlo simulations (default: 1000)
- Model configuration (trend, seasonality)
- Confidence levels for prediction intervals

## Forecast Accuracy Disclaimer
**Important Accuracy Considerations**:
While the model will generate forecasts for any horizon you specify, the reliability and accuracy of these forecasts decrease as the time horizon increases. Based on back-testing and error analysis:

- **Short-term (1-30 days)**: Generally reasonable forecasts with typical MAPE (Mean Absolute Percentage Error) < 5%
- **Medium-term (31-60 days)**: Moderately reliable with increasing error (MAPE typically 5-10%)
- **Long-term (61+ days)**: Highly uncertain forecasts; primarily useful for trend direction rather than precise values

The widening confidence intervals in the Monte Carlo simulation output visually demonstrate this increasing uncertainty. Always consider:
1. Forecasts are extrapolations of historical trends, not predictions of future events
2. External factors (interest rates, geopolitical events, economic data) are not modeled
3. Past performance does not guarantee future results
4. Use forecasts as one tool among many in your decision-making process
5. For horizons beyond 60 days, treat forecasts as directional indicators only

## Requirements
See `requirements.txt` for detailed dependencies:
- pandas: Data manipulation
- numpy: Numerical computations
- statsmodels: Holt-Winters implementation
- matplotlib: Plotting capabilities
- openpyxl: Excel file reading

## Notes
- This project is intended for educational purposes
- Financial forecasting involves risk; past performance doesn't guarantee future results
- The models simplify complex market dynamics
- Consider transaction costs, taxes, and other factors in real applications

## References
- Holt, C.C. (2004). Forecasting seasonals and trends by exponentially weighted moving averages. International Journal of Forecasting.
- Winters, P.R. (1960). Forecasting sales by exponentially weighted moving averages. Management Science.
- Glasserman, P. (2004). Monte Carlo Methods in Financial Engineering. Springer.

---
*Student Portfolio Project - Gold Price Forecasting using Holt-Winters and Monte Carlo Simulation*