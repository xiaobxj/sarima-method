# SARIMA Time Series Modeling for Treasury Cash Flow Data

## Project Overview

This project implements SARIMA time series modeling on U.S. Treasury cash flow data. It covers data collection, preprocessing, generation of individual time series, and batch SARIMA modeling.

## Project Structure

```
pythonProject/
├── data/
│   ├── raw/                              # Raw data files
│   │   ├── deposits_withdrawals_operating_cash.csv
│   │   ├── daily_cash_flows_*.csv
│   │   └── ...
│   └── processed/
│       ├── individual_time_series/       # 197 independent time series files
│       └── treasury_modeling_data_*.csv
├── src/
│   ├── data/
│   │   ├── data_collector.py             # Data collection module
│   │   ├── enhanced_field_mapper_complete.py
│   │   └── run_data_collection.py
│   └── model/
│       ├── sarima_model.py               # Core SARIMA modeling class
│       ├── batch_modeling.py             # Batch modeling processing
│       ├── run_batch_sarima.py           # Batch modeling execution script
│       └── fitted_models/                # 186 trained SARIMA models
├── config/
│   ├── config.py                         # Configuration file
│   └── env_template.txt
├── logs/                                 # Log files
├── requirements.txt                      # Dependency list
├── main.py                               # Main entry point
└── run_batch_sarima.py                   # Quick modeling script
```

## Features

### 1. Data Processing
- **Individual Time Series Generation**: Extracts 197 unique Treasury categories from raw data and generates independent time series datasets
- **Data Preprocessing**: Handles missing values, outlier detection, and data cleaning
- **Non-modeling Category Filtering**: Automatically excludes non-modeling categories such as Cash FTDs and Public Debt

### 2. SARIMA Modeling
- **Automatic Parameter Optimization**: Uses grid search to find the best SARIMA parameters
- **Batch Modeling**: Parallel processing of multiple time series for efficiency
- **Model Validation**: Includes residual analysis, Ljung-Box test, and other diagnostics
- **Forecasting**: Supports multi-step forecasting and confidence interval estimation

### 3. Major Categories Covered

**Deposits**:
- Withheld Income and Employment Taxes
- Individual Income Taxes
- Corporation Income Taxes
- Federal Reserve Earnings
- Other Deposits

**Withdrawals**:
- Social Security Benefits (EFT)
- Medicare & Medicaid
- Defense Vendor Payments (EFT)
- Education Department Programs
- Agriculture Department Programs
- Unemployment Insurance Benefits
- Other Withdrawals

## Installation and Usage

### Requirements
```bash
pip install -r requirements.txt
```

### Main Dependencies
- pandas >= 2.0.0
- numpy >= 1.24.0
- statsmodels >= 0.14.0
- scikit-learn
- matplotlib
- seaborn

### Quick Start

1. **Generate Individual Time Series**:
```python
# Already completed: 197 independent time series files saved in data/processed/individual_time_series/
```

2. **Batch SARIMA Modeling**:
```bash
python -m src.model.run_batch_sarima
```

3. **Use Individual Model**:
```python
from src.model.sarima_model import SARIMAModel

# Load specific time series
model = SARIMAModel('Defense_Vendor_Payments_EFT', 
                   'data/processed/individual_time_series/Defense_Vendor_Payments_EFT.csv')
model.load_data()
model.fit_model()
forecast = model.forecast(steps=30)
```

## Modeling Results

- **Successfully Modeled**: 186/195 time series (95.4% success rate)
- **Skipped Files**: 9 files skipped due to insufficient data
- **Model Type**: SARIMA(p,d,q)×(P,D,Q,s) where s=7 (weekly seasonality)
- **Evaluation Metric**: AIC used for model selection and comparison

### Top Models (by AIC)
- Change_in_Balance_of_Uncollected_Funds: AIC = -40742.81
- Transfers_to_Depositaries: AIC = -40742.81  
- Transfers_to_Federal_Reserve_Account: AIC = -37209.87
- Transfers_from_Federal_Reserve_Account: AIC = -37235.66
- Interest_recd_from_cash_investments: AIC = -35868.90

## Data Sources

This project uses U.S. Treasury Daily Treasury Statement data, including:
- Federal Reserve Account deposit and withdrawal records
- Public debt transaction data
- Tax deposit and refund data
- Various government department expenditure data
- Time span: 2016 to 2025

## Technical Features

- **Parallel Processing**: Supports multi-core parallel modeling
- **Automated Pipeline**: Fully automated process from data loading to model saving
- **Error Handling**: Comprehensive exception handling and logging
- **Model Persistence**: All trained models saved in pickle format
- **Diagnostic Tools**: Built-in model diagnostics and visualization functions

## Contributing

Welcome to submit Issues and Pull Requests to improve this project.

## License

MIT License

---

**Author**: xiaobxj  
**Project Link**: https://github.com/xiaobxj/sarima-method  
**Created**: January 2025