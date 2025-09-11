# US Treasury Debt Ceiling Analysis & Prediction System

## 🎯 Project Overview

This project implements a comprehensive **U.S. Treasury Debt Ceiling Analysis and Prediction System** that forecasts Treasury General Account (TGA) balance movements and identifies potential X-Date scenarios. The system integrates historical cash flow data, debt maturity schedules, and advanced simulation techniques to provide actionable insights for debt ceiling management.

## 🚀 Key Features

### 💰 **Cash Flow Analysis**
- **Historical Data Integration**: Processes 20+ years of Treasury cash flow data (2005-2025)
- **127+ Cash Flow Categories**: Comprehensive coverage of government receipts and outlays
- **Real-time Forecasting**: Generates 180-day forward predictions with confidence intervals

### 📊 **TGA Balance Simulation**
- **Advanced Simulation Engine**: Monte Carlo-style simulation with automatic debt issuance
- **X-Date Identification**: Predicts when Treasury cash balance may fall below critical thresholds
- **Debt Ceiling Constraints**: Incorporates statutory debt limits and automatic triggers

### 🗓️ **Debt Maturity Calendar**
- **Principal & Interest Tracking**: Monitors scheduled debt redemptions and coupon payments
- **CUSIP-Level Detail**: Tracks individual security characteristics and payment schedules
- **Integration Ready**: Seamlessly integrates with simulation engine for accurate cash flow projections

### 📈 **Advanced Analytics**
- **Interactive Visualizations**: Comprehensive charts and graphs for decision support
- **Scenario Analysis**: Multiple forecasting scenarios with sensitivity analysis
- **Risk Assessment**: Quantifies probability of debt ceiling breaches

## 🏗️ Project Structure

```
debt-prediction-system/
├── src/
│   └── debt_prediction/
│       ├── main.py                    # Main analysis entry point
│       ├── config.py                  # System configuration
│       ├── tga_simulator.py           # TGA balance simulation engine
│       ├── debt_analyzer.py           # Debt ceiling analysis
│       ├── debt_calendar.py           # Debt maturity tracking
│       └── cash_flow_converter.py     # Data conversion utilities
├── data/
│   └── raw/
│       ├── debt_subject_to_limit.csv  # Historical debt limit data
│       └── operating_cash_balance.csv # TGA balance history
├── src/model/fitted_models/forecasts/
│   └── cash_flow_forecasts_corrected_*.csv  # Processed forecast data
├── output/                            # Analysis results and visualizations
├── cash_flow_with_cat.csv            # Core 20-year cash flow dataset
├── requirements.txt                   # Python dependencies
└── README.md                         # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Git (for version control)

### Installation
```bash
# Clone the repository
git clone https://github.com/xiaobxj/sarima-method.git
cd sarima-method

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
```python
# Core Data Processing
pandas>=2.0.0
numpy>=1.24.0

# Financial Analysis
scipy>=1.10.0
statsmodels>=0.14.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.0.0

# Configuration & Utilities
python-dateutil>=2.8.0
pydantic>=2.0.0
tqdm>=4.65.0
```

## 🚦 Quick Start

### 1. **Basic Debt Analysis**
```python
from src.debt_prediction.main import run_complete_analysis

# Run full debt ceiling analysis
results = run_complete_analysis()
print(f"X-Date Prediction: {results['x_date']}")
print(f"Days Until X-Date: {results['days_to_x_date']}")
```

### 2. **TGA Balance Simulation**
```python
from src.debt_prediction.tga_simulator import TGABalanceSimulator
from src.debt_prediction.config import DebtPredictionConfig

# Initialize simulator
config = DebtPredictionConfig()
simulator = TGABalanceSimulator(config)

# Run 180-day simulation
results = simulator.run_simulation()
print(f"Final TGA Balance: ${results['opening_tga_balance'].iloc[-1]:,.0f}M")
```

### 3. **Debt Calendar Integration**
```python
from src.debt_prediction.debt_calendar import DebtEventsCalendar

# Create debt calendar
calendar = DebtEventsCalendar()
schedule = calendar.build_calendar()

# Query specific date
debt_service = calendar.get_scheduled_debt_service('2025-09-15')
print(f"Principal Due: ${debt_service['principal_due']:,.0f}M")
print(f"Interest Due: ${debt_service['interest_due']:,.0f}M")
```

## 📊 Data Sources & Methodology

### **Historical Cash Flow Data**
- **Source**: U.S. Treasury Daily Treasury Statements
- **Coverage**: October 2005 - July 2025 (418,000+ records)
- **Categories**: 127+ distinct cash flow types
- **Processing**: Automated sign correction and categorization

### **Forecasting Methodology**
- **Base Model**: Historical mean with seasonal adjustments
- **Validation**: 2024 data used for parameter estimation
- **Horizon**: 180-day rolling forecasts
- **Confidence Intervals**: 95% prediction bands

### **Key Cash Flow Categories**

#### 🔼 **Major Inflows**
- Public Debt Issuance: ~$118,491M/day
- Withheld Income Taxes: ~$13,508M/day  
- Corporate Income Taxes: ~$1,999M/day
- Federal Reserve Earnings: Variable

#### 🔽 **Major Outflows**
- Debt Redemptions: ~$113,860M/day
- Social Security Payments: ~$5,433M/day
- Medicare/Medicaid: ~$4,810M/day
- Interest on Treasury Securities: ~$2,194M/day

## 🎯 System Capabilities

### **X-Date Analysis**
- **Definition**: Date when Treasury cash balance falls below operational minimum
- **Threshold**: Configurable minimum balance (default: $30B)
- **Buffer**: Additional safety margin (default: $50B)
- **Probability**: Monte Carlo estimation of breach likelihood

### **Debt Ceiling Integration**
- **Current Limit**: $35,000B (configurable)
- **Outstanding Debt**: ~$33,000B
- **Available Capacity**: ~$2,000B
- **Automatic Issuance**: Triggered when TGA < minimum threshold

### **Scenario Analysis**
- **Base Case**: Historical patterns continue
- **Stress Testing**: Revenue shortfalls, expense surges
- **Policy Impact**: Tax policy changes, spending modifications
- **Seasonal Adjustments**: Holiday patterns, fiscal year effects

## 📈 Sample Output

```
🏛️  U.S. Treasury Debt Ceiling Analysis Report
═══════════════════════════════════════════════

📊 Current Status (2025-07-22):
   💰 TGA Balance: $215,160M
   📊 Public Debt: $33,000,000M  
   🎯 Debt Ceiling: $35,000,000M
   📏 Available Capacity: $2,000,000M

🔮 180-Day Forecast:
   📅 X-Date Estimate: 2026-01-15
   ⏰ Days to X-Date: 177 days
   📉 Minimum Balance: $25,847M (2025-12-31)
   
🎲 Risk Assessment:
   🟢 Low Risk (30+ days): 85%
   🟡 Medium Risk (7-30 days): 12%  
   🔴 High Risk (<7 days): 3%
```

## 🔧 Configuration

The system uses a centralized configuration approach:

```python
# src/debt_prediction/config.py
class DebtPredictionConfig:
    # Simulation Parameters
    START_DATE = date(2025, 7, 22)
    FORECAST_HORIZON = 180
    
    # Financial Thresholds  
    DEBT_CEILING_LIMIT = 35_000_000  # $35T
    MINIMUM_CASH_BALANCE = 30_000    # $30B
    CASH_BUFFER = 50_000            # $50B
    
    # Data Sources
    FORECASTS_FILE = "cash_flow_forecasts_corrected_*.csv"
    DEBT_DATA_FILE = "debt_subject_to_limit.csv"
```

## 🤝 Contributing

We welcome contributions to improve the debt prediction system:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Areas for Contribution
- Enhanced forecasting models (ARIMA, ML approaches)
- Real-time data integration
- Additional visualization capabilities
- Performance optimizations
- Documentation improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **U.S. Treasury**: For providing comprehensive financial data
- **Federal Reserve**: Economic data and research
- **Open Source Community**: Python libraries and tools

## 📞 Contact & Support

- **Author**: xiaobxj
- **Project**: [https://github.com/xiaobxj/sarima-method](https://github.com/xiaobxj/sarima-method)
- **Issues**: [GitHub Issues](https://github.com/xiaobxj/sarima-method/issues)

---

**⚠️ Disclaimer**: This system is for educational and research purposes only. It should not be used as the sole basis for financial or policy decisions. Always consult official Treasury guidance and professional financial analysis.

**🔄 Last Updated**: September 2025 | **Version**: 2.0.0 | **Status**: Active Development