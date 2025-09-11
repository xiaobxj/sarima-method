"""
Configuration Module for Treasury Debt Prediction System

This module contains all key parameters that may change during analysis,
enabling easy updates and scenario analysis.
"""

from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional
import os
import json


class DebtPredictionConfig:
    """
    Centralized configuration for Treasury debt prediction and analysis.
    
    This class manages all critical parameters including simulation dates,
    debt limits, cash thresholds, and file paths.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration with default values or from config file.
        
        Args:
            config_file (str, optional): Path to JSON configuration file
        """
        # === SIMULATION PARAMETERS ===
        
        # Simulation start date (last day with real data)
        # Updated to use cash_flow_with_cat data which ends 2025-07-21
        self.START_DATE = date(2025, 7, 22)
        
        # Forecast horizon in days
        self.FORECAST_HORIZON = 180
        
        # Current TGA balance at simulation start (in millions USD)
        self.CURRENT_TGA_BALANCE = 215_160.0  # $215.16 billion
        
        # Current total public debt outstanding (in millions USD)  
        self.CURRENT_PUBLIC_DEBT = 33_000_000.0  # $33 trillion
        
        # Legal debt ceiling limit (in millions USD)(check later)
        self.DEBT_CEILING_LIMIT = 35_000_000.0  # $35 trillion
        
        # Minimum cash balance for Treasury operations (in millions USD) (check later)
        self.MINIMUM_CASH_BALANCE = 30_000.0  # $30 billion
        
        # === OPERATIONAL PARAMETERS ===
        
        # Cash management buffer above minimum (in millions USD)
        self.CASH_BUFFER = 50_000.0  # $50 billion
        
        # Average maturity of outstanding debt (in days)
        self.AVERAGE_DEBT_MATURITY = 1825  # ~5 years
        
        # Daily debt rollover rate (fraction of total debt)
        self.DAILY_ROLLOVER_RATE = 1 / self.AVERAGE_DEBT_MATURITY
        
        # Interest rate assumptions (annual rates)
        self.SHORT_TERM_RATE = 0.05  # 5% for bills
        self.MEDIUM_TERM_RATE = 0.045  # 4.5% for notes
        self.LONG_TERM_RATE = 0.04  # 4% for bonds
        
        # === FILE PATHS ===
        
        # Base directories
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.OUTPUT_DIR = os.path.join(self.BASE_DIR, "output")
        self.MODEL_DIR = os.path.join(self.BASE_DIR, "src", "model", "fitted_models")
        
        # Input data files - Updated to use corrected cash flow data with proper signs
        self.FORECASTS_FILE = os.path.join(
            self.MODEL_DIR, "forecasts", "cash_flow_forecasts_corrected_20250911_125718.csv"
        )
        self.OPERATING_BALANCE_FILE = os.path.join(
            self.DATA_DIR, "raw", "operating_cash_balance.csv"
        )
        self.DEBT_DATA_FILE = os.path.join(
            self.DATA_DIR, "raw", "debt_subject_to_limit.csv"
        )
        
        # Debt calendar output file
        self.DEBT_CALENDAR_FILE = os.path.join(self.OUTPUT_DIR, "debt_events_calendar.csv")
        
        # === DEBT CALENDAR PARAMETERS ===
        
        # Whether to use live Treasury data (requires internet access)
        self.USE_LIVE_TREASURY_DATA = True
        
        # Treasury API endpoints (official U.S. Treasury Fiscal Data API - VERIFIED WORKING ENDPOINTS)
        self.TREASURY_API_ENDPOINTS = {
            'debt_to_penny': 'https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/debt_to_penny',
            'debt_subject_limit': 'https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/debt_subject_to_limit',
            'debt_auctions': 'https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query',
            'daily_treasury': 'https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/operating_cash_balance'
        }
        
        # Sample data parameters for demonstration
        self.SAMPLE_SECURITIES_COUNT = {
            'BILLS': 50,    # Treasury Bills
            'NOTES': 100,   # Treasury Notes  
            'BONDS': 30     # Treasury Bonds
        }
        
        # Interest payment frequency (payments per year)
        self.DEFAULT_INTEREST_FREQUENCY = 2  # Semi-annual for most securities
        
        # === SCENARIO ANALYSIS PARAMETERS ===
        
        # Multiple debt ceiling scenarios (in millions USD)
        self.DEBT_CEILING_SCENARIOS = {
            "current": 35_000_000.0,      # Current limit
            "raised_1t": 36_000_000.0,    # +$1T increase
            "raised_2t": 37_000_000.0,    # +$2T increase
            "suspended": float('inf')      # Suspended ceiling
        }
        
        # Cash flow shock scenarios (multipliers)
        self.CASH_FLOW_SCENARIOS = {
            "baseline": 1.0,              # Normal conditions
            "recession": 0.8,             # 20% revenue decline
            "expansion": 1.1,             # 10% revenue increase
            "crisis": 0.6                 # 40% revenue decline
        }
        
        # === ALERT THRESHOLDS ===
        
        # X-Date warning levels (days before exhaustion)
        self.X_DATE_WARNING_DAYS = {
            "red": 7,      # Critical: 7 days
            "orange": 30,  # Warning: 30 days  
            "yellow": 60   # Watch: 60 days
        }
        
        # Debt utilization warning levels (% of ceiling)
        self.DEBT_WARNING_LEVELS = {
            "red": 0.98,    # 98% of ceiling
            "orange": 0.95, # 95% of ceiling
            "yellow": 0.90  # 90% of ceiling
        }
        
        # Load from config file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str):
        """
        Load configuration from JSON file.
        
        Args:
            config_file (str): Path to JSON configuration file
        """
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update attributes from config file
            for key, value in config_data.items():
                if hasattr(self, key):
                    # Handle date conversion
                    if key == 'START_DATE' and isinstance(value, str):
                        self.START_DATE = datetime.strptime(value, '%Y-%m-%d').date()
                    else:
                        setattr(self, key, value)
                        
            print(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
            print("Using default configuration values.")
    
    def save_to_file(self, config_file: str):
        """
        Save current configuration to JSON file.
        
        Args:
            config_file (str): Path to save JSON configuration
        """
        config_data = {}
        
        # Get all public attributes
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                attr_value = getattr(self, attr_name)
                
                # Handle date serialization
                if isinstance(attr_value, date):
                    config_data[attr_name] = attr_value.strftime('%Y-%m-%d')
                elif isinstance(attr_value, (int, float, str, list, dict)):
                    config_data[attr_name] = attr_value
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            print(f"Configuration saved to {config_file}")
        except Exception as e:
            print(f"Error saving config file: {e}")
    
    def get_x_date_threshold(self) -> float:
        """
        Calculate X-Date threshold balance.
        
        Returns:
            float: Balance level that triggers X-Date warning (in millions USD)
        """
        return self.MINIMUM_CASH_BALANCE + self.CASH_BUFFER
    
    def get_debt_headroom(self) -> float:
        """
        Calculate remaining debt capacity.
        
        Returns:
            float: Available borrowing capacity (in millions USD)
        """
        return self.DEBT_CEILING_LIMIT - self.CURRENT_PUBLIC_DEBT
    
    def get_debt_utilization(self) -> float:
        """
        Calculate current debt utilization rate.
        
        Returns:
            float: Debt as percentage of ceiling (0.0 to 1.0)
        """
        return self.CURRENT_PUBLIC_DEBT / self.DEBT_CEILING_LIMIT
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate configuration parameters and return status.
        
        Returns:
            Dict[str, Any]: Validation results and warnings
        """
        validation = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check debt ceiling consistency
        if self.CURRENT_PUBLIC_DEBT >= self.DEBT_CEILING_LIMIT:
            validation["errors"].append(
                "Current debt exceeds or equals debt ceiling limit"
            )
            validation["valid"] = False
        
        # Check cash balance reasonableness
        if self.CURRENT_TGA_BALANCE < self.MINIMUM_CASH_BALANCE:
            validation["warnings"].append(
                "Current TGA balance is below minimum operational threshold"
            )
        
        # Check forecast horizon
        if self.FORECAST_HORIZON < 30:
            validation["warnings"].append(
                "Forecast horizon less than 30 days may not capture seasonal patterns"
            )
        
        # Check file existence
        critical_files = [
            self.FORECASTS_FILE,
            self.OPERATING_BALANCE_FILE
        ]
        
        for file_path in critical_files:
            if not os.path.exists(file_path):
                validation["errors"].append(f"Critical file not found: {file_path}")
                validation["valid"] = False
        
        return validation
    
    def print_summary(self):
        """Print configuration summary."""
        print("="*60)
        print("TREASURY DEBT PREDICTION CONFIGURATION")
        print("="*60)
        
        print(f"\n📅 SIMULATION PARAMETERS:")
        print(f"  Start Date: {self.START_DATE}")
        print(f"  Forecast Horizon: {self.FORECAST_HORIZON} days")
        end_date = self.START_DATE + timedelta(days=self.FORECAST_HORIZON)
        print(f"  End Date: {end_date}")
        
        print(f"\n💰 FINANCIAL PARAMETERS:")
        print(f"  Current TGA Balance: ${self.CURRENT_TGA_BALANCE:,.0f} million")
        print(f"  Current Public Debt: ${self.CURRENT_PUBLIC_DEBT:,.0f} million")
        print(f"  Debt Ceiling Limit: ${self.DEBT_CEILING_LIMIT:,.0f} million")
        print(f"  Debt Utilization: {self.get_debt_utilization():.1%}")
        print(f"  Available Headroom: ${self.get_debt_headroom():,.0f} million")
        
        print(f"\n🚨 OPERATIONAL THRESHOLDS:")
        print(f"  Minimum Cash Balance: ${self.MINIMUM_CASH_BALANCE:,.0f} million")
        print(f"  Cash Buffer: ${self.CASH_BUFFER:,.0f} million")
        print(f"  X-Date Threshold: ${self.get_x_date_threshold():,.0f} million")
        
        print(f"\n📊 INTEREST RATE ASSUMPTIONS:")
        print(f"  Short-term Rate: {self.SHORT_TERM_RATE:.1%}")
        print(f"  Medium-term Rate: {self.MEDIUM_TERM_RATE:.1%}")
        print(f"  Long-term Rate: {self.LONG_TERM_RATE:.1%}")
        
        print("="*60)


# Create default configuration instance
default_config = DebtPredictionConfig()


def load_config(config_file: str = None) -> DebtPredictionConfig:
    """
    Load configuration from file or return default.
    
    Args:
        config_file (str, optional): Path to configuration file
        
    Returns:
        DebtPredictionConfig: Configuration instance
    """
    if config_file:
        return DebtPredictionConfig(config_file)
    else:
        return default_config


if __name__ == "__main__":
    # Example usage and testing
    config = DebtPredictionConfig()
    
    # Print configuration summary
    config.print_summary()
    
    # Validate configuration
    validation = config.validate_configuration()
    print(f"\nConfiguration Valid: {validation['valid']}")
    
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  ⚠️  {warning}")
    
    if validation['errors']:
        print("Errors:")
        for error in validation['errors']:
            print(f"  ❌ {error}")
    
    # Save example configuration
    config.save_to_file("config_example.json")
