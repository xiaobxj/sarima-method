"""
Cash Flow Data Converter for Debt Prediction System

This module converts the cash_flow_with_cat.csv historical data into the forecast format
expected by the debt prediction system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import os

class CashFlowConverter:
    """
    Converts historical cash flow data to forecast format for debt prediction system.
    """
    
    def __init__(self, cash_flow_file: str, output_dir: str = None):
        """
        Initialize the converter.
        
        Args:
            cash_flow_file (str): Path to cash_flow_with_cat.csv
            output_dir (str): Output directory for converted data
        """
        self.cash_flow_file = cash_flow_file
        self.output_dir = output_dir or os.path.dirname(cash_flow_file)
        
        # Data containers
        self.raw_data = None
        self.processed_data = None
        self.forecast_data = None
        
        print(f"Cash Flow Converter initialized")
        print(f"Input file: {cash_flow_file}")
        print(f"Output directory: {self.output_dir}")
    
    def load_cash_flow_data(self) -> pd.DataFrame:
        """
        Load the cash flow data from CSV.
        
        Returns:
            pd.DataFrame: Raw cash flow data
        """
        print("Loading cash flow data...")
        
        try:
            self.raw_data = pd.read_csv(self.cash_flow_file)
            self.raw_data['record_date'] = pd.to_datetime(self.raw_data['record_date'])
            
            print(f"✅ Loaded {len(self.raw_data)} records")
            print(f"   Date range: {self.raw_data['record_date'].min().date()} to {self.raw_data['record_date'].max().date()}")
            print(f"   Unique transaction types: {self.raw_data['transaction_catg_desc'].nunique()}")
            
            return self.raw_data
            
        except Exception as e:
            print(f"❌ Error loading cash flow data: {e}")
            return None
    
    def aggregate_daily_flows(self, start_date: date = None, end_date: date = None) -> pd.DataFrame:
        """
        Aggregate cash flows by transaction category and date.
        
        Args:
            start_date (date): Start date for aggregation
            end_date (date): End date for aggregation
            
        Returns:
            pd.DataFrame: Aggregated daily cash flows
        """
        if self.raw_data is None:
            self.load_cash_flow_data()
        
        print("Aggregating daily cash flows...")
        
        # Filter date range if specified
        data = self.raw_data.copy()
        if start_date:
            data = data[data['record_date'] >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data['record_date'] <= pd.to_datetime(end_date)]
        
        # Use signed_amount for proper directional flows
        # Positive = inflows (receipts), Negative = outflows (outlays)
        data['flow_amount'] = data['signed_amount']
        
        # Create consistent series names from transaction descriptions
        data['series_name'] = data['transaction_catg_desc'].str.replace(' ', '_').str.replace(',', '').str.replace('(', '').str.replace(')', '').str.replace('-', '_')
        
        # Aggregate by date and series
        aggregated = data.groupby(['record_date', 'series_name']).agg({
            'flow_amount': 'sum',
            'category_type': 'first',  # Keep category type for reference
            'class_name': 'first'      # Keep classification for reference
        }).reset_index()
        
        # Rename columns to match expected format
        aggregated = aggregated.rename(columns={
            'record_date': 'date',
            'flow_amount': 'amount'
        })
        
        print(f"✅ Aggregated to {len(aggregated)} daily flow records")
        print(f"   Unique series: {aggregated['series_name'].nunique()}")
        print(f"   Date range: {aggregated['date'].min().date()} to {aggregated['date'].max().date()}")
        
        self.processed_data = aggregated
        return aggregated
    
    def convert_to_forecast_format(self, forecast_start_date: date, forecast_horizon: int = 180) -> pd.DataFrame:
        """
        Convert aggregated data to forecast format expected by debt prediction system.
        
        Args:
            forecast_start_date (date): Start date for forecast period
            forecast_horizon (int): Number of days to forecast
            
        Returns:
            pd.DataFrame: Data in forecast format
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Run aggregate_daily_flows first.")
        
        print(f"Converting to forecast format...")
        print(f"Forecast start: {forecast_start_date}")
        print(f"Forecast horizon: {forecast_horizon} days")
        
        # Generate forecast dates
        forecast_dates = pd.date_range(
            start=forecast_start_date,
            periods=forecast_horizon,
            freq='D'
        )
        
        # Get unique series names
        unique_series = self.processed_data['series_name'].unique()
        
        forecast_records = []
        
        for series_name in unique_series:
            series_data = self.processed_data[self.processed_data['series_name'] == series_name].copy()
            
            # Calculate historical statistics for this series
            historical_mean = series_data['amount'].mean()
            historical_std = series_data['amount'].std()
            
            # Use recent data (last 365 days) for better forecasting if available
            recent_cutoff = pd.to_datetime(forecast_start_date) - timedelta(days=365)
            recent_data = series_data[series_data['date'] >= recent_cutoff]
            
            if len(recent_data) > 0:
                forecast_mean = recent_data['amount'].mean()
                forecast_std = recent_data['amount'].std()
            else:
                forecast_mean = historical_mean
                forecast_std = historical_std
            
            # Handle NaN values
            if pd.isna(forecast_mean):
                forecast_mean = 0.0
            if pd.isna(forecast_std) or forecast_std == 0:
                forecast_std = abs(forecast_mean) * 0.1  # 10% of mean as default std
            
            # Create forecast records for each date
            for i, forecast_date in enumerate(forecast_dates):
                day_ahead = i + 1
                
                # Use historical mean as forecast value
                forecast_value = forecast_mean
                
                # Calculate confidence intervals (±2 standard deviations)
                lower_ci = forecast_value - 2 * forecast_std
                upper_ci = forecast_value + 2 * forecast_std
                
                forecast_records.append({
                    'series_name': series_name,
                    'forecast_date': forecast_date,
                    'forecast_value': forecast_value,
                    'lower_ci': lower_ci,
                    'upper_ci': upper_ci,
                    'day_ahead': day_ahead,
                    'model_params': 'historical_mean',
                    'aic': 0.0  # Not applicable for simple historical mean
                })
        
        # Create forecast DataFrame
        self.forecast_data = pd.DataFrame(forecast_records)
        
        print(f"✅ Generated {len(self.forecast_data)} forecast records")
        print(f"   Series count: {self.forecast_data['series_name'].nunique()}")
        print(f"   Date range: {self.forecast_data['forecast_date'].min().date()} to {self.forecast_data['forecast_date'].max().date()}")
        
        return self.forecast_data
    
    def save_forecast_data(self, filename: str = None) -> str:
        """
        Save forecast data to CSV file.
        
        Args:
            filename (str): Output filename (optional)
            
        Returns:
            str: Path to saved file
        """
        if self.forecast_data is None:
            raise ValueError("No forecast data available. Run convert_to_forecast_format first.")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cash_flow_forecasts_{timestamp}.csv"
        
        output_path = os.path.join(self.output_dir, filename)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save to CSV
        self.forecast_data.to_csv(output_path, index=False)
        
        print(f"✅ Forecast data saved to: {output_path}")
        return output_path
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics of the converted data.
        
        Returns:
            Dict: Summary statistics
        """
        if self.forecast_data is None:
            return {}
        
        summary = {
            'total_records': len(self.forecast_data),
            'unique_series': self.forecast_data['series_name'].nunique(),
            'date_range': {
                'start': self.forecast_data['forecast_date'].min().date(),
                'end': self.forecast_data['forecast_date'].max().date()
            },
            'forecast_value_stats': {
                'mean': self.forecast_data['forecast_value'].mean(),
                'std': self.forecast_data['forecast_value'].std(),
                'min': self.forecast_data['forecast_value'].min(),
                'max': self.forecast_data['forecast_value'].max()
            },
            'top_series_by_volume': self.forecast_data.groupby('series_name')['forecast_value'].mean().abs().sort_values(ascending=False).head(10).to_dict()
        }
        
        return summary

def main():
    """
    Main function for testing the converter.
    """
    # Example usage
    cash_flow_file = "cash_flow_with_cat.csv"
    
    if os.path.exists(cash_flow_file):
        converter = CashFlowConverter(cash_flow_file)
        
        # Load and process data
        converter.load_cash_flow_data()
        
        # Aggregate daily flows (use recent data for better forecasting)
        start_date = date(2024, 1, 1)  # Use recent year for aggregation
        converter.aggregate_daily_flows(start_date=start_date)
        
        # Convert to forecast format
        forecast_start = date(2025, 7, 22)  # Start after last data point
        converter.convert_to_forecast_format(forecast_start, forecast_horizon=180)
        
        # Save results
        output_file = converter.save_forecast_data()
        
        # Print summary
        summary = converter.get_data_summary()
        print(f"\n📊 Data Summary:")
        print(f"   Total forecast records: {summary['total_records']}")
        print(f"   Unique series: {summary['unique_series']}")
        print(f"   Forecast period: {summary['date_range']['start']} to {summary['date_range']['end']}")
        
        print(f"\n💰 Top 5 series by volume:")
        for series, value in list(summary['top_series_by_volume'].items())[:5]:
            print(f"   {series}: ${value:,.0f}")
    
    else:
        print(f"❌ Cash flow file not found: {cash_flow_file}")

if __name__ == "__main__":
    main()

