#!/usr/bin/env python3
"""
Treasury Data Collection Main Program - Simplified Version

Focuses only on collecting deposits_withdrawals_operating_cash data.
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Add src to path
current_dir = Path(__file__).parent
src_path = current_dir / "src"
sys.path.insert(0, str(src_path))

# Import the enhanced data collector
from data.data_collector import EnhancedTreasuryCollector

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"data_collection_{datetime.now().strftime('%Y%m%d')}.log"),
            logging.StreamHandler()
        ]
    )

def main():
    """Main function"""
    print("=" * 70)
    print("Treasury Data Collection System")
    print("=" * 70)
    
    # Setup logging
    setup_logging()
    
    # Initialize data collector
    data_dir = Path("data/raw")
    collector = EnhancedTreasuryCollector(data_dir=str(data_dir))
    
    # Set date range (from 2016 to present for complete historical data)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = "2016-01-01"  # Complete historical coverage
    
    print(f"Data collection date range: {start_date} to {end_date}")
    print(f"Data storage directory: {data_dir}")
    print()
    
    try:
        print("Starting Treasury data collection...")
        
        # Use enhanced collection
        all_data = collector.collect_all_enhanced_data(start_date, end_date)
        
        print("\nData collection completed!")
        print("=" * 70)
        
        # Display collection statistics
        raw_data = all_data['raw_data']
        total_records = 0
        
        print("Dataset Collection Results:")
        for name, df in raw_data.items():
            if hasattr(df, '__len__') and len(df) > 0:
                record_count = len(df)
                total_records += record_count
                print(f"   {name}: {record_count:,} records")
            else:
                print(f"   {name}: No data")
        
        # TGA balance records
        tga_balance = all_data['tga_balance']
        if hasattr(tga_balance, '__len__') and len(tga_balance) > 0:
            print(f"\nTGA Balance Analysis:")
            print(f"   Records: {len(tga_balance):,} entries")
            if not tga_balance.empty and 'tga_balance' in tga_balance.columns:
                latest_balance = tga_balance.iloc[-1]['tga_balance']
                if isinstance(latest_balance, (int, float)):
                    print(f"   Latest balance: ${latest_balance:,.0f} million USD")
        
        # Categorized cash flows
        categorized_flows = all_data['categorized_flows']
        if categorized_flows:
            print(f"\nCash Flow Categorization:")
            for flow_type, df in categorized_flows.items():
                if hasattr(df, '__len__') and len(df) > 0:
                    unique_categories = df['transaction_group'].nunique() if 'transaction_group' in df.columns else 0
                    print(f"   {flow_type.capitalize()}: {len(df):,} transactions, {unique_categories} categories")
        
        # Summary statistics
        summary = all_data['summary']
        print(f"\nCollection Summary:")
        print(f"   Total datasets: {len(raw_data)}")
        print(f"   Total records: {total_records:,}")
        print(f"   Category mappings: {summary.get('category_mapping_size', 0)}")
        print(f"   Collection timestamp: {summary.get('collection_timestamp', 'N/A')}")
        
        print(f"\nData location: {data_dir}")
        print("Treasury data collection completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError during data collection: {e}")
        logging.error(f"Data collection failed: {e}", exc_info=True)
        print("Please check your internet connection and API access.")
        sys.exit(1)

if __name__ == "__main__":
    main()
