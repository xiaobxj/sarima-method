#!/usr/bin/env python3
"""
Run Enhanced Treasury Data Collector
Enhanced version with detailed DTS data collection and categorization
"""

import sys
import os
from pathlib import Path

# Add src path for imports
src_path = Path(__file__).parent.parent  # Go up to src directory
sys.path.insert(0, str(src_path))

from data.data_collector import EnhancedTreasuryCollector
import logging
from datetime import datetime, timedelta

def main():
    """Main function to run enhanced Treasury data collection"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize collector
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    collector = EnhancedTreasuryCollector(data_dir=str(data_dir))
    
    # Set date range (from 2016 to present for complete historical data)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = "2016-01-01"  # Start from 2016 for complete historical coverage
    
    print(f'Starting enhanced Treasury data collection...')
    print(f'Date range: {start_date} to {end_date}')
    print(f'Data directory: {data_dir}')
    print(f'Enhanced features: TGA balance fix, category deduplication, subtotal filtering')
    print('=' * 70)
    
    try:
        # Collect all enhanced data
        print('Connecting to Treasury API...')
        all_data = collector.collect_all_enhanced_data(start_date, end_date)
        
        print('\nData collection completed!')
        print('=' * 70)
        
        # Output collection statistics
        raw_data = all_data['raw_data']
        total_records = 0
        
        print('Dataset Collection Results:')
        for name, df in raw_data.items():
            if hasattr(df, '__len__') and len(df) > 0:
                record_count = len(df)
                total_records += record_count
                print(f'   {name}: {record_count:,} rows')
            else:
                print(f'   {name}: No data available')
        
        # TGA balance records
        tga_balance = all_data['tga_balance']
        if hasattr(tga_balance, '__len__') and len(tga_balance) > 0:
            print(f'\nTGA Balance Analysis:')
            print(f'   Records: {len(tga_balance):,} entries')
            if not tga_balance.empty:
                latest_balance = tga_balance.iloc[-1]['tga_balance'] if 'tga_balance' in tga_balance.columns else 'N/A'
                print(f'   Latest balance: ${latest_balance:,.0f}' if isinstance(latest_balance, (int, float)) else f'   Latest balance: {latest_balance}')
        
        # Categorized cash flows
        categorized_flows = all_data['categorized_flows']
        if categorized_flows:
            print(f'\nCash Flow Categorization:')
            for flow_type, df in categorized_flows.items():
                if hasattr(df, '__len__') and len(df) > 0:
                    unique_categories = df['transaction_group'].nunique() if 'transaction_group' in df.columns else 0
                    print(f'   {flow_type.capitalize()}: {len(df):,} transactions, {unique_categories} categories')
        
        # Summary statistics
        summary = all_data['summary']
        print(f'\nCollection Summary:')
        print(f'   Total datasets collected: {len(raw_data)}')
        print(f'   Total records: {total_records:,}')
        print(f'   Category mappings: {summary.get("category_mapping_size", 0)}')
        print(f'   Collection timestamp: {summary.get("collection_timestamp", "N/A")}')
        
        # Check subtotal filtering effectiveness
        subtotal_check = summary.get('subtotal_check', {})
        if subtotal_check.get('subtotal_records_count', 0) > 0:
            print(f'\nData Quality Checks:')
            print(f'   Subtotal records identified: {subtotal_check["subtotal_records_count"]:,}')
            print(f'   Categories filtered: {subtotal_check.get("subtotal_categories_found", [])}')
            print(f'   Subtotals excluded from analysis to prevent double-counting')
        else:
            print(f'\nData Quality: No subtotal/transfer records found')
        
        print(f'\nData Location: {data_dir}')
        print('Enhanced Treasury data collection completed successfully!')
        print('=' * 70)
        
    except Exception as e:
        print(f'\nError during data collection: {e}')
        logging.error(f"Data collection failed: {e}", exc_info=True)
        print('Please check your internet connection and API access.')
        sys.exit(1)

if __name__ == "__main__":
    main() 