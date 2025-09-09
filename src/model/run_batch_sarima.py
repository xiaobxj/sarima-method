"""
Main script to run batch SARIMA modeling on all individual time series files.
"""

import os
import sys
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from batch_modeling import BatchSARIMAModeler

def main():
    """
    Main function to run batch SARIMA modeling.
    """
    print("="*60)
    print("BATCH SARIMA MODELING FOR TREASURY TIME SERIES")
    print("="*60)
    
    # Define directories
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_dir = os.path.join(base_dir, 'data', 'processed', 'individual_time_series')
    output_dir = os.path.join(base_dir, 'src', 'model', 'fitted_models')
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    # Verify input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    # Initialize batch modeler
    batch_modeler = BatchSARIMAModeler(
        input_dir=input_dir,
        output_dir=output_dir
    )
    
    # Configure modeling parameters
    modeling_params = {
        'max_workers': 4,  # Adjust based on your system
        'min_records': 50,  # Minimum data points required
        'max_p': 2,  # Reduced for efficiency
        'max_d': 2,
        'max_q': 2,
        'max_P': 1,  # Seasonal parameters
        'max_D': 1,
        'max_Q': 1,
        'seasonal_period': 7  # Weekly seasonality for daily data
    }
    
    print("Modeling Parameters:")
    for key, value in modeling_params.items():
        print(f"  {key}: {value}")
    print("-" * 60)
    
    try:
        # Run batch modeling
        results = batch_modeler.run_batch_modeling(**modeling_params)
        
        print("\nBATCH MODELING COMPLETED!")
        print("=" * 60)
        print(f"Processing time: {results['processing_time']:.2f} seconds")
        print(f"Total files processed: {results['total_files']}")
        print(f"Successful models: {results['successful_models']}")
        print(f"Failed models: {results['failed_models']}")
        print(f"Success rate: {results['success_rate']:.1f}%")
        
        # Show top performing models
        if results['successful_results']:
            print(f"\nTop 5 Models (by AIC):")
            print("-" * 40)
            sorted_results = sorted(results['successful_results'], key=lambda x: x['aic'])
            for i, result in enumerate(sorted_results[:5], 1):
                print(f"{i}. {result['series_name']}")
                print(f"   AIC: {result['aic']:.4f}, Parameters: {result['params']}")
        
        # Show failed models if any
        if results['failed_models'] > 0:
            print(f"\nFailed Models:")
            print("-" * 40)
            for result in results['failed_results'][:10]:  # Show first 10 failures
                print(f"- {result['series_name']}: {result.get('error', 'Unknown error')}")
            
            if results['failed_models'] > 10:
                print(f"... and {results['failed_models'] - 10} more failures")
            
            # Ask if user wants to retry failed models
            retry_response = input(f"\nWould you like to retry {results['failed_models']} failed models with simpler parameters? (y/n): ")
            if retry_response.lower().startswith('y'):
                print("Retrying failed models...")
                retry_results = batch_modeler.retry_failed_models()
                
                if retry_results.get('status') == 'completed':
                    print(f"Retry completed:")
                    print(f"  Additional successful models: {retry_results['successful_models']}")
                    print(f"  Still failed: {retry_results['failed_models']}")
        
        print(f"\nAll models saved to: {output_dir}")
        print("Batch processing completed successfully!")
        
    except Exception as e:
        print(f"Error during batch processing: {str(e)}")
        return
    
    print("=" * 60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
