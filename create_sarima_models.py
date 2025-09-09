"""
Script to create SARIMA models for all individual time series files.
"""

import os
import pandas as pd
import numpy as np
import warnings
import pickle
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# Suppress warnings
warnings.filterwarnings('ignore')

def create_sarima_models():
    """Create SARIMA models for all time series files."""
    
    print("="*60)
    print("CREATING SARIMA MODELS FOR TREASURY TIME SERIES")
    print("="*60)
    
    # Define directories
    input_dir = 'data/processed/individual_time_series'
    output_dir = 'src/model/fitted_models'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CSV files (excluding summary)
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv') and f != 'time_series_summary.txt']
    
    print(f"Found {len(csv_files)} time series files")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    successful_models = 0
    failed_models = 0
    results = []
    
    for i, filename in enumerate(csv_files, 1):
        series_name = filename.replace('.csv', '')
        file_path = os.path.join(input_dir, filename)
        
        print(f"[{i}/{len(csv_files)}] Processing: {series_name}")
        
        try:
            # Load data
            df = pd.read_csv(file_path)
            
            # Skip files with insufficient data
            if len(df) < 50:
                print(f"  Skipped: Only {len(df)} records (minimum 50 required)")
                failed_models += 1
                continue
            
            # Prepare data
            df['record_date'] = pd.to_datetime(df['record_date'])
            df = df.set_index('record_date').sort_index()
            
            # Handle missing values
            df['transaction_today_amt'] = df['transaction_today_amt'].fillna(method='ffill')
            
            series = df['transaction_today_amt']
            
            # Simple parameter selection (for efficiency)
            best_aic = float('inf')
            best_params = None
            
            # Limited grid search for efficiency
            for p in [0, 1, 2]:
                for d in [0, 1]:
                    for q in [0, 1, 2]:
                        for P in [0, 1]:
                            for D in [0, 1]:
                                for Q in [0, 1]:
                                    try:
                                        model = SARIMAX(series,
                                                       order=(p, d, q),
                                                       seasonal_order=(P, D, Q, 7),  # Weekly seasonality
                                                       enforce_stationarity=False,
                                                       enforce_invertibility=False)
                                        
                                        fitted = model.fit(disp=False, maxiter=100)
                                        
                                        if fitted.aic < best_aic:
                                            best_aic = fitted.aic
                                            best_params = (p, d, q, P, D, Q, 7)
                                            
                                    except:
                                        continue
            
            if best_params is None:
                print(f"  Failed: Could not find suitable parameters")
                failed_models += 1
                continue
            
            # Fit final model with best parameters
            final_model = SARIMAX(series,
                                order=best_params[:3],
                                seasonal_order=best_params[3:],
                                enforce_stationarity=False,
                                enforce_invertibility=False)
            
            fitted_model = final_model.fit(disp=False, maxiter=100)
            
            # Create model data to save
            model_data = {
                'series_name': series_name,
                'fitted_model': fitted_model,
                'model_params': best_params,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'data_shape': df.shape,
                'date_range': f"{df.index.min()} to {df.index.max()}",
                'mean_value': series.mean(),
                'std_value': series.std(),
                'created_at': datetime.now().isoformat()
            }
            
            # Save model
            model_filename = f"{series_name}_sarima_model.pkl"
            model_path = os.path.join(output_dir, model_filename)
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"  Success: SARIMA{best_params[:3]}x{best_params[3:]} - AIC: {fitted_model.aic:.2f}")
            
            results.append({
                'series_name': series_name,
                'status': 'success',
                'params': best_params,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'data_points': len(df)
            })
            
            successful_models += 1
            
        except Exception as e:
            print(f"  Failed: {str(e)}")
            results.append({
                'series_name': series_name,
                'status': 'failed',
                'error': str(e)
            })
            failed_models += 1
    
    # Save results summary
    results_df = pd.DataFrame(results)
    summary_path = os.path.join(output_dir, f'modeling_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    results_df.to_csv(summary_path, index=False)
    
    print("="*60)
    print("BATCH MODELING COMPLETED!")
    print(f"Total files: {len(csv_files)}")
    print(f"Successful models: {successful_models}")
    print(f"Failed models: {failed_models}")
    print(f"Success rate: {successful_models/len(csv_files)*100:.1f}%")
    print(f"Models saved to: {output_dir}")
    print(f"Summary saved to: {summary_path}")
    print("="*60)
    
    return results

if __name__ == "__main__":
    create_sarima_models()
