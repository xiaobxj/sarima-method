"""
Batch SARIMA Modeling for Multiple Time Series
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

from .sarima_model import SARIMAModel

class BatchSARIMAModeler:
    """
    A class to batch process multiple time series files and create SARIMA models for each.
    """
    
    def __init__(self, input_dir: str, output_dir: str, log_dir: str = None):
        """
        Initialize the batch modeler.
        
        Args:
            input_dir (str): Directory containing individual time series CSV files
            output_dir (str): Directory to save the fitted models
            log_dir (str): Directory to save logs (optional)
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.log_dir = log_dir or os.path.join(output_dir, 'logs')
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        self.results = {}
        self.failed_models = []
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_filename = os.path.join(self.log_dir, f'batch_modeling_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Batch SARIMA modeling initialized")
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Log directory: {self.log_dir}")
    
    def get_time_series_files(self) -> List[str]:
        """
        Get list of time series CSV files from input directory.
        
        Returns:
            List[str]: List of CSV file paths
        """
        csv_files = []
        
        for filename in os.listdir(self.input_dir):
            if filename.endswith('.csv') and filename != 'time_series_summary.txt':
                csv_files.append(os.path.join(self.input_dir, filename))
        
        self.logger.info(f"Found {len(csv_files)} time series files")
        return csv_files
    
    def filter_files_by_size(self, files: List[str], min_records: int = 50) -> List[str]:
        """
        Filter files based on minimum number of records.
        
        Args:
            files (List[str]): List of CSV file paths
            min_records (int): Minimum number of records required
            
        Returns:
            List[str]: Filtered list of files
        """
        valid_files = []
        
        for file_path in files:
            try:
                df = pd.read_csv(file_path)
                if len(df) >= min_records:
                    valid_files.append(file_path)
                else:
                    self.logger.warning(f"Skipping {os.path.basename(file_path)}: Only {len(df)} records (minimum {min_records} required)")
            except Exception as e:
                self.logger.error(f"Error reading {file_path}: {str(e)}")
        
        self.logger.info(f"Filtered to {len(valid_files)} files with sufficient data")
        return valid_files
    
    def process_single_series(self, file_path: str, 
                            max_p: int = 2, max_d: int = 2, max_q: int = 2,
                            max_P: int = 1, max_D: int = 1, max_Q: int = 1,
                            seasonal_period: int = 7) -> Dict[str, Any]:
        """
        Process a single time series file and create SARIMA model.
        
        Args:
            file_path (str): Path to the time series CSV file
            max_p, max_d, max_q: Maximum values for non-seasonal parameters
            max_P, max_D, max_Q: Maximum values for seasonal parameters
            seasonal_period (int): Seasonal period (7 for weekly, 30 for monthly)
            
        Returns:
            Dict[str, Any]: Processing results
        """
        series_name = os.path.splitext(os.path.basename(file_path))[0]
        
        try:
            self.logger.info(f"Processing {series_name}...")
            
            # Initialize model
            model = SARIMAModel(series_name=series_name, data_path=file_path)
            
            # Load data
            data = model.load_data()
            if data is None or len(data) < 50:
                return {
                    'series_name': series_name,
                    'status': 'failed',
                    'error': f'Insufficient data: {len(data) if data is not None else 0} records'
                }
            
            # Analyze stationarity
            stationarity = model.analyze_stationarity()
            
            # Find optimal parameters with reduced search space for efficiency
            params = model.find_optimal_parameters(
                max_p=max_p, max_d=max_d, max_q=max_q,
                max_P=max_P, max_D=max_D, max_Q=max_Q,
                seasonal_period=seasonal_period
            )
            
            if params is None:
                return {
                    'series_name': series_name,
                    'status': 'failed',
                    'error': 'Could not find optimal parameters'
                }
            
            # Fit model
            fitted_model = model.fit_model(params)
            if fitted_model is None:
                return {
                    'series_name': series_name,
                    'status': 'failed',
                    'error': 'Model fitting failed'
                }
            
            # Validate model
            validation = model.validate_model()
            
            # Generate forecast
            forecast = model.forecast(steps=30)
            
            # Save model
            model_path = model.save_model(self.output_dir)
            
            result = {
                'series_name': series_name,
                'status': 'success',
                'model_path': model_path,
                'params': params,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'data_points': len(data),
                'date_range': f"{data.index.min()} to {data.index.max()}",
                'stationarity': stationarity,
                'validation': validation,
                'forecast': forecast
            }
            
            self.logger.info(f"Successfully processed {series_name} - AIC: {fitted_model.aic:.4f}")
            return result
            
        except Exception as e:
            error_msg = f"Error processing {series_name}: {str(e)}"
            self.logger.error(error_msg)
            return {
                'series_name': series_name,
                'status': 'failed',
                'error': error_msg
            }
    
    def run_batch_modeling(self, max_workers: int = None, 
                          min_records: int = 50,
                          max_p: int = 2, max_d: int = 2, max_q: int = 2,
                          max_P: int = 1, max_D: int = 1, max_Q: int = 1,
                          seasonal_period: int = 7) -> Dict[str, Any]:
        """
        Run batch SARIMA modeling on all time series files.
        
        Args:
            max_workers (int): Maximum number of parallel workers
            min_records (int): Minimum number of records required
            max_p, max_d, max_q: Maximum values for non-seasonal parameters
            max_P, max_D, max_Q: Maximum values for seasonal parameters
            seasonal_period (int): Seasonal period
            
        Returns:
            Dict[str, Any]: Batch processing results
        """
        start_time = datetime.now()
        
        # Get and filter files
        all_files = self.get_time_series_files()
        valid_files = self.filter_files_by_size(all_files, min_records)
        
        if not valid_files:
            self.logger.error("No valid files found for processing")
            return {'status': 'failed', 'error': 'No valid files found'}
        
        # Determine number of workers
        if max_workers is None:
            max_workers = min(mp.cpu_count() - 1, len(valid_files))
        
        self.logger.info(f"Starting batch processing with {max_workers} workers")
        
        successful_models = []
        failed_models = []
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(
                    self.process_single_series, 
                    file_path, max_p, max_d, max_q, max_P, max_D, max_Q, seasonal_period
                ): file_path 
                for file_path in valid_files
            }
            
            # Collect results
            for future in as_completed(future_to_file):
                result = future.result()
                
                if result['status'] == 'success':
                    successful_models.append(result)
                else:
                    failed_models.append(result)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Compile results
        batch_results = {
            'status': 'completed',
            'processing_time': processing_time,
            'total_files': len(valid_files),
            'successful_models': len(successful_models),
            'failed_models': len(failed_models),
            'success_rate': len(successful_models) / len(valid_files) * 100,
            'successful_results': successful_models,
            'failed_results': failed_models
        }
        
        self.results = batch_results
        self.failed_models = failed_models
        
        # Log summary
        self.logger.info(f"Batch processing completed in {processing_time:.2f} seconds")
        self.logger.info(f"Successfully processed: {len(successful_models)}/{len(valid_files)} files")
        self.logger.info(f"Success rate: {batch_results['success_rate']:.1f}%")
        
        # Save results summary
        self.save_results_summary(batch_results)
        
        return batch_results
    
    def save_results_summary(self, results: Dict[str, Any]):
        """
        Save batch processing results summary to files.
        
        Args:
            results (Dict[str, Any]): Batch processing results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        import json
        
        # Convert non-serializable objects to strings for JSON
        json_results = results.copy()
        for result in json_results.get('successful_results', []):
            if 'stationarity' in result:
                # Convert numpy types to Python types
                for key, value in result['stationarity'].items():
                    if hasattr(value, 'item'):
                        result['stationarity'][key] = value.item()
            # Remove complex objects that can't be serialized
            result.pop('validation', None)
            result.pop('forecast', None)
        
        json_path = os.path.join(self.output_dir, f'batch_results_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Save summary CSV
        summary_data = []
        for result in results.get('successful_results', []):
            summary_data.append({
                'series_name': result['series_name'],
                'status': result['status'],
                'params': str(result['params']),
                'aic': result['aic'],
                'bic': result['bic'],
                'data_points': result['data_points'],
                'date_range': result['date_range']
            })
        
        for result in results.get('failed_results', []):
            summary_data.append({
                'series_name': result['series_name'],
                'status': result['status'],
                'params': None,
                'aic': None,
                'bic': None,
                'data_points': None,
                'date_range': None,
                'error': result.get('error', 'Unknown error')
            })
        
        summary_df = pd.DataFrame(summary_data)
        csv_path = os.path.join(self.output_dir, f'batch_summary_{timestamp}.csv')
        summary_df.to_csv(csv_path, index=False)
        
        self.logger.info(f"Results saved to: {json_path}")
        self.logger.info(f"Summary saved to: {csv_path}")
    
    def retry_failed_models(self, **kwargs) -> Dict[str, Any]:
        """
        Retry processing failed models with potentially different parameters.
        
        Returns:
            Dict[str, Any]: Retry results
        """
        if not self.failed_models:
            self.logger.info("No failed models to retry")
            return {'status': 'no_failures', 'message': 'No failed models to retry'}
        
        self.logger.info(f"Retrying {len(self.failed_models)} failed models...")
        
        # Extract file paths from failed models
        failed_files = []
        for failed_result in self.failed_models:
            series_name = failed_result['series_name']
            file_path = os.path.join(self.input_dir, f"{series_name}.csv")
            if os.path.exists(file_path):
                failed_files.append(file_path)
        
        # Process with more relaxed parameters
        retry_kwargs = {
            'max_p': 1, 'max_d': 1, 'max_q': 1,
            'max_P': 1, 'max_D': 1, 'max_Q': 1,
            'seasonal_period': 7,
            'min_records': 30
        }
        retry_kwargs.update(kwargs)
        
        # Clear previous failed models
        self.failed_models = []
        
        # Run batch modeling on failed files only
        original_input_dir = self.input_dir
        
        # Temporarily create a directory with only failed files
        retry_dir = os.path.join(self.output_dir, 'retry_temp')
        os.makedirs(retry_dir, exist_ok=True)
        
        for file_path in failed_files:
            import shutil
            shutil.copy2(file_path, retry_dir)
        
        self.input_dir = retry_dir
        
        try:
            retry_results = self.run_batch_modeling(**retry_kwargs)
            return retry_results
        finally:
            # Restore original input directory
            self.input_dir = original_input_dir
            # Clean up temporary directory
            import shutil
            shutil.rmtree(retry_dir)
