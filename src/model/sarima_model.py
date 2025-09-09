"""
SARIMA Time Series Model for Treasury Cash Flow Data
"""

import pandas as pd
import numpy as np
import warnings
import pickle
import os
from datetime import datetime
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Time series analysis libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class SARIMAModel:
    """
    A comprehensive SARIMA model class for time series forecasting of treasury cash flows.
    """
    
    def __init__(self, series_name: str, data_path: str = None):
        """
        Initialize the SARIMA model.
        
        Args:
            series_name (str): Name of the time series (used for saving/loading)
            data_path (str): Path to the CSV file containing the time series data
        """
        self.series_name = series_name
        self.data_path = data_path
        self.data = None
        self.model = None
        self.fitted_model = None
        self.forecast_result = None
        self.model_params = None
        self.diagnostics = {}
        
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load time series data from CSV file.
        
        Args:
            data_path (str): Path to CSV file (optional if provided in __init__)
            
        Returns:
            pd.DataFrame: Loaded time series data
        """
        if data_path:
            self.data_path = data_path
            
        if not self.data_path:
            raise ValueError("Data path must be provided either in __init__ or load_data method")
            
        try:
            # Load data
            df = pd.read_csv(self.data_path)
            
            # Convert date column and set as index
            df['record_date'] = pd.to_datetime(df['record_date'])
            df = df.set_index('record_date')
            
            # Sort by date
            df = df.sort_index()
            
            # Handle missing values and outliers
            df = self._preprocess_data(df)
            
            self.data = df
            print(f"Data loaded successfully for {self.series_name}")
            print(f"Data shape: {df.shape}")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            
            return df
            
        except Exception as e:
            print(f"Error loading data for {self.series_name}: {str(e)}")
            return None
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the time series data.
        
        Args:
            df (pd.DataFrame): Raw time series data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        # Handle missing values
        if df['transaction_today_amt'].isnull().sum() > 0:
            print(f"Found {df['transaction_today_amt'].isnull().sum()} missing values, forward filling...")
            df['transaction_today_amt'] = df['transaction_today_amt'].fillna(method='ffill')
        
        # Handle extreme outliers (beyond 3 standard deviations)
        mean_val = df['transaction_today_amt'].mean()
        std_val = df['transaction_today_amt'].std()
        outlier_threshold = 3 * std_val
        
        outliers = abs(df['transaction_today_amt'] - mean_val) > outlier_threshold
        if outliers.sum() > 0:
            print(f"Found {outliers.sum()} extreme outliers, capping them...")
            df.loc[outliers & (df['transaction_today_amt'] > mean_val), 'transaction_today_amt'] = mean_val + outlier_threshold
            df.loc[outliers & (df['transaction_today_amt'] < mean_val), 'transaction_today_amt'] = mean_val - outlier_threshold
        
        return df
    
    def analyze_stationarity(self) -> Dict[str, Any]:
        """
        Analyze stationarity of the time series using ADF and KPSS tests.
        
        Returns:
            Dict: Stationarity test results
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        series = self.data['transaction_today_amt']
        
        # Augmented Dickey-Fuller test
        adf_result = adfuller(series.dropna())
        
        # KPSS test
        kpss_result = kpss(series.dropna())
        
        results = {
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'adf_critical_values': adf_result[4],
            'adf_is_stationary': adf_result[1] < 0.05,
            'kpss_statistic': kpss_result[0],
            'kpss_pvalue': kpss_result[1],
            'kpss_critical_values': kpss_result[3],
            'kpss_is_stationary': kpss_result[1] > 0.05
        }
        
        self.diagnostics['stationarity'] = results
        
        print(f"Stationarity Analysis for {self.series_name}:")
        print(f"ADF Test - Statistic: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}")
        print(f"ADF Test - Is Stationary: {results['adf_is_stationary']}")
        print(f"KPSS Test - Statistic: {kpss_result[0]:.4f}, p-value: {kpss_result[1]:.4f}")
        print(f"KPSS Test - Is Stationary: {results['kpss_is_stationary']}")
        
        return results
    
    def find_optimal_parameters(self, max_p: int = 3, max_d: int = 2, max_q: int = 3,
                               max_P: int = 2, max_D: int = 1, max_Q: int = 2, 
                               seasonal_period: int = 12) -> Tuple[int, int, int, int, int, int, int]:
        """
        Find optimal SARIMA parameters using grid search with AIC criterion.
        
        Args:
            max_p, max_d, max_q: Maximum values for non-seasonal parameters
            max_P, max_D, max_Q: Maximum values for seasonal parameters
            seasonal_period: Seasonal period (default 12 for monthly data)
            
        Returns:
            Tuple: Optimal (p, d, q, P, D, Q, s) parameters
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        series = self.data['transaction_today_amt']
        
        best_aic = float('inf')
        best_params = None
        results = []
        
        print(f"Searching for optimal SARIMA parameters for {self.series_name}...")
        
        # Grid search
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    for P in range(max_P + 1):
                        for D in range(max_D + 1):
                            for Q in range(max_Q + 1):
                                try:
                                    model = SARIMAX(series,
                                                   order=(p, d, q),
                                                   seasonal_order=(P, D, Q, seasonal_period),
                                                   enforce_stationarity=False,
                                                   enforce_invertibility=False)
                                    
                                    fitted_model = model.fit(disp=False)
                                    aic = fitted_model.aic
                                    
                                    results.append({
                                        'params': (p, d, q, P, D, Q, seasonal_period),
                                        'aic': aic
                                    })
                                    
                                    if aic < best_aic:
                                        best_aic = aic
                                        best_params = (p, d, q, P, D, Q, seasonal_period)
                                        
                                except Exception as e:
                                    continue
        
        self.model_params = best_params
        self.diagnostics['parameter_search'] = {
            'best_params': best_params,
            'best_aic': best_aic,
            'all_results': results
        }
        
        print(f"Optimal parameters found: {best_params}")
        print(f"Best AIC: {best_aic:.4f}")
        
        return best_params
    
    def fit_model(self, params: Tuple = None) -> Any:
        """
        Fit SARIMA model with given or optimal parameters.
        
        Args:
            params: SARIMA parameters (p, d, q, P, D, Q, s). If None, uses optimal parameters.
            
        Returns:
            Fitted SARIMA model
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")
        
        if params is None:
            if self.model_params is None:
                print("No parameters provided, searching for optimal parameters...")
                params = self.find_optimal_parameters()
            else:
                params = self.model_params
        
        series = self.data['transaction_today_amt']
        
        try:
            # Fit SARIMA model
            self.model = SARIMAX(series,
                               order=params[:3],
                               seasonal_order=params[3:],
                               enforce_stationarity=False,
                               enforce_invertibility=False)
            
            self.fitted_model = self.model.fit(disp=False)
            self.model_params = params
            
            print(f"SARIMA{params[:3]}x{params[3:]} model fitted successfully for {self.series_name}")
            print(f"AIC: {self.fitted_model.aic:.4f}")
            print(f"BIC: {self.fitted_model.bic:.4f}")
            
            # Store model diagnostics
            self.diagnostics['model_fit'] = {
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'llf': self.fitted_model.llf,
                'params': params
            }
            
            return self.fitted_model
            
        except Exception as e:
            print(f"Error fitting model for {self.series_name}: {str(e)}")
            return None
    
    def validate_model(self) -> Dict[str, Any]:
        """
        Validate the fitted model using residual analysis.
        
        Returns:
            Dict: Validation results
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        residuals = self.fitted_model.resid
        
        # Ljung-Box test for residual autocorrelation
        lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
        
        # Residual statistics
        residual_stats = {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'skewness': residuals.skew(),
            'kurtosis': residuals.kurtosis()
        }
        
        validation_results = {
            'ljung_box_test': lb_test,
            'residual_stats': residual_stats,
            'residuals': residuals
        }
        
        self.diagnostics['validation'] = validation_results
        
        print(f"Model validation for {self.series_name}:")
        print(f"Residual mean: {residual_stats['mean']:.6f}")
        print(f"Residual std: {residual_stats['std']:.4f}")
        print(f"Ljung-Box test p-value (lag 10): {lb_test['lb_pvalue'].iloc[9]:.4f}")
        
        return validation_results
    
    def forecast(self, steps: int = 30, confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Generate forecasts using the fitted model.
        
        Args:
            steps: Number of periods to forecast
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            Dict: Forecast results
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        # Generate forecast
        forecast = self.fitted_model.get_forecast(steps=steps)
        forecast_df = forecast.summary_frame(alpha=1-confidence_level)
        
        # Create forecast results
        forecast_results = {
            'forecast': forecast_df['mean'],
            'lower_ci': forecast_df['mean_ci_lower'],
            'upper_ci': forecast_df['mean_ci_upper'],
            'forecast_dates': pd.date_range(
                start=self.data.index[-1] + pd.Timedelta(days=1),
                periods=steps,
                freq='D'
            )
        }
        
        self.forecast_result = forecast_results
        
        print(f"Generated {steps}-step forecast for {self.series_name}")
        
        return forecast_results
    
    def save_model(self, save_dir: str) -> str:
        """
        Save the fitted model and diagnostics to disk.
        
        Args:
            save_dir: Directory to save the model
            
        Returns:
            str: Path to saved model file
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Create model data to save
        model_data = {
            'series_name': self.series_name,
            'fitted_model': self.fitted_model,
            'model_params': self.model_params,
            'diagnostics': self.diagnostics,
            'forecast_result': self.forecast_result,
            'data_path': self.data_path
        }
        
        # Save model
        model_filename = f"{self.series_name}_sarima_model.pkl"
        model_path = os.path.join(save_dir, model_filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {model_path}")
        
        return model_path
    
    def load_model(self, model_path: str):
        """
        Load a previously saved model.
        
        Args:
            model_path: Path to the saved model file
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.series_name = model_data['series_name']
        self.fitted_model = model_data['fitted_model']
        self.model_params = model_data['model_params']
        self.diagnostics = model_data['diagnostics']
        self.forecast_result = model_data.get('forecast_result')
        self.data_path = model_data.get('data_path')
        
        print(f"Model loaded successfully: {self.series_name}")
    
    def plot_diagnostics(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Plot model diagnostics including residuals, ACF, PACF, and Q-Q plot.
        
        Args:
            figsize: Figure size for the plots
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'SARIMA Model Diagnostics - {self.series_name}', fontsize=16)
        
        residuals = self.fitted_model.resid
        
        # Residuals plot
        axes[0, 0].plot(residuals)
        axes[0, 0].set_title('Residuals')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Residuals')
        
        # ACF of residuals
        plot_acf(residuals, ax=axes[0, 1], lags=20)
        axes[0, 1].set_title('ACF of Residuals')
        
        # PACF of residuals
        plot_pacf(residuals, ax=axes[1, 0], lags=20)
        axes[1, 0].set_title('PACF of Residuals')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot')
        
        plt.tight_layout()
        plt.show()
    
    def summary(self) -> str:
        """
        Generate a summary of the model and its performance.
        
        Returns:
            str: Model summary
        """
        if self.fitted_model is None:
            return f"No model fitted for {self.series_name}"
        
        summary_text = f"""
        SARIMA Model Summary - {self.series_name}
        {'='*50}
        Model Parameters: SARIMA{self.model_params[:3]}x{self.model_params[3:]}
        AIC: {self.fitted_model.aic:.4f}
        BIC: {self.fitted_model.bic:.4f}
        Log Likelihood: {self.fitted_model.llf:.4f}
        
        Data Information:
        - Data points: {len(self.data)}
        - Date range: {self.data.index.min()} to {self.data.index.max()}
        - Mean transaction amount: {self.data['transaction_today_amt'].mean():.2f}
        - Std transaction amount: {self.data['transaction_today_amt'].std():.2f}
        """
        
        return summary_text
