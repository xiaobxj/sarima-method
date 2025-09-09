"""
Treasury Data Collection Project Configuration
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pydantic import BaseSettings
from dataclasses import dataclass


class ProjectSettings(BaseSettings):
    """Project basic settings"""
    
    # Project basic information
    PROJECT_NAME: str = "US Treasury Data Collection"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "US Treasury data collection and processing system"
    
    # Data storage paths
    DATA_DIR: str = "./data"
    RAW_DATA_DIR: str = "./data/raw"
    PROCESSED_DATA_DIR: str = "./data/processed"
    OUTPUT_DIR: str = "./output"
    LOG_DIR: str = "./logs"
    
    # Treasury API configuration
    TREASURY_API_BASE_URL: str = "https://api.fiscaldata.treasury.gov/services/api/v1"
    
    # BEA API configuration  
    BEA_API_BASE_URL: str = "https://apps.bea.gov/api/data"
    BEA_API_KEY: Optional[str] = None
    
    # Time configuration
    DEFAULT_START_DATE: str = "2020-01-01"
    DEFAULT_END_DATE: str = "2025-12-31"
    
    class Config:
        env_file = ".env"


@dataclass
class DataCollectionConfig:
    """Data collection configuration"""
    
    # Collection settings
    request_delay: float = 0.5  # Delay between API requests (seconds)
    page_size: int = 1000  # Records per API request
    timeout: int = 60  # Request timeout (seconds)
    
    # Date range settings
    default_days_back: int = 730  # Default to collect past 2 years
    
    # Data quality settings
    enable_data_validation: bool = True
    remove_duplicates: bool = True
    fill_missing_values: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration"""
    
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(levelname)s - %(message)s"
    max_log_files: int = 10
    log_file_max_size: int = 10 * 1024 * 1024  # 10MB


# Create global configuration instances
settings = ProjectSettings()
data_config = DataCollectionConfig()
logging_config = LoggingConfig()


# Validation function
def validate_project_setup():
    """Validate project setup and dependencies"""
    validation_results = {
        "directories_created": False,
        "api_access_tested": False,
        "dependencies_installed": False
    }
    
    # Check directory structure
    required_dirs = [
        settings.DATA_DIR, 
        settings.RAW_DATA_DIR, 
        settings.PROCESSED_DATA_DIR,
        settings.OUTPUT_DIR, 
        settings.LOG_DIR
    ]
    
    all_dirs_exist = all(os.path.exists(d) for d in required_dirs)
    validation_results["directories_created"] = all_dirs_exist
    
    return validation_results


if __name__ == "__main__":
    print("=== Treasury Data Collection Project Configuration ===")
    print(f"Project: {settings.PROJECT_NAME}")
    print(f"Version: {settings.VERSION}")
    print(f"Description: {settings.DESCRIPTION}")
    
    print("\n=== Data Collection Settings ===")
    print(f"Request delay: {data_config.request_delay}s")
    print(f"Page size: {data_config.page_size}")
    print(f"Default collection period: {data_config.default_days_back} days")
    
    print("\n=== Validation ===")
    results = validate_project_setup()
    for check, status in results.items():
        status_symbol = "✓" if status else "✗"
        print(f"{status_symbol} {check}")