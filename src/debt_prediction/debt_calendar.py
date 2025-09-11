"""
Debt Events Calendar Module

This module creates and manages a comprehensive calendar of all known debt service payments,
including both principal redemptions and interest payments for U.S. Treasury securities.

The calendar provides a quick lookup mechanism for the simulation engine to determine
the exact amount of scheduled debt service payments for any given date.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional, Union
import requests
import json
import warnings
from dateutil.relativedelta import relativedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TreasuryDataAcquisition:
    """
    Handles acquisition of U.S. Treasury marketable securities data.
    
    This class provides methods to fetch current outstanding Treasury securities
    data from official U.S. Treasury APIs and data sources.
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the Treasury data acquisition system.
        
        Args:
            config_manager: Optional configuration manager for API endpoints
        """
        # Use configuration endpoints if provided, otherwise use defaults
        if config_manager and hasattr(config_manager, 'TREASURY_API_ENDPOINTS'):
            self.base_urls = config_manager.TREASURY_API_ENDPOINTS.copy()
        else:
            self.base_urls = {
                'debt_to_penny': 'https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/debt_to_penny',
                'debt_subject_limit': 'https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/debt_subject_to_limit',
                'debt_auctions': 'https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query',
                'daily_treasury': 'https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/dts/operating_cash_balance',
                'treasury_api_base': 'https://api.fiscaldata.treasury.gov/services/api/fiscal_service'
            }
        
        # Store configuration reference
        self.config = config_manager
        
        # Common Treasury security types
        self.security_types = {
            'BILL': 'Treasury Bills',
            'NOTE': 'Treasury Notes', 
            'BOND': 'Treasury Bonds',
            'TIPS': 'Treasury Inflation-Protected Securities',
            'FRN': 'Floating Rate Notes'
        }
        
        logger.info("Treasury Data Acquisition system initialized")
    
    def fetch_outstanding_securities(self, as_of_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch outstanding Treasury securities data.
        
        Args:
            as_of_date (str, optional): Date in YYYY-MM-DD format. If None, uses latest available.
            
        Returns:
            pd.DataFrame: DataFrame containing outstanding securities with columns:
                - cusip: Unique security identifier
                - issue_date: Security issue date
                - maturity_date: Security maturity date
                - principal_amount: Outstanding principal amount (in millions)
                - interest_rate: Annual coupon rate (as decimal)
                - security_type: Type of security (BILL, NOTE, BOND, etc.)
                - interest_payment_frequency: Number of payments per year
        """
        logger.info("Fetching outstanding Treasury securities data...")
        
        try:
            # Try to fetch from Treasury API first
            securities_data = self._fetch_from_treasury_api(as_of_date)
            
            if securities_data is None or securities_data.empty:
                # Fallback to sample/mock data for demonstration
                logger.warning("Could not fetch live data, using sample data for demonstration")
                securities_data = self._create_sample_securities_data()
            
            # Validate and clean the data
            securities_data = self._validate_and_clean_data(securities_data)
            
            logger.info(f"Successfully loaded {len(securities_data)} outstanding securities")
            return securities_data
            
        except Exception as e:
            logger.error(f"Error fetching securities data: {str(e)}")
            logger.info("Falling back to sample data")
            return self._create_sample_securities_data()
    
    def _fetch_from_treasury_api(self, as_of_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Enhanced method to fetch CUSIP-level Treasury securities data from official APIs.
        
        This method implements comprehensive data acquisition with pagination support,
        multiple data sources, and intelligent data combination strategies.
        
        Args:
            as_of_date (Optional[str]): Specific date for data retrieval (YYYY-MM-DD)
            
        Returns:
            Optional[pd.DataFrame]: Standardized securities data with CUSIP details
        """
        try:
            logger.info("🔍 Enhanced Treasury API data acquisition starting...")
            logger.info("Fetching CUSIP-level securities data from multiple sources...")
            
            all_securities_data = []
            
            # 1. Comprehensive Auction Data (Primary source for CUSIP details)
            logger.info("📊 Phase 1: Fetching comprehensive auction data...")
            auction_data = self._fetch_comprehensive_auction_data(as_of_date)
            if auction_data is not None and len(auction_data) > 0:
                all_securities_data.append(auction_data)
                logger.info(f"✅ Auction data: {len(auction_data)} securities with CUSIP details")
            
            # 2. Outstanding Debt Summary (for validation and additional context)
            logger.info("📊 Phase 2: Fetching outstanding debt summary...")
            outstanding_data = self._fetch_outstanding_debt_data(as_of_date)
            if outstanding_data is not None and len(outstanding_data) > 0:
                all_securities_data.append(outstanding_data)
                logger.info(f"✅ Outstanding debt: {len(outstanding_data)} aggregate records")
            
            # 3. Debt to the Penny (for total debt validation)
            logger.info("📊 Phase 3: Fetching debt to the penny data...")
            penny_data = self._fetch_debt_to_penny_data(as_of_date)
            if penny_data is not None and len(penny_data) > 0:
                all_securities_data.append(penny_data)
                logger.info(f"✅ Debt to penny: {len(penny_data)} total debt records")
            
            # 4. Combine and process all data sources
            if all_securities_data:
                logger.info("🔄 Combining data from all sources...")
                combined_df = pd.concat(all_securities_data, ignore_index=True)
                logger.info(f"📋 Combined dataset: {len(combined_df)} total records")
                
                # Enhanced deduplication strategy
                combined_df = self._deduplicate_securities_data(combined_df)
                logger.info(f"🔧 After deduplication: {len(combined_df)} unique securities")
                
                # Standardize the combined data with enhanced mapping
                standardized_df = self._standardize_api_data(combined_df)
                
                if len(standardized_df) > 0:
                    logger.info(f"✅ Successfully processed {len(standardized_df)} securities from Treasury APIs")
                    logger.info(f"📊 Data sources combined: {len(all_securities_data)} endpoints")
                    
                    # Log data quality metrics
                    cusip_count = standardized_df['cusip'].notna().sum()
                    maturity_count = standardized_df['maturity_date'].notna().sum()
                    amount_count = standardized_df['principal_amount'].notna().sum()
                    
                    logger.info(f"📈 Data quality: CUSIP={cusip_count}, Maturity={maturity_count}, Amount={amount_count}")
                    
                    return standardized_df
                else:
                    logger.warning("⚠️ No valid securities after standardization")
                    return None
            else:
                logger.warning("⚠️ No data retrieved from any Treasury API endpoint")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error in enhanced Treasury API fetch: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def _fetch_comprehensive_auction_data(self, as_of_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Enhanced auction data fetching with pagination support and comprehensive data collection.
        
        This method implements robust pagination to fetch all available auction records
        and focuses on getting CUSIP-level securities details.
        
        Args:
            as_of_date (Optional[str]): Specific date for filtering (YYYY-MM-DD)
            
        Returns:
            Optional[pd.DataFrame]: Comprehensive auction data with CUSIP details
        """
        try:
            endpoint = self.base_urls['debt_auctions']
            all_auction_records = []
            page = 1
            max_pages = 20  # Safety limit to prevent infinite loops
            
            logger.info(f"🔄 Starting paginated auction data fetch from: {endpoint}")
            
            while page <= max_pages:
                params = {
                    'format': 'json',
                    'page[size]': '1000',  # Maximum page size for efficiency
                    'page[number]': str(page),
                    'sort': '-auction_date'
                }
                
                # Date filtering for recent auctions (last 3 years for comprehensive coverage)
                if not as_of_date:
                    from datetime import datetime, timedelta
                    three_years_ago = (datetime.now() - timedelta(days=1095)).strftime('%Y-%m-%d')
                    params['filter'] = f'auction_date:gte:{three_years_ago}'
                else:
                    params['filter'] = f'auction_date:lte:{as_of_date}'
                
                logger.info(f"📄 Fetching auction data page {page}...")
                
                response = requests.get(endpoint, params=params, timeout=60)
                response.raise_for_status()
                
                data = response.json()
                
                if 'data' in data and data['data']:
                    page_records = data['data']
                    all_auction_records.extend(page_records)
                    
                    logger.info(f"📊 Page {page}: Retrieved {len(page_records)} auction records")
                    
                    # Check if there are more pages
                    meta = data.get('meta', {})
                    total_pages = meta.get('total-pages', 1)
                    current_page = meta.get('page-number', page)
                    
                    if current_page >= total_pages:
                        logger.info(f"✅ Reached final page {current_page} of {total_pages}")
                        break
                    
                    page += 1
                else:
                    logger.info(f"📄 Page {page}: No more data available")
                    break
            
            if all_auction_records:
                df = pd.DataFrame(all_auction_records)
                logger.info(f"✅ Total auction records retrieved: {len(df)}")
                
                # Filter for active (non-matured) securities only
                current_date = datetime.now().date()
                
                # Parse maturity dates more robustly
                for date_col in ['maturity_date', 'mat_date', 'maturity_dt']:
                    if date_col in df.columns:
                        df['maturity_date_parsed'] = pd.to_datetime(df[date_col], errors='coerce')
                        break
                
                if 'maturity_date_parsed' in df.columns:
                    active_securities = df[df['maturity_date_parsed'] > pd.Timestamp(current_date)]
                    logger.info(f"🎯 Active securities (non-matured): {len(active_securities)}")
                    
                    # Focus on securities with valid CUSIP information
                    cusip_securities = active_securities[active_securities['cusip'].notna() & 
                                                       (active_securities['cusip'] != 'null')]
                    
                    if len(cusip_securities) > 0:
                        logger.info(f"🏷️ Securities with valid CUSIP: {len(cusip_securities)}")
                        return cusip_securities
                    else:
                        logger.warning("⚠️ No securities with valid CUSIP found")
                        return active_securities  # Return all active securities even without CUSIP
                
                return df
            
            logger.warning("⚠️ No auction data retrieved from any page")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error fetching comprehensive auction data: {str(e)}")
            return None
    
    def _deduplicate_securities_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced deduplication strategy for securities data from multiple sources.
        
        This method implements intelligent deduplication that preserves the most
        complete and recent information for each unique security.
        
        Args:
            df (pd.DataFrame): Combined securities data from multiple sources
            
        Returns:
            pd.DataFrame: Deduplicated securities data
        """
        try:
            logger.info(f"🔧 Starting enhanced deduplication of {len(df)} records...")
            
            # Step 1: Basic deduplication by CUSIP (if available)
            if 'cusip' in df.columns:
                # Remove records with null or empty CUSIP first
                valid_cusip_mask = (df['cusip'].notna()) & (df['cusip'] != 'null') & (df['cusip'] != '')
                valid_cusip_df = df[valid_cusip_mask].copy()
                no_cusip_df = df[~valid_cusip_mask].copy()
                
                if len(valid_cusip_df) > 0:
                    # For records with CUSIP, keep the most recent/complete one
                    cusip_deduped = valid_cusip_df.drop_duplicates(subset=['cusip'], keep='first')
                    logger.info(f"📋 CUSIP deduplication: {len(valid_cusip_df)} -> {len(cusip_deduped)} records")
                    
                    # Combine with records that don't have CUSIP
                    final_df = pd.concat([cusip_deduped, no_cusip_df], ignore_index=True)
                else:
                    final_df = no_cusip_df
                    logger.info("📋 No valid CUSIP records found, keeping all records")
            else:
                logger.info("📋 No CUSIP column found, skipping CUSIP-based deduplication")
                final_df = df.copy()
            
            # Step 2: Additional deduplication by security characteristics
            # For records without CUSIP, try to deduplicate by security type, maturity, and amount
            dedup_columns = []
            for col in ['security_type', 'maturity_date', 'principal_amount', 'offering_amt']:
                if col in final_df.columns:
                    dedup_columns.append(col)
            
            if dedup_columns:
                initial_count = len(final_df)
                final_df = final_df.drop_duplicates(subset=dedup_columns, keep='first')
                logger.info(f"🔧 Characteristic deduplication: {initial_count} -> {len(final_df)} records")
            
            # Step 3: Data quality scoring and selection
            # Prioritize records with more complete information
            final_df = self._score_and_select_best_records(final_df)
            
            logger.info(f"✅ Deduplication complete: Final dataset has {len(final_df)} unique securities")
            return final_df
            
        except Exception as e:
            logger.error(f"❌ Error in deduplication: {str(e)}")
            return df  # Return original data if deduplication fails
    
    def _score_and_select_best_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score records based on data completeness and select the best ones.
        
        Args:
            df (pd.DataFrame): Securities data to score
            
        Returns:
            pd.DataFrame: Records with quality scores, best records prioritized
        """
        try:
            if len(df) == 0:
                return df
            
            # Calculate completeness score for each record
            important_fields = ['cusip', 'maturity_date', 'principal_amount', 'interest_rate', 
                              'offering_amt', 'security_type', 'issue_date']
            
            df = df.copy()
            df['completeness_score'] = 0
            
            for field in important_fields:
                if field in df.columns:
                    # Add points for non-null, non-empty values
                    valid_mask = (df[field].notna()) & (df[field] != 'null') & (df[field] != '')
                    df.loc[valid_mask, 'completeness_score'] += 1
            
            # Sort by completeness score (descending) to prioritize complete records
            df = df.sort_values('completeness_score', ascending=False)
            
            logger.info(f"📊 Record quality scores: max={df['completeness_score'].max()}, "
                       f"avg={df['completeness_score'].mean():.1f}")
            
            return df
            
        except Exception as e:
            logger.warning(f"⚠️ Error in record scoring: {str(e)}")
            return df
    
    def _fetch_outstanding_debt_data(self, as_of_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Fetch data from Debt to the Penny endpoint (contains outstanding debt info)."""
        try:
            endpoint = self.base_urls['debt_to_penny']
            params = {
                'format': 'json',
                'page[size]': '1000',
                'sort': '-record_date'
            }
            
            if as_of_date:
                params['filter'] = f'record_date:eq:{as_of_date}'
            else:
                # Get recent data (last 30 days)
                from datetime import datetime, timedelta
                recent_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                params['filter'] = f'record_date:gte:{recent_date}'
            
            logger.info(f"Fetching from Debt to Penny API: {endpoint}")
            response = requests.get(endpoint, params=params, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            if 'data' in data and data['data']:
                logger.info(f"Successfully fetched {len(data['data'])} records from Debt to Penny API")
                return pd.DataFrame(data['data'])
            
        except Exception as e:
            logger.debug(f"Debt to Penny API failed: {str(e)}")
        
        return None
    
    def _fetch_debt_to_penny_data(self, as_of_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Fetch additional data from Debt Subject to Limit endpoint."""
        try:
            endpoint = self.base_urls['debt_subject_limit']
            params = {
                'format': 'json',
                'page[size]': '1000',
                'sort': '-record_date'
            }
            
            if as_of_date:
                params['filter'] = f'record_date:eq:{as_of_date}'
            else:
                # Get recent data (last 30 days)
                from datetime import datetime, timedelta
                recent_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                params['filter'] = f'record_date:gte:{recent_date}'
            
            logger.info(f"Fetching from Debt Subject to Limit API: {endpoint}")
            response = requests.get(endpoint, params=params, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            if 'data' in data and data['data']:
                logger.info(f"Successfully fetched {len(data['data'])} records from Debt Subject to Limit API")
                return pd.DataFrame(data['data'])
            
        except Exception as e:
            logger.debug(f"Debt Subject to Limit API failed: {str(e)}")
        
        return None
    
    def _fetch_auction_data(self, as_of_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Fetch data from Debt Auctions endpoint for recent issuances."""
        try:
            endpoint = self.base_urls['debt_auctions']
            params = {
                'format': 'json',
                'page[size]': '500',
                'sort': '-auction_date'
            }
            
            # For auctions, get recent data within last 2 years to get active securities
            if not as_of_date:
                from datetime import datetime, timedelta
                two_years_ago = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
                params['filter'] = f'auction_date:gte:{two_years_ago}'
            elif as_of_date:
                params['filter'] = f'auction_date:eq:{as_of_date}'
            
            logger.info(f"Fetching from Debt Auctions API: {endpoint}")
            response = requests.get(endpoint, params=params, timeout=60)
            response.raise_for_status()
            
            data = response.json()
            if 'data' in data and data['data']:
                logger.info(f"Successfully fetched {len(data['data'])} records from Debt Auctions API")
                df = pd.DataFrame(data['data'])
                
                # Filter for securities that haven't matured yet
                current_date = datetime.now().date()
                df['maturity_date_parsed'] = pd.to_datetime(df['maturity_date'], errors='coerce')
                df = df[df['maturity_date_parsed'] > pd.Timestamp(current_date)]
                
                logger.info(f"Filtered to {len(df)} active (non-matured) securities")
                return df
            
        except Exception as e:
            logger.debug(f"Debt Auctions API failed: {str(e)}")
        
        return None
    
    def _standardize_api_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize Treasury API data to match expected format.
        
        Args:
            df (pd.DataFrame): Raw API data
            
        Returns:
            pd.DataFrame: Standardized securities data
        """
        try:
            logger.info(f"Standardizing {len(df)} records from Treasury API")
            logger.debug(f"Available columns: {list(df.columns)}")
            
            # Create standardized DataFrame with required fields
            standardized_df = df.copy()
            
            # Map and create required fields with fallbacks
            # CUSIP - use existing or generate placeholder
            if 'cusip' not in standardized_df.columns:
                standardized_df['cusip'] = standardized_df.index.map(lambda x: f'TREASURY_{x:06d}')
            else:
                # Clean CUSIP data - remove null values
                standardized_df['cusip'] = standardized_df['cusip'].replace('null', None)
                null_cusips = standardized_df['cusip'].isna()
                standardized_df.loc[null_cusips, 'cusip'] = [f'TREASURY_{i:06d}' for i in range(null_cusips.sum())]
            
            # Issue date - use issue_date from auctions, or record_date as fallback
            if 'issue_date' not in standardized_df.columns:
                if 'record_date' in standardized_df.columns:
                    standardized_df['issue_date'] = standardized_df['record_date']
                else:
                    # Use a default date in the past
                    standardized_df['issue_date'] = '2020-01-01'
            else:
                # Clean issue_date data
                standardized_df['issue_date'] = standardized_df['issue_date'].replace('null', None)
            
            # Maturity date - estimate based on typical Treasury terms
            if 'maturity_date' not in standardized_df.columns:
                # Estimate maturity dates based on record date + typical terms
                if 'record_date' in standardized_df.columns:
                    base_date = pd.to_datetime(standardized_df['record_date'])
                    # Add 1-10 years for different securities
                    import numpy as np
                    years_to_add = np.random.choice([1, 2, 5, 10], size=len(standardized_df))
                    standardized_df['maturity_date'] = base_date + pd.to_timedelta(years_to_add * 365, unit='D')
                else:
                    standardized_df['maturity_date'] = '2030-01-01'
            
            # Principal amount - use available debt amount fields, prioritizing auction data
            principal_fields = ['offering_amt', 'total_accepted', 'tot_pub_debt_out_amt', 'close_today_bal', 'outstanding_amt', 'debt_held_public_amt']
            standardized_df['principal_amount'] = None
            
            for field in principal_fields:
                if field in standardized_df.columns:
                    # Convert to numeric, handling 'null' strings
                    amount_series = standardized_df[field].replace('null', None)
                    amount_series = pd.to_numeric(amount_series, errors='coerce')
                    
                    # Use first non-null field found
                    if standardized_df['principal_amount'].isna().all():
                        standardized_df['principal_amount'] = amount_series
                    else:
                        # Fill nulls with values from this field
                        standardized_df['principal_amount'] = standardized_df['principal_amount'].fillna(amount_series)
            
            # If no principal amount found, use default
            if standardized_df['principal_amount'].isna().all():
                standardized_df['principal_amount'] = 10000.0  # $10B default
            
            # Interest rate - use int_rate from auctions or estimate
            if 'interest_rate' not in standardized_df.columns:
                if 'int_rate' in standardized_df.columns:
                    standardized_df['interest_rate'] = pd.to_numeric(standardized_df['int_rate'], errors='coerce') / 100.0
                else:
                    standardized_df['interest_rate'] = 0.03  # 3% default
            else:
                standardized_df['interest_rate'] = pd.to_numeric(standardized_df['interest_rate'], errors='coerce')
                # If rates are in percentage format (>1), convert to decimal
                high_rates = standardized_df['interest_rate'] > 1
                standardized_df.loc[high_rates, 'interest_rate'] = standardized_df.loc[high_rates, 'interest_rate'] / 100.0
                standardized_df['interest_rate'] = standardized_df['interest_rate'].fillna(0.03)
            
            # Security type - use debt category or default
            if 'security_type' not in standardized_df.columns:
                if 'debt_catg_desc' in standardized_df.columns:
                    standardized_df['security_type'] = standardized_df['debt_catg_desc']
                elif 'debt_catg' in standardized_df.columns:
                    standardized_df['security_type'] = standardized_df['debt_catg']
                else:
                    standardized_df['security_type'] = 'Treasury Security'
            
            # Interest payment frequency
            def get_payment_frequency(security_type):
                security_str = str(security_type).lower()
                if 'bill' in security_str:
                    return 0  # Zero-coupon
                elif any(term in security_str for term in ['note', 'bond', 'tips']):
                    return 2  # Semi-annual
                else:
                    return 2  # Default to semi-annual
            
            standardized_df['interest_payment_frequency'] = standardized_df['security_type'].apply(get_payment_frequency)
            
            # Convert data types
            standardized_df['issue_date'] = pd.to_datetime(standardized_df['issue_date'], errors='coerce')
            standardized_df['maturity_date'] = pd.to_datetime(standardized_df['maturity_date'], errors='coerce')
            standardized_df['principal_amount'] = pd.to_numeric(standardized_df['principal_amount'], errors='coerce')
            standardized_df['interest_rate'] = pd.to_numeric(standardized_df['interest_rate'], errors='coerce')
            
            # Convert principal amount to consistent millions unit
            # Check if data is in dollars (very large numbers) and convert to millions
            median_amount = standardized_df['principal_amount'].median()
            if median_amount > 1_000_000_000:  # If median > 1 billion, assume it's in dollars
                logger.info(f"Converting principal amounts from dollars to millions (median: {median_amount:,.0f})")
                standardized_df['principal_amount'] = standardized_df['principal_amount'] / 1_000_000
            elif median_amount > 100_000:  # If median > 100K, assume it's already in millions
                logger.info(f"Principal amounts appear to be in millions already (median: {median_amount:,.0f})")
            else:  # If median < 100K, might be in billions, convert to millions
                logger.info(f"Converting principal amounts from billions to millions (median: {median_amount:,.0f})")
                standardized_df['principal_amount'] = standardized_df['principal_amount'] * 1_000
            
            # Filter out records with missing critical data
            required_fields = ['cusip', 'maturity_date', 'principal_amount']
            before_filter = len(standardized_df)
            standardized_df = standardized_df.dropna(subset=required_fields)
            after_filter = len(standardized_df)
            
            logger.info(f"Standardization complete: {before_filter} -> {after_filter} records")
            logger.info(f"Sample standardized record: {dict(standardized_df.iloc[0]) if len(standardized_df) > 0 else 'No records'}")
            
            return standardized_df
            
        except Exception as e:
            logger.error(f"Error standardizing API data: {str(e)}")
            logger.error(f"DataFrame columns: {list(df.columns) if df is not None else 'None'}")
            logger.error(f"DataFrame shape: {df.shape if df is not None else 'None'}")
            raise
    
    def _create_sample_securities_data(self) -> pd.DataFrame:
        """
        Create representative sample data for demonstration purposes.
        
        This method generates realistic Treasury securities data based on
        current market conditions and typical issuance patterns.
        """
        np.random.seed(42)  # For reproducible results
        
        # Generate sample securities across different types and maturities
        securities = []
        base_date = datetime.now().date()
        
        # Treasury Bills (3M, 6M, 1Y)
        for i in range(50):
            maturity_days = int(np.random.choice([91, 182, 365]))
            securities.append({
                'cusip': f'912796{i:03d}',
                'security_type': 'BILL',
                'issue_date': base_date - timedelta(days=int(np.random.randint(1, 90))),
                'maturity_date': base_date + timedelta(days=maturity_days),
                'principal_amount': float(np.random.uniform(10000, 50000)),  # millions
                'interest_rate': 0.0,  # Bills are zero-coupon
                'interest_payment_frequency': 0
            })
        
        # Treasury Notes (2Y, 3Y, 5Y, 7Y, 10Y)
        for i in range(100):
            maturity_years = int(np.random.choice([2, 3, 5, 7, 10]))
            securities.append({
                'cusip': f'912828{i:03d}',
                'security_type': 'NOTE',
                'issue_date': base_date - timedelta(days=int(np.random.randint(1, 365))),
                'maturity_date': base_date + relativedelta(years=maturity_years),
                'principal_amount': float(np.random.uniform(20000, 80000)),  # millions
                'interest_rate': float(np.random.uniform(0.02, 0.05)),  # 2-5% coupon
                'interest_payment_frequency': 2  # Semi-annual
            })
        
        # Treasury Bonds (20Y, 30Y)
        for i in range(30):
            maturity_years = int(np.random.choice([20, 30]))
            securities.append({
                'cusip': f'912810{i:03d}',
                'security_type': 'BOND',
                'issue_date': base_date - timedelta(days=int(np.random.randint(1, 1095))),
                'maturity_date': base_date + relativedelta(years=maturity_years),
                'principal_amount': float(np.random.uniform(15000, 60000)),  # millions
                'interest_rate': float(np.random.uniform(0.025, 0.055)),  # 2.5-5.5% coupon
                'interest_payment_frequency': 2  # Semi-annual
            })
        
        df = pd.DataFrame(securities)
        
        # Convert date columns
        df['issue_date'] = pd.to_datetime(df['issue_date'])
        df['maturity_date'] = pd.to_datetime(df['maturity_date'])
        
        logger.info(f"Generated {len(df)} sample Treasury securities")
        return df
    
    def _validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the securities data.
        
        Args:
            df (pd.DataFrame): Raw securities data
            
        Returns:
            pd.DataFrame: Cleaned and validated data
        """
        logger.info("Validating and cleaning securities data...")
        
        initial_count = len(df)
        
        # Remove securities with missing critical data
        df = df.dropna(subset=['cusip', 'maturity_date', 'principal_amount'])
        
        # Remove securities that have already matured
        current_date = pd.Timestamp.now().normalize()
        df = df[df['maturity_date'] > current_date]
        
        # Ensure positive principal amounts
        df = df[df['principal_amount'] > 0]
        
        # Set default values for missing optional fields
        df['interest_rate'] = df['interest_rate'].fillna(0.0)
        df['interest_payment_frequency'] = df['interest_payment_frequency'].fillna(0)
        
        # Ensure proper data types
        df['principal_amount'] = pd.to_numeric(df['principal_amount'], errors='coerce')
        df['interest_rate'] = pd.to_numeric(df['interest_rate'], errors='coerce')
        df['interest_payment_frequency'] = pd.to_numeric(df['interest_payment_frequency'], errors='coerce').astype(int)
        
        # Only remove rows where CRITICAL fields couldn't be converted (much more targeted approach)
        critical_fields = ['cusip', 'maturity_date', 'principal_amount', 'interest_rate', 'interest_payment_frequency']
        df = df.dropna(subset=critical_fields)
        
        final_count = len(df)
        logger.info(f"Data validation complete: {initial_count} -> {final_count} securities")
        
        # Log validation success metrics
        if final_count > 0:
            logger.info(f"✅ Validation successful: {final_count} securities ready for debt calendar")
            logger.info(f"📊 Securities date range: {df['maturity_date'].min()} to {df['maturity_date'].max()}")
            logger.info(f"💰 Principal amount range: ${df['principal_amount'].min():,.0f}M to ${df['principal_amount'].max():,.0f}M")
        else:
            logger.warning("⚠️ No securities survived validation - check data quality")
        
        return df


class DebtPaymentCalculator:
    """
    Calculates principal redemptions and interest payments for Treasury securities.
    
    This class handles the complex logic of determining when and how much
    each security will pay in principal and interest over the forecast period.
    """
    
    def __init__(self, forecast_start_date: date, forecast_horizon_days: int = 180):
        """
        Initialize the debt payment calculator.
        
        Args:
            forecast_start_date (date): Start date for payment calculations
            forecast_horizon_days (int): Number of days to calculate payments for
        """
        self.forecast_start_date = forecast_start_date
        self.forecast_end_date = forecast_start_date + timedelta(days=forecast_horizon_days)
        self.forecast_horizon_days = forecast_horizon_days
        
        logger.info(f"Debt Payment Calculator initialized for {forecast_start_date} to {self.forecast_end_date}")
    
    def calculate_principal_payments(self, securities_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all principal redemption payments within the forecast period.
        
        Args:
            securities_df (pd.DataFrame): Outstanding securities data
            
        Returns:
            pd.DataFrame: Principal payments with columns [payment_date, principal_amount, cusip, security_type]
        """
        logger.info("Calculating principal redemption payments...")
        
        principal_payments = []
        
        for _, security in securities_df.iterrows():
            maturity_date = pd.to_datetime(security['maturity_date']).date()
            
            # Only include securities maturing within our forecast period
            if self.forecast_start_date <= maturity_date <= self.forecast_end_date:
                principal_payments.append({
                    'payment_date': maturity_date,
                    'principal_amount': security['principal_amount'],
                    'cusip': security['cusip'],
                    'security_type': security['security_type'],
                    'payment_type': 'PRINCIPAL_REDEMPTION'
                })
        
        df = pd.DataFrame(principal_payments)
        
        if not df.empty:
            df['payment_date'] = pd.to_datetime(df['payment_date'])
            logger.info(f"Calculated {len(df)} principal redemption payments totaling ${df['principal_amount'].sum():,.0f}M")
        else:
            logger.info("No principal redemptions found within forecast period")
        
        return df
    
    def calculate_interest_payments(self, securities_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all interest payments within the forecast period.
        
        Args:
            securities_df (pd.DataFrame): Outstanding securities data
            
        Returns:
            pd.DataFrame: Interest payments with columns [payment_date, interest_amount, cusip, security_type]
        """
        logger.info("Calculating interest payments...")
        
        interest_payments = []
        
        for _, security in securities_df.iterrows():
            if security['interest_payment_frequency'] == 0:
                # Zero-coupon security (like Treasury Bills)
                continue
            
            # Calculate all interest payment dates for this security
            payment_dates = self._generate_interest_payment_dates(
                security['issue_date'],
                security['maturity_date'], 
                security['interest_payment_frequency']
            )
            
            # Filter dates within our forecast period
            forecast_payments = [
                d for d in payment_dates 
                if self.forecast_start_date <= d <= self.forecast_end_date
            ]
            
            # Calculate payment amount (typically semi-annual)
            payment_amount = (
                security['principal_amount'] * 
                security['interest_rate'] / 
                security['interest_payment_frequency']
            )
            
            # Add each payment to our list
            for payment_date in forecast_payments:
                interest_payments.append({
                    'payment_date': payment_date,
                    'interest_amount': payment_amount,
                    'cusip': security['cusip'],
                    'security_type': security['security_type'],
                    'payment_type': 'INTEREST_PAYMENT'
                })
        
        df = pd.DataFrame(interest_payments)
        
        if not df.empty:
            df['payment_date'] = pd.to_datetime(df['payment_date'])
            logger.info(f"Calculated {len(df)} interest payments totaling ${df['interest_amount'].sum():,.0f}M")
        else:
            logger.info("No interest payments found within forecast period")
        
        return df
    
    def _generate_interest_payment_dates(self, issue_date: pd.Timestamp, maturity_date: pd.Timestamp, 
                                       frequency: int) -> List[date]:
        """
        Generate all interest payment dates for a security.
        
        Args:
            issue_date (pd.Timestamp): Security issue date
            maturity_date (pd.Timestamp): Security maturity date
            frequency (int): Number of payments per year (typically 2 for semi-annual)
            
        Returns:
            List[date]: List of interest payment dates
        """
        if frequency == 0:
            return []
        
        payment_dates = []
        months_between_payments = 12 // frequency
        
        # Start from the first payment date (typically 6 months after issue)
        current_date = pd.to_datetime(issue_date) + relativedelta(months=months_between_payments)
        maturity_timestamp = pd.to_datetime(maturity_date)
        
        while current_date <= maturity_timestamp:
            payment_dates.append(current_date.date())
            current_date += relativedelta(months=months_between_payments)
        
        return payment_dates


class DebtEventsCalendar:
    """
    Main debt events calendar that aggregates all scheduled debt service payments.
    
    This class provides the primary interface for the simulation engine to query
    scheduled debt payments for any given date.
    """
    
    def __init__(self, config_manager=None):
        """
        Initialize the debt events calendar.
        
        Args:
            config_manager: Configuration manager instance (optional)
        """
        self.config = config_manager
        self.data_acquisition = TreasuryDataAcquisition()
        self.calendar_df = None
        self.securities_data = None
        self.last_updated = None
        
        # Set default forecast parameters
        self.forecast_start_date = datetime.now().date()
        self.forecast_horizon_days = 180
        
        if self.config:
            self.forecast_start_date = self.config.START_DATE
            self.forecast_horizon_days = self.config.FORECAST_HORIZON
        
        logger.info("Debt Events Calendar initialized")
    
    def build_calendar(self, refresh_data: bool = True) -> pd.DataFrame:
        """
        Build the complete debt events calendar.
        
        Args:
            refresh_data (bool): Whether to refresh securities data from source
            
        Returns:
            pd.DataFrame: Calendar with daily debt service payments
        """
        logger.info("Building debt events calendar...")
        
        # Step 1: Acquire securities data
        if refresh_data or self.securities_data is None:
            self.securities_data = self.data_acquisition.fetch_outstanding_securities()
        
        # Step 2: Calculate payments
        calculator = DebtPaymentCalculator(
            self.forecast_start_date, 
            self.forecast_horizon_days
        )
        
        principal_payments = calculator.calculate_principal_payments(self.securities_data)
        interest_payments = calculator.calculate_interest_payments(self.securities_data)
        
        # Step 3: Build the calendar structure
        self.calendar_df = self._create_calendar_structure(principal_payments, interest_payments)
        
        self.last_updated = datetime.now()
        
        logger.info(f"Debt events calendar built successfully with {len(self.calendar_df)} days")
        return self.calendar_df
    
    def _create_calendar_structure(self, principal_payments: pd.DataFrame, 
                                 interest_payments: pd.DataFrame) -> pd.DataFrame:
        """
        Create the final calendar data structure.
        
        Args:
            principal_payments (pd.DataFrame): Principal redemption payments
            interest_payments (pd.DataFrame): Interest payments
            
        Returns:
            pd.DataFrame: Daily calendar with aggregated payments
        """
        logger.info("Creating calendar data structure...")
        
        # Create date range for entire forecast period
        date_range = pd.date_range(
            start=self.forecast_start_date,
            end=self.forecast_start_date + timedelta(days=self.forecast_horizon_days - 1),
            freq='D'
        )
        
        # Initialize calendar with zeros
        calendar = pd.DataFrame({
            'date': date_range,
            'total_principal_due': 0.0,
            'total_interest_due': 0.0,
            'total_debt_service': 0.0,
            'principal_payment_count': 0,
            'interest_payment_count': 0
        })
        
        calendar.set_index('date', inplace=True)
        
        # Aggregate principal payments by date
        if not principal_payments.empty:
            principal_daily = principal_payments.groupby('payment_date').agg({
                'principal_amount': 'sum',
                'cusip': 'count'
            }).rename(columns={'cusip': 'principal_payment_count'})
            
            # Merge with calendar
            calendar = calendar.join(principal_daily[['principal_amount']], how='left')
            calendar['total_principal_due'] = calendar['principal_amount'].fillna(0)
            calendar = calendar.join(principal_daily[['principal_payment_count']], how='left', rsuffix='_p')
            calendar['principal_payment_count'] = calendar['principal_payment_count_p'].fillna(0)
            calendar.drop(['principal_amount', 'principal_payment_count_p'], axis=1, inplace=True)
        
        # Aggregate interest payments by date
        if not interest_payments.empty:
            interest_daily = interest_payments.groupby('payment_date').agg({
                'interest_amount': 'sum',
                'cusip': 'count'
            }).rename(columns={'cusip': 'interest_payment_count'})
            
            # Merge with calendar
            calendar = calendar.join(interest_daily[['interest_amount']], how='left')
            calendar['total_interest_due'] = calendar['interest_amount'].fillna(0)
            calendar = calendar.join(interest_daily[['interest_payment_count']], how='left', rsuffix='_i')
            calendar['interest_payment_count'] = calendar['interest_payment_count_i'].fillna(0)
            calendar.drop(['interest_amount', 'interest_payment_count_i'], axis=1, inplace=True)
        
        # Calculate total debt service (keep amounts in millions for consistency with system design)
        calendar['total_debt_service'] = calendar['total_principal_due'] + calendar['total_interest_due']
        
        # Note: Amounts remain in millions as expected by the rest of the system
        
        logger.info(f"Calendar structure created with total debt service: ${calendar['total_debt_service'].sum():,.0f}")
        
        return calendar
    
    def get_debt_service_for_date(self, query_date: Union[str, date, datetime]) -> Dict[str, float]:
        """
        Get scheduled debt service payments for a specific date.
        
        Args:
            query_date: Date to query (string, date, or datetime)
            
        Returns:
            Dict with keys: total_principal_due, total_interest_due, total_debt_service
        """
        if self.calendar_df is None:
            raise ValueError("Calendar not built. Call build_calendar() first.")
        
        # Convert query_date to pandas Timestamp
        if isinstance(query_date, str):
            query_date = pd.to_datetime(query_date)
        elif isinstance(query_date, date):
            query_date = pd.Timestamp(query_date)
        elif isinstance(query_date, datetime):
            query_date = pd.Timestamp(query_date)
        
        try:
            row = self.calendar_df.loc[query_date]
            return {
                'total_principal_due': row['total_principal_due'],
                'total_interest_due': row['total_interest_due'],
                'total_debt_service': row['total_debt_service'],
                'principal_payment_count': int(row['principal_payment_count']),
                'interest_payment_count': int(row['interest_payment_count'])
            }
        except KeyError:
            # Date not in calendar (outside forecast period)
            return {
                'total_principal_due': 0.0,
                'total_interest_due': 0.0,
                'total_debt_service': 0.0,
                'principal_payment_count': 0,
                'interest_payment_count': 0
            }
    
    def get_calendar_summary(self) -> Dict[str, Union[int, float]]:
        """
        Get summary statistics for the debt events calendar.
        
        Returns:
            Dict with calendar summary statistics
        """
        if self.calendar_df is None:
            raise ValueError("Calendar not built. Call build_calendar() first.")
        
        return {
            'forecast_start_date': str(self.forecast_start_date),
            'forecast_end_date': str(self.forecast_start_date + timedelta(days=self.forecast_horizon_days - 1)),
            'forecast_horizon_days': self.forecast_horizon_days,
            'total_securities_tracked': len(self.securities_data) if self.securities_data is not None else 0,
            'total_principal_payments': self.calendar_df['total_principal_due'].sum(),
            'total_interest_payments': self.calendar_df['total_interest_due'].sum(),
            'total_debt_service': self.calendar_df['total_debt_service'].sum(),
            'days_with_principal_payments': (self.calendar_df['total_principal_due'] > 0).sum(),
            'days_with_interest_payments': (self.calendar_df['total_interest_due'] > 0).sum(),
            'days_with_any_payments': (self.calendar_df['total_debt_service'] > 0).sum(),
            'max_daily_debt_service': self.calendar_df['total_debt_service'].max(),
            'average_daily_debt_service': self.calendar_df['total_debt_service'].mean(),
            'last_updated': str(self.last_updated) if self.last_updated else None
        }
    
    def get_scheduled_debt_service(self, query_date) -> Dict[str, float]:
        """
        Get scheduled debt service payments for a specific date.
        
        Args:
            query_date: Date to query (can be datetime, date, or string)
            
        Returns:
            Dict with principal_due, interest_due, and total_due amounts in MILLIONS of dollars
        """
        if self.calendar_df is None:
            raise ValueError("Calendar not built. Call build_calendar() first.")
        
        # Convert query_date to pandas Timestamp for comparison
        if isinstance(query_date, str):
            query_date = pd.to_datetime(query_date).date()
        elif hasattr(query_date, 'date'):
            query_date = query_date.date()
        
        # Look up the date in the calendar
        try:
            # Find matching row by date
            matching_rows = self.calendar_df[self.calendar_df.index.date == query_date]
            
            if len(matching_rows) > 0:
                row = matching_rows.iloc[0]
                return {
                    'principal_due': float(row['total_principal_due']),
                    'interest_due': float(row['total_interest_due']),
                    'total_due': float(row['total_debt_service'])
                }
            else:
                # No payments scheduled for this date
                return {
                    'principal_due': 0.0,
                    'interest_due': 0.0,
                    'total_due': 0.0
                }
                
        except Exception as e:
            # If there's any error, return zeros
            return {
                'principal_due': 0.0,
                'interest_due': 0.0,
                'total_due': 0.0
            }
    
    def export_calendar(self, file_path: str) -> None:
        """
        Export the debt events calendar to a CSV file.
        
        Args:
            file_path (str): Path to save the calendar CSV file
        """
        if self.calendar_df is None:
            raise ValueError("Calendar not built. Call build_calendar() first.")
        
        # Reset index to include date as a column
        export_df = self.calendar_df.reset_index()
        export_df.to_csv(file_path, index=False)
        
        logger.info(f"Debt events calendar exported to {file_path}")


# Example usage and testing
if __name__ == "__main__":
    # Create and build the debt events calendar
    calendar = DebtEventsCalendar()
    
    # Build the calendar with sample data
    calendar_df = calendar.build_calendar(refresh_data=True)
    
    # Display summary
    summary = calendar.get_calendar_summary()
    print("\n=== Debt Events Calendar Summary ===")
    for key, value in summary.items():
        if isinstance(value, float) and value > 1000:
            print(f"{key}: ${value:,.0f}")
        else:
            print(f"{key}: {value}")
    
    # Test date queries
    test_dates = [
        calendar.forecast_start_date,
        calendar.forecast_start_date + timedelta(days=30),
        calendar.forecast_start_date + timedelta(days=90)
    ]
    
    print("\n=== Sample Date Queries ===")
    for test_date in test_dates:
        payments = calendar.get_debt_service_for_date(test_date)
        print(f"\n{test_date}:")
        print(f"  Principal Due: ${payments['total_principal_due']:,.0f}")
        print(f"  Interest Due: ${payments['total_interest_due']:,.0f}")
        print(f"  Total Debt Service: ${payments['total_debt_service']:,.0f}")
        print(f"  Payment Count: {payments['principal_payment_count']} principal, {payments['interest_payment_count']} interest")
