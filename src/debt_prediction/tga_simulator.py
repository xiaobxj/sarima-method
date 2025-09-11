"""
Treasury General Account (TGA) Balance Simulator

This module simulates future TGA balance changes using operational cash flow forecasts
and configuration parameters for scenario analysis.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import Dict, List, Tuple, Optional, Any

from .config import DebtPredictionConfig, load_config


class TGABalanceSimulator:
    """
    Simulates Treasury General Account (TGA) balance using operational cash flow forecasts.
    
    This class provides comprehensive TGA balance simulation capabilities including:
    - Cash flow categorization and aggregation
    - Daily balance projections
    - X-Date identification
    - Scenario analysis
    - Visualization and reporting
    """
    
    def __init__(self, config: Optional[DebtPredictionConfig] = None):
        """
        Initialize the TGA Balance Simulator.
        
        Args:
            config (DebtPredictionConfig, optional): Configuration object
        """
        self.config = config or load_config()
        
        # Unit management - all internal calculations in actual dollars
        self.internal_unit = "dollars"  # Internal calculations use actual dollars
        self.display_unit = "millions"  # Display and storage use millions
        self.unit_scale = 1_000_000     # Conversion factor
        
        # Data containers
        self.forecast_df = None
        self.daily_cash_flows = None
        self.tga_simulation = None
        self.starting_balance = None
        
        # Results storage
        self.simulation_results = {}
        self.x_date_analysis = {}
        
        # Enhanced cash flow categorization keywords with comprehensive coverage
        self.deposit_keywords = {
            # Tax-related inflows
            'Tax', 'Income', 'Corporation', 'Withheld', 'Employment', 'Individual',
            'Estate', 'Gift', 'FUTA', 'FICA', 'Customs', 'Excise',
            # Government earnings and receipts
            'Federal_Reserve_Earnings', 'Earnings', 'Revenue', 'Receipt', 'Deposits',
            'Unemployment_Insurance', 'Railroad_Retirement', 'Proceeds',
            # Loan repayments and recoveries
            'Repayments', 'Repayment', 'Recovery', 'Collections',
            # Fees and fines
            'Fees', 'Fines', 'Penalties'
        }
        
        self.withdrawal_keywords = {
            # Benefit payments
            'Benefits', 'Medicare', 'Medicaid', 'Social_Security', 'Unemployment_Benefits',
            'Retirement', 'Disability', 'Supplemental',
            # Government operations
            'Defense', 'Military', 'Veterans', 'Salaries', 'Payroll',
            # Department spending
            'Education', 'Agriculture', 'Health', 'Transportation', 'Energy',
            'Justice', 'Housing', 'NASA', 'EPA', 'Interior', 'Commerce', 'Labor',
            # Payment types
            'Refunds', 'Payment', 'Payments', 'Grant', 'Grants', 'Assistance',
            'Support', 'Relief', 'Aid', 'Subsidies',
            # Administrative and operational
            'Program', 'Programs', 'Admin', 'Administration', 'Services',
            'Operations', 'Maintenance', 'Contracts', 'Vendor'
        }
        
        print(f"TGA Balance Simulator initialized")
        print(f"Forecast horizon: {self.config.FORECAST_HORIZON} days")
        print(f"Starting TGA balance: ${self.config.CURRENT_TGA_BALANCE:,.0f} million")
        print(f"Internal calculations: {self.internal_unit}, Display: {self.display_unit}")
    
    def _to_internal_units(self, value_in_millions: float) -> float:
        """Convert from millions to internal dollars."""
        return value_in_millions * self.unit_scale
    
    def _to_display_units(self, value_in_dollars: float) -> float:
        """Convert from internal dollars to millions for display."""
        return value_in_dollars / self.unit_scale
    
    def _sanity_check_value(self, value: float, name: str, max_reasonable: float = 1e13) -> float:
        """Check if value is reasonable and warn if not."""
        if abs(value) > max_reasonable:
            print(f"⚠️ WARNING: Unreasonably large value detected for {name}: ${value:,.0f}")
            print(f"   This may indicate a unit conversion error")
        return value
    
    def classify_cash_flow(self, series_name: str) -> str:
        """
        Enhanced intelligent classification function for determining cash flow direction.
        Uses word-level matching with priority rules for more accurate categorization.
        
        Args:
            series_name (str): Name of the cash flow series
            
        Returns:
            str: Classification result ('deposit', 'withdrawal', or 'uncategorized')
        """
        # Convert series name to standardized word set for full-word matching
        # Replace underscores and hyphens with spaces, then split into words
        words = set(series_name.replace('_', ' ').replace('-', ' ').split())
        
        # Check for keyword matches using set operations (more efficient)
        has_deposit_keywords = not self.deposit_keywords.isdisjoint(words)
        has_withdrawal_keywords = not self.withdrawal_keywords.isdisjoint(words)
        
        # Apply priority rules for classification
        # Rule 1: If contains explicit withdrawal words (like Refunds, Benefits, Payments), 
        #         prioritize as 'withdrawal'
        if has_withdrawal_keywords:
            return 'withdrawal'
        
        # Rule 2: If no withdrawal words but has deposit words, classify as 'deposit'
        if has_deposit_keywords:
            return 'deposit'
            
        # Rule 3: If neither category matches, mark as uncategorized
        return 'uncategorized'
    
    def load_forecast_data(self) -> pd.DataFrame:
        """
        Load operational cash flow forecast data.
        
        Returns:
            pd.DataFrame: Loaded forecast data
        """
        print("Loading operational cash flow forecasts...")
        
        try:
            self.forecast_df = pd.read_csv(self.config.FORECASTS_FILE)
            self.forecast_df['forecast_date'] = pd.to_datetime(self.forecast_df['forecast_date'])
            
            # Filter to forecast horizon
            start_date = pd.to_datetime(self.config.START_DATE)
            end_date = start_date + timedelta(days=self.config.FORECAST_HORIZON)
            
            self.forecast_df = self.forecast_df[
                (self.forecast_df['forecast_date'] >= start_date) &
                (self.forecast_df['forecast_date'] <= end_date)
            ].copy()
            
            print(f"✅ Loaded {len(self.forecast_df)} forecast records")
            print(f"   Date range: {self.forecast_df['forecast_date'].min().date()} to {self.forecast_df['forecast_date'].max().date()}")
            print(f"   Unique categories: {self.forecast_df['series_name'].nunique()}")
            
            return self.forecast_df
            
        except FileNotFoundError:
            print(f"❌ Error: Forecast file not found: {self.config.FORECASTS_FILE}")
            return None
        except Exception as e:
            print(f"❌ Error loading forecast data: {e}")
            return None
    
    def get_starting_balance(self) -> float:
        """
        Get or validate the starting TGA balance.
        
        Returns:
            float: Starting TGA balance in millions USD
        """
        print("Validating starting TGA balance...")
        
        # Try to get live TGA balance from API if enabled
        if self.config.USE_LIVE_TREASURY_DATA:
            live_balance = self._get_live_tga_balance()
            if live_balance is not None:
                print(f"🌐 Latest live TGA balance: ${live_balance:,.0f} million")
                print(f"📋 Configured starting balance: ${self.config.CURRENT_TGA_BALANCE:,.0f} million")
                
                # Use configured balance for consistency with scenario analysis
                self.starting_balance = self.config.CURRENT_TGA_BALANCE * 1_000_000
                return self.starting_balance
        
        try:
            # Try to load actual balance data for validation
            balance_df = pd.read_csv(self.config.OPERATING_BALANCE_FILE)
            balance_df['record_date'] = pd.to_datetime(balance_df['record_date'])
            
            # Look for TGA closing balance
            tga_data = balance_df[
                balance_df['account_type'] == 'Treasury General Account (TGA) Closing Balance'
            ].copy()
            
            if len(tga_data) > 0:
                latest_tga = tga_data.loc[tga_data['record_date'].idxmax()]
                actual_balance = float(latest_tga['close_today_bal']) * 1_000_000  # Convert to actual dollars
                
                print(f"📊 Latest actual TGA balance: ${actual_balance/1e6:,.0f} million ({latest_tga['record_date'].date()})")
                print(f"📋 Configured starting balance: ${self.config.CURRENT_TGA_BALANCE:,.0f} million")
                
                # Use configured balance (allows for scenario analysis)
                self.starting_balance = self.config.CURRENT_TGA_BALANCE * 1_000_000  # Convert to actual dollars
                
            else:
                # Fallback to Federal Reserve Account
                fra_data = balance_df[balance_df['account_type'] == 'Federal Reserve Account'].copy()
                if len(fra_data) > 0:
                    latest_fra = fra_data.loc[fra_data['record_date'].idxmax()]
                    self.starting_balance = float(latest_fra['close_today_bal']) * 1_000_000
                    print(f"📊 Using Federal Reserve Account balance: ${self.starting_balance/1e6:,.0f} million")
                else:
                    # Use configured balance
                    self.starting_balance = self.config.CURRENT_TGA_BALANCE * 1_000_000
                    print(f"📋 Using configured starting balance: ${self.starting_balance/1e6:,.0f} million")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load balance data ({e})")
            self.starting_balance = self.config.CURRENT_TGA_BALANCE * 1_000_000
            print(f"📋 Using configured starting balance: ${self.starting_balance/1e6:,.0f} million")
        
        return self.starting_balance
    
    def _get_live_tga_balance(self) -> Optional[float]:
        """
        Get the latest TGA balance from the live Treasury API.
        
        Returns:
            float: Latest TGA balance in millions USD, or None if failed
        """
        try:
            import requests
            from datetime import datetime, timedelta
            
            # Use the operating cash balance API
            endpoint = self.config.TREASURY_API_ENDPOINTS['daily_treasury']
            
            # Get recent data (last 7 days)
            recent_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            params = {
                'format': 'json',
                'page[size]': '50',
                'filter': f'record_date:gte:{recent_date}',
                'sort': '-record_date'
            }
            
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'data' in data and data['data']:
                # Look for TGA Opening Balance (most recent available balance data)
                for record in data['data']:
                    account_type = record.get('account_type', '')
                    if 'TGA' in account_type and 'Opening Balance' in account_type:
                        # Try open_today_bal first (this has actual data)
                        balance_str = record.get('open_today_bal', '0')
                        record_date = record.get('record_date')
                        
                        # Handle 'null' strings and convert to float
                        if balance_str == 'null' or balance_str is None:
                            continue  # Skip null records
                        
                        try:
                            balance = float(balance_str)
                            print(f"🌐 Live API - TGA Opening Balance: ${balance:,.0f} million ({record_date})")
                            return balance
                        except (ValueError, TypeError):
                            continue  # Skip invalid balance values
                
                # Fallback to Federal Reserve Account if available
                for record in data['data']:
                    if record.get('account_type') == 'Federal Reserve Account':
                        balance_str = record.get('close_today_bal', '0')
                        record_date = record.get('record_date')
                        
                        # Handle 'null' strings and convert to float
                        if balance_str == 'null' or balance_str is None:
                            continue  # Skip null records
                        
                        try:
                            balance = float(balance_str)
                            print(f"🌐 Live API - Federal Reserve Account: ${balance:,.0f} million ({record_date})")
                            return balance
                        except (ValueError, TypeError):
                            continue  # Skip invalid balance values
                
                print(f"⚠️ No TGA or Federal Reserve Account data found in live API response")
                return None
            
        except Exception as e:
            print(f"⚠️ Failed to fetch live TGA balance: {str(e)}")
            return None
    
    def categorize_cash_flows(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Enhanced categorization of forecast series into deposits (inflows) and withdrawals (outflows).
        Uses the improved classify_cash_flow function with word-level matching and priority rules.
        
        Returns:
            Tuple[List[str], List[str], List[str]]: (deposits, withdrawals, unclassified)
        """
        print("Categorizing cash flows with enhanced classification...")
        
        series_names = self.forecast_df['series_name'].unique()
        
        deposits = []
        withdrawals = []
        unclassified = []
        
        # Classification statistics for reporting
        classification_stats = {'deposit': 0, 'withdrawal': 0, 'uncategorized': 0}
        
        for series in series_names:
            classification = self.classify_cash_flow(series)
            classification_stats[classification] += 1
            
            if classification == 'deposit':
                deposits.append(series)
            elif classification == 'withdrawal':
                withdrawals.append(series)
            else:  # uncategorized
                unclassified.append(series)
        
        print(f"📈 Enhanced Cash Flow Classification Results:")
        print(f"   Deposits (Inflows): {len(deposits)} categories")
        print(f"   Withdrawals (Outflows): {len(withdrawals)} categories")
        print(f"   Unclassified: {len(unclassified)} categories")
        
        # Show classification improvement
        total_classified = len(deposits) + len(withdrawals)
        classification_rate = (total_classified / len(series_names)) * 100
        print(f"   Classification Success Rate: {classification_rate:.1f}%")
        
        if unclassified:
            print(f"   ⚠️  Unclassified items will be treated as outflows (conservative approach)")
            print(f"      Examples: {', '.join(unclassified[:3])}{'...' if len(unclassified) > 3 else ''}")
            
            # Show some examples of why items might be unclassified
            if len(unclassified) > 0:
                sample_unclassified = unclassified[0]
                words = set(sample_unclassified.replace('_', ' ').replace('-', ' ').split())
                print(f"      Sample analysis for '{sample_unclassified}':")
                print(f"        Words: {sorted(words)}")
                print(f"        No matching keywords found in either category")
        
        return deposits, withdrawals, unclassified
    
    def generate_classification_report(self, deposits: List[str], withdrawals: List[str], 
                                     unclassified: List[str]) -> Dict[str, Any]:
        """
        Generate detailed classification report for analysis and validation.
        
        Args:
            deposits (List[str]): List of deposit series names
            withdrawals (List[str]): List of withdrawal series names
            unclassified (List[str]): List of unclassified series names
            
        Returns:
            Dict[str, Any]: Detailed classification report
        """
        report = {
            'summary': {
                'total_series': len(deposits) + len(withdrawals) + len(unclassified),
                'deposits_count': len(deposits),
                'withdrawals_count': len(withdrawals),
                'unclassified_count': len(unclassified),
                'classification_rate': (len(deposits) + len(withdrawals)) / (len(deposits) + len(withdrawals) + len(unclassified)) * 100
            },
            'deposit_categories': {
                'tax_related': [s for s in deposits if any(word in s.lower() for word in ['tax', 'income', 'withheld'])],
                'customs_excise': [s for s in deposits if any(word in s.lower() for word in ['customs', 'excise'])],
                'loan_repayments': [s for s in deposits if 'repayment' in s.lower()],
                'federal_reserve': [s for s in deposits if 'federal_reserve' in s.lower()],
                'other_deposits': []
            },
            'withdrawal_categories': {
                'benefits_payments': [s for s in withdrawals if any(word in s.lower() for word in ['benefit', 'social_security', 'medicare', 'medicaid'])],
                'defense_military': [s for s in withdrawals if any(word in s.lower() for word in ['defense', 'military'])],
                'department_programs': [s for s in withdrawals if any(word in s.lower() for word in ['education', 'agriculture', 'veterans', 'health'])],
                'tax_refunds': [s for s in withdrawals if 'refund' in s.lower()],
                'salaries_payments': [s for s in withdrawals if any(word in s.lower() for word in ['salaries', 'payment'])],
                'other_withdrawals': []
            },
            'unclassified_analysis': {
                'potential_deposits': [],
                'potential_withdrawals': [],
                'truly_ambiguous': []
            }
        }
        
        # Categorize remaining items
        for category in ['deposit_categories', 'withdrawal_categories']:
            category_items = report[category]
            if category == 'deposit_categories':
                all_categorized = sum([items for items in category_items.values()], [])
                category_items['other_deposits'] = [s for s in deposits if s not in all_categorized]
            else:
                all_categorized = sum([items for items in category_items.values()], [])
                category_items['other_withdrawals'] = [s for s in withdrawals if s not in all_categorized]
        
        # Analyze unclassified items for potential manual classification
        for series in unclassified:
            words = set(series.replace('_', ' ').replace('-', ' ').lower().split())
            
            # Check for potential deposit indicators
            if any(word in words for word in ['fund', 'account', 'earnings', 'proceeds']):
                report['unclassified_analysis']['potential_deposits'].append(series)
            # Check for potential withdrawal indicators  
            elif any(word in words for word in ['support', 'assistance', 'relief', 'corp', 'commission']):
                report['unclassified_analysis']['potential_withdrawals'].append(series)
            else:
                report['unclassified_analysis']['truly_ambiguous'].append(series)
        
        return report
    
    def calculate_daily_cash_flows(self) -> pd.DataFrame:
        """
        Calculate daily aggregated cash flows from forecasts.
        
        Returns:
            pd.DataFrame: Daily cash flow summary
        """
        print("Calculating daily cash flows...")
        
        deposits, withdrawals, unclassified = self.categorize_cash_flows()
        
        # Generate detailed classification report
        classification_report = self.generate_classification_report(deposits, withdrawals, unclassified)
        
        # Print detailed breakdown for transparency
        print(f"\n📋 Detailed Classification Breakdown:")
        print(f"   Tax-related deposits: {len(classification_report['deposit_categories']['tax_related'])}")
        print(f"   Loan repayments: {len(classification_report['deposit_categories']['loan_repayments'])}")
        print(f"   Benefits payments: {len(classification_report['withdrawal_categories']['benefits_payments'])}")
        print(f"   Tax refunds: {len(classification_report['withdrawal_categories']['tax_refunds'])}")
        print(f"   Department programs: {len(classification_report['withdrawal_categories']['department_programs'])}")
        
        if classification_report['unclassified_analysis']['potential_deposits']:
            print(f"   Potential deposits in unclassified: {len(classification_report['unclassified_analysis']['potential_deposits'])}")
        if classification_report['unclassified_analysis']['potential_withdrawals']:
            print(f"   Potential withdrawals in unclassified: {len(classification_report['unclassified_analysis']['potential_withdrawals'])}")
        
        # Store classification report for later analysis
        self.classification_report = classification_report
        
        # Pivot forecast data to have dates as index and series as columns
        pivot_df = self.forecast_df.pivot(
            index='forecast_date', 
            columns='series_name', 
            values='forecast_value'
        ).fillna(0)
        
        # Calculate daily totals (convert from millions to actual dollars)
        daily_deposits = pivot_df[deposits].sum(axis=1) * 1_000_000
        daily_withdrawals = pivot_df[withdrawals].sum(axis=1) * 1_000_000
        
        # Add unclassified to withdrawals (conservative approach)
        if unclassified:
            daily_unclassified = pivot_df[unclassified].sum(axis=1) * 1_000_000
            daily_withdrawals += daily_unclassified
        
        # Calculate net cash flow (positive = surplus, negative = deficit)
        # Note: withdrawals are already negative in our corrected data, so we add them
        daily_net_cash_flow = daily_deposits + daily_withdrawals
        
        # Create summary DataFrame
        self.daily_cash_flows = pd.DataFrame({
            'date': daily_deposits.index,
            'total_deposits': daily_deposits.values,
            'total_withdrawals': daily_withdrawals.values,
            'net_cash_flow': daily_net_cash_flow.values
        })
        
        # Calculate statistics
        avg_deposits = self.daily_cash_flows['total_deposits'].mean()
        avg_withdrawals = self.daily_cash_flows['total_withdrawals'].mean()
        avg_net_flow = self.daily_cash_flows['net_cash_flow'].mean()
        
        print(f"💰 Daily Cash Flow Summary:")
        print(f"   Average daily deposits: ${avg_deposits/1e6:,.0f} million")
        print(f"   Average daily withdrawals: ${avg_withdrawals/1e6:,.0f} million")
        print(f"   Average daily net flow: ${avg_net_flow/1e6:,.0f} million")
        
        if avg_net_flow < 0:
            print(f"   ⚠️  Average daily deficit detected")
        
        return self.daily_cash_flows
    
    def run_simulation(self, debt_calendar=None) -> pd.DataFrame:
        """
        Run the enhanced TGA balance simulation with automatic debt issuance logic.
        
        This is the core simulation loop that integrates:
        - Operational cash flows (from SARIMA forecasts)
        - Known debt service payments (from debt calendar)
        - Automatic debt issuance to maintain minimum cash levels
        - X-Date identification when debt ceiling is breached
        
        Args:
            debt_calendar: DebtEventsCalendar instance for scheduled debt payments
            
        Returns:
            pd.DataFrame: Complete simulation results with debt issuance tracking
        """
        print("🔄 Running enhanced TGA simulation with automatic debt issuance...")
        
        # Prerequisites validation
        if self.starting_balance is None:
            self.get_starting_balance()
        
        if self.daily_cash_flows is None:
            self.calculate_daily_cash_flows()
        
        # Initialize simulation state
        simulation_results = []
        
        # Current state (in actual dollars, not millions)
        current_tga_balance = self.starting_balance
        current_debt_total = self._to_internal_units(self.config.CURRENT_PUBLIC_DEBT)  # Convert to internal dollars
        
        # Thresholds (convert to internal dollars with clear documentation)
        minimum_cash_balance = self._to_internal_units(self.config.MINIMUM_CASH_BALANCE)  # Operational minimum - triggers automatic debt issuance
        debt_ceiling_limit = self._to_internal_units(self.config.DEBT_CEILING_LIMIT)      # Legal debt limit - cannot be exceeded
        x_date_threshold = self._to_internal_units(self.config.get_x_date_threshold())    # Political/market warning threshold (typically lower than minimum_cash)
        
        # Simulation tracking
        x_date_identified = False
        x_date = None
        total_new_debt_issued = 0
        debt_issuance_events = []
        
        print(f"📊 Initial State:")
        print(f"   TGA Balance: ${current_tga_balance/1e6:,.0f} million")
        print(f"   Public Debt: ${current_debt_total/1e6:,.0f} million")
        print(f"   Debt Ceiling: ${debt_ceiling_limit/1e6:,.0f} million")
        print(f"   Available Headroom: ${(debt_ceiling_limit - current_debt_total)/1e6:,.0f} million")
        print(f"   Minimum Cash Threshold: ${minimum_cash_balance/1e6:,.0f} million")
        
        # Main simulation loop - process each day
        for idx, row in self.daily_cash_flows.iterrows():
            current_date = row['date']
            
            # Step 1: Get daily inputs
            operational_net_flow = row['net_cash_flow']  # Already in dollars
            
            # Get scheduled debt service from calendar if available
            scheduled_redemption = 0
            scheduled_interest_payment = 0
            
            if debt_calendar is not None:
                try:
                    # Ensure date consistency for debt calendar query
                    query_date = current_date
                    if hasattr(current_date, 'date'):
                        query_date = current_date.date()
                    elif hasattr(current_date, 'to_pydatetime'):
                        query_date = current_date.to_pydatetime().date()
                    
                    debt_service = debt_calendar.get_scheduled_debt_service(query_date)
                    # Debt calendar returns values in millions, convert to internal dollars
                    scheduled_redemption = self._to_internal_units(debt_service.get('principal_due', 0))
                    scheduled_interest_payment = self._to_internal_units(debt_service.get('interest_due', 0))
                except Exception as e:
                    # If debt calendar query fails, continue with zeros
                    # Could add logging here in production
                    pass
            
            total_debt_service = scheduled_redemption + scheduled_interest_payment
            
            # Sanity check debt service amounts
            if total_debt_service > 0:
                self._sanity_check_value(scheduled_redemption, f"scheduled_redemption on {current_date.date()}", 5e12)  # $5T max
                self._sanity_check_value(scheduled_interest_payment, f"scheduled_interest on {current_date.date()}", 1e12)  # $1T max
            
            # Step 2: Calculate "no new debt" provisional state
            provisional_tga = current_tga_balance + operational_net_flow - total_debt_service
            provisional_debt = current_debt_total - scheduled_redemption  # Principal payments reduce debt
            
            # Step 3: Automatic debt issuance trigger (only if X-Date not reached)
            new_debt_issuance = 0
            debt_issuance_triggered = False
            
            # Check if we can still issue debt (haven't reached X-Date)
            potential_debt_after_issuance = provisional_debt + (minimum_cash_balance - provisional_tga if provisional_tga < minimum_cash_balance else 0)
            
            if not x_date_identified and provisional_tga < minimum_cash_balance and potential_debt_after_issuance <= debt_ceiling_limit:
                # Cash shortfall detected and we can still issue debt - trigger automatic debt issuance
                cash_shortfall = minimum_cash_balance - provisional_tga
                
                # Limit debt issuance to stay within debt ceiling
                max_available_debt_capacity = debt_ceiling_limit - provisional_debt
                new_debt_issuance = min(cash_shortfall, max_available_debt_capacity)
                debt_issuance_triggered = True
                
                # Record debt issuance event
                debt_issuance_events.append({
                    'date': current_date,
                    'reason': 'minimum_cash_maintenance',
                    'cash_shortfall': cash_shortfall,
                    'debt_issued': new_debt_issuance,
                    'debt_capacity_used': max_available_debt_capacity,
                    'provisional_tga': provisional_tga,
                    'final_tga': provisional_tga + new_debt_issuance
                })
                
                total_new_debt_issued += new_debt_issuance
            elif not x_date_identified and provisional_tga < minimum_cash_balance:
                # Cash shortfall but cannot issue more debt - this triggers X-Date
                print(f"🚨 DEBT CEILING CONSTRAINT: Cannot issue ${(minimum_cash_balance - provisional_tga)/1e6:,.0f}M needed")
                print(f"   Available capacity: ${(debt_ceiling_limit - provisional_debt)/1e6:,.0f}M")
                print(f"   Required amount: ${(minimum_cash_balance - provisional_tga)/1e6:,.0f}M")
            
            # Step 4: Calculate final daily state
            final_tga_balance = provisional_tga + new_debt_issuance
            final_debt_total = provisional_debt + new_debt_issuance
            
            # Step 5: X-Date check (debt ceiling breach or inability to meet cash needs)
            debt_ceiling_breached = final_debt_total > debt_ceiling_limit
            cash_needs_unmet = provisional_tga < minimum_cash_balance and new_debt_issuance == 0 and not debt_issuance_triggered
            
            if not x_date_identified and (debt_ceiling_breached or cash_needs_unmet):
                x_date = current_date
                x_date_identified = True
                print(f"🚨 X-DATE IDENTIFIED: {x_date.date()}")
                print(f"   Final debt: ${final_debt_total/1e6:,.0f} million")
                print(f"   Debt ceiling: ${debt_ceiling_limit/1e6:,.0f} million")
                
                if debt_ceiling_breached:
                    print(f"   Reason: Debt ceiling breach by ${(final_debt_total - debt_ceiling_limit)/1e6:,.0f} million")
                else:
                    print(f"   Reason: Cannot meet minimum cash needs (${minimum_cash_balance/1e6:,.0f}M required)")
                    print(f"   TGA balance would fall to: ${final_tga_balance/1e6:,.0f} million")
                
                print(f"🚨 SIMULATION ENTERING POST-X-DATE MODE")
                print(f"   No additional debt can be issued beyond this point")
            
            # Step 6: Post-X-Date behavior adjustments (before storing results)
            post_x_date_mode = False
            if x_date_identified:
                # After X-Date, no new debt can be issued
                # TGA balance will decline based only on operational flows and debt service
                # This represents the "extraordinary measures exhausted" scenario
                
                if current_date == x_date:
                    # On X-Date itself, debt reaches ceiling but no further issuance
                    final_debt_total = min(final_debt_total, debt_ceiling_limit)
                    # TGA balance reflects the shortfall (cannot be supported by new debt)
                    final_tga_balance = provisional_tga
                else:
                    # After X-Date, debt stays at ceiling, no new issuance possible
                    final_debt_total = debt_ceiling_limit
                    final_tga_balance = provisional_tga  # No debt support available
                    new_debt_issuance = 0
                    debt_issuance_triggered = False
                
                post_x_date_mode = True
            
            # Step 7: Store daily results (with corrected post-X-Date values)
            simulation_results.append({
                'date': current_date,
                'opening_tga_balance': self._to_display_units(current_tga_balance),  # Convert to millions for storage
                'opening_debt_total': self._to_display_units(current_debt_total),
                
                # Daily flows (convert to millions for storage)
                'operational_net_flow': self._to_display_units(operational_net_flow),
                'scheduled_redemption': self._to_display_units(scheduled_redemption),
                'scheduled_interest': self._to_display_units(scheduled_interest_payment),
                'total_debt_service': self._to_display_units(total_debt_service),
                
                # Provisional state (before debt issuance)
                'provisional_tga': self._to_display_units(provisional_tga),
                'provisional_debt': self._to_display_units(provisional_debt),
                
                # Debt issuance
                'debt_issuance_triggered': debt_issuance_triggered,
                'new_debt_issued': self._to_display_units(new_debt_issuance),
                'cash_shortfall': self._to_display_units(minimum_cash_balance - provisional_tga) if debt_issuance_triggered else 0,
                
                # Final state (corrected for post-X-Date)
                'closing_tga_balance': self._to_display_units(final_tga_balance),
                'closing_debt_total': self._to_display_units(final_debt_total),
                
                # Status indicators
                'below_minimum_cash': final_tga_balance < minimum_cash_balance,
                'below_x_date_threshold': final_tga_balance < x_date_threshold,
                'debt_ceiling_breached': final_debt_total > debt_ceiling_limit,
                'debt_utilization': (final_debt_total / debt_ceiling_limit) * 100,
                'available_headroom': self._to_display_units(debt_ceiling_limit - final_debt_total),
                
                # Post-X-Date indicators
                'post_x_date_mode': post_x_date_mode,                    # Boolean: True if we're in post-X-Date simulation mode
                'debt_ceiling_reached': final_debt_total >= debt_ceiling_limit,  # Boolean: True if debt has reached ceiling
                'extraordinary_measures_available': not post_x_date_mode, # Boolean: True if extraordinary measures still available
                
                # Metadata
                'days_from_start': (current_date - self.daily_cash_flows['date'].iloc[0]).days,
                'is_x_date': current_date == x_date if x_date else False
            })
            
            # Step 8: Update state for next iteration
            current_tga_balance = final_tga_balance
            current_debt_total = final_debt_total
            
            # Optional: Early termination if X-Date found and TGA goes significantly negative
            if x_date_identified and final_tga_balance < -100_000_000_000:  # -$100B threshold
                print(f"⏹️ Simulation terminated: TGA balance critically low (${final_tga_balance/1e6:,.0f}M)")
                print(f"   Government operations would cease at this point")
                break
        
        # Create simulation results DataFrame
        self.tga_simulation = pd.DataFrame(simulation_results)
        
        # Calculate comprehensive summary statistics
        self._calculate_enhanced_summary_statistics(
            total_new_debt_issued, debt_issuance_events, x_date
        )
        
        return self.tga_simulation
    
    def _calculate_enhanced_summary_statistics(self, total_new_debt_issued: float, 
                                             debt_issuance_events: list, x_date) -> None:
        """
        Calculate comprehensive summary statistics for the enhanced simulation.
        
        Args:
            total_new_debt_issued (float): Total amount of new debt issued during simulation
            debt_issuance_events (list): List of debt issuance events
            x_date: X-Date if identified, None otherwise
        """
        if self.tga_simulation is None or len(self.tga_simulation) == 0:
            return
        
        # Basic balance statistics (using consistent unit conversion)
        starting_balance = self.starting_balance
        final_balance = self._to_internal_units(self.tga_simulation['closing_tga_balance'].iloc[-1])
        balance_change = final_balance - starting_balance
        
        min_balance = self._to_internal_units(self.tga_simulation['closing_tga_balance'].min())
        min_balance_date = self.tga_simulation.loc[
            self.tga_simulation['closing_tga_balance'].idxmin(), 'date'
        ]
        
        max_balance = self._to_internal_units(self.tga_simulation['closing_tga_balance'].max())
        max_balance_date = self.tga_simulation.loc[
            self.tga_simulation['closing_tga_balance'].idxmax(), 'date'
        ]
        
        # Debt statistics (using consistent unit conversion)
        starting_debt = self._to_internal_units(self.config.CURRENT_PUBLIC_DEBT)
        final_debt = self._to_internal_units(self.tga_simulation['closing_debt_total'].iloc[-1])
        debt_change = final_debt - starting_debt
        
        max_debt = self._to_internal_units(self.tga_simulation['closing_debt_total'].max())
        max_debt_date = self.tga_simulation.loc[
            self.tga_simulation['closing_debt_total'].idxmax(), 'date'
        ]
        
        # Debt issuance statistics
        debt_issuance_days = self.tga_simulation['debt_issuance_triggered'].sum()
        total_debt_issued_millions = total_new_debt_issued / 1_000_000
        
        # Cash flow statistics
        total_operational_flow = self.tga_simulation['operational_net_flow'].sum() * 1_000_000
        total_debt_service = self.tga_simulation['total_debt_service'].sum() * 1_000_000
        total_redemptions = self.tga_simulation['scheduled_redemption'].sum() * 1_000_000
        total_interest_payments = self.tga_simulation['scheduled_interest'].sum() * 1_000_000
        
        # Risk indicators
        days_below_minimum = self.tga_simulation['below_minimum_cash'].sum()
        days_below_x_threshold = self.tga_simulation['below_x_date_threshold'].sum()
        debt_ceiling_breaches = self.tga_simulation['debt_ceiling_breached'].sum()
        
        # X-Date analysis with proper datetime handling
        # Ensure x_date is proper datetime type, not numpy.datetime64
        if x_date is not None:
            if hasattr(x_date, 'to_pydatetime'):  # pandas Timestamp
                x_date = x_date.to_pydatetime().date()
            elif hasattr(x_date, 'date'):  # datetime
                x_date = x_date.date()
            # If it's already date, keep as is
        
        x_date_analysis = {
            'x_date_identified': x_date is not None,
            'x_date': x_date,
            'days_to_x_date': None,
            'debt_at_x_date': None,
            'tga_at_x_date': None
        }
        
        if x_date:
            start_date = self.tga_simulation['date'].iloc[0]
            # Ensure start_date is also proper date for comparison
            if hasattr(start_date, 'date'):
                start_date = start_date.date()
                
            x_date_analysis['days_to_x_date'] = (x_date - start_date).days
            
            # Convert x_date back to datetime for pandas comparison
            x_date_datetime = pd.to_datetime(x_date)
            x_date_row = self.tga_simulation[self.tga_simulation['date'] == x_date_datetime]
            if not x_date_row.empty:
                x_date_analysis['debt_at_x_date'] = self._to_internal_units(x_date_row['closing_debt_total'].iloc[0])
                x_date_analysis['tga_at_x_date'] = self._to_internal_units(x_date_row['closing_tga_balance'].iloc[0])
        
        # Print comprehensive summary
        print(f"\n📊 ENHANCED SIMULATION SUMMARY:")
        print(f"=" * 60)
        
        print(f"💰 TGA Balance Analysis:")
        print(f"   Starting balance: ${starting_balance/1e6:,.0f} million")
        print(f"   Final balance: ${final_balance/1e6:,.0f} million")
        print(f"   Balance change: ${balance_change/1e6:,.0f} million")
        print(f"   Minimum balance: ${min_balance/1e6:,.0f} million on {min_balance_date.date()}")
        print(f"   Maximum balance: ${max_balance/1e6:,.0f} million on {max_balance_date.date()}")
        
        print(f"\n🏛️ Public Debt Analysis:")
        print(f"   Starting debt: ${starting_debt/1e6:,.0f} million")
        print(f"   Final debt: ${final_debt/1e6:,.0f} million")
        print(f"   Debt change: ${debt_change/1e6:,.0f} million")
        print(f"   Maximum debt: ${max_debt/1e6:,.0f} million on {max_debt_date.date()}")
        print(f"   Debt ceiling: ${self.config.DEBT_CEILING_LIMIT * 1_000:,.0f} million")
        print(f"   Final utilization: {(final_debt/(self.config.DEBT_CEILING_LIMIT * 1_000_000))*100:.1f}%")
        
        print(f"\n💳 Automatic Debt Issuance:")
        print(f"   Total new debt issued: ${total_debt_issued_millions:,.0f} million")
        print(f"   Debt issuance events: {len(debt_issuance_events)}")
        print(f"   Days with debt issuance: {debt_issuance_days}")
        if len(debt_issuance_events) > 0:
            avg_issuance = total_debt_issued_millions / len(debt_issuance_events)
            print(f"   Average issuance per event: ${avg_issuance:,.0f} million")
        
        print(f"\n💸 Cash Flow Summary:")
        print(f"   Total operational flow: ${total_operational_flow/1e6:,.0f} million")
        print(f"   Total debt service: ${total_debt_service/1e6:,.0f} million")
        print(f"   Principal redemptions: ${total_redemptions/1e6:,.0f} million")
        print(f"   Interest payments: ${total_interest_payments/1e6:,.0f} million")
        
        print(f"\n🚨 Risk Indicators:")
        print(f"   Days below minimum cash: {days_below_minimum}")
        print(f"   Days below X-Date threshold: {days_below_x_threshold}")
        print(f"   Debt ceiling breaches: {debt_ceiling_breaches}")
        
        if x_date_analysis['x_date_identified']:
            print(f"\n🚨 X-DATE ANALYSIS:")
            print(f"   X-Date: {x_date}")
            print(f"   Days to X-Date: {x_date_analysis['days_to_x_date']}")
            print(f"   Debt at X-Date: ${x_date_analysis['debt_at_x_date']/1e6:,.0f} million")
            print(f"   TGA at X-Date: ${x_date_analysis['tga_at_x_date']/1e6:,.0f} million")
        else:
            print(f"\n✅ No X-Date within forecast horizon")
        
        # Store comprehensive results
        self.simulation_results = {
            # Balance metrics
            'starting_balance': starting_balance,
            'final_balance': final_balance,
            'balance_change': balance_change,
            'minimum_balance': min_balance,
            'minimum_balance_date': min_balance_date,
            'maximum_balance': max_balance,
            'maximum_balance_date': max_balance_date,
            
            # Debt metrics
            'starting_debt': starting_debt,
            'final_debt': final_debt,
            'debt_change': debt_change,
            'maximum_debt': max_debt,
            'maximum_debt_date': max_debt_date,
            'final_debt_utilization': (final_debt/(self.config.DEBT_CEILING_LIMIT * 1_000_000))*100,
            
            # Debt issuance metrics
            'total_new_debt_issued': total_new_debt_issued,
            'debt_issuance_events_count': len(debt_issuance_events),
            'debt_issuance_days': debt_issuance_days,
            'debt_issuance_events': debt_issuance_events,
            
            # Cash flow metrics
            'total_operational_flow': total_operational_flow,
            'total_debt_service': total_debt_service,
            'total_redemptions': total_redemptions,
            'total_interest_payments': total_interest_payments,
            
            # Risk metrics
            'days_below_minimum': days_below_minimum,
            'days_below_x_threshold': days_below_x_threshold,
            'debt_ceiling_breaches': debt_ceiling_breaches,
            
            # X-Date analysis
            'x_date_analysis': x_date_analysis,
            
            # Raw data
            'simulation_data': self.tga_simulation
        }
    
    def analyze_x_date_scenarios(self) -> Dict[str, Any]:
        """
        Analyze X-Date under different scenarios.
        
        Returns:
            Dict[str, Any]: X-Date analysis results
        """
        print("Analyzing X-Date scenarios...")
        
        if self.tga_simulation is None:
            print("❌ Error: Run simulation first")
            return {}
        
        analysis = {
            'base_case': {},
            'scenarios': {},
            'sensitivity': {}
        }
        
        # Base case analysis
        x_date_rows = self.tga_simulation[
            self.tga_simulation['below_x_date_threshold'] == True
        ]
        
        if len(x_date_rows) > 0:
            first_x_date = x_date_rows.iloc[0]['date']
            days_to_x_date = (first_x_date - pd.to_datetime(self.config.START_DATE)).days
            
            analysis['base_case'] = {
                'x_date': first_x_date,
                'days_to_x_date': days_to_x_date,
                'balance_at_x_date': x_date_rows.iloc[0]['closing_tga_balance'],
                'warning_level': self._get_warning_level(days_to_x_date)
            }
        else:
            analysis['base_case'] = {
                'x_date': None,
                'days_to_x_date': None,
                'balance_at_x_date': None,
                'warning_level': 'green'
            }
        
        # Scenario analysis would go here
        # (Revenue shocks, spending changes, etc.)
        
        self.x_date_analysis = analysis
        return analysis
    
    def _get_warning_level(self, days_to_x_date: int) -> str:
        """
        Determine warning level based on days to X-Date.
        
        Args:
            days_to_x_date (int): Days until X-Date
            
        Returns:
            str: Warning level ('red', 'orange', 'yellow', 'green')
        """
        if days_to_x_date is None:
            return 'green'
        elif days_to_x_date <= self.config.X_DATE_WARNING_DAYS['red']:
            return 'red'
        elif days_to_x_date <= self.config.X_DATE_WARNING_DAYS['orange']:
            return 'orange'
        elif days_to_x_date <= self.config.X_DATE_WARNING_DAYS['yellow']:
            return 'yellow'
        else:
            return 'green'
    
    def save_results(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Save simulation results to files.
        
        Args:
            output_dir (str, optional): Output directory path
            
        Returns:
            Dict[str, str]: Dictionary of saved file paths
        """
        output_dir = output_dir or self.config.OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        try:
            # Save TGA simulation results
            tga_file = os.path.join(output_dir, f"tga_simulation_{timestamp}.csv")
            self.tga_simulation.to_csv(tga_file, index=False)
            saved_files['tga_simulation'] = tga_file
            
            # Save daily cash flows
            cash_flow_file = os.path.join(output_dir, f"daily_cash_flows_{timestamp}.csv")
            self.daily_cash_flows.to_csv(cash_flow_file, index=False)
            saved_files['daily_cash_flows'] = cash_flow_file
            
            # Save summary and analysis
            summary = {
                "simulation_timestamp": timestamp,
                "configuration": {
                    "start_date": self.config.START_DATE.strftime('%Y-%m-%d'),
                    "forecast_horizon": self.config.FORECAST_HORIZON,
                    "starting_balance_million": self.config.CURRENT_TGA_BALANCE,
                    "minimum_cash_threshold_million": self.config.MINIMUM_CASH_BALANCE,
                    "x_date_threshold_million": self.config.get_x_date_threshold()
                },
                "results": {
                    "starting_balance_million": self.starting_balance / 1e6,
                    "final_balance_million": self.simulation_results['final_balance'] / 1e6,
                    "balance_change_million": self.simulation_results['balance_change'] / 1e6,
                    "minimum_balance_million": self.simulation_results['minimum_balance'] / 1e6,
                    "minimum_balance_date": self.simulation_results['minimum_balance_date'].strftime('%Y-%m-%d'),
                    "x_date": self.simulation_results['x_date_analysis']['x_date'].strftime('%Y-%m-%d') if self.simulation_results['x_date_analysis']['x_date'] else None,
                    "total_forecast_days": len(self.tga_simulation)
                },
                "cash_flows": {
                    "average_daily_deposits_million": self.daily_cash_flows['total_deposits'].mean() / 1e6,
                    "average_daily_withdrawals_million": self.daily_cash_flows['total_withdrawals'].mean() / 1e6,
                    "average_daily_net_flow_million": self.daily_cash_flows['net_cash_flow'].mean() / 1e6
                }
            }
            
            summary_file = os.path.join(output_dir, f"tga_simulation_summary_{timestamp}.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            saved_files['summary'] = summary_file
            
            print(f"📁 Results saved:")
            for file_type, file_path in saved_files.items():
                print(f"   {file_type}: {os.path.basename(file_path)}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
        
        return saved_files
    
    def create_visualizations(self, output_dir: Optional[str] = None) -> str:
        """
        Create comprehensive visualizations of simulation results.
        
        Args:
            output_dir (str, optional): Output directory path
            
        Returns:
            str: Path to saved visualization file
        """
        output_dir = output_dir or self.config.OUTPUT_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Treasury General Account (TGA) Balance Simulation', fontsize=16, fontweight='bold')
        
        # Convert to millions for display
        tga_balance_millions = self.tga_simulation['closing_tga_balance']
        deposits_millions = self.daily_cash_flows['total_deposits'] / 1e6
        withdrawals_millions = self.daily_cash_flows['total_withdrawals'] / 1e6
        net_flow_millions = self.daily_cash_flows['net_cash_flow'] / 1e6
        
        # 1. TGA Balance Over Time
        axes[0, 0].plot(self.tga_simulation['date'], tga_balance_millions, 
                       linewidth=2, color='navy', label='TGA Balance')
        
        # Add threshold lines
        axes[0, 0].axhline(y=self.config.MINIMUM_CASH_BALANCE, color='red', 
                          linestyle='--', alpha=0.7, label='Minimum Cash')
        axes[0, 0].axhline(y=self.config.get_x_date_threshold(), color='orange', 
                          linestyle='--', alpha=0.7, label='X-Date Threshold')
        
        axes[0, 0].set_title('TGA Balance Projection')
        axes[0, 0].set_ylabel('Balance ($ Billions)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Daily Cash Flows
        axes[0, 1].plot(self.daily_cash_flows['date'], deposits_millions, 
                       label='Deposits', color='green', alpha=0.8, linewidth=1)
        axes[0, 1].plot(self.daily_cash_flows['date'], withdrawals_millions, 
                       label='Withdrawals', color='red', alpha=0.8, linewidth=1)
        axes[0, 1].set_title('Daily Cash Flows')
        axes[0, 1].set_ylabel('Amount ($ Billions)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Net Cash Flow
        axes[1, 0].plot(self.daily_cash_flows['date'], net_flow_millions, 
                       linewidth=1.5, color='purple')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 0].fill_between(self.daily_cash_flows['date'], net_flow_millions, 0, 
                               where=(net_flow_millions >= 0), alpha=0.3, color='green', label='Surplus')
        axes[1, 0].fill_between(self.daily_cash_flows['date'], net_flow_millions, 0, 
                               where=(net_flow_millions < 0), alpha=0.3, color='red', label='Deficit')
        axes[1, 0].set_title('Daily Net Cash Flow')
        axes[1, 0].set_ylabel('Net Flow ($ Billions)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Balance Distribution and Statistics
        axes[1, 1].hist(tga_balance_millions, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        axes[1, 1].axvline(x=self.config.MINIMUM_CASH_BALANCE, color='red', 
                          linestyle='--', label='Minimum Cash')
        axes[1, 1].axvline(x=tga_balance_millions.mean(), color='blue', 
                          linestyle='-', label='Average Balance')
        axes[1, 1].set_title('TGA Balance Distribution')
        axes[1, 1].set_xlabel('Balance ($ Billions)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(output_dir, f"tga_simulation_analysis_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Visualization saved: {os.path.basename(plot_file)}")
        return plot_file


def main():
    """
    Main function to run TGA balance simulation with configuration.
    """
    print("="*80)
    print("TREASURY GENERAL ACCOUNT (TGA) BALANCE SIMULATION")
    print("="*80)
    
    # Load configuration
    config = load_config()
    
    # Print configuration summary
    config.print_summary()
    
    # Validate configuration
    validation = config.validate_configuration()
    if not validation['valid']:
        print("❌ Configuration validation failed:")
        for error in validation['errors']:
            print(f"   {error}")
        return
    
    if validation['warnings']:
        print("⚠️  Configuration warnings:")
        for warning in validation['warnings']:
            print(f"   {warning}")
    
    try:
        # Initialize simulator
        simulator = TGABalanceSimulator(config)
        
        # Load forecast data
        forecast_data = simulator.load_forecast_data()
        if forecast_data is None:
            print("❌ Failed to load forecast data. Exiting.")
            return
        
        # Get starting balance
        simulator.get_starting_balance()
        
        # Calculate daily cash flows
        simulator.calculate_daily_cash_flows()
        
        # Run simulation
        simulator.run_simulation()
        
        # Analyze X-Date scenarios
        simulator.analyze_x_date_scenarios()
        
        # Save results
        simulator.save_results()
        
        # Create visualizations
        simulator.create_visualizations()
        
        print("\n" + "="*80)
        print("🎉 TGA SIMULATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
