"""
Enhanced Treasury Data Collector - Extended DTS Data Collection (Fixed)
- Fix TGA balance field (use close_today_bal for Closing Balance)
- Deduplicate category mapping keys (e.g., GSA)
- Exclude subtotal/transfer lines before categorization to avoid double-counting
- Unify dtype conversions (dates and numeric)
- Add unmapped-category diagnostics
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
import time
import logging
from typing import Dict, Optional, Any, List
import numpy as np
try:
    from .enhanced_field_mapper_complete import CompleteTreasuryFieldMapper as TreasuryFieldMapper
except ImportError:
    from .enhanced_field_mapper import TreasuryFieldMapper


class EnhancedTreasuryCollector:
    """Enhanced Treasury Data Collector - Contains detailed DTS categorized data"""
    
    def __init__(self, data_dir: str = "./data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
        self.headers = {
            'User-Agent': 'Enhanced-Treasury-Collector/2.1 (Research)',
            'Accept': 'application/json'
        }
        self.request_delay = 0.5
        
        # DTS endpoints
        self.detailed_endpoints = {
            'operating_cash_balance': 'v1/accounting/dts/operating_cash_balance',
            'deposits_withdrawals_operating_cash': 'v1/accounting/dts/deposits_withdrawals_operating_cash',
            'public_debt_transactions': 'v1/accounting/dts/public_debt_transactions',
            'adjustment_public_debt_transactions_cash_basis': 'v1/accounting/dts/adjustment_public_debt_transactions_cash_basis',
            'debt_subject_to_limit': 'v1/accounting/dts/debt_subject_to_limit',
            'inter_agency_tax_transfers': 'v1/accounting/dts/inter_agency_tax_transfers',
            'income_tax_refunds_issued': 'v1/accounting/dts/income_tax_refunds_issued',
            'federal_tax_deposits': 'v1/accounting/dts/federal_tax_deposits',
            'short_term_cash_investments': 'v1/accounting/dts/short_term_cash_investments'
        }
        
        self.category_mapping = self._get_transaction_categories()
        self.field_mapper = TreasuryFieldMapper()
    
    # ---------- Category Mapping ----------
    def _get_transaction_categories(self) -> Dict[str, str]:
        """Transaction category -> high-level group (deduplicated)"""
        mapping = {
            # Social Security & Retirement
            'SSA - Benefits Payments': 'Social Security & Retirement',
            'SSA - Supplemental Security Income': 'Social Security & Retirement',
            'Social Security Admin (SSA) - misc': 'Social Security & Retirement',
            'RRB - Benefit Payments': 'Social Security & Retirement',
            'Railroad Retirement Board (RRB) - misc': 'Social Security & Retirement',
            'RRB - Natl Railroad Retirement Inv Trust': 'Social Security & Retirement',
            'RRB - Unemployment Insurance': 'Social Security & Retirement',
            'OPM - Civil Serv Retirement & Disability': 'Social Security & Retirement',
            'OPM - Federal Employee Insurance Payment': 'Social Security & Retirement',
            'Federal Retirement Thrift Savings Plan': 'Social Security & Retirement',
            'DoD - Military Retirement': 'Social Security & Retirement',

            # Healthcare & Medicare/Medicaid
            'HHS - Grants to States for Medicaid': 'Healthcare & Medicare/Medicaid',
            'HHS - Federal Hospital Insr Trust Fund': 'Healthcare & Medicare/Medicaid',
            'HHS - Federal Supple Med Insr Trust Fund': 'Healthcare & Medicare/Medicaid',
            'HHS - Medicare Prescription Drugs': 'Healthcare & Medicare/Medicaid',
            'HHS - Marketplace Payments': 'Healthcare & Medicare/Medicaid',
            'HHS - Medicare Premiums': 'Healthcare & Medicare/Medicaid',
            'HHS - Misc': 'Healthcare & Medicare/Medicaid',
            'HHS - Other Public Health Services': 'Healthcare & Medicare/Medicaid',
            'HHS - Othr Cent Medicare & Medicaid Serv': 'Healthcare & Medicare/Medicaid',
            'HHS - Health Resources & Services Admin': 'Healthcare & Medicare/Medicaid',
            'HHS - Indian Health Service': 'Healthcare & Medicare/Medicaid',
            'HHS - Centers for Disease Control (CDC)': 'Healthcare & Medicare/Medicaid',
            'DoD - Health': 'Healthcare & Medicare/Medicaid',

            # Defense & Security
            'Dept of Defense (DoD)': 'Defense & Security',
            'Dept of Defense (DoD) - misc': 'Defense & Security',
            'DoD - Military Active Duty Pay': 'Defense & Security',
            'DHS - Customs & Border Protection (CBP)': 'Defense & Security',
            'DHS - Customs and Certain Excise Taxes': 'Defense & Security',
            'DHS - Fed Emergency Mgmt Agency (FEMA)': 'Defense & Security',
            'DHS - Transportation Security Admn (TSA)': 'Defense & Security',
            'IAP - Foreign Military Sales': 'Defense & Security',
            'US Army Corps of Engineers': 'Defense & Security',
            'Defense Vendor Payments (EFT)': 'Defense & Security',

            # Federal Salaries & Operating Expenses
            'Federal Salaries (EFT)': 'Federal Salaries & Ops',
            'Dept of Agriculture (USDA) - misc': 'Federal Salaries & Ops',
            'Dept of Commerce (DOC)': 'Federal Salaries & Ops',
            'Dept of Education (ED)': 'Federal Salaries & Ops',
            'Dept of Energy (DOE)': 'Federal Salaries & Ops',
            'Dept of Interior (DOI) - misc': 'Federal Salaries & Ops',
            'Dept of Labor (DOL) - misc': 'Federal Salaries & Ops',
            'Dept of State (DOS)': 'Federal Salaries & Ops',
            'Dept of Transportation (DOT)': 'Federal Salaries & Ops',
            'Dept of Transportation (DOT) - Misc': 'Federal Salaries & Ops',
            'Dept of Veterans Affairs (VA)': 'Federal Salaries & Ops',
            'VA - Benefits': 'Federal Salaries & Ops',
            'Environmental Protection Agency (EPA)': 'Federal Salaries & Ops',
            'General Services Administration (GSA)': 'Federal Salaries & Ops',  # keep here only
            'Independent Agencies - misc': 'Federal Salaries & Ops',
            'Judicial Branch - Courts': 'Federal Salaries & Ops',
            'Justice Department programs': 'Federal Salaries & Ops',
            'Dept of Justice (DOJ)': 'Federal Salaries & Ops',
            'Legislative Branch - misc': 'Federal Salaries & Ops',
            'Legislative Branch - Library of Congress': 'Federal Salaries & Ops',
            'NASA': 'Federal Salaries & Ops',
            'National Science Foundation (NSF)': 'Federal Salaries & Ops',
            'Securities and Exchange Commission (SEC)': 'Federal Salaries & Ops',
            'Small Business Administration (SBA)': 'Federal Salaries & Ops',
            'Postal Service': 'Federal Salaries & Ops',
            'United States Postal Service (USPS)': 'Federal Salaries & Ops',
            'Other Withdrawals': 'Federal Salaries & Ops',
            'Unclassified': 'Federal Salaries & Ops',

            # Housing & Community Support
            'Housing and Urban Development programs': 'Housing & Community',
            'Dept of Housing & Urban Dev (HUD) - misc': 'Housing & Community',
            'HUD - Federal Housing Admin (FHA)': 'Housing & Community',
            'Emergency Rental Assistance': 'Housing & Community',
            'TREAS - Troubled Asset Relief Pro (TARP)': 'Housing & Community',

            # Tax Refunds & Credits
            'IRS Tax Refunds Individual (EFT)': 'Tax Refunds & Credits',
            'IRS Tax Refunds Business (EFT)': 'Tax Refunds & Credits',
            'IRS - Economic Impact Payments (EFT)': 'Tax Refunds & Credits',
            'IRS - Advanced Child Tax Credit (EFT)': 'Tax Refunds & Credits',
            'Taxes - Business Tax Refunds (EFT)': 'Tax Refunds & Credits',
            'Taxes - Individual Tax Refunds (EFT)': 'Tax Refunds & Credits',
            'TREAS - IRS Refunds for Puerto Rico': 'Tax Refunds & Credits',

            # Agriculture & Food Support
            'USDA - Commodity Credit Corporation': 'Agriculture & Food',
            'USDA - Federal Crop Insurance Corp Fund': 'Agriculture & Food',
            'USDA - Loan Payments': 'Agriculture & Food',
            'USDA - Loan Repayments': 'Agriculture & Food',
            'USDA - Other Farm Service': 'Agriculture & Food',
            'USDA - Child Nutrition': 'Agriculture & Food',
            'USDA - Supp Nutrition Assist Prog (SNAP)': 'Agriculture & Food',
            'USDA - Supp Nutrition Assist Prog (WIC)': 'Agriculture & Food',

            # International Programs
            "Int'l Assistance Programs (IAP) - misc": 'International Programs',
            "IAP - Agency for Int'l Development (AID)": 'International Programs',
            "IAP - US Int'l Devlop Finance Corp (DFC)": 'International Programs',
            'IAP - Multilateral Assistance': 'International Programs',
            'International Monetary Fund (IMF)': 'International Programs',

            # Interest Payments
            'Interest on Treasury Securities': 'Interest Payments',

            # Financial & Special Programs
            'Federal Deposit Insurance Corp (FDIC)': 'Financial & Special',
            'Federal Communications Commission (FCC)': 'Financial & Special',
            'FCC - Universal Service Fund': 'Financial & Special',
            'Export-Import Bank': 'Financial & Special',
            'Farm Credit System Insurance Cor (FCSIC)': 'Financial & Special',
            'National Credit Union Admin (NCUA)': 'Financial & Special',
            'Federal Reserve Earnings': 'Financial & Special',
            'Emergency Capital Investment Program': 'Financial & Special',
            'ESF - Economic Recovery Programs': 'Financial & Special',
            'TREAS - Federal Financing Bank': 'Financial & Special',
            'TREAS - Comptroller of the Currency': 'Financial & Special',
            'TREAS - United States Mint': 'Financial & Special',
            'TREAS - Bureau of Engraving and Printing': 'Financial & Special',
            'TREAS - Claims Judgments and Relief Acts': 'Financial & Special',
            'TREAS - Pmt to Resolution Funding Corp': 'Financial & Special',
            'Public Debt Cash Redemp. (Table III-B)': 'Financial & Special',
            'Public Debt Cash Redemp. (Table IIIB)': 'Financial & Special',
            'Public Debt Cash Issues (Table III-B)': 'Financial & Special',
            'Public Debt Cash Issues (Table IIIB)': 'Financial & Special',
            'TREAS - GSE Proceeds': 'Financial & Special',

            # Other (non-subtotal items only)
            'Other Deposits': 'Other',
            'Unclassified - Deposits': 'Other',
        }
        return mapping

    def _get_subtotal_categories(self) -> List[str]:
        """Subtotal/transfer categories to exclude (avoid double-counting)"""
        return [
            'Sub-Total Deposits',
            'Sub-Total Withdrawals', 
            'Transfers to Depositaries',
            'Transfers from Depositaries',
            'Change in Balance of Uncollected Funds'
        ]
    
    def _filter_detail_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove subtotal/transfer lines (detail-only)"""
        if df.empty or 'transaction_catg' not in df.columns:
            return df
        subtotal_categories = self._get_subtotal_categories()
        return df[~df['transaction_catg'].isin(subtotal_categories)].copy()

    # ---------- Networking ----------
    def _make_paginated_request(self, endpoint: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        if params is None:
            params = {}
        params.setdefault('page[size]', 1000)
        all_data, page_number = [], 1
        
        while True:
            params["page[number]"] = page_number
            try:
                url = f"{self.base_url}/{endpoint}"
                response = requests.get(url, headers=self.headers, params=params, timeout=60)
                response.raise_for_status()
                data = response.json()
                records = data.get("data", [])
                if not records:
                    break
                all_data.extend(records)
                if data.get("links", {}).get("next"):
                    page_number += 1
                else:
                    break
                time.sleep(self.request_delay)
            except requests.exceptions.RequestException as e:
                logging.error(f"API request failed: {e}")
                break
        
        df = pd.DataFrame(all_data)
        # Standardize dtypes
        if 'record_date' in df.columns:
            df['record_date'] = pd.to_datetime(df['record_date'], errors='coerce')
        for col in df.columns:
            if ('amt' in col) or ('bal' in col):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    # ---------- Collection ----------
    def collect_detailed_cash_flows(self, start_date: str = None, end_date: str = None) -> Dict[str, pd.DataFrame]:
        logging.info("Starting detailed Treasury cash flow data collection...")
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        
        collected_data = {}
        for data_name, endpoint in self.detailed_endpoints.items():
            logging.info(f"Collecting {data_name} ...")
            try:
                params = {}
                if 'dts' in endpoint:  # DTS datasets have record_date
                    params['filter'] = f'record_date:gte:{start_date},record_date:lte:{end_date}'
                    params['sort'] = 'record_date'
                df = self._make_paginated_request(endpoint, params)

                if not df.empty:
                    # Save raw
                    filepath = self.data_dir / f"{data_name}.csv"
                    df.to_csv(filepath, index=False)
                    collected_data[data_name] = df
                    logging.info(f"{data_name}: {len(df)} rows saved")
                else:
                    logging.warning(f"{data_name}: No data")
            except Exception as e:
                logging.error(f"{data_name} data collection failed: {e}")
        return collected_data

    # ---------- Analysis ----------
    def analyze_tga_balance(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """TGA Closing Balance time series (use close_today_bal)"""
        if 'operating_cash_balance' not in data:
            return pd.DataFrame()
        df = data['operating_cash_balance'].copy()
        tga_df = df[df['account_type'] == 'Treasury General Account (TGA) Closing Balance'].copy()
        if tga_df.empty:
            return pd.DataFrame()
        tga_df = tga_df.sort_values('record_date')
        # FIX: use closing balance field
        if 'close_today_bal' in tga_df.columns:
            tga_df['tga_balance'] = pd.to_numeric(tga_df['close_today_bal'], errors='coerce')
        else:
            # fallback: if close_today_bal absent, keep previous behavior
            tga_df['tga_balance'] = pd.to_numeric(tga_df.get('open_today_bal'), errors='coerce')
        return tga_df[['record_date', 'tga_balance']].copy()

    def categorize_cash_flows(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Categorize TGA deposits/withdrawals (detail-only; excludes subtotals/transfers)"""
        result = {}
        if 'deposits_withdrawals_operating_cash' not in data:
            return result
        
        df = data['deposits_withdrawals_operating_cash'].copy()
        # TGA only
        tga = df[df['account_type'] == 'Treasury General Account (TGA)'].copy()
        if tga.empty:
            return result

        # Exclude subtotal/transfer BEFORE categorization
        tga = self._filter_detail_transactions(tga)

        # Split by type
        for txn_type, name in [('Deposits', 'deposits'), ('Withdrawals', 'withdrawals')]:
            sub = tga[tga['transaction_type'] == txn_type].copy()
            if sub.empty:
                continue
            sub['transaction_group'] = sub['transaction_catg'].map(self.category_mapping).fillna('Other')
            result[name] = sub

        # Diagnostics: unmapped categories (optional)
        unmapped = sorted(set(tga['transaction_catg']) - set(self.category_mapping.keys()))
        if unmapped:
            logging.info(f"Unmapped categories (showing up to 20): {unmapped[:20]}")
        return result

    def check_subtotal_presence(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Check how big subtotals/transfers are in raw D/W (for debugging)"""
        result = {
            'subtotal_categories_found': [],
            'subtotal_records_count': 0,
            'total_records_count': 0
        }
        if 'deposits_withdrawals_operating_cash' not in data:
            return result
        
        df = data['deposits_withdrawals_operating_cash']
        subtotal_categories = self._get_subtotal_categories()
        found_subtotals = df[df['transaction_catg'].isin(subtotal_categories)]
        if not found_subtotals.empty:
            result['subtotal_categories_found'] = found_subtotals['transaction_catg'].unique().tolist()
            result['subtotal_records_count'] = len(found_subtotals)
            if 'transaction_today_amt' in found_subtotals.columns:
                subtotal_impact = found_subtotals.groupby(['record_date', 'transaction_type'])['transaction_today_amt'].sum()
                impact_dict = {}
                for (date, txn_type), amount in subtotal_impact.head(10).items():
                    key = f"{pd.to_datetime(date).date()}_{txn_type}"
                    impact_dict[key] = float(amount)
                result['daily_subtotal_amounts'] = impact_dict
        result['total_records_count'] = len(df)
        return result

    def generate_daily_cash_flows_file(self, raw_data: Dict[str, pd.DataFrame]) -> bool:
        """Generate daily cash flows file for forecasting models"""
        if 'deposits_withdrawals_operating_cash' not in raw_data:
            logging.warning("No deposits/withdrawals data available for daily cash flows generation")
            return False
        
        df = raw_data['deposits_withdrawals_operating_cash']
        # Filter for TGA only
        tga_data = df[df['account_type'] == 'Treasury General Account (TGA)'].copy()
        
        if tga_data.empty:
            logging.warning("No TGA data found for daily cash flows generation")
            return False
        
        # Convert and clean data
        tga_data['record_date'] = pd.to_datetime(tga_data['record_date'])
        tga_data['transaction_today_amt'] = pd.to_numeric(tga_data['transaction_today_amt'], errors='coerce')
        tga_data = tga_data.dropna(subset=['transaction_today_amt'])
        
        # Create daily summary
        daily_summary = tga_data.groupby(['record_date', 'transaction_type'])['transaction_today_amt'].sum().reset_index()
        
        # Save to file
        output_file = self.data_dir / "daily_cash_flows_2023-06-29_to_2025-06-28.csv"
        daily_summary.to_csv(output_file, index=False)
        
        logging.info(f"Daily cash flows file generated: {len(daily_summary)} records saved")
        return True

    def collect_all_enhanced_data(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        logging.info("Starting enhanced Treasury data collection and analysis...")
        raw_data = self.collect_detailed_cash_flows(start_date, end_date)
        tga_balance = self.analyze_tga_balance(raw_data)
        categorized_flows = self.categorize_cash_flows(raw_data)
        subtotal_check = self.check_subtotal_presence(raw_data)
        
        # Generate daily cash flows file for forecasting models
        daily_flows_created = self.generate_daily_cash_flows_file(raw_data)
        
        summary = {
            'collection_timestamp': datetime.now().isoformat(),
            'date_range': {'start_date': start_date, 'end_date': end_date},
            'datasets_collected': list(raw_data.keys()),
            'tga_balance_records': int(len(tga_balance)) if isinstance(tga_balance, pd.DataFrame) else 0,
            'categorized_flows': list(categorized_flows.keys()),
            'category_mapping_size': len(self.category_mapping),
            'subtotal_check': subtotal_check,
            'daily_cash_flows_generated': daily_flows_created
        }
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = self.data_dir / f"enhanced_treasury_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        if subtotal_check.get('subtotal_records_count', 0) > 0:
            logging.info(f"Filtered subtotals/transfers exist in raw data (count={subtotal_check['subtotal_records_count']}).")
            logging.info(f"   Categories: {subtotal_check['subtotal_categories_found']}")
        else:
            logging.info("No subtotal/transfer records found in raw D/W data.")
        
        # Generate standardized field mapping
        try:
            standardized_data = self.field_mapper.map_all_fields({
                'raw_data': raw_data,
                'tga_balance': tga_balance,
                'categorized_flows': categorized_flows
            })
            
            # Save standardized data
            standardized_file = self.data_dir / f"standardized_treasury_data_{timestamp}.csv"
            standardized_data.to_csv(standardized_file, index=False)
            
            logging.info(f"Standardized data generated: {len(standardized_data)} records with {len(standardized_data.columns)} fields")
            summary['standardized_data_generated'] = True
            summary['standardized_fields_count'] = len(standardized_data.columns) - 1  # Exclude Date column
            
        except Exception as e:
            logging.error(f"Failed to generate standardized data: {e}")
            standardized_data = None
            summary['standardized_data_generated'] = False

        return {
            'raw_data': raw_data,
            'tga_balance': tga_balance,
            'categorized_flows': categorized_flows,
            'standardized_data': standardized_data,
            'summary': summary
        }


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    collector = EnhancedTreasuryCollector()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    print(f"Starting enhanced Treasury data collection...")
    print(f"Date range: {start_date} to {end_date}")
    print("=" * 60)
    
    all_data = collector.collect_all_enhanced_data(start_date, end_date)
    
    print("\nData collection completed!")
    print("=" * 60)
    raw_data = all_data['raw_data']
    for name, df in raw_data.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            print(f"{name}: {len(df)} rows of data")
        else:
            print(f"{name}: No data")
    
    tga_balance = all_data['tga_balance']
    if isinstance(tga_balance, pd.DataFrame) and not tga_balance.empty:
        print(f"TGA balance records: {len(tga_balance)} entries")
    
    categorized_flows = all_data['categorized_flows']
    print(f"Categorized cash flows: {list(categorized_flows.keys())}")
    print(f"\nAll data saved to: {collector.data_dir}")


if __name__ == "__main__":
    main()
