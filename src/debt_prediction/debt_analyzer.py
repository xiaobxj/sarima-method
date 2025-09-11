"""
Debt Ceiling Analysis Module

This module provides comprehensive analysis of U.S. Treasury debt dynamics,
debt ceiling scenarios, and borrowing capacity assessment.
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
from .debt_calendar import DebtEventsCalendar


class DebtCeilingAnalyzer:
    """
    Analyzes debt ceiling scenarios and borrowing capacity.
    
    This class provides comprehensive debt analysis including:
    - Debt utilization tracking
    - Borrowing capacity assessment  
    - Debt ceiling scenario analysis
    - Extraordinary measures simulation
    - Debt sustainability metrics
    """
    
    def __init__(self, config: Optional[DebtPredictionConfig] = None):
        """
        Initialize the Debt Ceiling Analyzer.
        
        Args:
            config (DebtPredictionConfig, optional): Configuration object
        """
        self.config = config or load_config()
        
        # Data containers
        self.debt_data = None
        self.current_debt = self.config.CURRENT_PUBLIC_DEBT
        self.debt_ceiling = self.config.DEBT_CEILING_LIMIT
        
        # Debt calendar for scheduled payments
        self.debt_calendar = DebtEventsCalendar(config_manager=self.config)
        self.calendar_data = None
        
        # Analysis results
        self.debt_analysis = {}
        self.scenario_results = {}
        self.extraordinary_measures = {}
        
        print(f"Debt Ceiling Analyzer initialized")
        print(f"Current debt: ${self.current_debt:,.0f} million")
        print(f"Debt ceiling: ${self.debt_ceiling:,.0f} million")
        print(f"Current utilization: {self.get_debt_utilization():.1%}")
    
    def load_debt_data(self) -> pd.DataFrame:
        """
        Load historical debt subject to limit data.
        
        Returns:
            pd.DataFrame: Historical debt data
        """
        print("Loading debt subject to limit data...")
        
        try:
            self.debt_data = pd.read_csv(self.config.DEBT_DATA_FILE)
            self.debt_data['record_date'] = pd.to_datetime(self.debt_data['record_date'])
            
            # Sort by date
            self.debt_data = self.debt_data.sort_values('record_date')
            
            print(f"✅ Loaded {len(self.debt_data)} debt records")
            print(f"   Date range: {self.debt_data['record_date'].min().date()} to {self.debt_data['record_date'].max().date()}")
            
            return self.debt_data
            
        except FileNotFoundError:
            print(f"⚠️  Warning: Debt data file not found: {self.config.DEBT_DATA_FILE}")
            return None
        except Exception as e:
            print(f"❌ Error loading debt data: {e}")
            return None
    
    def initialize_debt_calendar(self) -> None:
        """
        Initialize the debt events calendar for scheduled payment tracking.
        """
        print("Initializing debt events calendar...")
        
        try:
            # Build the debt calendar
            self.calendar_data = self.debt_calendar.build_calendar(refresh_data=True)
            
            # Get calendar summary
            calendar_summary = self.debt_calendar.get_calendar_summary()
            
            print(f"✅ Debt calendar initialized successfully")
            print(f"   📊 Tracking {calendar_summary['total_securities_tracked']} securities")
            print(f"   💰 Total debt service: ${calendar_summary['total_debt_service']:,.0f}")
            print(f"   📅 Days with payments: {calendar_summary['days_with_any_payments']}")
            
            # Store calendar summary in analysis results
            self.debt_analysis['debt_calendar_summary'] = calendar_summary
            
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize debt calendar: {str(e)}")
            self.calendar_data = None
    
    def get_scheduled_debt_service(self, query_date: date) -> Dict[str, float]:
        """
        Get scheduled debt service payments for a specific date.
        
        Args:
            query_date (date): Date to query for scheduled payments
            
        Returns:
            Dict with scheduled payment amounts
        """
        if self.calendar_data is None:
            self.initialize_debt_calendar()
        
        if self.calendar_data is not None:
            return self.debt_calendar.get_debt_service_for_date(query_date)
        else:
            # Return zeros if calendar is not available
            return {
                'total_principal_due': 0.0,
                'total_interest_due': 0.0,
                'total_debt_service': 0.0,
                'principal_payment_count': 0,
                'interest_payment_count': 0
            }

    def get_debt_utilization(self) -> float:
        """
        Calculate current debt utilization rate.
        
        Returns:
            float: Debt utilization as percentage (0.0 to 1.0)
        """
        return self.current_debt / self.debt_ceiling
    
    def get_available_headroom(self) -> float:
        """
        Calculate available borrowing capacity.
        
        Returns:
            float: Available headroom in millions USD
        """
        return self.debt_ceiling - self.current_debt
    
    def analyze_debt_trajectory(self, tga_simulation: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze debt trajectory based on TGA simulation results.
        
        Args:
            tga_simulation (pd.DataFrame): TGA simulation results
            
        Returns:
            Dict[str, Any]: Debt trajectory analysis
        """
        print("Analyzing debt trajectory...")
        
        analysis = {
            'baseline_scenario': {},
            'debt_issuance_needed': {},
            'ceiling_breach_risk': {},
            'recommendations': []
        }
        
        # Calculate financing needs based on TGA balance changes
        tga_simulation = tga_simulation.copy()
        tga_simulation['balance_change'] = tga_simulation['closing_tga_balance'].diff()
        tga_simulation['cumulative_financing_need'] = -tga_simulation['balance_change'].cumsum()
        
        # Estimate debt issuance needed to maintain minimum TGA balance
        min_balance_threshold = self.config.MINIMUM_CASH_BALANCE * 1_000_000  # Convert to dollars
        
        tga_simulation['financing_needed'] = np.maximum(
            min_balance_threshold - tga_simulation['closing_tga_balance'] * 1_000_000, 0
        )
        
        # Calculate projected debt levels
        max_financing_need = tga_simulation['financing_needed'].max()
        projected_peak_debt = self.current_debt + (max_financing_need / 1_000_000)  # Convert back to millions
        
        analysis['baseline_scenario'] = {
            'current_debt_million': self.current_debt,
            'projected_peak_debt_million': projected_peak_debt,
            'max_financing_need_million': max_financing_need / 1_000_000,
            'peak_debt_utilization': projected_peak_debt / self.debt_ceiling,
            'days_to_ceiling_breach': self._calculate_days_to_breach(tga_simulation)
        }
        
        # Debt issuance analysis
        total_issuance_needed = tga_simulation['financing_needed'].sum() / 1_000_000
        
        analysis['debt_issuance_needed'] = {
            'total_issuance_million': total_issuance_needed,
            'average_weekly_issuance_million': total_issuance_needed / (len(tga_simulation) / 7),
            'peak_single_day_need_million': tga_simulation['financing_needed'].max() / 1_000_000
        }
        
        # Ceiling breach risk assessment
        headroom = self.get_available_headroom()
        breach_risk = projected_peak_debt > self.debt_ceiling
        
        analysis['ceiling_breach_risk'] = {
            'breach_likely': breach_risk,
            'available_headroom_million': headroom,
            'headroom_exhausted_days': self._calculate_headroom_exhaustion(tga_simulation),
            'risk_level': self._assess_breach_risk(projected_peak_debt, headroom)
        }
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        self.debt_analysis = analysis
        return analysis
    
    def _calculate_days_to_breach(self, tga_simulation: pd.DataFrame) -> Optional[int]:
        """
        Calculate days until debt ceiling breach.
        
        Args:
            tga_simulation (pd.DataFrame): TGA simulation data
            
        Returns:
            Optional[int]: Days to breach or None if no breach
        """
        # Estimate debt increases from financing needs
        cumulative_debt_increase = tga_simulation['financing_needed'].cumsum() / 1_000_000
        projected_debt = self.current_debt + cumulative_debt_increase
        
        breach_days = tga_simulation[projected_debt > self.debt_ceiling]
        
        if len(breach_days) > 0:
            first_breach_date = breach_days.iloc[0]['date']
            start_date = pd.to_datetime(self.config.START_DATE)
            return (first_breach_date - start_date).days
        
        return None
    
    def _calculate_headroom_exhaustion(self, tga_simulation: pd.DataFrame) -> Optional[int]:
        """
        Calculate when available headroom is exhausted.
        
        Args:
            tga_simulation (pd.DataFrame): TGA simulation data
            
        Returns:
            Optional[int]: Days until headroom exhaustion
        """
        headroom = self.get_available_headroom() * 1_000_000  # Convert to dollars
        cumulative_need = tga_simulation['financing_needed'].cumsum()
        
        exhaustion_days = tga_simulation[cumulative_need > headroom]
        
        if len(exhaustion_days) > 0:
            return exhaustion_days.iloc[0]['days_from_start']
        
        return None
    
    def _assess_breach_risk(self, projected_peak_debt: float, headroom: float) -> str:
        """
        Assess debt ceiling breach risk level.
        
        Args:
            projected_peak_debt (float): Projected peak debt level
            headroom (float): Available headroom
            
        Returns:
            str: Risk level ('low', 'medium', 'high', 'critical')
        """
        utilization_at_peak = projected_peak_debt / self.debt_ceiling
        
        if utilization_at_peak >= 1.0:
            return 'critical'
        elif utilization_at_peak >= self.config.DEBT_WARNING_LEVELS['red']:
            return 'high'
        elif utilization_at_peak >= self.config.DEBT_WARNING_LEVELS['orange']:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate policy recommendations based on analysis.
        
        Args:
            analysis (Dict[str, Any]): Debt analysis results
            
        Returns:
            List[str]: List of recommendations
        """
        recommendations = []
        
        breach_risk = analysis['ceiling_breach_risk']
        baseline = analysis['baseline_scenario']
        
        if breach_risk['breach_likely']:
            recommendations.append(
                "🚨 CRITICAL: Debt ceiling breach likely - immediate action required"
            )
            recommendations.append(
                "Consider implementing extraordinary measures to extend borrowing capacity"
            )
            recommendations.append(
                "Engage Congress for debt ceiling increase or suspension"
            )
        
        if breach_risk['risk_level'] in ['high', 'critical']:
            recommendations.append(
                "Monitor daily debt levels and cash flows closely"
            )
            recommendations.append(
                "Prepare contingency plans for debt ceiling scenarios"
            )
        
        if baseline['peak_debt_utilization'] > self.config.DEBT_WARNING_LEVELS['orange']:
            recommendations.append(
                "Consider reducing discretionary spending to lower borrowing needs"
            )
        
        if analysis['debt_issuance_needed']['total_issuance_million'] > 1_000_000:  # $1T
            recommendations.append(
                "Significant debt issuance required - coordinate with market operations"
            )
        
        return recommendations
    
    def analyze_scenarios(self) -> Dict[str, Any]:
        """
        Analyze multiple debt ceiling scenarios.
        
        Returns:
            Dict[str, Any]: Scenario analysis results
        """
        print("Analyzing debt ceiling scenarios...")
        
        scenarios = {}
        
        for scenario_name, ceiling_level in self.config.DEBT_CEILING_SCENARIOS.items():
            print(f"   Analyzing scenario: {scenario_name}")
            
            # Temporarily update ceiling for analysis
            original_ceiling = self.debt_ceiling
            self.debt_ceiling = ceiling_level
            
            scenario_analysis = {
                'ceiling_level_million': ceiling_level,
                'current_utilization': self.get_debt_utilization() if ceiling_level != float('inf') else 0.0,
                'available_headroom_million': self.get_available_headroom() if ceiling_level != float('inf') else float('inf'),
                'risk_assessment': self._assess_scenario_risk(ceiling_level)
            }
            
            scenarios[scenario_name] = scenario_analysis
            
            # Restore original ceiling
            self.debt_ceiling = original_ceiling
        
        self.scenario_results = scenarios
        return scenarios
    
    def _assess_scenario_risk(self, ceiling_level: float) -> Dict[str, Any]:
        """
        Assess risk for a specific ceiling scenario.
        
        Args:
            ceiling_level (float): Debt ceiling level
            
        Returns:
            Dict[str, Any]: Risk assessment
        """
        if ceiling_level == float('inf'):
            return {
                'risk_level': 'none',
                'breach_probability': 0.0,
                'time_to_breach_days': None,
                'description': 'No debt ceiling constraint'
            }
        
        utilization = self.current_debt / ceiling_level
        headroom = ceiling_level - self.current_debt
        
        # Simple risk assessment based on current position
        if utilization >= 0.99:
            risk_level = 'critical'
            breach_probability = 0.9
        elif utilization >= 0.95:
            risk_level = 'high'
            breach_probability = 0.7
        elif utilization >= 0.90:
            risk_level = 'medium'
            breach_probability = 0.3
        else:
            risk_level = 'low'
            breach_probability = 0.1
        
        return {
            'risk_level': risk_level,
            'breach_probability': breach_probability,
            'headroom_million': headroom,
            'utilization_rate': utilization,
            'description': f'Debt ceiling at ${ceiling_level:,.0f}M with {utilization:.1%} utilization'
        }
    
    def estimate_extraordinary_measures(self) -> Dict[str, Any]:
        """
        Estimate capacity and duration of extraordinary measures.
        
        Returns:
            Dict[str, Any]: Extraordinary measures analysis
        """
        print("Estimating extraordinary measures capacity...")
        
        # Typical extraordinary measures and their estimated capacity
        measures = {
            'civil_service_retirement_fund': {
                'capacity_million': 200_000,  # ~$200B
                'duration_days': 60,
                'description': 'Suspend investments in Civil Service Retirement Fund'
            },
            'postal_service_retiree_fund': {
                'capacity_million': 50_000,   # ~$50B
                'duration_days': 30,
                'description': 'Suspend investments in Postal Service Retiree Health Fund'
            },
            'government_securities_fund': {
                'capacity_million': 150_000,  # ~$150B
                'duration_days': 45,
                'description': 'Suspend investments in G Fund (TSP)'
            },
            'exchange_stabilization_fund': {
                'capacity_million': 25_000,   # ~$25B
                'duration_days': 15,
                'description': 'Utilize Exchange Stabilization Fund'
            }
        }
        
        # Calculate total capacity and estimated duration
        total_capacity = sum(measure['capacity_million'] for measure in measures.values())
        weighted_avg_duration = sum(
            measure['capacity_million'] * measure['duration_days'] 
            for measure in measures.values()
        ) / total_capacity
        
        analysis = {
            'total_capacity_million': total_capacity,
            'estimated_duration_days': int(weighted_avg_duration),
            'individual_measures': measures,
            'effectiveness_assessment': self._assess_measure_effectiveness(total_capacity)
        }
        
        self.extraordinary_measures = analysis
        return analysis
    
    def _assess_measure_effectiveness(self, total_capacity: float) -> Dict[str, Any]:
        """
        Assess effectiveness of extraordinary measures.
        
        Args:
            total_capacity (float): Total capacity in millions USD
            
        Returns:
            Dict[str, Any]: Effectiveness assessment
        """
        headroom_needed = max(0, self.current_debt - self.debt_ceiling)
        
        if headroom_needed == 0:
            effectiveness = 'not_needed'
            coverage_ratio = float('inf')
        else:
            coverage_ratio = total_capacity / headroom_needed
            if coverage_ratio >= 2.0:
                effectiveness = 'highly_effective'
            elif coverage_ratio >= 1.0:
                effectiveness = 'effective'
            elif coverage_ratio >= 0.5:
                effectiveness = 'partially_effective'
            else:
                effectiveness = 'insufficient'
        
        return {
            'effectiveness_level': effectiveness,
            'coverage_ratio': coverage_ratio,
            'additional_capacity_needed_million': max(0, headroom_needed - total_capacity),
            'estimated_extension_days': int(total_capacity / (self.config.CURRENT_TGA_BALANCE / 30))  # Rough estimate
        }
    
    def create_debt_visualizations(self, output_dir: Optional[str] = None) -> str:
        """
        Create comprehensive debt analysis visualizations.
        
        Args:
            output_dir (str, optional): Output directory path
            
        Returns:
            str: Path to saved visualization file
        """
        output_dir = output_dir or self.config.OUTPUT_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('U.S. Treasury Debt Ceiling Analysis', fontsize=16, fontweight='bold')
        
        # 1. Debt Utilization by Scenario
        if self.scenario_results:
            scenarios = list(self.scenario_results.keys())
            utilizations = [
                self.scenario_results[s]['current_utilization'] * 100 
                for s in scenarios if self.scenario_results[s]['current_utilization'] != 0
            ]
            scenario_names = [s for s in scenarios if self.scenario_results[s]['current_utilization'] != 0]
            
            colors = ['red' if u >= 95 else 'orange' if u >= 90 else 'green' for u in utilizations]
            
            axes[0, 0].bar(scenario_names, utilizations, color=colors, alpha=0.7)
            axes[0, 0].axhline(y=90, color='orange', linestyle='--', alpha=0.7, label='Warning (90%)')
            axes[0, 0].axhline(y=95, color='red', linestyle='--', alpha=0.7, label='Critical (95%)')
            axes[0, 0].set_title('Debt Utilization by Scenario')
            axes[0, 0].set_ylabel('Utilization (%)')
            axes[0, 0].legend()
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Available Headroom
        if self.scenario_results:
            headrooms = [
                self.scenario_results[s]['available_headroom_million'] / 1000  # Convert to billions
                for s in scenario_names
            ]
            
            axes[0, 1].bar(scenario_names, headrooms, color='lightblue', alpha=0.7, edgecolor='navy')
            axes[0, 1].set_title('Available Borrowing Capacity')
            axes[0, 1].set_ylabel('Headroom ($ Trillions)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Extraordinary Measures Capacity
        if self.extraordinary_measures:
            measures = self.extraordinary_measures['individual_measures']
            measure_names = list(measures.keys())
            capacities = [measures[m]['capacity_million'] / 1000 for m in measure_names]  # Convert to billions
            
            axes[1, 0].barh(measure_names, capacities, color='gold', alpha=0.7, edgecolor='darkorange')
            axes[1, 0].set_title('Extraordinary Measures Capacity')
            axes[1, 0].set_xlabel('Capacity ($ Billions)')
        
        # 4. Risk Assessment Summary
        if self.debt_analysis:
            risk_metrics = {
                'Current Utilization': self.get_debt_utilization() * 100,
                'Projected Peak Util.': self.debt_analysis.get('baseline_scenario', {}).get('peak_debt_utilization', 0) * 100,
                'Breach Risk Score': 50 if self.debt_analysis.get('ceiling_breach_risk', {}).get('breach_likely', False) else 20
            }
            
            metrics = list(risk_metrics.keys())
            values = list(risk_metrics.values())
            colors = ['red' if v >= 95 else 'orange' if v >= 90 else 'green' for v in values]
            
            axes[1, 1].bar(metrics, values, color=colors, alpha=0.7)
            axes[1, 1].set_title('Risk Assessment Summary')
            axes[1, 1].set_ylabel('Risk Level (%)')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(output_dir, f"debt_analysis_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Debt analysis visualization saved: {os.path.basename(plot_file)}")
        return plot_file
    
    def save_analysis(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Save debt analysis results to files.
        
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
            # Comprehensive analysis summary
            analysis_summary = {
                "analysis_timestamp": timestamp,
                "configuration": {
                    "current_debt_million": self.current_debt,
                    "debt_ceiling_million": self.debt_ceiling,
                    "current_utilization": self.get_debt_utilization(),
                    "available_headroom_million": self.get_available_headroom()
                },
                "debt_analysis": self.debt_analysis,
                "scenario_results": self.scenario_results,
                "extraordinary_measures": self.extraordinary_measures
            }
            
            analysis_file = os.path.join(output_dir, f"debt_analysis_{timestamp}.json")
            with open(analysis_file, 'w') as f:
                json.dump(analysis_summary, f, indent=2, default=str)
            saved_files['debt_analysis'] = analysis_file
            
            print(f"📁 Debt analysis saved: {os.path.basename(analysis_file)}")
            
        except Exception as e:
            print(f"❌ Error saving debt analysis: {e}")
        
        return saved_files


def main():
    """
    Main function to run comprehensive debt ceiling analysis.
    """
    print("="*80)
    print("U.S. TREASURY DEBT CEILING ANALYSIS")
    print("="*80)
    
    # Load configuration
    config = load_config()
    
    try:
        # Initialize analyzer
        analyzer = DebtCeilingAnalyzer(config)
        
        # Load historical debt data
        analyzer.load_debt_data()
        
        # Analyze scenarios
        analyzer.analyze_scenarios()
        
        # Estimate extraordinary measures
        analyzer.estimate_extraordinary_measures()
        
        # Create visualizations
        analyzer.create_debt_visualizations()
        
        # Save analysis
        analyzer.save_analysis()
        
        print("\n" + "="*80)
        print("🎉 DEBT CEILING ANALYSIS COMPLETED!")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
