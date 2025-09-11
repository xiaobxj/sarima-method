"""
Main Execution Script for Treasury Debt Prediction System

This script orchestrates the complete debt prediction and analysis workflow,
including TGA balance simulation and debt ceiling analysis.
"""

import os
import sys
from datetime import datetime, timedelta
import argparse
from typing import Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.debt_prediction.config import DebtPredictionConfig, load_config
from src.debt_prediction.tga_simulator import TGABalanceSimulator  
from src.debt_prediction.debt_analyzer import DebtCeilingAnalyzer
from src.debt_prediction.debt_calendar import DebtEventsCalendar


def run_complete_analysis(config_file: Optional[str] = None, 
                         output_dir: Optional[str] = None) -> dict:
    """
    Run complete treasury debt prediction and analysis workflow.
    
    Args:
        config_file (str, optional): Path to configuration file
        output_dir (str, optional): Output directory for results
        
    Returns:
        dict: Complete analysis results
    """
    print("="*100)
    print("🏛️  U.S. TREASURY DEBT PREDICTION & ANALYSIS SYSTEM")
    print("="*100)
    
    # Load configuration
    print("\n📋 LOADING CONFIGURATION...")
    config = load_config(config_file)
    config.print_summary()
    
    # Validate configuration
    validation = config.validate_configuration()
    if not validation['valid']:
        print("\n❌ CONFIGURATION VALIDATION FAILED:")
        for error in validation['errors']:
            print(f"   • {error}")
        return {'status': 'failed', 'reason': 'configuration_invalid'}
    
    if validation['warnings']:
        print("\n⚠️  CONFIGURATION WARNINGS:")
        for warning in validation['warnings']:
            print(f"   • {warning}")
    
    # Override output directory if specified
    if output_dir:
        config.OUTPUT_DIR = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    results = {
        'status': 'success',
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'debt_calendar': None,
        'tga_simulation': None,
        'debt_analysis': None,
        'files_created': [],
        'summary': {}
    }
    
    try:
        # ========== PHASE 1: TGA BALANCE SIMULATION ==========
        print("\n" + "="*100)
        print("📊 PHASE 1: TGA BALANCE SIMULATION")
        print("="*100)
        
        # Initialize TGA simulator
        tga_simulator = TGABalanceSimulator(config)
        
        # Initialize debt events calendar
        print("\n🗓️  Initializing Debt Events Calendar...")
        debt_calendar = DebtEventsCalendar(config_manager=config)
        try:
            calendar_data = debt_calendar.build_calendar(refresh_data=True)
            calendar_summary = debt_calendar.get_calendar_summary()
            print(f"   ✅ Debt calendar built successfully")
            print(f"   📊 Tracking {calendar_summary['total_securities_tracked']} securities")
            print(f"   💰 Total scheduled debt service: ${calendar_summary['total_debt_service']:,.0f}")
            
            # Store calendar in results for later use
            debt_calendar_results = {
                'calendar_summary': calendar_summary,
                'calendar_data': calendar_data
            }
        except Exception as e:
            print(f"   ⚠️ Warning: Debt calendar initialization failed: {str(e)}")
            debt_calendar = None
            debt_calendar_results = None
        
        # Store debt calendar results
        results['debt_calendar'] = debt_calendar_results
        
        # Load forecast data
        forecast_data = tga_simulator.load_forecast_data()
        if forecast_data is None:
            return {'status': 'failed', 'reason': 'forecast_data_load_failed'}
        
        # Get starting balance
        tga_simulator.get_starting_balance()
        
        # Calculate daily cash flows
        tga_simulator.calculate_daily_cash_flows()
        
        # Run enhanced TGA simulation with debt calendar integration
        tga_simulation = tga_simulator.run_simulation(debt_calendar=debt_calendar)
        
        # Analyze X-Date scenarios
        x_date_analysis = tga_simulator.analyze_x_date_scenarios()
        
        # Save TGA results
        tga_files = tga_simulator.save_results(config.OUTPUT_DIR)
        results['files_created'].extend(tga_files.values())
        
        # Create TGA visualizations
        tga_plot = tga_simulator.create_visualizations(config.OUTPUT_DIR)
        results['files_created'].append(tga_plot)
        
        # Store TGA results
        results['tga_simulation'] = {
            'simulation_data': tga_simulation,
            'x_date_analysis': x_date_analysis,
            'summary': tga_simulator.simulation_results
        }
        
        print(f"✅ TGA simulation completed successfully")
        
        # ========== PHASE 2: DEBT CEILING ANALYSIS ==========
        print("\n" + "="*100)
        print("💰 PHASE 2: DEBT CEILING ANALYSIS")
        print("="*100)
        
        # Initialize debt analyzer
        debt_analyzer = DebtCeilingAnalyzer(config)
        
        # Load debt data
        debt_analyzer.load_debt_data()
        
        # Analyze debt trajectory using TGA results
        debt_trajectory = debt_analyzer.analyze_debt_trajectory(tga_simulation)
        
        # Analyze scenarios
        scenario_analysis = debt_analyzer.analyze_scenarios()
        
        # Estimate extraordinary measures
        extraordinary_measures = debt_analyzer.estimate_extraordinary_measures()
        
        # Create debt visualizations
        debt_plot = debt_analyzer.create_debt_visualizations(config.OUTPUT_DIR)
        results['files_created'].append(debt_plot)
        
        # Save debt analysis
        debt_files = debt_analyzer.save_analysis(config.OUTPUT_DIR)
        results['files_created'].extend(debt_files.values())
        
        # Store debt analysis results
        results['debt_analysis'] = {
            'debt_trajectory': debt_trajectory,
            'scenario_analysis': scenario_analysis,
            'extraordinary_measures': extraordinary_measures
        }
        
        print(f"✅ Debt ceiling analysis completed successfully")
        
        # ========== PHASE 3: INTEGRATED ANALYSIS & SUMMARY ==========
        print("\n" + "="*100)
        print("🔍 PHASE 3: INTEGRATED ANALYSIS & SUMMARY")
        print("="*100)
        
        # Generate integrated summary
        integrated_summary = generate_integrated_summary(
            tga_simulator.simulation_results,
            x_date_analysis,
            debt_trajectory,
            scenario_analysis,
            extraordinary_measures,
            config
        )
        
        results['summary'] = integrated_summary
        
        # Save integrated summary
        summary_file = save_integrated_summary(integrated_summary, config.OUTPUT_DIR)
        results['files_created'].append(summary_file)
        
        # Print executive summary
        print_executive_summary(integrated_summary)
        
        print(f"\n✅ Integrated analysis completed successfully")
        
    except Exception as e:
        print(f"\n❌ ERROR DURING ANALYSIS: {e}")
        import traceback
        traceback.print_exc()
        results['status'] = 'failed'
        results['error'] = str(e)
        return results
    
    # Final summary
    print("\n" + "="*100)
    print("🎉 TREASURY DEBT PREDICTION ANALYSIS COMPLETED!")
    print("="*100)
    print(f"📁 Output directory: {config.OUTPUT_DIR}")
    print(f"📄 Files created: {len(results['files_created'])}")
    for file_path in results['files_created']:
        print(f"   • {os.path.basename(file_path)}")
    print("="*100)
    
    return results


def generate_integrated_summary(tga_results: dict, x_date_analysis: dict,
                               debt_trajectory: dict, scenario_analysis: dict,
                               extraordinary_measures: dict, 
                               config: DebtPredictionConfig) -> dict:
    """
    Generate integrated analysis summary combining all results.
    
    Args:
        tga_results (dict): TGA simulation results
        x_date_analysis (dict): X-Date analysis results
        debt_trajectory (dict): Debt trajectory analysis
        scenario_analysis (dict): Scenario analysis results
        extraordinary_measures (dict): Extraordinary measures analysis
        config (DebtPredictionConfig): Configuration object
        
    Returns:
        dict: Integrated summary
    """
    print("Generating integrated analysis summary...")
    
    # Extract key metrics - prioritize TGA simulation results for consistency
    tga_x_date_analysis = tga_results.get('summary', {}).get('x_date_analysis', {})
    x_date = tga_x_date_analysis.get('x_date') or x_date_analysis.get('base_case', {}).get('x_date')
    days_to_x_date = tga_x_date_analysis.get('days_to_x_date') or x_date_analysis.get('base_case', {}).get('days_to_x_date')
    
    # If we still don't have days_to_x_date but have x_date, calculate it consistently
    if x_date and not days_to_x_date:
        start_date = pd.to_datetime(config.START_DATE).date()
        if hasattr(x_date, 'date'):
            x_date = x_date.date()
        days_to_x_date = (x_date - start_date).days
    
    debt_breach_likely = debt_trajectory.get('ceiling_breach_risk', {}).get('breach_likely', False)
    days_to_debt_breach = debt_trajectory.get('ceiling_breach_risk', {}).get('headroom_exhausted_days')
    
    extraordinary_capacity = extraordinary_measures.get('total_capacity_million', 0)
    extraordinary_duration = extraordinary_measures.get('estimated_duration_days', 0)
    
    # Determine overall risk level
    overall_risk = determine_overall_risk(days_to_x_date, debt_breach_likely, 
                                        days_to_debt_breach, config)
    
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'forecast_period': {
            'start_date': config.START_DATE.isoformat(),
            'horizon_days': config.FORECAST_HORIZON,
            'end_date': (config.START_DATE + timedelta(days=config.FORECAST_HORIZON)).isoformat()
        },
        
        'executive_summary': {
            'overall_risk_level': overall_risk,
            'x_date': x_date.isoformat() if x_date else None,
            'days_to_x_date': days_to_x_date,
            'debt_ceiling_breach_likely': debt_breach_likely,
            'days_to_debt_breach': days_to_debt_breach,
            'extraordinary_measures_available': extraordinary_capacity > 0
        },
        
        'cash_flow_outlook': {
            'starting_balance_billion': tga_results['starting_balance'] / 1e9,
            'final_balance_billion': tga_results['final_balance'] / 1e9,
            'minimum_balance_billion': tga_results['minimum_balance'] / 1e9,
            'minimum_balance_date': tga_results['minimum_balance_date'].isoformat(),
            'average_daily_deficit_billion': (
                (tga_results['starting_balance'] - tga_results['final_balance']) / 
                config.FORECAST_HORIZON / 1e9
            )
        },
        
        'debt_outlook': {
            'current_debt_trillion': config.CURRENT_PUBLIC_DEBT / 1e6,
            'debt_ceiling_trillion': config.DEBT_CEILING_LIMIT / 1e6,
            'current_utilization_pct': (config.CURRENT_PUBLIC_DEBT / config.DEBT_CEILING_LIMIT) * 100,
            'available_headroom_billion': (config.DEBT_CEILING_LIMIT - config.CURRENT_PUBLIC_DEBT) / 1e3,
            'projected_peak_debt_trillion': debt_trajectory.get('baseline_scenario', {}).get(
                'projected_peak_debt_million', 0) / 1e6
        },
        
        'risk_assessment': {
            'x_date_warning_level': x_date_analysis.get('base_case', {}).get('warning_level', 'green'),
            'debt_breach_risk_level': debt_trajectory.get('ceiling_breach_risk', {}).get('risk_level', 'low'),
            'overall_risk_level': overall_risk,
            'key_risks': generate_key_risks(x_date_analysis, debt_trajectory, config)
        },
        
        'extraordinary_measures': {
            'total_capacity_billion': extraordinary_capacity / 1e3,
            'estimated_duration_days': extraordinary_duration,
            'effectiveness_level': extraordinary_measures.get('effectiveness_assessment', {}).get(
                'effectiveness_level', 'unknown')
        },
        
        'policy_recommendations': debt_trajectory.get('recommendations', []),
        
        'scenario_comparison': {
            scenario: {
                'ceiling_trillion': data['ceiling_level_million'] / 1e6,
                'utilization_pct': data['current_utilization'] * 100,
                'headroom_billion': data['available_headroom_million'] / 1e3,
                'risk_level': data['risk_assessment']['risk_level']
            }
            for scenario, data in scenario_analysis.items()
            if data['ceiling_level_million'] != float('inf')
        }
    }
    
    return summary


def determine_overall_risk(days_to_x_date: Optional[int], debt_breach_likely: bool,
                          days_to_debt_breach: Optional[int], 
                          config: DebtPredictionConfig) -> str:
    """
    Determine overall risk level based on multiple factors.
    
    Args:
        days_to_x_date (int, optional): Days to X-Date
        debt_breach_likely (bool): Whether debt breach is likely
        days_to_debt_breach (int, optional): Days to debt breach
        config (DebtPredictionConfig): Configuration object
        
    Returns:
        str: Overall risk level
    """
    if debt_breach_likely or (days_to_x_date and days_to_x_date <= config.X_DATE_WARNING_DAYS['red']):
        return 'critical'
    elif days_to_x_date and days_to_x_date <= config.X_DATE_WARNING_DAYS['orange']:
        return 'high'
    elif days_to_x_date and days_to_x_date <= config.X_DATE_WARNING_DAYS['yellow']:
        return 'medium'
    else:
        return 'low'


def generate_key_risks(x_date_analysis: dict, debt_trajectory: dict, 
                      config: DebtPredictionConfig) -> list:
    """
    Generate list of key risks based on analysis.
    
    Args:
        x_date_analysis (dict): X-Date analysis results
        debt_trajectory (dict): Debt trajectory analysis
        config (DebtPredictionConfig): Configuration object
        
    Returns:
        list: List of key risk descriptions
    """
    risks = []
    
    days_to_x_date = x_date_analysis.get('base_case', {}).get('days_to_x_date')
    if days_to_x_date and days_to_x_date <= 30:
        risks.append(f"X-Date approaching in {days_to_x_date} days")
    
    if debt_trajectory.get('ceiling_breach_risk', {}).get('breach_likely'):
        risks.append("Debt ceiling breach likely within forecast period")
    
    current_utilization = config.CURRENT_PUBLIC_DEBT / config.DEBT_CEILING_LIMIT
    if current_utilization >= 0.95:
        risks.append(f"High debt utilization ({current_utilization:.1%})")
    
    avg_deficit = debt_trajectory.get('debt_issuance_needed', {}).get('total_issuance_million', 0)
    if avg_deficit > 500_000:  # $500B
        risks.append("Significant financing needs require substantial debt issuance")
    
    return risks


def save_integrated_summary(summary: dict, output_dir: str) -> str:
    """
    Save integrated summary to JSON file.
    
    Args:
        summary (dict): Integrated summary data
        output_dir (str): Output directory
        
    Returns:
        str: Path to saved summary file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(output_dir, f"integrated_analysis_summary_{timestamp}.json")
    
    try:
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"📄 Integrated summary saved: {os.path.basename(summary_file)}")
    except Exception as e:
        print(f"❌ Error saving integrated summary: {e}")
    
    return summary_file


def print_executive_summary(summary: dict):
    """
    Print executive summary to console.
    
    Args:
        summary (dict): Integrated summary data
    """
    print("\n📋 EXECUTIVE SUMMARY")
    print("-" * 50)
    
    exec_summary = summary['executive_summary']
    cash_outlook = summary['cash_flow_outlook']
    debt_outlook = summary['debt_outlook']
    risk_assessment = summary['risk_assessment']
    
    # Overall risk level
    risk_level = exec_summary['overall_risk_level']
    risk_emoji = {'low': '✅', 'medium': '⚠️', 'high': '🚨', 'critical': '🔴'}
    print(f"{risk_emoji.get(risk_level, '❓')} Overall Risk Level: {risk_level.upper()}")
    
    # X-Date status
    if exec_summary['x_date']:
        print(f"🗓️  X-Date: {exec_summary['x_date']} ({exec_summary['days_to_x_date']} days)")
    else:
        print(f"✅ No X-Date within {summary['forecast_period']['horizon_days']}-day forecast")
    
    # Cash flow outlook
    print(f"💰 TGA Balance: ${cash_outlook['starting_balance_billion']:.1f}B → ${cash_outlook['final_balance_billion']:.1f}B")
    print(f"📉 Minimum Balance: ${cash_outlook['minimum_balance_billion']:.1f}B on {cash_outlook['minimum_balance_date'][:10]}")
    
    # Debt outlook
    print(f"📊 Debt Utilization: {debt_outlook['current_utilization_pct']:.1f}% of ${debt_outlook['debt_ceiling_trillion']:.1f}T ceiling")
    print(f"💳 Available Headroom: ${debt_outlook['available_headroom_billion']:.1f}B")
    
    # Key risks
    if risk_assessment['key_risks']:
        print(f"\n🚨 Key Risks:")
        for risk in risk_assessment['key_risks']:
            print(f"   • {risk}")
    
    # Policy recommendations
    if summary.get('policy_recommendations'):
        print(f"\n💡 Policy Recommendations:")
        for rec in summary['policy_recommendations'][:3]:  # Top 3
            print(f"   • {rec}")


def main():
    """
    Main function with command-line interface.
    """
    parser = argparse.ArgumentParser(
        description="U.S. Treasury Debt Prediction & Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run with default configuration
  python main.py --config custom.json     # Run with custom configuration
  python main.py --output ./results       # Specify output directory
  python main.py --config custom.json --output ./results  # Both options
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration JSON file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Run complete analysis
    results = run_complete_analysis(
        config_file=args.config,
        output_dir=args.output
    )
    
    # Exit with appropriate code
    if results['status'] == 'success':
        print(f"\n🎉 Analysis completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Analysis failed: {results.get('reason', 'unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
