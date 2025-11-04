"""
Run both technical and fundamental analysis engines
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engines.technical import TechnicalAnalysisEngine
from src.engines.fundamental import FundamentalAnalysisEngine
from src.utils.logger import log
from config.settings import PROCESSED_DATA_DIR
import pandas as pd

def main():
    log.info("Starting complete analysis pipeline...")

    # Ensure processed data directory exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Run Technical Analysis
    log.info("\n" + "="*50)
    log.info("RUNNING TECHNICAL ANALYSIS ENGINE")
    log.info("="*50)
    tech_engine = TechnicalAnalysisEngine()
    tech_results = tech_engine.analyze_all_funds()

    # Run Fundamental Analysis
    log.info("\n" + "="*50)
    log.info("RUNNING FUNDAMENTAL ANALYSIS ENGINE")
    log.info("="*50)
    fund_engine = FundamentalAnalysisEngine()
    fund_results = fund_engine.analyze_all_funds()

    # Rename columns before merge to avoid conflicts
    tech_results = tech_results.rename(columns={
        'status': 'status_tech',
        'date': 'date_tech',
        'latest_nav': 'latest_nav',
        'score': 'score_tech',
        'signals': 'signals_tech',
        'confidence': 'confidence_tech'
    })

    fund_results = fund_results.rename(columns={
        'status': 'status_fund',
        'score': 'score_fund'
    })

    # Combine results
    combined = tech_results.merge(
        fund_results,
        on='scheme_code',
        how='inner'
    )

    log.info(f"\nCombined columns: {combined.columns.tolist()}")

    # Add combined score (simple average for now)
    # Normalize technical score to 0-100 scale (from -5 to +5)
    combined['tech_normalized'] = ((combined['score_tech'] + 5) / 10) * 100
    combined['combined_score'] = (combined['tech_normalized'] + combined['score_fund']) / 2

    # Sort by combined score
    combined = combined.sort_values('combined_score', ascending=False)

    # Display top performers
    print("\n" + "="*70)
    print("🏆 TOP 10 MUTUAL FUNDS (COMBINED TECHNICAL + FUNDAMENTAL ANALYSIS)")
    print("="*70)

    # Select display columns
    display_cols = [
        'scheme_code',
        'score_tech',
        'score_fund',
        'combined_score'
    ]

    # Add optional columns if they exist
    optional_cols = ['sharpe_ratio', 'cagr_1y', 'rsi', 'confidence_tech']

    for col in optional_cols:
        if col in combined.columns:
            display_cols.append(col)

    top_10 = combined.head(10)[display_cols]

    # Format numeric columns for better display
    pd.set_option('display.float_format', '{:.2f}'.format)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print(top_10.to_string(index=False))

    # Save combined results
    output_file = PROCESSED_DATA_DIR / 'combined_analysis.csv'
    combined.to_csv(output_file, index=False)
    log.info(f"\n✓ Combined results saved to {output_file}")

    print("\n📊 Analysis Summary:")
    print(f"   Total funds analyzed: {len(combined)}")
    print(f"   Technical Analysis:")
    print(f"      - Avg Score: {combined['score_tech'].mean():.2f}/5")
    print(f"      - Max Score: {combined['score_tech'].max():.2f}/5")
    print(f"      - Min Score: {combined['score_tech'].min():.2f}/5")

    print(f"\n   Fundamental Analysis:")
    print(f"      - Avg Score: {combined['score_fund'].mean():.2f}/100")
    print(f"      - Max Score: {combined['score_fund'].max():.2f}/100")
    print(f"      - Min Score: {combined['score_fund'].min():.2f}/100")

    if 'sharpe_ratio' in combined.columns:
        # Filter out None/NaN values for Sharpe ratio
        sharpe_valid = combined['sharpe_ratio'].dropna()
        if len(sharpe_valid) > 0:
            print(f"\n   Average Sharpe Ratio: {sharpe_valid.mean():.2f}")
            print(f"   Max Sharpe Ratio: {sharpe_valid.max():.2f}")

    if 'cagr_1y' in combined.columns:
        cagr_valid = combined['cagr_1y'].dropna()
        if len(cagr_valid) > 0:
            cagr_mean = cagr_valid.mean() * 100
            print(f"   Average 1Y CAGR: {cagr_mean:.2f}%")

    print(f"\n   Combined Score:")
    print(f"      - Avg: {combined['combined_score'].mean():.2f}/100")
    print(f"      - Top fund: {combined['combined_score'].max():.2f}/100")

    # Show distribution of scores
    print("\n📈 Score Distribution:")
    print(f"   Funds with Combined Score > 60: {len(combined[combined['combined_score'] > 60])}")
    print(f"   Funds with Combined Score 40-60: {len(combined[(combined['combined_score'] >= 40) & (combined['combined_score'] <= 60)])}")
    print(f"   Funds with Combined Score < 40: {len(combined[combined['combined_score'] < 40])}")

    # Show funds with best Sharpe ratios
    if 'sharpe_ratio' in combined.columns:
        print("\n⭐ TOP 5 FUNDS BY SHARPE RATIO:")
        sharpe_df = combined[combined['sharpe_ratio'].notna()].copy()
        if len(sharpe_df) > 0:
            sharpe_top = sharpe_df.nlargest(5, 'sharpe_ratio')[
                ['scheme_code', 'sharpe_ratio', 'cagr_1y', 'score_tech', 'score_fund']
            ]
            print(sharpe_top.to_string(index=False))

    # Show funds with best CAGR
    if 'cagr_1y' in combined.columns:
        print("\n💰 TOP 5 FUNDS BY 1-YEAR CAGR:")
        cagr_df = combined[combined['cagr_1y'].notna()].copy()
        if len(cagr_df) > 0:
            cagr_top = cagr_df.nlargest(5, 'cagr_1y')[
                ['scheme_code', 'cagr_1y', 'sharpe_ratio', 'score_tech', 'score_fund']
            ]
            print(cagr_top.to_string(index=False))

    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
