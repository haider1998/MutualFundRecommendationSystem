"""
FULLY OPTIMIZED MASTER PIPELINE
All engines optimized for 2.2M NAV records
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime

from src.ingestion.mfapi_fetcher import UltraOptimizedMFAPIFetcher
from src.engines.technical import TechnicalAnalysisEngine
from src.engines.fundamental import FundamentalAnalysisEngine
from src.engines.peer_comparison import PeerComparisonEngine
from src.engines.advanced_scoring import AdvancedScoringEngine
from src.signals.signal_generator import SignalGenerator, SignalConfig
from src.utils.logger import log
from config.settings import PROCESSED_DATA_DIR


def main():
    start_time = datetime.now()

    print("\n" + "="*80)
    print("🚀 OPTIMIZED MASTER MUTUAL FUND INTELLIGENCE PIPELINE")
    print("="*80)

    # STEP 1: Check for new funds
    print("\n📥 STEP 1: Checking for new mutual funds...")
    fetcher = UltraOptimizedMFAPIFetcher()
    new_funds = fetcher.detect_new_funds()

    if len(new_funds) > 0:
        print(f"   ✓ Found {len(new_funds)} new funds")
    else:
        print(f"   ✓ No new funds detected")

    # STEP 2: Technical Analysis (OPTIMIZED)
    print("\n📊 STEP 2: Running Optimized Technical Analysis...")
    step_start = datetime.now()

    tech_engine = TechnicalAnalysisEngine()
    tech_results = tech_engine.analyze_all_funds()

    step_time = (datetime.now() - step_start).total_seconds()
    print(f"   ✓ Analyzed {len(tech_results)} funds in {step_time:.1f}s")

    # STEP 3: Fundamental Analysis (OPTIMIZED)
    print("\n📈 STEP 3: Running Optimized Fundamental Analysis...")
    step_start = datetime.now()

    fund_engine = FundamentalAnalysisEngine()
    fund_results = fund_engine.analyze_all_funds()

    step_time = (datetime.now() - step_start).total_seconds()
    print(f"   ✓ Analyzed {len(fund_results)} funds in {step_time:.1f}s")

    # STEP 4: Combine Results
    print("\n🔗 STEP 4: Combining Analysis Results...")

    tech_results = tech_results.rename(columns={
        'status': 'status_tech',
        'score': 'score_tech',
        'confidence': 'confidence_tech'
    })

    fund_results = fund_results.rename(columns={
        'status': 'status_fund',
        'score': 'score_fund'
    })

    combined = tech_results.merge(fund_results, on='scheme_code', how='inner')
    print(f"   ✓ Combined results: {len(combined)} funds")

    # STEP 5: Comprehensive Scoring (OPTIMIZED)
    print("\n⭐ STEP 5: Calculating Comprehensive Scores...")
    step_start = datetime.now()

    advanced_engine = AdvancedScoringEngine()
    comprehensive = advanced_engine.analyze_all_funds(combined)

    step_time = (datetime.now() - step_start).total_seconds()
    print(f"   ✓ Comprehensive scoring complete in {step_time:.1f}s")

    # STEP 6: Peer Comparison (OPTIMIZED)
    print("\n🏅 STEP 6: Peer Comparison & Category Ranking...")
    step_start = datetime.now()

    peer_engine = PeerComparisonEngine()

    ranked = peer_engine.rank_within_category(
        comprehensive,
        metric_columns=['comprehensive_score', 'sharpe_ratio', 'cagr_1y']
    )

    step_time = (datetime.now() - step_start).total_seconds()
    print(f"   ✓ Ranked funds within categories in {step_time:.1f}s")

    # Get category statistics
    category_stats = peer_engine.get_category_statistics_bulk(ranked)

    # Get category leaders
    category_leaders = peer_engine.identify_category_leaders(ranked, top_n=3)

    # STEP 7: Generate Signals (OPTIMIZED)
    print("\n🎯 STEP 7: Generating Buy/Sell/Hold Signals...")
    step_start = datetime.now()

    config = SignalConfig(
        min_votes_for_buy=2,
        technical_buy_threshold=1.0,
        fundamental_buy_threshold=50
    )

    signal_gen = SignalGenerator(config)
    signals = signal_gen.generate_signals_for_all_funds()

    step_time = (datetime.now() - step_start).total_seconds()

    buy_signals = [s for s in signals if s.signal_type == "BUY"]
    sell_signals = [s for s in signals if s.signal_type == "SELL"]
    hold_signals = [s for s in signals if s.signal_type == "HOLD"]

    print(f"   ✓ Generated signals in {step_time:.1f}s: {len(buy_signals)} BUY, {len(sell_signals)} SELL, {len(hold_signals)} HOLD")

    # STEP 8: Create Portfolio
    print("\n💼 STEP 8: Creating Optimized Portfolio...")
    portfolio = signal_gen.create_portfolio_recommendation(signals, total_capital=100000)
    print(f"   ✓ Portfolio created with {portfolio['num_funds']} funds")

    # DISPLAY RESULTS
    print("\n" + "="*80)
    print("📋 ANALYSIS SUMMARY")
    print("="*80)

    print(f"\n📊 Overall Statistics:")
    print(f"   Total Funds Analyzed: {len(comprehensive)}")
    print(f"   Average Comprehensive Score: {comprehensive['comprehensive_score'].mean():.2f}/100")
    print(f"   Top Score: {comprehensive['comprehensive_score'].max():.2f}/100")

    print(f"\n🎯 Signals Generated:")
    print(f"   BUY: {len(buy_signals)}")
    print(f"   SELL: {len(sell_signals)}")
    print(f"   HOLD: {len(hold_signals)}")

    print(f"\n💼 Portfolio Allocation:")
    print(f"   Equity: {portfolio['equity_allocation']:.1f}%")
    print(f"   Cash: {portfolio['cash_allocation']:.1f}%")

    if portfolio['expected_portfolio_return_1y']:
        print(f"\n📈 Expected Portfolio Returns:")
        print(f"   1 Year: {portfolio['expected_portfolio_return_1y']*100:.2f}%")
        if portfolio['expected_portfolio_return_3y']:
            print(f"   3 Years: {portfolio['expected_portfolio_return_3y']*100:.2f}%")
        if portfolio['expected_portfolio_return_5y']:
            print(f"   5 Years: {portfolio['expected_portfolio_return_5y']*100:.2f}%")

    # Category statistics
    print("\n" + "="*80)
    print("📁 CATEGORY STATISTICS")
    print("="*80)
    print(category_stats.to_string(index=False))

    # Category leaders
    print("\n" + "="*80)
    print("🏆 CATEGORY LEADERS (Top 3 in each category)")
    print("="*80)

    for category, leaders in list(category_leaders.items())[:5]:  # Show top 5 categories
        if len(leaders) > 0:
            print(f"\n📁 {category}:")
            for i, fund in enumerate(leaders[:3], 1):
                comp_score = fund.get('comprehensive_score', fund.get('score_fund', 0))
                sharpe = fund.get('sharpe_ratio', 0)
                cagr = fund.get('cagr_1y', 0)

                sharpe_str = f"{sharpe:.2f}" if sharpe and not pd.isna(sharpe) else "N/A"
                cagr_str = f"{cagr*100:.1f}%" if cagr and not pd.isna(cagr) else "N/A"

                print(f"   {i}. Score: {comp_score:.1f}/100 | "
                      f"Sharpe: {sharpe_str} | "
                      f"CAGR: {cagr_str} | "
                      f"Code: {fund['scheme_code']}")

    # Top 10 overall
    print("\n" + "="*80)
    print("🥇 TOP 10 FUNDS (BY COMPREHENSIVE SCORE)")
    print("="*80)

    top_10 = comprehensive.nlargest(10, 'comprehensive_score')

    for i, row in enumerate(top_10.itertuples(), 1):
        print(f"\n{i}. Scheme Code: {row.scheme_code}")
        print(f"   Comprehensive Score: {row.comprehensive_score:.1f}/100")
        print(f"   ├─ Performance: {row.performance:.1f}/100")
        print(f"   ├─ Risk-Adjusted: {row.risk_adjusted:.1f}/100")
        print(f"   ├─ Consistency: {row.consistency:.1f}/100")
        print(f"   ├─ Cost Efficiency: {row.cost_efficiency:.1f}/100")
        print(f"   ├─ Fund Quality: {row.fund_quality:.1f}/100")
        print(f"   └─ Momentum: {row.momentum:.1f}/100")

        if hasattr(row, 'sharpe_ratio') and pd.notna(row.sharpe_ratio):
            print(f"   Sharpe Ratio: {row.sharpe_ratio:.2f}")
        if hasattr(row, 'cagr_1y') and pd.notna(row.cagr_1y):
            print(f"   1Y CAGR: {row.cagr_1y*100:.2f}%")

    # Portfolio recommendation
    if len(portfolio['portfolio']) > 0:
        print("\n" + "="*80)
        print(f"💰 RECOMMENDED PORTFOLIO (₹1,00,000 Investment)")
        print("="*80)

        for i, fund in enumerate(portfolio['portfolio'][:10], 1):  # Show top 10
            print(f"\n{i}. {fund['scheme_name'][:65]}")
            print(f"   Allocation: {fund['allocation_percent']:.1f}% (₹{fund['allocation_amount']:,.0f})")
            print(f"   Signal: {fund['signal_strength']} | Score: {fund['combined_score']:.1f}/100")

            if fund['expected_1y_return'] and pd.notna(fund['expected_1y_return']):
                print(f"   Expected Returns:")
                print(f"   ├─ 1 Year: {fund['expected_1y_return']*100:.1f}%")
                if fund['expected_3y_return'] and pd.notna(fund['expected_3y_return']):
                    print(f"   ├─ 3 Years: {fund['expected_3y_return']*100:.1f}%")
                if fund['expected_5y_return'] and pd.notna(fund['expected_5y_return']):
                    print(f"   └─ 5 Years: {fund['expected_5y_return']*100:.1f}%")

    # SAVE RESULTS
    print("\n" + "="*80)
    print("💾 SAVING RESULTS")
    print("="*80)

    comprehensive.to_csv(PROCESSED_DATA_DIR / 'comprehensive_analysis.csv', index=False)
    print(f"   ✓ Comprehensive analysis saved")

    ranked.to_csv(PROCESSED_DATA_DIR / 'ranked_funds.csv', index=False)
    print(f"   ✓ Ranked funds saved")

    category_stats.to_csv(PROCESSED_DATA_DIR / 'category_statistics.csv', index=False)
    print(f"   ✓ Category statistics saved")

    if len(portfolio['portfolio']) > 0:
        portfolio_df = pd.DataFrame(portfolio['portfolio'])
        portfolio_df.to_csv(PROCESSED_DATA_DIR / 'recommended_portfolio.csv', index=False)
        print(f"   ✓ Portfolio recommendation saved")

    signals_df = pd.DataFrame([{
        'scheme_code': s.scheme_code,
        'scheme_name': s.scheme_name,
        'signal_type': s.signal_type,
        'signal_strength': s.signal_strength,
        'comprehensive_score': s.combined_score,
        'allocation_pct': s.recommended_allocation * 100,
        'reasons': '; '.join(s.reasons),
        'warnings': '; '.join(s.warnings)
    } for s in signals])

    signals_df.to_csv(PROCESSED_DATA_DIR / 'signals.csv', index=False)
    print(f"   ✓ Signals saved")

    # PERFORMANCE SUMMARY
    total_time = (datetime.now() - start_time).total_seconds()

    print("\n" + "="*80)
    print("✅ MASTER PIPELINE COMPLETE!")
    print("="*80)
    print(f"\n⏱️  Total Execution Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"\n📁 All results saved to: {PROCESSED_DATA_DIR}")
    print(f"\n💡 Next Steps:")
    print(f"   1. Review comprehensive_analysis.csv for detailed fund analysis")
    print(f"   2. Check signals.csv for Buy/Sell/Hold recommendations")
    print(f"   3. See recommended_portfolio.csv for optimized allocation")
    print(f"   4. Review category_statistics.csv for category insights")


if __name__ == "__main__":
    main()
