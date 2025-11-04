"""
Daily automated update script
Run this daily (via cron/Task Scheduler) to:
1. Fetch latest NAV data
2. Update analysis
3. Generate signals
4. Send email alerts
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from src.ingestion.mfapi_fetcher import MFAPIFetcher
from src.engines.technical import TechnicalAnalysisEngine
from src.engines.fundamental import FundamentalAnalysisEngine
from src.engines.advanced_scoring import AdvancedScoringEngine
from src.engines.peer_comparison import PeerComparisonEngine
from src.signals.signal_generator import SignalGenerator, SignalConfig
from src.utils.logger import log
from config.settings import PROCESSED_DATA_DIR
import pandas as pd


def daily_update():
    """Run daily update pipeline"""

    log.info("=" * 80)
    log.info(f"DAILY UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 80)

    # STEP 1: Update NAV data
    log.info("\n📥 Step 1: Updating NAV data...")
    fetcher = MFAPIFetcher()

    try:
        # Update latest NAVs for existing funds
        fetcher.update_latest_navs()

        # Check for new funds (once a week)
        if datetime.now().weekday() == 0:  # Monday
            fetcher.update_fund_database(include_new=True)

    except Exception as e:
        log.error(f"Error updating NAV data: {e}")

    # STEP 2: Run analysis
    log.info("\n📊 Step 2: Running analysis...")

    try:
        tech_engine = TechnicalAnalysisEngine()
        fund_engine = FundamentalAnalysisEngine()

        tech_results = tech_engine.analyze_all_funds()
        fund_results = fund_engine.analyze_all_funds()

        # Combine
        tech_results = tech_results.rename(
            columns={'status': 'status_tech', 'score': 'score_tech', 'confidence': 'confidence_tech'})
        fund_results = fund_results.rename(columns={'status': 'status_fund', 'score': 'score_fund'})

        combined = tech_results.merge(fund_results, on='scheme_code', how='inner')

        # Advanced scoring
        advanced_engine = AdvancedScoringEngine()
        comprehensive = advanced_engine.analyze_all_funds(combined)

        # Peer ranking
        peer_engine = PeerComparisonEngine()
        ranked = peer_engine.rank_within_category(
            comprehensive,
            metric_columns=['comprehensive_score', 'sharpe_ratio', 'cagr_1y']
        )

        # Save
        ranked.to_csv(PROCESSED_DATA_DIR / 'comprehensive_analysis.csv', index=False)
        log.info("✓ Analysis complete and saved")

    except Exception as e:
        log.error(f"Error running analysis: {e}")
        return

    # STEP 3: Generate signals
    log.info("\n🎯 Step 3: Generating signals...")

    try:
        config = SignalConfig(min_votes_for_buy=2, technical_buy_threshold=1.0, fundamental_buy_threshold=50)
        signal_gen = SignalGenerator(config)
        signals = signal_gen.generate_signals_for_all_funds()

        # Save signals
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

        # Count signals
        buy_count = len([s for s in signals if s.signal_type == "BUY"])
        sell_count = len([s for s in signals if s.signal_type == "SELL"])

        log.info(f"✓ Signals generated: {buy_count} BUY, {sell_count} SELL")

    except Exception as e:
        log.error(f"Error generating signals: {e}")
        return

    # STEP 4: Generate portfolio
    log.info("\n💼 Step 4: Creating portfolio recommendation...")

    try:
        portfolio = signal_gen.create_portfolio_recommendation(signals, total_capital=100000)

        if len(portfolio['portfolio']) > 0:
            portfolio_df = pd.DataFrame(portfolio['portfolio'])
            portfolio_df.to_csv(PROCESSED_DATA_DIR / 'recommended_portfolio.csv', index=False)
            log.info(f"✓ Portfolio created with {portfolio['num_funds']} funds")
        else:
            log.info("⚠ No portfolio recommendations (no BUY signals)")

    except Exception as e:
        log.error(f"Error creating portfolio: {e}")

    log.info("\n" + "=" * 80)
    log.info("✅ DAILY UPDATE COMPLETE")
    log.info("=" * 80)


if __name__ == "__main__":
    daily_update()
