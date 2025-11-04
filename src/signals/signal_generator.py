"""
Advanced Signal Generator with Multi-Confirmation Logic
Optimized through backtesting for maximum returns
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from src.engines.technical import TechnicalAnalysisEngine
from src.engines.fundamental import FundamentalAnalysisEngine
from src.utils.database import db, DailyNAV, FundMetadata, Signal
from src.utils.logger import log
from config.settings import PROCESSED_DATA_DIR


@dataclass
class SignalConfig:
    """Configuration for signal generation thresholds"""
    # Entry thresholds (optimized through backtesting)
    technical_buy_threshold: float = 2.0  # Technical score >= 2/5
    fundamental_buy_threshold: float = 60  # Fundamental score >= 60/100
    min_votes_for_buy: int = 2  # Minimum engines agreeing

    # Exit thresholds (optimized for capital preservation)
    stop_loss_percent: float = -0.15  # -15% stop loss
    trailing_stop_percent: float = -0.10  # -10% trailing stop after gains
    profit_target_percent: float = 0.25  # 25% profit target

    # Consecutive down days exit
    max_consecutive_down_days: int = 3  # Exit after 3 consecutive red days
    max_consecutive_drop_percent: float = -0.10  # Exit if down 10% in 3 days

    # Hold criteria
    min_sharpe_for_hold: float = 0.5  # Minimum Sharpe ratio to hold
    technical_hold_threshold: float = 0  # Technical score >= 0 to hold

    # Risk management
    max_portfolio_equity: float = 0.70  # Max 70% in equity
    max_single_position: float = 0.30  # Max 30% in one fund
    min_position_size: float = 0.05  # Min 5% position


@dataclass
class FundSignal:
    """Signal for a specific fund"""
    scheme_code: str
    scheme_name: str
    signal_type: str  # BUY, SELL, HOLD, AVOID
    signal_strength: str  # STRONG, MODERATE, WEAK

    # Scores
    technical_score: float
    fundamental_score: float
    combined_score: float
    confirmation_votes: int

    # Metrics
    latest_nav: float
    sharpe_ratio: Optional[float]
    cagr_1y: Optional[float]
    max_drawdown: Optional[float]
    rsi: Optional[float]

    # Allocation
    recommended_allocation: float  # Percentage of portfolio

    # Rationale
    reasons: List[str]
    warnings: List[str]

    # Expected returns (from historical analysis)
    expected_return_1m: Optional[float] = None
    expected_return_3m: Optional[float] = None
    expected_return_6m: Optional[float] = None
    expected_return_1y: Optional[float] = None
    expected_return_3y: Optional[float] = None
    expected_return_5y: Optional[float] = None


class SignalGenerator:
    """Optimized signal generation with bulk queries"""

    def __init__(self, config: SignalConfig = None):
        self.config = config or SignalConfig()
        self.all_nav_data = None
        self.historical_returns_cache = {}

    def load_all_data_bulk(self):
        """Load all necessary data once"""
        if self.all_nav_data is not None:
            return

        log.info("Loading all data for signal generation...")
        session = db.get_session()

        # Load 10 years of data for historical returns
        cutoff = datetime.now().date() - timedelta(days=365 * 10 + 30)

        query = session.query(
            DailyNAV.scheme_code,
            DailyNAV.date,
            DailyNAV.nav
        ).filter(
            DailyNAV.date >= cutoff
        ).order_by(
            DailyNAV.scheme_code,
            DailyNAV.date
        )

        self.all_nav_data = pd.read_sql(query.statement, session.bind)
        self.all_nav_data['date'] = pd.to_datetime(self.all_nav_data['date'])

        session.close()
        log.info(f"✓ Loaded {len(self.all_nav_data):,} NAV records for signals")

    def calculate_all_historical_returns_bulk(self, scheme_codes: List[str]):
        """
        Calculate historical returns for ALL funds at once
        MUCH faster than per-fund calculation
        """
        self.load_all_data_bulk()

        log.info("Calculating historical returns for all funds...")

        grouped = self.all_nav_data.groupby('scheme_code')

        periods_days = {
            '1m': 30,
            '3m': 90,
            '6m': 180,
            '1y': 365,
            '3y': 365 * 3,
            '5y': 365 * 5,
            '10y': 365 * 10
        }

        today = datetime.now()

        for scheme_code in scheme_codes:
            if scheme_code not in grouped.groups:
                continue

            df = grouped.get_group(scheme_code).sort_values('date')

            if df.empty:
                continue

            returns = {}
            latest_nav = df.iloc[-1]['nav']

            for period, days in periods_days.items():
                target_date = today - timedelta(days=days)

                # Find closest date (vectorized)
                date_diffs = (df['date'] - target_date).abs()
                closest_idx = date_diffs.idxmin()
                closest_date = df.loc[closest_idx, 'date']

                # Must be within 7 days tolerance
                if abs((closest_date - target_date).days) <= 7:
                    start_nav = df.loc[closest_idx, 'nav']
                    period_return = (latest_nav - start_nav) / start_nav
                    returns[period] = period_return
                else:
                    returns[period] = None

            self.historical_returns_cache[scheme_code] = returns

        log.info(f"✓ Calculated returns for {len(self.historical_returns_cache)} funds")

    def generate_signals_for_all_funds(self) -> List[FundSignal]:
        """Generate signals (optimized)"""
        from src.engines.technical import TechnicalAnalysisEngine
        from src.engines.fundamental import FundamentalAnalysisEngine

        log.info("Running analysis engines...")

        tech_engine = TechnicalAnalysisEngine()
        fund_engine = FundamentalAnalysisEngine()

        tech_results = tech_engine.analyze_all_funds()
        fund_results = fund_engine.analyze_all_funds()

        # Pre-calculate all historical returns
        scheme_codes = tech_results['scheme_code'].unique().tolist()
        self.calculate_all_historical_returns_bulk(scheme_codes)

        # Convert to dicts for lookup
        tech_dict = tech_results.set_index('scheme_code').to_dict('index')
        fund_dict = fund_results.set_index('scheme_code').to_dict('index')

        # Load fund metadata
        session = db.get_session()
        metadata = session.query(FundMetadata.scheme_code, FundMetadata.scheme_name).all()
        metadata_dict = dict(metadata)
        session.close()

        log.info("Generating signals...")
        signals = []

        for scheme_code in scheme_codes:
            if scheme_code in fund_dict:
                try:
                    signal = self._generate_signal_optimized(
                        scheme_code,
                        metadata_dict.get(scheme_code, scheme_code),
                        tech_dict[scheme_code],
                        fund_dict[scheme_code]
                    )
                    signals.append(signal)
                except Exception as e:
                    log.error(f"Error generating signal for {scheme_code}: {e}")

        log.info(f"✓ Generated {len(signals)} signals")
        return signals

    def _generate_signal_optimized(self, scheme_code, scheme_name, tech_analysis, fund_analysis):
        """Generate signal (uses cached data)"""
        tech_score = tech_analysis.get('score', 0)
        fund_score = fund_analysis.get('score', 0)

        tech_normalized = ((tech_score + 5) / 10) * 100
        combined_score = (tech_normalized + fund_score) / 2

        votes = 0
        reasons = []
        warnings = []

        # Voting logic (same as before)
        if tech_score >= self.config.technical_buy_threshold:
            votes += 1
            reasons.append(f"✓ Strong technical signal ({tech_score:.1f}/5)")

        sharpe = fund_analysis.get('sharpe_ratio')
        if sharpe and sharpe > 1.0:
            votes += 1
            reasons.append(f"✓ Excellent Sharpe ratio ({sharpe:.2f})")

        cagr = fund_analysis.get('cagr_1y')
        if cagr and cagr > 0.10:
            votes += 1
            reasons.append(f"✓ Strong 1Y CAGR ({cagr * 100:.1f}%)")

        # Determine signal
        if votes >= self.config.min_votes_for_buy:
            signal_type = "BUY"
            signal_strength = "STRONG" if votes >= 3 else "MODERATE"
        elif len(warnings) >= 2:
            signal_type = "SELL"
            signal_strength = "MODERATE"
        else:
            signal_type = "HOLD"
            signal_strength = "WEAK"

        # Get historical returns from cache
        hist_returns = self.historical_returns_cache.get(scheme_code, {})

        # Create signal
        signal = FundSignal(
            scheme_code=scheme_code,
            scheme_name=scheme_name,
            signal_type=signal_type,
            signal_strength=signal_strength,
            technical_score=tech_score,
            fundamental_score=fund_score,
            combined_score=combined_score,
            confirmation_votes=votes,
            latest_nav=tech_analysis.get('latest_nav', 0),
            sharpe_ratio=sharpe,
            cagr_1y=cagr,
            max_drawdown=fund_analysis.get('max_drawdown'),
            rsi=tech_analysis.get('rsi'),
            recommended_allocation=0.1 if signal_type == "BUY" else 0.0,
            reasons=reasons,
            warnings=warnings,
            expected_return_1m=hist_returns.get('1m'),
            expected_return_3m=hist_returns.get('3m'),
            expected_return_6m=hist_returns.get('6m'),
            expected_return_1y=hist_returns.get('1y'),
            expected_return_3y=hist_returns.get('3y'),
            expected_return_5y=hist_returns.get('5y')
        )

        return signal


def main():
    """Test signal generation"""

    # Create signal generator with default config
    config = SignalConfig(
        min_votes_for_buy=2,
        technical_buy_threshold=1.5,
        fundamental_buy_threshold=50
    )

    generator = SignalGenerator(config)

    # Generate signals
    signals = generator.generate_signals_for_all_funds()

    # Show BUY signals
    buy_signals = [s for s in signals if s.signal_type == "BUY"]

    print("\n" + "=" * 80)
    print("🟢 BUY SIGNALS (Sorted by Combined Score)")
    print("=" * 80)

    buy_signals.sort(key=lambda x: x.combined_score, reverse=True)

    for i, signal in enumerate(buy_signals[:10], 1):
        print(f"\n{i}. {signal.scheme_name[:60]}")
        print(f"   Scheme Code: {signal.scheme_code}")
        print(f"   Signal Strength: {signal.signal_strength} ({signal.confirmation_votes}/4 votes)")
        print(
            f"   Scores: Tech={signal.technical_score:.1f}/5 | Fund={signal.fundamental_score:.0f}/100 | Combined={signal.combined_score:.1f}/100")

        if signal.sharpe_ratio:
            print(f"   Sharpe Ratio: {signal.sharpe_ratio:.2f}")

        if signal.cagr_1y:
            print(f"   1Y CAGR: {signal.cagr_1y * 100:.2f}%")

        print(f"   Recommended Allocation: {signal.recommended_allocation * 100:.1f}%")

        if signal.expected_return_1y:
            print(
                f"   Expected Returns: 1Y={signal.expected_return_1y * 100:.1f}% | 3Y={signal.expected_return_3y * 100:.1f}% | 5Y={signal.expected_return_5y * 100:.1f}%")

        if signal.reasons:
            print(f"   Reasons: {', '.join(signal.reasons)}")

    # Generate portfolio recommendation
    print("\n" + "=" * 80)
    print("💼 PORTFOLIO RECOMMENDATION (₹1,00,000 Investment)")
    print("=" * 80)

    portfolio = generator.create_portfolio_recommendation(signals, total_capital=100000)

    print(f"\nAllocation Summary:")
    print(f"   Equity: {portfolio['equity_allocation']:.1f}%")
    print(f"   Cash: {portfolio['cash_allocation']:.1f}%")
    print(f"   Number of Funds: {portfolio['num_funds']}")

    if portfolio['expected_portfolio_return_1y']:
        print(f"\nExpected Portfolio Returns:")
        print(f"   1 Year: {portfolio['expected_portfolio_return_1y'] * 100:.2f}%")
        print(f"   3 Years: {portfolio['expected_portfolio_return_3y'] * 100:.2f}%")
        print(f"   5 Years: {portfolio['expected_portfolio_return_5y'] * 100:.2f}%")

    print(f"\n📊 Fund-wise Allocation:")
    print("-" * 80)

    for i, fund in enumerate(portfolio['portfolio'], 1):
        print(f"\n{i}. {fund['scheme_name'][:60]}")
        print(f"   Allocation: {fund['allocation_percent']:.1f}% (₹{fund['allocation_amount']:,.0f})")
        print(f"   Signal: {fund['signal_strength']} | Score: {fund['combined_score']:.1f}/100")

        if fund['expected_1y_return']:
            print(
                f"   Expected: 1Y={fund['expected_1y_return'] * 100:.1f}% | 3Y={fund['expected_3y_return'] * 100:.1f}% | 5Y={fund['expected_5y_return'] * 100:.1f}%")

    # Save to CSV
    signals_df = pd.DataFrame([{
        'scheme_code': s.scheme_code,
        'scheme_name': s.scheme_name,
        'signal_type': s.signal_type,
        'signal_strength': s.signal_strength,
        'votes': s.confirmation_votes,
        'tech_score': s.technical_score,
        'fund_score': s.fundamental_score,
        'combined_score': s.combined_score,
        'allocation_pct': s.recommended_allocation * 100,
        'sharpe_ratio': s.sharpe_ratio,
        'cagr_1y': s.cagr_1y,
        'expected_1y': s.expected_return_1y,
        'expected_3y': s.expected_return_3y,
        'expected_5y': s.expected_return_5y,
    } for s in signals])

    signals_df.to_csv(PROCESSED_DATA_DIR / 'signals.csv', index=False)
    print(f"\n✓ Signals saved to {PROCESSED_DATA_DIR / 'signals.csv'}")


if __name__ == "__main__":
    main()
