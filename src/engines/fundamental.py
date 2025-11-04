"""
OPTIMIZED Fundamental Analysis Engine
Uses bulk queries + vectorized calculations
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta

from config.settings import RISK_FREE_RATE
from src.utils.database import db, DailyNAV, FundMetadata
from src.utils.logger import log


class FundamentalAnalysisEngine:
    """Ultra-fast fundamental analysis"""

    def __init__(self, risk_free_rate: float = RISK_FREE_RATE):
        self.risk_free_rate = risk_free_rate
        self.nav_data_1y = None
        self.nav_data_3y = None

    def load_all_nav_data_bulk(self):
        """Load all NAV data once"""
        if self.nav_data_1y is not None:
            return

        log.info("Loading NAV data for fundamental analysis...")
        session = db.get_session()

        # Load 3 years of data (covers 1y and 3y analysis)
        cutoff_date = datetime.now().date() - timedelta(days=365 * 3 + 30)

        query = session.query(
            DailyNAV.scheme_code,
            DailyNAV.date,
            DailyNAV.nav
        ).filter(
            DailyNAV.date >= cutoff_date
        ).order_by(
            DailyNAV.scheme_code,
            DailyNAV.date
        )

        self.nav_data_3y = pd.read_sql(query.statement, session.bind)
        self.nav_data_3y['date'] = pd.to_datetime(self.nav_data_3y['date'])

        # Create 1y subset
        cutoff_1y = datetime.now().date() - timedelta(days=365)
        self.nav_data_1y = self.nav_data_3y[self.nav_data_3y['date'] >= pd.to_datetime(cutoff_1y)].copy()

        session.close()
        log.info(f"✓ Loaded {len(self.nav_data_3y):,} NAV records")

    def calculate_all_metrics_vectorized(self, scheme_codes: List[str]) -> pd.DataFrame:
        """
        Calculate ALL fundamental metrics using vectorized operations
        Processes all funds simultaneously - MASSIVE speedup
        """
        self.load_all_nav_data_bulk()

        results = []

        # Group data by scheme
        grouped_1y = self.nav_data_1y.groupby('scheme_code')
        grouped_3y = self.nav_data_3y.groupby('scheme_code')

        for scheme_code in scheme_codes:
            try:
                # Get data for this fund
                df_1y = grouped_1y.get_group(scheme_code) if scheme_code in grouped_1y.groups else pd.DataFrame()
                df_3y = grouped_3y.get_group(scheme_code) if scheme_code in grouped_3y.groups else pd.DataFrame()

                if df_1y.empty:
                    results.append({
                        'scheme_code': scheme_code,
                        'status': 'insufficient_data',
                        'score': 0
                    })
                    continue

                # Sort by date
                df_1y = df_1y.sort_values('date').reset_index(drop=True)
                df_3y = df_3y.sort_values('date').reset_index(drop=True) if not df_3y.empty else pd.DataFrame()

                # Calculate returns (vectorized)
                df_1y['returns'] = df_1y['nav'].pct_change()

                # CAGR calculations
                cagr_1y = self._calculate_cagr_fast(df_1y, years=1)
                cagr_3y = self._calculate_cagr_fast(df_3y, years=3) if not df_3y.empty else None

                # Risk metrics (vectorized)
                returns_clean = df_1y['returns'].dropna()

                if len(returns_clean) < 30:
                    results.append({
                        'scheme_code': scheme_code,
                        'status': 'insufficient_data',
                        'score': 0
                    })
                    continue

                # Annualized metrics
                mean_return = returns_clean.mean() * 252
                std_dev = returns_clean.std() * np.sqrt(252)

                # Sharpe ratio
                sharpe = (mean_return - self.risk_free_rate) / std_dev if std_dev > 0 else None

                # Sortino ratio
                downside_returns = returns_clean[returns_clean < 0]
                downside_dev = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else None
                sortino = (mean_return - self.risk_free_rate) / downside_dev if downside_dev and downside_dev > 0 else None

                # Volatility
                volatility = std_dev

                # Max drawdown (vectorized)
                cummax = df_1y['nav'].cummax()
                drawdown = (df_1y['nav'] - cummax) / cummax
                max_drawdown = drawdown.min()

                # Generate score
                metrics = {
                    'scheme_code': scheme_code,
                    'status': 'success',
                    'cagr_1y': cagr_1y,
                    'cagr_3y': cagr_3y,
                    'sharpe_ratio': sharpe,
                    'sortino_ratio': sortino,
                    'volatility': volatility,
                    'max_drawdown': max_drawdown,
                    'beta': None,  # Would need benchmark
                    'alpha': None,
                    'upside_capture': None,
                    'downside_capture': None
                }

                metrics['score'] = self.generate_fundamental_score(metrics)

                results.append(metrics)

            except Exception as e:
                log.warning(f"Error processing {scheme_code}: {e}")
                results.append({
                    'scheme_code': scheme_code,
                    'status': 'error',
                    'score': 0
                })

        return pd.DataFrame(results)

    def _calculate_cagr_fast(self, df, years):
        """Fast CAGR calculation"""
        if df.empty or len(df) < 30:
            return None

        beginning_value = df.iloc[0]['nav']
        ending_value = df.iloc[-1]['nav']

        # Calculate actual time period
        days = (df.iloc[-1]['date'] - df.iloc[0]['date']).days
        actual_years = days / 365.25

        if actual_years < years * 0.9:  # Need at least 90% of period
            return None

        cagr = (ending_value / beginning_value) ** (1 / actual_years) - 1
        return cagr

    def generate_fundamental_score(self, metrics: Dict) -> int:
        """Generate score (same logic as before)"""
        score = 50

        sharpe = metrics.get('sharpe_ratio')
        if sharpe is not None:
            if sharpe > 2.0:
                score += 20
            elif sharpe > 1.5:
                score += 15
            elif sharpe > 1.0:
                score += 10
            elif sharpe > 0.5:
                score += 5
            elif sharpe < 0:
                score -= 20

        cagr_1y = metrics.get('cagr_1y')
        if cagr_1y is not None:
            if cagr_1y > 0.20:
                score += 15
            elif cagr_1y > 0.15:
                score += 10
            elif cagr_1y > 0.10:
                score += 5
            elif cagr_1y < 0:
                score -= 15

        max_dd = metrics.get('max_drawdown')
        if max_dd is not None:
            if max_dd > -0.10:
                score += 10
            elif max_dd > -0.20:
                score += 5
            elif max_dd < -0.40:
                score -= 15

        return max(0, min(100, score))

    def analyze_all_funds(self) -> pd.DataFrame:
        """Analyze all funds (vectorized)"""
        session = db.get_session()
        scheme_codes = [f.scheme_code for f in session.query(FundMetadata.scheme_code).all()]
        session.close()

        log.info(f"Analyzing {len(scheme_codes)} funds...")

        results = self.calculate_all_metrics_vectorized(scheme_codes)

        log.info(f"✓ Fundamental analysis complete")

        return results


def main():
    engine = FundamentalAnalysisEngine()
    results = engine.analyze_all_funds()

    top_funds = results.nlargest(10, 'score')[
        ['scheme_code', 'score', 'sharpe_ratio', 'cagr_1y', 'max_drawdown']
    ]
    print("\n📊 TOP 10 FUNDS BY FUNDAMENTAL SCORE:")
    print(top_funds.to_string(index=False))

    results.to_csv('data/processed/fundamental_scores.csv', index=False)


if __name__ == "__main__":
    main()
