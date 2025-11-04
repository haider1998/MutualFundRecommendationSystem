"""
OPTIMIZED Advanced Scoring Engine
Bulk calculations + vectorization for multi-factor scoring
"""
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from src.utils.database import db, DailyNAV, FundMetadata
from src.utils.logger import log


class AdvancedScoringEngine:
    """
    Ultra-fast comprehensive scoring with vectorized operations
    """

    def __init__(self, num_workers=None):
        self.weights = {
            'performance': 0.30,
            'risk_adjusted': 0.25,
            'consistency': 0.15,
            'cost_efficiency': 0.10,
            'fund_quality': 0.10,
            'momentum': 0.10
        }
        self.num_workers = num_workers or max(1, mp.cpu_count() - 1)

        # Caches
        self.nav_data = None
        self.metadata_df = None
        self.expense_ratios = None

    def load_all_data_bulk(self):
        """Load all necessary data in bulk"""
        if self.nav_data is not None:
            return

        log.info("Loading data for advanced scoring...")
        session = db.get_session()

        # Load NAV data (18 months for rolling returns)
        cutoff_date = datetime.now().date() - timedelta(days=365 + 180)

        nav_query = session.query(
            DailyNAV.scheme_code,
            DailyNAV.date,
            DailyNAV.nav
        ).filter(
            DailyNAV.date >= cutoff_date
        ).order_by(
            DailyNAV.scheme_code,
            DailyNAV.date
        )

        self.nav_data = pd.read_sql(nav_query.statement, session.bind)
        self.nav_data['date'] = pd.to_datetime(self.nav_data['date'])

        # Load metadata (for expense ratios)
        metadata_query = session.query(
            FundMetadata.scheme_code,
            FundMetadata.scheme_category,
            FundMetadata.expense_ratio,
            FundMetadata.fund_house
        )

        self.metadata_df = pd.read_sql(metadata_query.statement, session.bind)

        session.close()

        log.info(f"✓ Loaded {len(self.nav_data):,} NAV records")
        log.info(f"✓ Loaded {len(self.metadata_df)} fund metadata")

    def calculate_consistency_scores_bulk(self, scheme_codes: List[str]) -> pd.DataFrame:
        """
        Calculate rolling return consistency for ALL funds at once
        Uses vectorized operations - MUCH faster
        """
        log.info("Calculating consistency scores...")

        self.load_all_data_bulk()

        window = 365  # 1 year rolling window

        grouped = self.nav_data.groupby('scheme_code')

        results = []

        for scheme_code in scheme_codes:
            if scheme_code not in grouped.groups:
                results.append({
                    'scheme_code': scheme_code,
                    'consistency': 50  # Default
                })
                continue

            df = grouped.get_group(scheme_code).sort_values('date').reset_index(drop=True)

            if len(df) < window + 90:  # Need at least window + 3 months
                results.append({
                    'scheme_code': scheme_code,
                    'consistency': 50
                })
                continue

            # Calculate rolling 1-year returns (vectorized)
            rolling_returns = []

            for i in range(len(df) - window):
                start_nav = df.iloc[i]['nav']
                end_nav = df.iloc[i + window]['nav']
                annual_return = (end_nav - start_nav) / start_nav
                rolling_returns.append(annual_return)

            if len(rolling_returns) < 2:
                results.append({
                    'scheme_code': scheme_code,
                    'consistency': 50
                })
                continue

            # Consistency score: inverse of standard deviation
            std_dev = np.std(rolling_returns)
            consistency_score = max(0, 100 - (std_dev * 1000))

            results.append({
                'scheme_code': scheme_code,
                'consistency': round(consistency_score, 2)
            })

        return pd.DataFrame(results)

    def calculate_momentum_scores_bulk(self, scheme_codes: List[str]) -> pd.DataFrame:
        """
        Calculate momentum for ALL funds using vectorized operations
        """
        log.info("Calculating momentum scores...")

        self.load_all_data_bulk()

        today = datetime.now()
        date_3m = today - timedelta(days=90)
        date_6m = today - timedelta(days=180)

        grouped = self.nav_data.groupby('scheme_code')

        results = []

        for scheme_code in scheme_codes:
            if scheme_code not in grouped.groups:
                results.append({
                    'scheme_code': scheme_code,
                    'momentum': 50
                })
                continue

            df = grouped.get_group(scheme_code).sort_values('date')

            if len(df) < 30:
                results.append({
                    'scheme_code': scheme_code,
                    'momentum': 50
                })
                continue

            # Get NAVs at different points (vectorized lookup)
            nav_latest = df.iloc[-1]['nav']

            # Find closest dates
            df_3m = df[df['date'] <= date_3m]
            df_6m = df[df['date'] <= date_6m]

            if len(df_3m) == 0 or len(df_6m) == 0:
                results.append({
                    'scheme_code': scheme_code,
                    'momentum': 50
                })
                continue

            nav_3m = df_3m.iloc[-1]['nav']
            nav_6m = df_6m.iloc[-1]['nav']

            # Calculate returns
            return_3m = (nav_latest - nav_3m) / nav_3m
            return_6m = (nav_latest - nav_6m) / nav_6m

            # Annualize
            return_3m_annual = return_3m * 4
            return_6m_annual = return_6m * 2

            # Momentum: is recent performance accelerating?
            if return_3m_annual > return_6m_annual:
                momentum = 50 + min(50, (return_3m_annual - return_6m_annual) * 100)
            else:
                momentum = 50 - min(50, (return_6m_annual - return_3m_annual) * 100)

            results.append({
                'scheme_code': scheme_code,
                'momentum': round(max(0, min(100, momentum)), 2)
            })

        return pd.DataFrame(results)

    def calculate_cost_efficiency_scores_bulk(self, scheme_codes: List[str]) -> pd.DataFrame:
        """
        Calculate cost efficiency for ALL funds
        Compares expense ratio to category average
        """
        log.info("Calculating cost efficiency scores...")

        self.load_all_data_bulk()

        # Calculate category average expense ratios (vectorized)
        category_avg = self.metadata_df.groupby('scheme_category')['expense_ratio'].mean().to_dict()

        results = []

        for scheme_code in scheme_codes:
            fund_meta = self.metadata_df[self.metadata_df['scheme_code'] == scheme_code]

            if len(fund_meta) == 0 or pd.isna(fund_meta.iloc[0]['expense_ratio']):
                results.append({
                    'scheme_code': scheme_code,
                    'cost_efficiency': 50
                })
                continue

            fund_meta = fund_meta.iloc[0]
            expense_ratio = fund_meta['expense_ratio']
            category = fund_meta['scheme_category']

            avg_expense = category_avg.get(category, expense_ratio)

            if avg_expense == 0:
                results.append({
                    'scheme_code': scheme_code,
                    'cost_efficiency': 50
                })
                continue

            # Score: lower expense = higher score
            ratio = expense_ratio / avg_expense
            score = 100 - ((ratio - 0.5) * 100)
            score = max(0, min(100, score))

            results.append({
                'scheme_code': scheme_code,
                'cost_efficiency': round(score, 2)
            })

        return pd.DataFrame(results)

    def calculate_fund_quality_scores_bulk(self, scheme_codes: List[str]) -> pd.DataFrame:
        """
        Calculate fund quality (AUM momentum proxy)
        Uses NAV data frequency as proxy for fund activity
        """
        log.info("Calculating fund quality scores...")

        self.load_all_data_bulk()

        today = datetime.now()
        six_months_ago = today - timedelta(days=180)
        twelve_months_ago = today - timedelta(days=365)

        # Count NAV records in each period (vectorized)
        nav_recent = self.nav_data[self.nav_data['date'] >= six_months_ago]
        nav_previous = self.nav_data[
            (self.nav_data['date'] >= twelve_months_ago) &
            (self.nav_data['date'] < six_months_ago)
        ]

        recent_counts = nav_recent.groupby('scheme_code').size().to_dict()
        previous_counts = nav_previous.groupby('scheme_code').size().to_dict()

        results = []

        for scheme_code in scheme_codes:
            recent_count = recent_counts.get(scheme_code, 0)
            previous_count = previous_counts.get(scheme_code, 0)

            if previous_count == 0:
                quality_score = 50  # Neutral for new funds
            else:
                growth_rate = (recent_count - previous_count) / previous_count
                quality_score = 50 + (growth_rate * 250)
                quality_score = max(0, min(100, quality_score))

            results.append({
                'scheme_code': scheme_code,
                'fund_quality': round(quality_score, 2)
            })

        return pd.DataFrame(results)

    def analyze_all_funds(self, basic_results: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive scoring to existing analysis results
        Uses vectorized bulk operations for maximum speed
        """
        log.info("Calculating comprehensive scores...")

        scheme_codes = basic_results['scheme_code'].unique().tolist()

        # Calculate all component scores in parallel
        log.info(f"Processing {len(scheme_codes)} funds...")

        # Get all component scores (vectorized bulk operations)
        consistency_df = self.calculate_consistency_scores_bulk(scheme_codes)
        momentum_df = self.calculate_momentum_scores_bulk(scheme_codes)
        cost_efficiency_df = self.calculate_cost_efficiency_scores_bulk(scheme_codes)
        fund_quality_df = self.calculate_fund_quality_scores_bulk(scheme_codes)

        # Merge all component scores
        comprehensive_df = basic_results[['scheme_code']].copy()

        # Performance score (from fundamental analysis)
        if 'score_fund' in basic_results.columns:
            comprehensive_df = comprehensive_df.merge(
                basic_results[['scheme_code', 'score_fund']],
                on='scheme_code',
                how='left'
            )
            comprehensive_df = comprehensive_df.rename(columns={'score_fund': 'performance'})
        else:
            comprehensive_df['performance'] = 50

        # Risk-adjusted score (from Sharpe ratio)
        if 'sharpe_ratio' in basic_results.columns:
            comprehensive_df = comprehensive_df.merge(
                basic_results[['scheme_code', 'sharpe_ratio']],
                on='scheme_code',
                how='left'
            )
            # Normalize Sharpe to 0-100 (vectorized)
            comprehensive_df['risk_adjusted'] = comprehensive_df['sharpe_ratio'].apply(
                lambda x: min(100, max(0, x * 50 + 50)) if pd.notna(x) else 50
            )
        else:
            comprehensive_df['risk_adjusted'] = 50

        # Merge component scores
        comprehensive_df = comprehensive_df.merge(consistency_df, on='scheme_code', how='left')
        comprehensive_df = comprehensive_df.merge(cost_efficiency_df, on='scheme_code', how='left')
        comprehensive_df = comprehensive_df.merge(fund_quality_df, on='scheme_code', how='left')
        comprehensive_df = comprehensive_df.merge(momentum_df, on='scheme_code', how='left')

        # Fill NaN values with neutral score
        score_columns = ['performance', 'risk_adjusted', 'consistency',
                        'cost_efficiency', 'fund_quality', 'momentum']
        comprehensive_df[score_columns] = comprehensive_df[score_columns].fillna(50)

        # Calculate weighted comprehensive score (vectorized)
        comprehensive_df['comprehensive_score'] = (
            comprehensive_df['performance'] * self.weights['performance'] +
            comprehensive_df['risk_adjusted'] * self.weights['risk_adjusted'] +
            comprehensive_df['consistency'] * self.weights['consistency'] +
            comprehensive_df['cost_efficiency'] * self.weights['cost_efficiency'] +
            comprehensive_df['fund_quality'] * self.weights['fund_quality'] +
            comprehensive_df['momentum'] * self.weights['momentum']
        ).round(2)

        # Merge back with basic results
        enhanced = basic_results.merge(
            comprehensive_df[['scheme_code', 'performance', 'risk_adjusted',
                            'consistency', 'cost_efficiency', 'fund_quality',
                            'momentum', 'comprehensive_score']],
            on='scheme_code',
            how='left'
        )

        log.info(f"✓ Comprehensive scoring complete for {len(enhanced)} funds")

        return enhanced

    def get_score_breakdown(self, scheme_code: str, comprehensive_results: pd.DataFrame) -> Dict:
        """
        Get detailed score breakdown for a single fund
        Useful for understanding what drives the score
        """
        fund_data = comprehensive_results[comprehensive_results['scheme_code'] == scheme_code]

        if len(fund_data) == 0:
            return {'error': 'Fund not found'}

        fund = fund_data.iloc[0]

        breakdown = {
            'scheme_code': scheme_code,
            'comprehensive_score': fund['comprehensive_score'],
            'components': {
                'performance': {
                    'score': fund['performance'],
                    'weight': self.weights['performance'] * 100,
                    'contribution': fund['performance'] * self.weights['performance']
                },
                'risk_adjusted': {
                    'score': fund['risk_adjusted'],
                    'weight': self.weights['risk_adjusted'] * 100,
                    'contribution': fund['risk_adjusted'] * self.weights['risk_adjusted']
                },
                'consistency': {
                    'score': fund['consistency'],
                    'weight': self.weights['consistency'] * 100,
                    'contribution': fund['consistency'] * self.weights['consistency']
                },
                'cost_efficiency': {
                    'score': fund['cost_efficiency'],
                    'weight': self.weights['cost_efficiency'] * 100,
                    'contribution': fund['cost_efficiency'] * self.weights['cost_efficiency']
                },
                'fund_quality': {
                    'score': fund['fund_quality'],
                    'weight': self.weights['fund_quality'] * 100,
                    'contribution': fund['fund_quality'] * self.weights['fund_quality']
                },
                'momentum': {
                    'score': fund['momentum'],
                    'weight': self.weights['momentum'] * 100,
                    'contribution': fund['momentum'] * self.weights['momentum']
                }
            }
        }

        # Identify strengths and weaknesses
        component_scores = {k: v['score'] for k, v in breakdown['components'].items()}
        sorted_components = sorted(component_scores.items(), key=lambda x: x[1], reverse=True)

        breakdown['strengths'] = [k for k, v in sorted_components[:3]]
        breakdown['weaknesses'] = [k for k, v in sorted_components[-3:]]

        return breakdown


def main():
    """Test optimized advanced scoring"""
    from src.engines.technical import TechnicalAnalysisEngine
    from src.engines.fundamental import FundamentalAnalysisEngine

    log.info("Running analysis engines...")

    tech_engine = TechnicalAnalysisEngine()
    fund_engine = FundamentalAnalysisEngine()

    tech_results = tech_engine.analyze_all_funds()
    fund_results = fund_engine.analyze_all_funds()

    # Combine
    combined = tech_results.merge(
        fund_results,
        on='scheme_code',
        suffixes=('_tech', '_fund')
    )

    # Add comprehensive scoring
    advanced_engine = AdvancedScoringEngine()
    final_results = advanced_engine.analyze_all_funds(combined)

    # Show top funds
    print("\n" + "=" * 80)
    print("🏆 TOP 10 FUNDS BY COMPREHENSIVE SCORE")
    print("=" * 80)

    top_10 = final_results.nlargest(10, 'comprehensive_score')[[
        'scheme_code', 'comprehensive_score', 'performance', 'risk_adjusted',
        'consistency', 'cost_efficiency', 'fund_quality', 'momentum'
    ]]

    print(top_10.to_string(index=False))

    # Show score breakdown for top fund
    if len(top_10) > 0:
        top_scheme = top_10.iloc[0]['scheme_code']

        print(f"\n📊 SCORE BREAKDOWN FOR TOP FUND ({top_scheme}):")
        print("=" * 80)

        breakdown = advanced_engine.get_score_breakdown(top_scheme, final_results)

        print(f"\nComprehensive Score: {breakdown['comprehensive_score']:.1f}/100\n")
        print(f"{'Component':<20} {'Score':>10} {'Weight':>10} {'Contribution':>15}")
        print("-" * 60)

        for component, data in breakdown['components'].items():
            print(f"{component:<20} {data['score']:>10.1f} {data['weight']:>9.1f}% {data['contribution']:>14.1f}")

        print(f"\nStrengths: {', '.join(breakdown['strengths'])}")
        print(f"Weaknesses: {', '.join(breakdown['weaknesses'])}")

    # Distribution analysis
    print("\n" + "=" * 80)
    print("📈 SCORE DISTRIBUTION")
    print("=" * 80)

    print(f"\nComprehensive Score Statistics:")
    print(f"  Mean: {final_results['comprehensive_score'].mean():.2f}")
    print(f"  Median: {final_results['comprehensive_score'].median():.2f}")
    print(f"  Std Dev: {final_results['comprehensive_score'].std():.2f}")
    print(f"  Min: {final_results['comprehensive_score'].min():.2f}")
    print(f"  Max: {final_results['comprehensive_score'].max():.2f}")

    # Quartile analysis
    quartiles = final_results['comprehensive_score'].quantile([0.25, 0.5, 0.75])
    print(f"\nQuartiles:")
    print(f"  25th percentile: {quartiles[0.25]:.2f}")
    print(f"  50th percentile: {quartiles[0.50]:.2f}")
    print(f"  75th percentile: {quartiles[0.75]:.2f}")

    # Save
    final_results.to_csv('data/processed/comprehensive_analysis.csv', index=False)
    print(f"\n✓ Saved to data/processed/comprehensive_analysis.csv")


if __name__ == "__main__":
    main()
