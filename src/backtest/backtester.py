"""
Backtesting Engine for Strategy Validation
Tests strategy across historical data to validate performance
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json

from src.utils.database import db, DailyNAV, FundMetadata
from src.utils.logger import log
from src.signals.signal_generator import SignalGenerator, SignalConfig
from config.settings import PROCESSED_DATA_DIR


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    start_date: str  # "2020-01-01"
    end_date: str  # "2024-01-01"
    initial_capital: float = 100000
    rebalance_frequency: int = 30  # days
    transaction_cost: float = 0.001  # 0.1% per transaction

    # Strategy parameters
    min_votes_for_buy: int = 2
    technical_threshold: float = 1.0
    fundamental_threshold: float = 50
    stop_loss: float = -0.15
    profit_target: float = 0.25


@dataclass
class BacktestResult:
    """Results from backtesting"""
    config: BacktestConfig

    # Performance metrics
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float

    # Time series data
    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame
    monthly_returns: pd.Series

    # Benchmark comparison
    benchmark_return: float
    alpha: float
    beta: float

    # Risk metrics
    volatility: float
    downside_deviation: float
    max_consecutive_losses: int

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'config': asdict(self.config),
            'total_return': self.total_return,
            'cagr': self.cagr,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'benchmark_return': self.benchmark_return,
            'alpha': self.alpha,
            'beta': self.beta,
            'volatility': self.volatility,
            'downside_deviation': self.downside_deviation,
            'max_consecutive_losses': self.max_consecutive_losses
        }


class Backtester:
    """Backtest trading strategy on historical data"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.signal_config = SignalConfig(
            min_votes_for_buy=config.min_votes_for_buy,
            technical_buy_threshold=config.technical_threshold,
            fundamental_buy_threshold=config.fundamental_threshold,
            stop_loss_percent=config.stop_loss,
            profit_target_percent=config.profit_target
        )

    def get_historical_nav_data(self, scheme_code: str,
                                start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical NAV data for a fund"""
        session = db.get_session()

        navs = session.query(
            DailyNAV.date,
            DailyNAV.nav
        ).filter(
            DailyNAV.scheme_code == scheme_code,
            DailyNAV.date >= start_date,
            DailyNAV.date <= end_date
        ).order_by(
            DailyNAV.date
        ).all()

        session.close()

        if not navs:
            return pd.DataFrame()

        df = pd.DataFrame(navs, columns=['date', 'nav'])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)

        return df

    def run_backtest(self, scheme_codes: List[str] = None) -> BacktestResult:
        """
        Run backtest on selected funds

        Args:
            scheme_codes: List of funds to include (None = use top ranked funds)
        """
        log.info(f"Running backtest from {self.config.start_date} to {self.config.end_date}")

        # Initialize portfolio
        cash = self.config.initial_capital
        positions = {}  # {scheme_code: {'shares': X, 'entry_price': Y, 'entry_date': Z}}

        # Track equity over time
        equity_curve = []
        trade_log = []

        # Generate trading dates (monthly rebalancing)
        start = pd.to_datetime(self.config.start_date)
        end = pd.to_datetime(self.config.end_date)

        trading_dates = pd.date_range(start, end, freq=f'{self.config.rebalance_frequency}D')

        # Run simulation
        for trade_date in trading_dates:
            log.info(f"Processing date: {trade_date.date()}")

            # Get current portfolio value
            portfolio_value = cash

            for scheme_code, position in positions.items():
                # Get current NAV
                nav_data = self.get_historical_nav_data(
                    scheme_code,
                    trade_date.strftime('%Y-%m-%d'),
                    trade_date.strftime('%Y-%m-%d')
                )

                if not nav_data.empty:
                    current_nav = nav_data.iloc[0]['nav']
                    position_value = position['shares'] * current_nav
                    portfolio_value += position_value

            # Record equity
            equity_curve.append({
                'date': trade_date,
                'equity': portfolio_value,
                'cash': cash,
                'positions_value': portfolio_value - cash
            })

            # Check for exits (stop-loss, profit target)
            for scheme_code in list(positions.keys()):
                position = positions[scheme_code]

                nav_data = self.get_historical_nav_data(
                    scheme_code,
                    trade_date.strftime('%Y-%m-%d'),
                    trade_date.strftime('%Y-%m-%d')
                )

                if nav_data.empty:
                    continue

                current_nav = nav_data.iloc[0]['nav']
                entry_price = position['entry_price']

                pnl_pct = (current_nav - entry_price) / entry_price

                # Check exit conditions
                should_exit = False
                exit_reason = ""

                if pnl_pct <= self.config.stop_loss:
                    should_exit = True
                    exit_reason = "Stop-loss"
                elif pnl_pct >= self.config.profit_target:
                    should_exit = True
                    exit_reason = "Profit target"

                # Exit position
                if should_exit:
                    position_value = position['shares'] * current_nav
                    cash += position_value * (1 - self.config.transaction_cost)

                    # Log trade
                    trade_log.append({
                        'scheme_code': scheme_code,
                        'entry_date': position['entry_date'],
                        'exit_date': trade_date,
                        'entry_price': entry_price,
                        'exit_price': current_nav,
                        'shares': position['shares'],
                        'pnl': position_value - (position['shares'] * entry_price),
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason
                    })

                    del positions[scheme_code]
                    log.info(f"Exited {scheme_code}: {exit_reason} (P&L: {pnl_pct * 100:.2f}%)")

            # Generate new signals (simplified - would normally use full signal generation)
            # For now, use a simple rule: buy top funds if we have cash
            if cash > self.config.initial_capital * 0.1:  # Keep 10% cash reserve

                # Get available funds
                if scheme_codes:
                    available_funds = scheme_codes[:5]  # Top 5
                else:
                    session = db.get_session()
                    available_funds = [f.scheme_code for f in
                                       session.query(FundMetadata).limit(10).all()]
                    session.close()

                # Allocate to top funds (equal weight)
                funds_to_buy = [f for f in available_funds if f not in positions][:3]

                if funds_to_buy:
                    allocation_per_fund = (cash * 0.8) / len(funds_to_buy)  # Use 80% of cash

                    for scheme_code in funds_to_buy:
                        nav_data = self.get_historical_nav_data(
                            scheme_code,
                            trade_date.strftime('%Y-%m-%d'),
                            trade_date.strftime('%Y-%m-%d')
                        )

                        if not nav_data.empty:
                            entry_nav = nav_data.iloc[0]['nav']
                            shares = allocation_per_fund / entry_nav

                            # Deduct from cash
                            cost = allocation_per_fund * (1 + self.config.transaction_cost)
                            cash -= cost

                            # Add position
                            positions[scheme_code] = {
                                'shares': shares,
                                'entry_price': entry_nav,
                                'entry_date': trade_date
                            }

                            log.info(f"Entered {scheme_code} at ₹{entry_nav:.2f}")

        # Close all positions at end
        final_date = pd.to_datetime(self.config.end_date)

        for scheme_code, position in positions.items():
            nav_data = self.get_historical_nav_data(
                scheme_code,
                final_date.strftime('%Y-%m-%d'),
                final_date.strftime('%Y-%m-%d')
            )

            if not nav_data.empty:
                exit_nav = nav_data.iloc[0]['nav']
                position_value = position['shares'] * exit_nav
                cash += position_value * (1 - self.config.transaction_cost)

                pnl_pct = (exit_nav - position['entry_price']) / position['entry_price']

                trade_log.append({
                    'scheme_code': scheme_code,
                    'entry_date': position['entry_date'],
                    'exit_date': final_date,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_nav,
                    'shares': position['shares'],
                    'pnl': position_value - (position['shares'] * position['entry_price']),
                    'pnl_pct': pnl_pct,
                    'exit_reason': 'End of backtest'
                })

        # Calculate metrics
        equity_df = pd.DataFrame(equity_curve)
        trades_df = pd.DataFrame(trade_log)

        result = self._calculate_metrics(equity_df, trades_df)

        log.info(f"✓ Backtest complete. Total Return: {result.total_return * 100:.2f}%")

        return result

    def _calculate_metrics(self, equity_df: pd.DataFrame,
                           trades_df: pd.DataFrame) -> BacktestResult:
        """Calculate performance metrics from backtest results"""

        # Total return
        initial = self.config.initial_capital
        final = equity_df.iloc[-1]['equity']
        total_return = (final - initial) / initial

        # CAGR
        days = (equity_df.iloc[-1]['date'] - equity_df.iloc[0]['date']).days
        years = days / 365.25
        cagr = (final / initial) ** (1 / years) - 1 if years > 0 else 0

        # Returns series
        equity_df['returns'] = equity_df['equity'].pct_change()

        # Sharpe ratio
        mean_return = equity_df['returns'].mean() * 252  # Annualized
        std_return = equity_df['returns'].std() * np.sqrt(252)
        sharpe = (mean_return - 0.065) / std_return if std_return > 0 else 0

        # Sortino ratio
        downside_returns = equity_df[equity_df['returns'] < 0]['returns']
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino = (mean_return - 0.065) / downside_std if downside_std > 0 else 0

        # Maximum drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()

        # Calmar ratio
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade statistics
        if not trades_df.empty:
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            total_trades = len(trades_df)

            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0

            profit_factor = (winning_trades * avg_win) / (losing_trades * avg_loss) if losing_trades > 0 else 0

            # Max consecutive losses
            trades_df['is_loss'] = trades_df['pnl'] < 0
            consecutive_losses = (trades_df['is_loss'] != trades_df['is_loss'].shift()).cumsum()
            max_consecutive = trades_df[trades_df['is_loss']].groupby(
                consecutive_losses).size().max() if losing_trades > 0 else 0
        else:
            total_trades = winning_trades = losing_trades = 0
            win_rate = avg_win = avg_loss = profit_factor = max_consecutive = 0

        # Monthly returns
        equity_df['month'] = equity_df['date'].dt.to_period('M')
        monthly_returns = equity_df.groupby('month')['returns'].sum()

        # Benchmark (simplified - would normally compare to actual index)
        benchmark_return = 0.12 * years  # Assume 12% benchmark CAGR

        alpha = cagr - benchmark_return
        beta = 1.0  # Simplified

        # Volatility
        volatility = std_return
        downside_deviation = downside_std

        return BacktestResult(
            config=self.config,
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            equity_curve=equity_df,
            trade_log=trades_df,
            monthly_returns=monthly_returns,
            benchmark_return=benchmark_return,
            alpha=alpha,
            beta=beta,
            volatility=volatility,
            downside_deviation=downside_deviation,
            max_consecutive_losses=max_consecutive
        )

    def save_results(self, result: BacktestResult, filename: str = "backtest_results"):
        """Save backtest results"""

        # Save metrics as JSON
        metrics = result.to_dict()

        with open(PROCESSED_DATA_DIR / f"{filename}_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save equity curve
        result.equity_curve.to_csv(PROCESSED_DATA_DIR / f"{filename}_equity.csv", index=False)

        # Save trade log
        result.trade_log.to_csv(PROCESSED_DATA_DIR / f"{filename}_trades.csv", index=False)

        log.info(f"✓ Backtest results saved to {PROCESSED_DATA_DIR}/{filename}_*")


def main():
    """Test backtesting"""

    config = BacktestConfig(
        start_date="2001-01-01",
        end_date="2025-10-01",
        initial_capital=100000,
        rebalance_frequency=30,
        min_votes_for_buy=2,
        stop_loss=-0.15,
        profit_target=0.25
    )

    backtester = Backtester(config)
    result = backtester.run_backtest()

    # Display results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)

    print(f"\n📊 Performance Metrics:")
    print(f"   Total Return: {result.total_return * 100:.2f}%")
    print(f"   CAGR: {result.cagr * 100:.2f}%")
    print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"   Sortino Ratio: {result.sortino_ratio:.2f}")
    print(f"   Max Drawdown: {result.max_drawdown * 100:.2f}%")
    print(f"   Calmar Ratio: {result.calmar_ratio:.2f}")

    print(f"\n📈 Trade Statistics:")
    print(f"   Total Trades: {result.total_trades}")
    print(f"   Winning Trades: {result.winning_trades}")
    print(f"   Losing Trades: {result.losing_trades}")
    print(f"   Win Rate: {result.win_rate * 100:.1f}%")
    print(f"   Avg Win: ₹{result.avg_win:,.2f}")
    print(f"   Avg Loss: ₹{result.avg_loss:,.2f}")
    print(f"   Profit Factor: {result.profit_factor:.2f}")

    print(f"\n🎯 Risk Metrics:")
    print(f"   Volatility: {result.volatility * 100:.2f}%")
    print(f"   Downside Deviation: {result.downside_deviation * 100:.2f}%")
    print(f"   Max Consecutive Losses: {result.max_consecutive_losses}")

    print(f"\n📊 vs Benchmark:")
    print(f"   Benchmark Return: {result.benchmark_return * 100:.2f}%")
    print(f"   Alpha: {result.alpha * 100:.2f}%")

    # Save results
    backtester.save_results(result)

    print("\n✓ Results saved!")


if __name__ == "__main__":
    main()
