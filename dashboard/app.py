"""
Streamlit Dashboard for Mutual Fund Intelligence System
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from src.backtest.backtester import Backtester, BacktestConfig

from config.settings import PROCESSED_DATA_DIR
from src.utils.database import db, FundMetadata

# Page configuration
st.set_page_config(
    page_title="MF Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .buy-signal {
        color: #00c853;
        font-weight: bold;
    }
    .sell-signal {
        color: #ff1744;
        font-weight: bold;
    }
    .hold-signal {
        color: #ffa726;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load fund names mapping
@st.cache_data
def load_fund_names():
    """Load fund names from database"""
    session = db.get_session()
    funds = session.query(FundMetadata.scheme_code, FundMetadata.scheme_name).all()
    session.close()

    return {str(code): name for code, name in funds}

# Load data
@st.cache_data
def load_data():
    """Load all processed data"""
    try:
        comprehensive = pd.read_csv(PROCESSED_DATA_DIR / 'comprehensive_analysis.csv')
        signals = pd.read_csv(PROCESSED_DATA_DIR / 'signals.csv')

        # Try to load portfolio (may not exist if no buy signals)
        try:
            portfolio = pd.read_csv(PROCESSED_DATA_DIR / 'recommended_portfolio.csv')
        except:
            portfolio = pd.DataFrame()

        # Add fund names to all dataframes
        fund_names = load_fund_names()

        comprehensive['fund_name'] = comprehensive['scheme_code'].astype(str).map(fund_names)

        if 'scheme_name' not in signals.columns:
            signals['fund_name'] = signals['scheme_code'].astype(str).map(fund_names)
        else:
            signals['fund_name'] = signals['scheme_name']

        if not portfolio.empty and 'scheme_name' not in portfolio.columns:
            portfolio['fund_name'] = portfolio['scheme_code'].astype(str).map(fund_names)
        elif not portfolio.empty:
            portfolio['fund_name'] = portfolio['scheme_name']

        return comprehensive, signals, portfolio
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Elite Mutual Fund Intelligence System</h1>',
                unsafe_allow_html=True)

    # Load data
    comprehensive, signals, portfolio = load_data()

    if comprehensive.empty:
        st.error("No data available. Please run the master pipeline first.")
        st.code("python scripts/master_pipeline.py")
        return

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "🏠 Overview",
            "📈 Fund Analysis",
            "💼 Portfolio",
            "🎯 Signals",
            "🔍 Fund Search",
            "📊 Backtest Results",
            "⚙️ Parameter Optimization"
        ]
    )

    # Route to pages
    if page == "🏠 Overview":
        show_overview(comprehensive, signals, portfolio)
    elif page == "📈 Fund Analysis":
        show_fund_analysis(comprehensive)
    elif page == "💼 Portfolio":
        show_portfolio(portfolio)
    elif page == "🎯 Signals":
        show_signals(signals, comprehensive)
    elif page == "🔍 Fund Search":
        show_fund_search(comprehensive, signals)
    elif page == "📊 Backtest Results":
        show_backtest_results()
    elif page == "⚙️ Parameter Optimization":
        show_parameter_optimization()


def show_overview(comprehensive, signals, portfolio):
    """Overview page with key metrics"""

    st.header("📊 System Overview")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Funds Analyzed",
            len(comprehensive),
            delta=None
        )

    with col2:
        buy_count = len(signals[signals['signal_type'] == 'BUY'])
        st.metric(
            "BUY Signals",
            buy_count,
            delta=f"{buy_count/len(signals)*100:.1f}%" if len(signals) > 0 else "0%"
        )

    with col3:
        avg_score = comprehensive['comprehensive_score'].mean()
        st.metric(
            "Avg Comprehensive Score",
            f"{avg_score:.1f}/100",
            delta=None
        )

    with col4:
        if not portfolio.empty and 'expected_1y_return' in portfolio.columns:
            portfolio_return = portfolio['expected_1y_return'].mean() * 100
            st.metric(
                "Expected Portfolio Return (1Y)",
                f"{portfolio_return:.2f}%",
                delta=None
            )
        else:
            st.metric("Portfolio Funds", len(portfolio) if not portfolio.empty else 0)

    st.markdown("---")

    # Score distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Comprehensive Score Distribution")

        fig = px.histogram(
            comprehensive,
            x='comprehensive_score',
            nbins=20,
            title="Distribution of Comprehensive Scores",
            labels={'comprehensive_score': 'Comprehensive Score', 'count': 'Number of Funds'},
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Signal Distribution")

        signal_counts = signals['signal_type'].value_counts()

        fig = px.pie(
            values=signal_counts.values,
            names=signal_counts.index,
            title="Signal Type Distribution",
            color_discrete_map={
                'BUY': '#00c853',
                'SELL': '#ff1744',
                'HOLD': '#ffa726',
                'AVOID': '#9e9e9e'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Top performers
    st.subheader("🏆 Top 10 Performing Funds")

    top_10 = comprehensive.nlargest(10, 'comprehensive_score')[[
        'fund_name', 'comprehensive_score', 'performance', 'risk_adjusted',
        'consistency', 'sharpe_ratio', 'cagr_1y'
    ]].copy()

    # Format percentages
    if 'cagr_1y' in top_10.columns:
        top_10['cagr_1y'] = top_10['cagr_1y'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")

    # Round scores
    for col in ['comprehensive_score', 'performance', 'risk_adjusted', 'consistency']:
        if col in top_10.columns:
            top_10[col] = top_10[col].round(1)

    if 'sharpe_ratio' in top_10.columns:
        top_10['sharpe_ratio'] = top_10['sharpe_ratio'].round(2)

    # Rename columns for display
    top_10 = top_10.rename(columns={
        'fund_name': 'Fund Name',
        'comprehensive_score': 'Score',
        'performance': 'Performance',
        'risk_adjusted': 'Risk-Adj',
        'consistency': 'Consistency',
        'sharpe_ratio': 'Sharpe',
        'cagr_1y': '1Y CAGR'
    })

    st.dataframe(
        top_10,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # Factor breakdown heatmap
    st.subheader("🔥 Top 20 Funds - Factor Breakdown Heatmap")

    top_20 = comprehensive.nlargest(20, 'comprehensive_score')

    factor_cols = ['performance', 'risk_adjusted', 'consistency',
                   'cost_efficiency', 'fund_quality', 'momentum']

    # Use fund names (truncated) for x-axis
    top_20['display_name'] = top_20['fund_name'].str[:30]

    heatmap_data = top_20[['display_name'] + factor_cols].set_index('display_name')

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values.T,
        x=heatmap_data.index,
        y=factor_cols,
        colorscale='RdYlGn',
        text=heatmap_data.values.T,
        texttemplate='%{text:.1f}',
        textfont={"size": 10},
        colorbar=dict(title="Score")
    ))

    fig.update_layout(
        title="Factor Scores for Top 20 Funds",
        xaxis_title="Fund Name",
        yaxis_title="Factor",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)


def show_fund_analysis(comprehensive):
    """Detailed fund analysis page"""

    st.header("📈 Detailed Fund Analysis")

    # Filters
    st.sidebar.subheader("Filters")

    # Score range filter
    score_range = st.sidebar.slider(
        "Comprehensive Score Range",
        0, 100,
        (0, 100)
    )

    # Category filter
    if 'category' in comprehensive.columns:
        categories = ['All'] + sorted(comprehensive['category'].dropna().unique().tolist())
        selected_category = st.sidebar.selectbox("Category", categories)
    else:
        selected_category = 'All'

    # Apply filters
    filtered = comprehensive[
        (comprehensive['comprehensive_score'] >= score_range[0]) &
        (comprehensive['comprehensive_score'] <= score_range[1])
    ]

    if selected_category != 'All' and 'category' in comprehensive.columns:
        filtered = filtered[filtered['category'] == selected_category]

    st.info(f"Showing {len(filtered)} funds")

    # Scatter plot: Risk vs Return
    st.subheader("📊 Risk vs Return Analysis")

    col1, col2 = st.columns(2)

    with col1:
        if 'sharpe_ratio' in filtered.columns and 'cagr_1y' in filtered.columns:
            fig = px.scatter(
                filtered,
                x='sharpe_ratio',
                y='cagr_1y',
                size='comprehensive_score',
                color='comprehensive_score',
                hover_data=['fund_name'],
                title="Sharpe Ratio vs CAGR",
                labels={
                    'sharpe_ratio': 'Sharpe Ratio',
                    'cagr_1y': '1-Year CAGR',
                    'comprehensive_score': 'Score'
                },
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Sharpe ratio or CAGR data not available")

    with col2:
        # Performance vs Risk-Adjusted scatter
        fig = px.scatter(
            filtered,
            x='performance',
            y='risk_adjusted',
            size='comprehensive_score',
            color='comprehensive_score',
            hover_data=['fund_name'],
            title="Performance vs Risk-Adjusted Score",
            labels={
                'performance': 'Performance Score',
                'risk_adjusted': 'Risk-Adjusted Score',
                'comprehensive_score': 'Score'
            },
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Factor comparison
    st.subheader("📊 Factor Score Comparison")

    factor_cols = ['performance', 'risk_adjusted', 'consistency',
                   'cost_efficiency', 'fund_quality', 'momentum']

    factor_avg = filtered[factor_cols].mean()

    fig = go.Figure(data=[
        go.Bar(
            x=factor_cols,
            y=factor_avg.values,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        )
    ])

    fig.update_layout(
        title="Average Factor Scores",
        xaxis_title="Factor",
        yaxis_title="Average Score",
        yaxis_range=[0, 100]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Data table
    st.subheader("📋 Detailed Fund List")

    # Select columns to display
    display_cols = ['fund_name', 'comprehensive_score', 'performance', 'risk_adjusted',
                   'consistency', 'cost_efficiency', 'fund_quality', 'momentum']

    # Add optional columns if available
    if 'sharpe_ratio' in filtered.columns:
        display_cols.append('sharpe_ratio')
    if 'cagr_1y' in filtered.columns:
        display_cols.append('cagr_1y')
    if 'category' in filtered.columns:
        display_cols.insert(1, 'category')

    # Filter to available columns
    display_cols = [col for col in display_cols if col in filtered.columns]

    display_df = filtered[display_cols].sort_values('comprehensive_score', ascending=False).copy()

    # Rename for better display
    display_df = display_df.rename(columns={
        'fund_name': 'Fund Name',
        'comprehensive_score': 'Score',
        'performance': 'Performance',
        'risk_adjusted': 'Risk-Adj',
        'consistency': 'Consistency',
        'cost_efficiency': 'Cost',
        'fund_quality': 'Quality',
        'momentum': 'Momentum',
        'sharpe_ratio': 'Sharpe',
        'cagr_1y': '1Y CAGR',
        'category': 'Category'
    })

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name="fund_analysis.csv",
        mime="text/csv"
    )


def show_portfolio(portfolio):
    """Portfolio recommendation page"""

    st.header("💼 Recommended Portfolio")

    if portfolio.empty:
        st.warning("No portfolio recommendations available. No funds met the BUY criteria.")
        st.info("💡 Tip: Try adjusting the SignalConfig thresholds in master_pipeline.py")
        return

    # Portfolio summary
    total_allocation = portfolio['allocation_percent'].sum()
    cash_allocation = 100 - total_allocation

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Funds", len(portfolio))

    with col2:
        st.metric("Equity Allocation", f"{total_allocation:.1f}%")

    with col3:
        st.metric("Cash Reserve", f"{cash_allocation:.1f}%")

    with col4:
        if 'expected_1y_return' in portfolio.columns:
            weighted_return = (portfolio['allocation_percent'] * portfolio['expected_1y_return']).sum() / 100
            st.metric("Expected 1Y Return", f"{weighted_return*100:.2f}%")

    st.markdown("---")

    # Allocation pie chart
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Portfolio Allocation")

        # Create data for pie chart
        pie_data = portfolio[['fund_name', 'allocation_percent']].copy()

        # Truncate long names
        pie_data['fund_name'] = pie_data['fund_name'].str[:50]

        fig = px.pie(
            pie_data,
            values='allocation_percent',
            names='fund_name',
            title="Fund Allocation Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📈 Expected Returns by Fund")

        if 'expected_1y_return' in portfolio.columns:
            # Prepare data
            chart_data = portfolio[['fund_name', 'expected_1y_return']].copy()
            chart_data['fund_name'] = chart_data['fund_name'].str[:40]

            fig = px.bar(
                chart_data,
                x='fund_name',
                y='expected_1y_return',
                title="1-Year Expected Returns",
                labels={'expected_1y_return': 'Expected Return', 'fund_name': 'Fund'},
                color='expected_1y_return',
                color_continuous_scale='RdYlGn'
            )

            # FIX: Use update_layout instead of update_xaxis
            fig.update_layout(
                xaxis=dict(tickangle=-45),
                xaxis_title="Fund Name",
                yaxis_title="Expected Return"
            )

            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Detailed portfolio table
    st.subheader("📋 Portfolio Details")

    # Format for display
    display_portfolio = portfolio.copy()

    # Select and reorder columns
    display_cols = ['fund_name', 'allocation_percent', 'allocation_amount']

    if 'signal_strength' in display_portfolio.columns:
        display_cols.append('signal_strength')

    if 'combined_score' in display_portfolio.columns:
        display_cols.append('combined_score')

    if 'expected_1y_return' in display_portfolio.columns:
        display_cols.append('expected_1y_return')

    if 'expected_3y_return' in display_portfolio.columns:
        display_cols.append('expected_3y_return')

    if 'expected_5y_return' in display_portfolio.columns:
        display_cols.append('expected_5y_return')

    # Filter to available columns
    display_cols = [col for col in display_cols if col in display_portfolio.columns]
    display_portfolio = display_portfolio[display_cols].copy()

    # Format values
    if 'allocation_percent' in display_portfolio.columns:
        display_portfolio['allocation_percent'] = display_portfolio['allocation_percent'].apply(lambda x: f"{x:.2f}%")

    if 'allocation_amount' in display_portfolio.columns:
        display_portfolio['allocation_amount'] = display_portfolio['allocation_amount'].apply(lambda x: f"₹{x:,.0f}")

    if 'expected_1y_return' in display_portfolio.columns:
        display_portfolio['expected_1y_return'] = display_portfolio['expected_1y_return'].apply(
            lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
        )

    if 'expected_3y_return' in display_portfolio.columns:
        display_portfolio['expected_3y_return'] = display_portfolio['expected_3y_return'].apply(
            lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
        )

    if 'expected_5y_return' in display_portfolio.columns:
        display_portfolio['expected_5y_return'] = display_portfolio['expected_5y_return'].apply(
            lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
        )

    # Rename columns
    display_portfolio = display_portfolio.rename(columns={
        'fund_name': 'Fund Name',
        'allocation_percent': 'Allocation %',
        'allocation_amount': 'Amount',
        'signal_strength': 'Signal',
        'combined_score': 'Score',
        'expected_1y_return': '1Y Return',
        'expected_3y_return': '3Y Return',
        'expected_5y_return': '5Y Return'
    })

    st.dataframe(display_portfolio, use_container_width=True, hide_index=True)

    # Investment calculator
    st.markdown("---")
    st.subheader("💰 Investment Calculator")

    investment_amount = st.number_input(
        "Enter investment amount (₹)",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=10000
    )

    calculated_portfolio = portfolio.copy()
    calculated_portfolio['investment'] = (calculated_portfolio['allocation_percent'] / 100) * investment_amount

    st.write(f"**Investment breakdown for ₹{investment_amount:,}:**")

    for _, row in calculated_portfolio.iterrows():
        st.write(f"- **{row['fund_name'][:70]}**: ₹{row['investment']:,.0f} ({row['allocation_percent']:.1f}%)")

    st.write(f"- **Cash Reserve**: ₹{cash_allocation/100 * investment_amount:,.0f} ({cash_allocation:.1f}%)")


def show_signals(signals, comprehensive):
    """Signals page"""

    st.header("🎯 Buy/Sell/Hold Signals")

    # Signal filter
    signal_type = st.sidebar.selectbox(
        "Filter by Signal Type",
        ['All', 'BUY', 'SELL', 'HOLD', 'AVOID']
    )

    if signal_type != 'All':
        filtered_signals = signals[signals['signal_type'] == signal_type]
    else:
        filtered_signals = signals

    st.info(f"Showing {len(filtered_signals)} signals")

    # Tabs for different signal types
    tab1, tab2, tab3, tab4 = st.tabs(["🟢 BUY", "🔴 SELL", "🟡 HOLD", "⚫ AVOID"])

    with tab1:
        show_signal_table(signals[signals['signal_type'] == 'BUY'], 'BUY', comprehensive)

    with tab2:
        show_signal_table(signals[signals['signal_type'] == 'SELL'], 'SELL', comprehensive)

    with tab3:
        show_signal_table(signals[signals['signal_type'] == 'HOLD'], 'HOLD', comprehensive)

    with tab4:
        show_signal_table(signals[signals['signal_type'] == 'AVOID'], 'AVOID', comprehensive)


def show_signal_table(signal_df, signal_type, comprehensive):
    """Display signal table with details"""

    if signal_df.empty:
        st.info(f"No {signal_type} signals generated.")
        return

    st.subheader(f"{signal_type} Signals ({len(signal_df)} funds)")

    # Merge with comprehensive data
    merge_cols = ['scheme_code']
    comp_display_cols = ['comprehensive_score', 'sharpe_ratio', 'cagr_1y']

    # Only merge columns that exist in comprehensive
    available_comp_cols = [col for col in comp_display_cols if col in comprehensive.columns]

    display_df = signal_df.merge(
        comprehensive[merge_cols + available_comp_cols],
        on='scheme_code',
        how='left'
    )

    # Determine which column to sort by
    sort_column = None
    if 'comprehensive_score' in display_df.columns:
        sort_column = 'comprehensive_score'
    elif 'comprehensive_score_x' in display_df.columns:
        sort_column = 'comprehensive_score_x'
    elif 'comprehensive_score_y' in display_df.columns:
        sort_column = 'comprehensive_score_y'

    # Sort if we have a sort column
    if sort_column:
        display_df = display_df.sort_values(sort_column, ascending=False)

    # Select columns to display
    display_cols = ['fund_name', 'signal_strength']

    # Add score column (with fallback)
    score_col = None
    for possible_col in ['comprehensive_score', 'comprehensive_score_x', 'comprehensive_score_y']:
        if possible_col in display_df.columns:
            score_col = possible_col
            display_cols.append(score_col)
            break

    if 'allocation_pct' in display_df.columns and signal_type == 'BUY':
        display_cols.append('allocation_pct')

    if 'sharpe_ratio' in display_df.columns:
        display_cols.append('sharpe_ratio')

    if 'cagr_1y' in display_df.columns:
        display_cols.append('cagr_1y')

    if 'reasons' in display_df.columns:
        display_cols.append('reasons')

    if 'warnings' in display_df.columns:
        display_cols.append('warnings')

    # Filter to available columns
    display_cols = [col for col in display_cols if col in display_df.columns]

    final_df = display_df[display_cols].copy()

    # Rename columns
    rename_map = {
        'fund_name': 'Fund Name',
        'signal_strength': 'Strength',
        'allocation_pct': 'Allocation %',
        'sharpe_ratio': 'Sharpe',
        'cagr_1y': '1Y CAGR',
        'reasons': 'Reasons',
        'warnings': 'Warnings'
    }

    # Add score column to rename map
    if score_col:
        rename_map[score_col] = 'Score'

    final_df = final_df.rename(columns=rename_map)

    st.dataframe(
        final_df,
        use_container_width=True,
        hide_index=True
    )


def show_fund_search(comprehensive, signals):
    """Fund search and deep dive"""

    st.header("🔍 Fund Search & Analysis")

    # Search box
    search_term = st.text_input(
        "Search by Fund Name or Scheme Code",
        placeholder="e.g., HDFC or 100078"
    )

    if search_term:
        # Find matching funds
        matches = comprehensive[
            (comprehensive['scheme_code'].astype(str).str.contains(search_term, case=False)) |
            (comprehensive['fund_name'].astype(str).str.contains(search_term, case=False))
        ]

        if matches.empty:
            st.warning("No funds found matching your search.")
        else:
            st.success(f"Found {len(matches)} matching fund(s)")

            for _, fund in matches.iterrows():
                show_fund_details(fund, signals)
    else:
        st.info("Enter a fund name or scheme code to search")


def show_fund_details(fund, signals):
    """Show detailed analysis for a single fund"""

    st.markdown("---")
    st.subheader(f"📊 {fund['fund_name']}")
    st.caption(f"Scheme Code: {fund['scheme_code']}")

    # Get signal for this fund
    fund_signal = signals[signals['scheme_code'] == fund['scheme_code']]

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Comprehensive Score", f"{fund['comprehensive_score']:.1f}/100")

    with col2:
        if not fund_signal.empty:
            signal = fund_signal.iloc[0]['signal_type']
            st.metric("Signal", signal)
        else:
            st.metric("Signal", "N/A")

    with col3:
        if 'sharpe_ratio' in fund.index and pd.notna(fund['sharpe_ratio']):
            st.metric("Sharpe Ratio", f"{fund['sharpe_ratio']:.2f}")
        else:
            st.metric("Sharpe Ratio", "N/A")

    with col4:
        if 'cagr_1y' in fund.index and pd.notna(fund['cagr_1y']):
            st.metric("1Y CAGR", f"{fund['cagr_1y']*100:.2f}%")
        else:
            st.metric("1Y CAGR", "N/A")

    # Factor breakdown
    st.subheader("Factor Breakdown")

    factors = {
        'Performance': fund['performance'],
        'Risk-Adjusted': fund['risk_adjusted'],
        'Consistency': fund['consistency'],
        'Cost Efficiency': fund['cost_efficiency'],
        'Fund Quality': fund['fund_quality'],
        'Momentum': fund['momentum']
    }

    fig = go.Figure(data=[
        go.Bar(
            x=list(factors.keys()),
            y=list(factors.values()),
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        )
    ])

    fig.update_layout(
        title="Factor Scores",
        yaxis_range=[0, 100],
        yaxis_title="Score"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Signal details
    if not fund_signal.empty:
        st.subheader("Signal Details")

        signal_data = fund_signal.iloc[0]

        col1, col2 = st.columns(2)

        with col1:
            if 'reasons' in signal_data.index and pd.notna(signal_data['reasons']):
                st.write("**Reasons:**")
                for reason in str(signal_data['reasons']).split(';'):
                    if reason.strip():
                        st.write(f"- {reason.strip()}")

        with col2:
            if 'warnings' in signal_data.index and pd.notna(signal_data['warnings']):
                st.write("**Warnings:**")
                for warning in str(signal_data['warnings']).split(';'):
                    if warning.strip():
                        st.write(f"- {warning.strip()}")


def show_backtest_results():
    """Display backtest results with visualizations"""

    st.header("📊 Backtest Results & Performance Analysis")

    # Load backtest results
    try:
        with open(PROCESSED_DATA_DIR / 'backtest_results_metrics.json', 'r') as f:
            metrics = json.load(f)

        equity_df = pd.read_csv(PROCESSED_DATA_DIR / 'backtest_results_equity.csv')
        equity_df['date'] = pd.to_datetime(equity_df['date'])

        trades_df = pd.read_csv(PROCESSED_DATA_DIR / 'backtest_results_trades.csv')
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])

    except Exception as e:
        st.warning("No backtest results found. Run backtest first.")
        st.code("python src/backtest/backtester.py")

        # Offer to run backtest
        st.subheader("Run Backtest Now")

        col1, col2 = st.columns(2)

        with col1:
            start_date = st.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
            initial_capital = st.number_input("Initial Capital (₹)", value=100000, step=10000)
            stop_loss = st.slider("Stop Loss %", -30, -5, -15)

        with col2:
            end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))
            rebalance_freq = st.number_input("Rebalance Frequency (days)", value=30, step=10)
            profit_target = st.slider("Profit Target %", 10, 50, 25)

        if st.button("🚀 Run Backtest"):
            with st.spinner("Running backtest... This may take a few minutes."):
                config = BacktestConfig(
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    initial_capital=initial_capital,
                    rebalance_frequency=rebalance_freq,
                    stop_loss=stop_loss / 100,
                    profit_target=profit_target / 100
                )

                backtester = Backtester(config)
                result = backtester.run_backtest()
                backtester.save_results(result)

                st.success("✓ Backtest complete! Refresh the page to see results.")
                st.rerun()

        return

    # Display key metrics
    st.subheader("📊 Performance Summary")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Total Return",
            f"{metrics['total_return'] * 100:.2f}%",
            delta=None
        )

    with col2:
        st.metric(
            "CAGR",
            f"{metrics['cagr'] * 100:.2f}%",
            delta=f"vs Benchmark: {metrics['alpha'] * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}",
            delta="Excellent" if metrics['sharpe_ratio'] > 1.5 else "Good" if metrics['sharpe_ratio'] > 1.0 else "Fair"
        )

    with col4:
        st.metric(
            "Max Drawdown",
            f"{metrics['max_drawdown'] * 100:.2f}%",
            delta=None
        )

    with col5:
        st.metric(
            "Win Rate",
            f"{metrics['win_rate'] * 100:.1f}%",
            delta=None
        )

    st.markdown("---")

    # Equity curve
    st.subheader("📈 Equity Curve")

    fig = go.Figure()

    # Add equity line
    fig.add_trace(go.Scatter(
        x=equity_df['date'],
        y=equity_df['equity'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2)
    ))

    # Add benchmark line (simplified)
    initial = equity_df.iloc[0]['equity']
    benchmark_cagr = metrics['benchmark_return'] / (
                (equity_df.iloc[-1]['date'] - equity_df.iloc[0]['date']).days / 365.25)
    equity_df['benchmark'] = initial * (1 + benchmark_cagr) ** (
                (equity_df['date'] - equity_df.iloc[0]['date']).dt.days / 365.25)

    fig.add_trace(go.Scatter(
        x=equity_df['date'],
        y=equity_df['benchmark'],
        mode='lines',
        name='Benchmark (12% CAGR)',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))

    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (₹)",
        hovermode='x unified',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Drawdown chart
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📉 Drawdown Analysis")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=equity_df['date'],
            y=equity_df['drawdown'] * 100,
            mode='lines',
            fill='tozeroy',
            name='Drawdown',
            line=dict(color='#d62728')
        ))

        fig.update_layout(
            title="Portfolio Drawdown Over Time",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            hovermode='x unified',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Monthly Returns Distribution")

        monthly_returns = equity_df.groupby(equity_df['date'].dt.to_period('M'))['returns'].sum() * 100

        fig = go.Figure(data=[go.Histogram(
            x=monthly_returns.values,
            nbinsx=20,
            marker_color='#2ca02c'
        )])

        fig.update_layout(
            title="Distribution of Monthly Returns",
            xaxis_title="Monthly Return (%)",
            yaxis_title="Frequency",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Trade statistics
    st.subheader("📈 Trade Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Trades", metrics['total_trades'])
        st.metric("Winning Trades", metrics['winning_trades'])
        st.metric("Losing Trades", metrics['losing_trades'])

    with col2:
        st.metric("Average Win", f"₹{metrics['avg_win']:,.2f}")
        st.metric("Average Loss", f"₹{metrics['avg_loss']:,.2f}")
        st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")

    with col3:
        st.metric("Win Rate", f"{metrics['win_rate'] * 100:.1f}%")
        st.metric("Max Consecutive Losses", metrics['max_consecutive_losses'])
        st.metric("Volatility", f"{metrics['volatility'] * 100:.2f}%")

    st.markdown("---")

    # Trade log
    st.subheader("📋 Trade Log")

    if not trades_df.empty:
        # Format display
        display_trades = trades_df.copy()
        display_trades['entry_date'] = display_trades['entry_date'].dt.date
        display_trades['exit_date'] = display_trades['exit_date'].dt.date
        display_trades['pnl'] = display_trades['pnl'].apply(lambda x: f"₹{x:,.2f}")
        display_trades['pnl_pct'] = display_trades['pnl_pct'].apply(lambda x: f"{x * 100:.2f}%")

        # Color code by profit/loss
        def color_pnl(val):
            if '₹-' in str(val):
                return 'background-color: #ffcccc'
            else:
                return 'background-color: #ccffcc'

        st.dataframe(
            display_trades[['scheme_code', 'entry_date', 'exit_date', 'entry_price', 'exit_price', 'pnl', 'pnl_pct',
                            'exit_reason']],
            use_container_width=True,
            hide_index=True
        )

        # Download button
        csv = trades_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trade Log",
            data=csv,
            file_name="trade_log.csv",
            mime="text/csv"
        )
    else:
        st.info("No trades executed during backtest period.")


def show_parameter_optimization():
    """Parameter optimization interface"""

    st.header("⚙️ Strategy Parameter Optimization")

    st.markdown("""
    Find the optimal strategy parameters through systematic testing.
    This runs multiple backtests with different parameter combinations to find the best settings.
    """)

    st.warning("⚠️ This process can take 10-30 minutes depending on the parameter range.")

    # Parameter ranges
    st.subheader("🎛️ Parameter Ranges to Test")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Buy Signal Thresholds**")

        min_votes_range = st.multiselect(
            "Min Votes for Buy",
            [1, 2, 3, 4],
            default=[2, 3]
        )

        tech_threshold_range = st.multiselect(
            "Technical Threshold",
            [0.5, 1.0, 1.5, 2.0, 2.5],
            default=[1.0, 1.5]
        )

        fund_threshold_range = st.multiselect(
            "Fundamental Threshold",
            [40, 50, 60, 70],
            default=[50, 60]
        )

    with col2:
        st.markdown("**Risk Management**")

        stop_loss_range = st.multiselect(
            "Stop Loss %",
            [-10, -12, -15, -18, -20],
            default=[-15, -18]
        )

        profit_target_range = st.multiselect(
            "Profit Target %",
            [15, 20, 25, 30],
            default=[20, 25]
        )

        rebalance_freq_range = st.multiselect(
            "Rebalance Frequency (days)",
            [15, 30, 60, 90],
            default=[30, 60]
        )

    # Date range
    st.subheader("📅 Backtest Period")

    col1, col2, col3 = st.columns(3)

    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2022-01-01"))

    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime("2024-01-01"))

    with col3:
        initial_capital = st.number_input("Initial Capital (₹)", value=100000, step=10000)

    # Calculate number of combinations
    total_combinations = (
            len(min_votes_range) *
            len(tech_threshold_range) *
            len(fund_threshold_range) *
            len(stop_loss_range) *
            len(profit_target_range) *
            len(rebalance_freq_range)
    )

    st.info(f"📊 Total parameter combinations to test: **{total_combinations}**")

    if total_combinations > 100:
        st.warning("⚠️ High number of combinations will take a long time. Consider reducing ranges.")

    # Run optimization
    if st.button("🚀 Run Optimization", type="primary"):

        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        combination_count = 0

        # Test all combinations
        for min_votes in min_votes_range:
            for tech_thresh in tech_threshold_range:
                for fund_thresh in fund_threshold_range:
                    for stop_loss in stop_loss_range:
                        for profit_target in profit_target_range:
                            for rebalance_freq in rebalance_freq_range:

                                combination_count += 1

                                status_text.text(f"Testing combination {combination_count}/{total_combinations}")
                                progress_bar.progress(combination_count / total_combinations)

                                # Create config
                                config = BacktestConfig(
                                    start_date=start_date.strftime('%Y-%m-%d'),
                                    end_date=end_date.strftime('%Y-%m-%d'),
                                    initial_capital=initial_capital,
                                    rebalance_frequency=rebalance_freq,
                                    min_votes_for_buy=min_votes,
                                    technical_threshold=tech_thresh,
                                    fundamental_threshold=fund_thresh,
                                    stop_loss=stop_loss / 100,
                                    profit_target=profit_target / 100
                                )

                                # Run backtest
                                try:
                                    backtester = Backtester(config)
                                    result = backtester.run_backtest()

                                    # Store results
                                    results.append({
                                        'min_votes': min_votes,
                                        'tech_threshold': tech_thresh,
                                        'fund_threshold': fund_thresh,
                                        'stop_loss': stop_loss,
                                        'profit_target': profit_target,
                                        'rebalance_freq': rebalance_freq,
                                        'total_return': result.total_return,
                                        'cagr': result.cagr,
                                        'sharpe_ratio': result.sharpe_ratio,
                                        'max_drawdown': result.max_drawdown,
                                        'win_rate': result.win_rate,
                                        'total_trades': result.total_trades,
                                        'profit_factor': result.profit_factor
                                    })

                                except Exception as e:
                                    st.error(f"Error testing combination: {e}")

        progress_bar.empty()
        status_text.empty()

        # Display results
        st.success(f"✓ Optimization complete! Tested {len(results)} combinations.")

        results_df = pd.DataFrame(results)

        # Save results
        results_df.to_csv(PROCESSED_DATA_DIR / 'optimization_results.csv', index=False)

        st.markdown("---")

        # Show top performers
        st.subheader("🏆 Top 10 Parameter Combinations")

        # Sort by Sharpe ratio (risk-adjusted returns)
        top_10 = results_df.nlargest(10, 'sharpe_ratio')

        st.dataframe(
            top_10,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        # Best by different metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🎯 Best Sharpe Ratio")
            best_sharpe = results_df.nlargest(1, 'sharpe_ratio').iloc[0]

            st.write(f"**Sharpe Ratio:** {best_sharpe['sharpe_ratio']:.2f}")
            st.write(f"**CAGR:** {best_sharpe['cagr'] * 100:.2f}%")
            st.write(f"**Max Drawdown:** {best_sharpe['max_drawdown'] * 100:.2f}%")
            st.write("**Parameters:**")
            st.write(f"- Min Votes: {best_sharpe['min_votes']}")
            st.write(f"- Tech Threshold: {best_sharpe['tech_threshold']}")
            st.write(f"- Fund Threshold: {best_sharpe['fund_threshold']}")
            st.write(f"- Stop Loss: {best_sharpe['stop_loss']}%")
            st.write(f"- Profit Target: {best_sharpe['profit_target']}%")

        with col2:
            st.subheader("📈 Best CAGR")
            best_cagr = results_df.nlargest(1, 'cagr').iloc[0]

            st.write(f"**CAGR:** {best_cagr['cagr'] * 100:.2f}%")
            st.write(f"**Sharpe Ratio:** {best_cagr['sharpe_ratio']:.2f}")
            st.write(f"**Max Drawdown:** {best_cagr['max_drawdown'] * 100:.2f}%")
            st.write("**Parameters:**")
            st.write(f"- Min Votes: {best_cagr['min_votes']}")
            st.write(f"- Tech Threshold: {best_cagr['tech_threshold']}")
            st.write(f"- Fund Threshold: {best_cagr['fund_threshold']}")
            st.write(f"- Stop Loss: {best_cagr['stop_loss']}%")
            st.write(f"- Profit Target: {best_cagr['profit_target']}%")

        with col3:
            st.subheader("🛡️ Lowest Drawdown")
            best_dd = results_df.nsmallest(1, 'max_drawdown').iloc[0]

            st.write(f"**Max Drawdown:** {best_dd['max_drawdown'] * 100:.2f}%")
            st.write(f"**CAGR:** {best_dd['cagr'] * 100:.2f}%")
            st.write(f"**Sharpe Ratio:** {best_dd['sharpe_ratio']:.2f}")
            st.write("**Parameters:**")
            st.write(f"- Min Votes: {best_dd['min_votes']}")
            st.write(f"- Tech Threshold: {best_dd['tech_threshold']}")
            st.write(f"- Fund Threshold: {best_dd['fund_threshold']}")
            st.write(f"- Stop Loss: {best_dd['stop_loss']}%")
            st.write(f"- Profit Target: {best_dd['profit_target']}%")

        st.markdown("---")

        # 3D scatter plot
        st.subheader("📊 Parameter Visualization")

        fig = px.scatter_3d(
            results_df,
            x='cagr',
            y='sharpe_ratio',
            z='max_drawdown',
            color='sharpe_ratio',
            size='total_trades',
            hover_data=['min_votes', 'tech_threshold', 'fund_threshold', 'stop_loss', 'profit_target'],
            title="CAGR vs Sharpe Ratio vs Max Drawdown",
            labels={
                'cagr': 'CAGR',
                'sharpe_ratio': 'Sharpe Ratio',
                'max_drawdown': 'Max Drawdown'
            },
            color_continuous_scale='RdYlGn'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Results",
            data=csv,
            file_name="optimization_results.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
