"""
Email alert system for critical signals
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List
import pandas as pd
from datetime import datetime

from src.utils.logger import log
from config.settings import PROCESSED_DATA_DIR


class EmailAlertSystem:
    """Send email alerts for trading signals"""

    def __init__(self):
        # Email configuration (use environment variables)
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', 587))
        self.sender_email = os.getenv('SENDER_EMAIL', 'your_email@gmail.com')
        self.sender_password = os.getenv('SENDER_PASSWORD', 'your_app_password')
        self.receiver_email = os.getenv('RECEIVER_EMAIL', 'your_email@gmail.com')

    def send_daily_report(self):
        """Send daily summary email"""

        # Load data
        try:
            comprehensive = pd.read_csv(PROCESSED_DATA_DIR / 'comprehensive_analysis.csv')
            signals = pd.read_csv(PROCESSED_DATA_DIR / 'signals.csv')

            try:
                portfolio = pd.read_csv(PROCESSED_DATA_DIR / 'recommended_portfolio.csv')
            except:
                portfolio = pd.DataFrame()
        except Exception as e:
            log.error(f"Error loading data for email: {e}")
            return

        # Count signals
        buy_signals = signals[signals['signal_type'] == 'BUY']
        sell_signals = signals[signals['signal_type'] == 'SELL']

        # Create email content
        subject = f"📊 MF Intelligence Daily Report - {datetime.now().strftime('%Y-%m-%d')}"

        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: #1f77b4; color: white; padding: 20px; text-align: center; }}
                .summary {{ padding: 20px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #f0f2f6; border-radius: 5px; }}
                .buy {{ color: #00c853; font-weight: bold; }}
                .sell {{ color: #ff1744; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #1f77b4; color: white; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Mutual Fund Intelligence System</h1>
                <p>Daily Report - {datetime.now().strftime('%B %d, %Y')}</p>
            </div>

            <div class="summary">
                <h2>📊 Summary</h2>

                <div class="metric">
                    <strong>Total Funds Analyzed:</strong> {len(comprehensive)}
                </div>

                <div class="metric">
                    <strong class="buy">BUY Signals:</strong> {len(buy_signals)}
                </div>

                <div class="metric">
                    <strong class="sell">SELL Signals:</strong> {len(sell_signals)}
                </div>

                <div class="metric">
                    <strong>Avg Score:</strong> {comprehensive['comprehensive_score'].mean():.1f}/100
                </div>
            </div>
        """

        # Add BUY signals
        if not buy_signals.empty:
            html += """
            <div class="summary">
                <h2 class="buy">🟢 BUY SIGNALS</h2>
                <table>
                    <tr>
                        <th>Fund Name</th>
                        <th>Signal Strength</th>
                        <th>Score</th>
                        <th>Allocation</th>
                        <th>Reasons</th>
                    </tr>
            """

            for _, signal in buy_signals.head(10).iterrows():
                html += f"""
                <tr>
                    <td>{signal.get('scheme_name', signal.get('fund_name', signal['scheme_code']))[:60]}</td>
                    <td>{signal['signal_strength']}</td>
                    <td>{signal['comprehensive_score']:.1f}</td>
                    <td>{signal['allocation_pct']:.1f}%</td>
                    <td>{signal.get('reasons', 'N/A')[:100]}</td>
                </tr>
                """

            html += "</table></div>"

        # Add SELL signals
        if not sell_signals.empty:
            html += """
            <div class="summary">
                <h2 class="sell">🔴 SELL SIGNALS</h2>
                <table>
                    <tr>
                        <th>Fund Name</th>
                        <th>Signal Strength</th>
                        <th>Warnings</th>
                    </tr>
            """

            for _, signal in sell_signals.head(10).iterrows():
                html += f"""
                <tr>
                    <td>{signal.get('scheme_name', signal.get('fund_name', signal['scheme_code']))[:60]}</td>
                    <td>{signal['signal_strength']}</td>
                    <td>{signal.get('warnings', 'N/A')[:150]}</td>
                </tr>
                """

            html += "</table></div>"

        # Add portfolio recommendation
        if not portfolio.empty:
            html += f"""
            <div class="summary">
                <h2>💼 Recommended Portfolio</h2>
                <p><strong>Total Funds:</strong> {len(portfolio)}</p>
                <table>
                    <tr>
                        <th>Fund Name</th>
                        <th>Allocation</th>
                        <th>Amount (₹1L)</th>
                        <th>Expected 1Y Return</th>
                    </tr>
            """

            for _, fund in portfolio.iterrows():
                expected_return = fund.get('expected_1y_return', 0)
                expected_str = f"{expected_return * 100:.2f}%" if pd.notna(expected_return) else "N/A"

                html += f"""
                <tr>
                    <td>{fund.get('scheme_name', fund.get('fund_name', fund['scheme_code']))[:60]}</td>
                    <td>{fund['allocation_percent']:.1f}%</td>
                    <td>₹{fund['allocation_amount']:,.0f}</td>
                    <td>{expected_str}</td>
                </tr>
                """

            html += "</table></div>"

        html += """
            <div class="summary">
                <p style="color: #666; font-size: 12px;">
                    This is an automated report from your Mutual Fund Intelligence System.
                    <br>For detailed analysis, visit your dashboard.
                </p>
            </div>
        </body>
        </html>
        """

        # Send email
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = self.receiver_email

            msg.attach(MIMEText(html, 'html'))

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            log.info("✓ Daily report email sent successfully")

        except Exception as e:
            log.error(f"Error sending email: {e}")
            log.info("Please configure email settings in environment variables:")
            log.info("  SMTP_SERVER, SMTP_PORT, SENDER_EMAIL, SENDER_PASSWORD, RECEIVER_EMAIL")


def main():
    """Test email alerts"""
    alert_system = EmailAlertSystem()
    alert_system.send_daily_report()


if __name__ == "__main__":
    main()
