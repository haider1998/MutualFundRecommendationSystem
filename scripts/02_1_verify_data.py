"""
Quick script to verify data in database
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import db, DailyNAV, FundMetadata
from sqlalchemy import func
import pandas as pd


def verify_data():
    session = db.get_session()

    # Count records
    fund_count = session.query(FundMetadata).count()
    nav_count = session.query(DailyNAV).count()

    print(f"\n📊 Database Summary:")
    print(f"   Total Funds: {fund_count}")
    print(f"   Total NAV Records: {nav_count}")

    # Show sample funds
    print(f"\n📋 Sample Funds:")
    funds = session.query(FundMetadata).limit(10).all()
    for fund in funds:
        print(f"   {fund.scheme_code}: {fund.scheme_name}")

    # Show date range
    min_date = session.query(func.min(DailyNAV.date)).scalar()
    max_date = session.query(func.max(DailyNAV.date)).scalar()
    print(f"\n📅 NAV Data Range: {min_date} to {max_date}")

    # Latest NAVs
    print(f"\n💰 Latest NAVs (sample):")
    latest_navs = session.query(DailyNAV).filter_by(date=max_date).limit(5).all()
    for nav in latest_navs:
        print(f"   {nav.scheme_code}: ₹{nav.nav:.2f} on {nav.date}")

    session.close()


if __name__ == "__main__":
    verify_data()
