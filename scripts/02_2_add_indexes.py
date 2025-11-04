# migration/add_indexes.py
"""
Add indexes for 100x faster queries
Run this ONCE before analysis
"""
from sqlalchemy import text
from src.utils.database import db


def add_performance_indexes():
    """Add critical indexes for fast queries"""
    session = db.get_session()

    try:
        # Index for scheme_code lookups
        session.execute(text("""
                             CREATE INDEX IF NOT EXISTS idx_daily_nav_scheme_code
                                 ON daily_navs(scheme_code)
                             """))

        # Index for date range queries
        session.execute(text("""
                             CREATE INDEX IF NOT EXISTS idx_daily_nav_date
                                 ON daily_navs(date)
                             """))

        # Composite index for scheme + date queries (MOST IMPORTANT)
        session.execute(text("""
                             CREATE INDEX IF NOT EXISTS idx_daily_nav_scheme_date
                                 ON daily_navs(scheme_code, date DESC)
                             """))

        # Index for fund metadata lookups
        session.execute(text("""
                             CREATE INDEX IF NOT EXISTS idx_fund_metadata_code
                                 ON fund_metadata(scheme_code)
                             """))

        session.commit()
        print("✓ Performance indexes created successfully!")

    except Exception as e:
        session.rollback()
        print(f"Error creating indexes: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    add_performance_indexes()
