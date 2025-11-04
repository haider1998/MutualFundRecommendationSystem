"""
Unit tests for data ingestion
"""
import pytest
from src.ingestion.mfapi_fetcher import MFAPIFetcher
from src.utils.database import db, DailyNAV, FundMetadata


def test_get_all_schemes():
    """Test fetching scheme list"""
    fetcher = MFAPIFetcher()
    schemes = fetcher.get_all_schemes()

    assert isinstance(schemes, list)
    assert len(schemes) > 0
    assert 'schemeCode' in schemes[0]
    assert 'schemeName' in schemes[0]


def test_get_scheme_details():
    """Test fetching single scheme details"""
    fetcher = MFAPIFetcher()

    # Test with a known scheme code (HDFC Balanced Advantage Fund)
    scheme_data = fetcher.get_scheme_details('119551')

    assert scheme_data is not None
    assert 'meta' in scheme_data
    assert 'data' in scheme_data
    assert len(scheme_data['data']) > 0


def test_database_connection():
    """Test database connectivity"""
    session = db.get_session()

    # Try a simple query
    count = session.query(DailyNAV).count()
    assert count >= 0  # Should work even if empty

    session.close()


def test_full_ingestion_small_sample():
    """Test full ingestion with 5 funds"""
    fetcher = MFAPIFetcher()
    metadata, navs = fetcher.fetch_and_store_all(limit=5)

    assert len(metadata) > 0
    assert len(navs) > 0

    # Verify data in database
    session = db.get_session()
    db_metadata_count = session.query(FundMetadata).count()
    db_nav_count = session.query(DailyNAV).count()

    assert db_metadata_count >= len(metadata)
    assert db_nav_count >= len(navs)

    session.close()
