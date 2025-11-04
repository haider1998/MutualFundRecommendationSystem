"""
Initialize local SQLite database
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.database import db
from loguru import logger


def setup_database():
    """Create database and all tables"""
    logger.info("Setting up local database...")

    # Create tables
    db.create_tables()

    logger.info("✓ Database setup complete!")
    logger.info(f"✓ Database location: {db.engine.url}")


if __name__ == "__main__":
    setup_database()
