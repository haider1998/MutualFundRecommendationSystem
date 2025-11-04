"""
Database management using SQLAlchemy ORM
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from config.settings import DATABASE_URL
from loguru import logger

# Create base class for models
Base = declarative_base()


# Database Models
class DailyNAV(Base):
    __tablename__ = 'daily_navs'

    id = Column(Integer, primary_key=True)
    scheme_code = Column(String(50), nullable=False, index=True)
    scheme_name = Column(String(500), nullable=False)
    nav = Column(Float, nullable=False)
    date = Column(Date, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<DailyNAV(scheme_code={self.scheme_code}, date={self.date}, nav={self.nav})>"


class FundMetadata(Base):
    __tablename__ = 'fund_metadata'

    id = Column(Integer, primary_key=True)
    scheme_code = Column(String(50), unique=True, nullable=False, index=True)
    scheme_name = Column(String(500), nullable=False)
    scheme_category = Column(String(200))
    scheme_type = Column(String(100))
    aum = Column(Float)  # in Crores
    expense_ratio = Column(Float)
    min_investment = Column(Float)
    fund_manager = Column(String(200))
    fund_house = Column(String(200))
    inception_date = Column(Date)
    last_updated = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<FundMetadata(scheme_code={self.scheme_code}, name={self.scheme_name})>"


class Signal(Base):
    __tablename__ = 'signals'

    id = Column(Integer, primary_key=True)
    scheme_code = Column(String(50), nullable=False, index=True)
    signal_type = Column(String(20), nullable=False)  # BUY, SELL, HOLD, REBALANCE
    signal_strength = Column(String(20))  # STRONG, MODERATE, WEAK
    confirmation_votes = Column(Integer)
    technical_score = Column(Float)
    fundamental_score = Column(Float)
    ml_score = Column(Float)
    sentiment_score = Column(Float)
    target_allocation = Column(Float)
    rationale = Column(Text)
    created_at = Column(DateTime, default=datetime.now, index=True)

    def __repr__(self):
        return f"<Signal(scheme_code={self.scheme_code}, type={self.signal_type}, date={self.created_at})>"


class EngineScore(Base):
    __tablename__ = 'engine_scores'

    id = Column(Integer, primary_key=True)
    scheme_code = Column(String(50), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)

    # Engine A: Technical Analysis
    technical_score = Column(Float)
    rsi = Column(Float)
    macd = Column(Float)
    ema_20 = Column(Float)
    ema_50 = Column(Float)
    ema_200 = Column(Float)

    # Engine B: Fundamental Analysis
    fundamental_score = Column(Float)
    sharpe_ratio = Column(Float)
    alpha = Column(Float)
    beta = Column(Float)

    # Engine C: ML Prediction
    ml_alpha_score = Column(Float)
    predicted_return = Column(Float)

    # Engine D: Sentiment
    sentiment_score = Column(Float)

    created_at = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<EngineScore(scheme_code={self.scheme_code}, date={self.date})>"


# Database connection and session management
class Database:
    """Database manager class"""

    def __init__(self, db_url=DATABASE_URL):
        self.engine = create_engine(db_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info(f"✓ Database engine created: {db_url}")

    def create_tables(self):
        """Create all tables"""
        Base.metadata.create_all(self.engine)
        logger.info("✓ Database tables created/verified")

    def get_session(self):
        """Get a new database session"""
        return self.SessionLocal()

    def drop_tables(self):
        """Drop all tables (use with caution!)"""
        Base.metadata.drop_all(self.engine)
        logger.warning("⚠ All tables dropped")


# Singleton instance
db = Database()
