"""
Configuration management for MF Intelligence System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Application Settings
APP_NAME = os.getenv('APP_NAME', 'MF Intelligence System')
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Database
DATABASE_PATH = BASE_DIR / os.getenv('DATABASE_PATH', 'data/database/mf_intelligence.db')
DATABASE_URL = f'sqlite:///{DATABASE_PATH}'

# Data Directories
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = DATA_DIR / 'models'
BACKTEST_DIR = DATA_DIR / 'backtest'
DATABASE_DIR = DATA_DIR / 'database'

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, BACKTEST_DIR, DATABASE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Logs
LOGS_DIR = BASE_DIR / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

# Data Sources
MFAPI_BASE_URL = os.getenv('MFAPI_BASE_URL', 'https://api.mfapi.in')
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
NEWS_API_URL = os.getenv('NEWS_API_URL', 'https://newsapi.org/v2/everything')
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY', '')
RBI_API_URL = os.getenv('RBI_API_URL', 'https://rbi.org.in/Scripts/api.aspx')

# Rate Limiting
MAX_REQUESTS_PER_MINUTE = int(os.getenv('MAX_REQUESTS_PER_MINUTE', 50))
REQUEST_DELAY_SECONDS = float(os.getenv('REQUEST_DELAY_SECONDS', 1))

# Feature Flags
ENABLE_NEWS_SENTIMENT = os.getenv('ENABLE_NEWS_SENTIMENT', 'False').lower() == 'true'
ENABLE_SOCIAL_MEDIA = os.getenv('ENABLE_SOCIAL_MEDIA', 'False').lower() == 'true'
ENABLE_ML_PREDICTIONS = os.getenv('ENABLE_ML_PREDICTIONS', 'True').lower() == 'true'

# Testing
TEST_MODE = os.getenv('TEST_MODE', 'True').lower() == 'true'
TEST_SAMPLE_SIZE = int(os.getenv('TEST_SAMPLE_SIZE', 100))

# Analysis Parameters
TECHNICAL_INDICATORS = {
    'ema_periods': [20, 50, 200],
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bollinger_period': 20,
    'bollinger_std': 2,
    'atr_period': 14
}

# Risk Parameters
RISK_FREE_RATE = 0.065  # 6.5% (10-year G-Sec approximate)
STOP_LOSS_PERCENT = {
    'conservative': 0.12,
    'moderate': 0.15,
    'aggressive': 0.20
}

# Portfolio Constraints
MAX_POSITION_SIZE = 0.30  # 30% max in single fund
MIN_POSITION_SIZE = 0.05  # 5% minimum or zero
MAX_EQUITY_ALLOCATION = {
    'conservative': 0.50,
    'moderate': 0.70,
    'aggressive': 0.85
}

print(f"✓ Configuration loaded for {APP_NAME} ({ENVIRONMENT} mode)")
