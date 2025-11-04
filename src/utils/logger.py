"""
Centralized logging configuration
"""
from loguru import logger
import sys
from config.settings import LOGS_DIR, LOG_LEVEL


def setup_logger():
    """Configure loguru logger"""

    # Remove default handler
    logger.remove()

    # Add console handler with custom format
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=LOG_LEVEL,
        colorize=True
    )

    # Add file handler for all logs
    logger.add(
        LOGS_DIR / "app_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level="DEBUG",
        rotation="1 day",
        retention="30 days",
        compression="zip"
    )

    # Add file handler for errors only
    logger.add(
        LOGS_DIR / "errors_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level="ERROR",
        rotation="1 week",
        retention="3 months"
    )

    return logger


# Initialize logger
log = setup_logger()
