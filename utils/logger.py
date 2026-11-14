"""
utils/logger.py
===============
Application-wide logger factory.

Usage in any module:
    from utils.logger import get_logger
    logger = get_logger(__name__)
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.  Call utils.config.setup_logging() once at startup."""
    return logging.getLogger(name)

