"""Loguru-based logging configuration for KUMI platform."""

import os
import sys
from pathlib import Path
from typing import Optional
from loguru import logger
from .settings import settings


def setup_logging(
    log_level: Optional[str] = None, log_file: Optional[str] = None
) -> None:
    """Configure loguru logging with console and file outputs."""
    level = (log_level or settings.LOG_LEVEL).upper()
    file_path = log_file or settings.LOG_FILE

    # Remove default handler
    logger.remove()

    # Add console handler with colored output
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        colorize=True,
    )

    # Add file handler if log file is specified
    if file_path:
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(file_path)
            if log_dir:
                Path(log_dir).mkdir(parents=True, exist_ok=True)

            logger.add(
                file_path,
                level=level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
                rotation="10 MB",
                retention="10 days",
                encoding="utf-8",
            )
        except Exception as e:
            # If file logging fails, log to console only
            logger.warning(f"Failed to set up file logging ({file_path}): {e}")
            logger.warning("Falling back to console-only logging")

    # Set third-party library log levels to WARNING to reduce noise
    logger.level("INFO", color="<yellow>")
    logger.level("WARNING", color="<yellow>")
    logger.level("ERROR", color="<red>")

    logger.info(f"Logging initialized at level: {level}")
    if file_path:
        logger.info(f"Log file: {file_path}")


def get_logger(name: Optional[str] = None):
    """Get a logger instance with the given name.

    Args:
        name: Optional logger name. If None, returns the root logger.

    Returns:
        A loguru logger instance.
    """
    if name:
        return logger.bind(name=name)
    return logger
