"""Logging configuration using loguru."""

import os

from loguru import logger


def disable_logging() -> None:
    """Disable all logging for the library."""
    os.environ["CAMEL_LOGGING_DISABLED"] = "true"
    logger.disable("rag_processor")


def enable_logging() -> None:
    """Enable logging for the library."""
    os.environ["CAMEL_LOGGING_DISABLED"] = "false"
    logger.enable("rag_processor")


def set_log_level(level: str | int) -> None:
    """Set the logging level.

    Args:
        level: Level name (e.g. 'INFO') or numeric level.
    """
    if isinstance(level, str):
        level = level.upper()
    logger.configure(handlers=[{"sink": "stderr", "level": level}])


def get_logger(name: str):
    """Return the loguru logger. Uses global logger (caller info auto-captured)."""
    return logger
