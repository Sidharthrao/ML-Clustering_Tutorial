"""
Logging utility module.
Configures logging for the e-commerce customer segmentation pipeline.
"""
import logging
import sys
from pathlib import Path
from typing import Optional

from config.config import LOG_CONFIG, ensure_directories

# Ensure logs directory exists
ensure_directories()


def setup_logger(
    name: str = "ecommerce_clustering",
    log_file: Optional[Path] = None,
    log_level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up and return a logger instance.
    
    Args:
        name: Logger name
        log_file: Path to log file (defaults to config log file)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Format
    if format_string is None:
        format_string = LOG_CONFIG.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = LOG_CONFIG.get("log_file", Path("logs/ecommerce_clustering.log"))
    
    # Ensure log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# Create default logger instance
logger = setup_logger()

