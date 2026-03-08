"""
Shared logging configuration for Explanation Lottery sessions.
"""

import os
import logging
from datetime import datetime


def setup_logging(log_dir, session_name):
    """
    Configure file + console logging for a session.

    Args:
        log_dir:      Directory to write log file into (created if absent).
        session_name: Identifier used for both the log filename and logger name.

    Returns:
        (logger, log_filename) tuple.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_filename = (
        f"{log_dir}/{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_filename, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(session_name)
    return logger, log_filename
