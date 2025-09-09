import datetime
import logging
import os
import sys
from contextlib import contextmanager
from typing import Optional

import urllib3


def current_timestamp() -> datetime.datetime:
    """
    Get the current time in utc timezone
    """
    return datetime.datetime.now(datetime.timezone.utc)


@contextmanager
def suppress_output(package: Optional[str] = None):
    """Context manager to suppress stdout and stderr output.

    Parameters
    ----------
    package : Optional[str]
        Logger name to suppress. If None, suppress root logger output which handles
        all loggers without their own handler.
    """
    # Save original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if package == "urllib3":
        urllib3.disable_warnings()
    logger = logging.getLogger(package)
    old_level = logger.level
    logger.setLevel(logging.CRITICAL)

    try:
        # Redirect to devnull or StringIO
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
    finally:
        # Restore original stdout and stderr
        logger.setLevel(old_level)
        sys.stdout = original_stdout
        sys.stderr = original_stderr
