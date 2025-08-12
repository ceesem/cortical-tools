import datetime
import logging
import os
import sys
from contextlib import contextmanager

import urllib3


def current_timestamp() -> datetime.datetime:
    """
    Get the current time in utc timezone
    """
    return datetime.datetime.now(datetime.timezone.utc)


@contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr output."""
    # Save original stdout and stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    urllib3.disable_warnings()
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)

    try:
        # Redirect to devnull or StringIO
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
    finally:
        # Restore original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
