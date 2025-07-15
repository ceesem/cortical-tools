import datetime
from collections import namedtuple


def current_timestamp() -> datetime.datetime:
    """
    Get the current time in utc timezone
    """
    return datetime.datetime.now(datetime.timezone.utc)


QueryTuple = namedtuple("Queries", ["views", "tables"])
