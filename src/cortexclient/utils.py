import datetime


def current_timestamp() -> datetime.datetime:
    """
    Get the current time in utc timezone
    """
    return datetime.datetime.now(datetime.timezone.utc)
