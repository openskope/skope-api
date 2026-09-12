from dateutil.relativedelta import relativedelta


def get_iso_key(dt, time_delta: dict[str, int]) -> str:
    """Returns a partially or fully ISO-8601 compliant string based on the time step resolution."""
    if any(k in time_delta for k in ["hours", "minutes", "seconds"]):
        return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}T{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}Z"
    if "days" in time_delta or "weeks" in time_delta:
        return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}"
    if "months" in time_delta:
        return f"{dt.year:04d}-{dt.month:02d}"
    return f"{dt.year:04d}"


def format_stac_datetime(dt) -> str:
    """Returns a strict STAC ISO-8601 compliant string with 4-digit year padding."""
    return f"{dt.year:04d}-{dt.month:02d}-{dt.day:02d}T{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}Z"


def generate_date_range(start_dt, end_dt, step: relativedelta):
    """Yields datetimes from start to end (inclusive) by a given step."""
    current = start_dt
    while current <= end_dt:
        yield current
        current += step


def singular_to_plural_for_relativedelta(time_delta: dict[str, int]) -> dict[str, int]:
    """Converts singular keys (e.g., 'year') to plural keys (e.g., 'years') for relativedelta."""
    return {k + 's' if not k.endswith('s') else k: v for k, v in time_delta.items()}
