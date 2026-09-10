"""Match frame times to a phone's GPS track without extrapolating."""

from bisect import bisect_left
from datetime import datetime

from .video import GPSPoint, Location


def location_at(track: list[GPSPoint], timestamp: datetime, max_gap_seconds: float) -> Location | None:
    """Interpolate a sorted, unique track; leave missing or distant samples unknown.

    Interpolated accuracy is unknown. Even exact samples locate the phone,
    not the damage on the road.
    """
    index = bisect_left(track, timestamp, key=lambda point: point.timestamp)
    if index < len(track) and track[index].timestamp == timestamp:
        return track[index].location
    if index == 0 or index == len(track):
        return None
    before, after = track[index - 1], track[index]
    gap = (after.timestamp - before.timestamp).total_seconds()
    if gap > max_gap_seconds:
        return None
    fraction = (timestamp - before.timestamp).total_seconds() / gap
    # Follow the short longitude arc if a survey crosses the date line.
    longitude_delta = (after.location.longitude - before.location.longitude + 180) % 360 - 180
    return Location(
        latitude=before.location.latitude + fraction * (after.location.latitude - before.location.latitude),
        longitude=(before.location.longitude + fraction * longitude_delta + 180) % 360 - 180,
    )
