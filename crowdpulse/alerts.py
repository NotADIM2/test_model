"""Alerts: a message each time a zone's warning level goes up, and when it settles back to green.

Each alert says what is happening in plain words and what to do about it, for the person watching the screen.
"""
from dataclasses import asdict, dataclass

from .risk import LEVELS, Status


@dataclass
class Alert:
    t: float                                   # seconds into the video or since the camera opened
    zone: str
    level: str                                 # green, amber or red
    message: str
    action: str

    def to_dict(self) -> dict:
        return asdict(self)


def describe(status: Status) -> tuple[str, str]:
    """(what is happening, what to do) for a zone at its current level."""
    filling = f", {status.inflow:.0f} people a minute walking in" if status.inflow >= 1 else ""
    countdown = ""
    if status.minutes_to_critical is not None:
        countdown = " It is at a crush-risk level now." if status.minutes_to_critical == 0 else \
            f" At this rate it reaches crush-risk density in about {status.minutes_to_critical:.0f} minutes."
    if status.level == "red":
        return (f"{status.name} is dangerously crowded: {status.density:.1f} people per m²{filling}.{countdown}",
                f"Stop people entering {status.name} now. Open every exit from it and send stewards to guide people out.")
    if status.level == "amber":
        return (f"{status.name} is filling up: {status.density:.1f} people per m²{filling}.{countdown}",
                f"Slow the flow into {status.name}: hold people at its entrances and point them to another route.")
    return (f"{status.name} is back to normal: {status.density:.1f} people per m².", "No action needed. Keep watching.")


class AlertLog:
    """Turns each frame's zone statuses into alerts when a level changes."""

    def __init__(self):
        self.levels: dict[str, str] = {}

    def update(self, t: float, statuses: dict[str, Status]) -> list[Alert]:
        new = []
        for name, status in statuses.items():
            before = self.levels.get(name, "green")
            self.levels[name] = status.level
            worse = LEVELS.index(status.level) > LEVELS.index(before)
            settled = status.level == "green" and before != "green"
            if worse or settled:
                message, action = describe(status)
                new.append(Alert(round(t, 1), name, status.level, message, action))
        return new
