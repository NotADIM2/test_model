"""Alerts: a message each time a zone's warning level goes up, and when it settles back to green.

Each alert says what is happening in plain words and what to do about it, for the person watching the screen.
"""
from dataclasses import asdict, dataclass

from .risk import AMBER_FROM, LEVELS, RED_FROM, Status, countdown

SETTLE = 30.0                                  # seconds a zone must stay green before it is called back to normal


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
    until = ""
    if status.minutes_to_critical is not None:
        until = " It is at a crush-risk level now." if status.minutes_to_critical == 0 else \
            f" At this rate it reaches crush-risk density in {countdown(status.minutes_to_critical)}."
    if status.level == "red":
        state = "is dangerously crowded" if status.density >= RED_FROM else "is heading for a crush"
        return (f"{status.name} {state}: {status.density:.1f} people per m²{filling}.{until}",
                f"Stop people entering {status.name} now. Open every exit from it and send stewards to guide people out.")
    if status.level == "amber":
        state = "is crowded" if status.density >= AMBER_FROM else "is filling fast"
        return (f"{status.name} {state}: {status.density:.1f} people per m²{filling}.{until}",
                f"Slow the flow into {status.name}: hold people at its entrances and point them to another route.")
    return (f"{status.name} is back to normal: {status.density:.1f} people per m².", "No action needed. Keep watching.")


class AlertLog:
    """Turns each frame's zone statuses into alerts: when a zone gets worse than it has been since it was last
    back to normal, and when it has stayed green for SETTLE seconds. A zone hovering on a boundary alerts once."""

    def __init__(self):
        self.worst: dict[str, str] = {}        # worst level announced for each zone since it was last back to normal
        self.green_since: dict[str, float] = {}

    def update(self, t: float, statuses: dict[str, Status]) -> list[Alert]:
        new = []
        for name, status in statuses.items():
            worst = self.worst.get(name, "green")
            if status.level == "green":
                self.green_since.setdefault(name, t)
            else:
                self.green_since.pop(name, None)
            worse = LEVELS.index(status.level) > LEVELS.index(worst)
            settled = worst != "green" and status.level == "green" and t - self.green_since[name] >= SETTLE
            if worse or settled:
                self.worst[name] = status.level
                message, action = describe(status)
                new.append(Alert(round(t, 1), name, status.level, message, action))
        return new
