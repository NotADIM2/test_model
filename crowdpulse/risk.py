"""How dangerous each zone is now, and how soon it will become critical.

Crowd-safety guidance measures density in people per square metre. Below about 2, people move freely. From 2
they are crowded and slow down; from 4 they cannot choose their own path; at about 5 and above, pressure can
pass through the crowd and a crush can start. CrowdPulse uses those bands:

    green  below 2      amber  2 to 4      red  4 and above      critical  5

A zone's warning also goes up before its density does: when it is filling fast enough to reach 5 within
AMBER_WITHIN minutes it is amber, and within RED_WITHIN minutes red. That early step is the point of the tool.
It only counts once it has held for HOLD seconds, because a warning that flickers on for a moment in a calm
crowd teaches people to ignore warnings.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from .zones import Reading

AMBER_FROM, RED_FROM, CRITICAL = 2.0, 4.0, 5.0
AMBER_WITHIN, RED_WITHIN = 10.0, 3.0           # minutes to critical that raise the warning
SMOOTHING = 5.0                                # seconds; readings are averaged over about this long
TREND_WINDOW = 30.0                            # seconds of history used to measure how fast density rises
MIN_RATE = 0.01                                # people per m2 per minute; slower than this is not "filling"
HOLD = 10.0                                    # seconds an early warning must last before it is raised
LEVELS = ("green", "amber", "red")


def level_of(density: float) -> str:
    return "red" if density >= RED_FROM else "amber" if density >= AMBER_FROM else "green"


@dataclass
class Status:
    name: str
    people: float
    density: float                             # people per m2, smoothed
    inflow: float                              # people per minute crossing in, smoothed
    rate: float                                # people per m2 per minute the zone is filling at (0 if not)
    minutes_to_critical: float | None          # None unless the zone is amber or red and heading for critical
    level: str                                 # green, amber or red, including the early warning

    def summary(self) -> str:
        words = f"{self.name}: {self.density:.1f} people/m2, {self.level}"
        if self.minutes_to_critical is not None:
            words += " (critical now)" if self.minutes_to_critical == 0 else f", critical in about {self.minutes_to_critical:.0f} min"
        return words


class ZoneTrend:
    """Follows one zone frame by frame: smooths the readings and works out where they are heading."""

    def __init__(self, name: str, area_m2: float):
        self.name, self.area_m2 = name, area_m2
        self.history: deque[tuple[float, float]] = deque()    # (seconds, smoothed density)
        self.density = self.inflow = self.people = None
        self.last_time = None
        self.early_since: dict[str, float] = {}               # early level -> when it was first seen without a break

    def update(self, seconds: float, reading: Reading) -> Status:
        if self.last_time is None:
            self.people, self.density, self.inflow = reading.people, reading.density, reading.inflow
        else:
            k = 1 - np.exp(-(seconds - self.last_time) / SMOOTHING)
            self.people += k * (reading.people - self.people)
            self.density += k * (reading.density - self.density)
            self.inflow += k * (reading.inflow - self.inflow)
        self.last_time = seconds
        self.history.append((seconds, self.density))
        while self.history[0][0] < seconds - TREND_WINDOW:
            self.history.popleft()
        return self.status()

    def slope(self) -> float:
        """People per m2 per minute over the trend window, by a straight-line fit."""
        if len(self.history) < 3 or self.history[-1][0] - self.history[0][0] < 5:
            return 0.0
        t, d = np.array(self.history).T
        return float(np.polyfit(t, d, 1)[0] * 60)

    def status(self) -> Status:
        # Two views of how fast the zone is filling: its density rising, and people crossing into it.
        # The larger one is used, so a zone that people are pouring into is flagged before it is packed.
        rate = max(self.slope(), self.inflow / self.area_m2, 0.0)
        if self.density >= CRITICAL:
            minutes = 0.0
        elif rate >= MIN_RATE:
            minutes = (CRITICAL - self.density) / rate
        else:
            minutes = None
        level = level_of(self.density)
        early = "green"
        if minutes is not None:
            early = "red" if minutes <= RED_WITHIN else "amber" if minutes <= AMBER_WITHIN else "green"
        for name in ("amber", "red"):
            if LEVELS.index(early) >= LEVELS.index(name):
                self.early_since.setdefault(name, self.last_time)
                if self.last_time - self.early_since[name] >= HOLD:
                    level = max(level, name, key=LEVELS.index)
            else:
                self.early_since.pop(name, None)
        if level == "green":
            minutes = None                                    # a countdown on a green zone would alarm for nothing
        return Status(self.name, self.people, self.density, self.inflow, rate if rate >= MIN_RATE else 0.0, minutes, level)
