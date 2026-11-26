"""The analysis loop shared by the command line (scripts/analyse.py) and the local dashboard.

A source gives pairs of frames a quarter of a second apart: the first is counted, and the pair together gives
the movement. The Analyser turns each pair into an annotated image and a record of every zone's readings.
"""
import time
from pathlib import Path

import cv2
import numpy as np

from . import overlay
from .density import DensityModel
from .flow import Steady, crowd_flow, people_per_cell
from .forecast import forecast, zone_ahead
from .risk import Status, ZoneTrend
from .thermal import ThermalModel
from .zones import Zone, load_zones, measure

FLOW_GAP = 0.25                                # seconds between the two frames compared for movement
OUTLINE_FLOOR = 1.0                            # people per movement cell: the forecast never outlines less


class VideoFile:
    """A recorded video, analysed every `every` seconds of video time from `start` to `end`."""

    def __init__(self, path: Path, every: float = 1.0, start: float = 0.0, end: float | None = None):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise OSError(f"Could not open {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.fps
        self.end = min(end or duration, duration - FLOW_GAP - 0.05)
        self.every, self.next_time = every, start
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def pair_at(self, seconds: float) -> tuple[np.ndarray, np.ndarray, float] | None:
        """The frame at `seconds`, the frame about FLOW_GAP later, and the time between them.

        Seeking in compressed video can land a frame or two off, so it seeks once and reads forward from there.
        """
        self.cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = self.cap.read()
        gap = max(1, round(FLOW_GAP * self.fps))
        later = None
        for _ in range(gap):
            ok, later = self.cap.read() if ok else (False, None)
        return (frame, later, gap / self.fps) if ok else None

    def next(self) -> tuple[float, np.ndarray, np.ndarray, float] | None:
        """(seconds, frame, later frame, gap), or None at the end of the video."""
        if self.next_time >= self.end:
            return None
        seconds = self.next_time
        self.next_time += self.every
        pair = self.pair_at(seconds)
        return (seconds, *pair) if pair else None


class Webcam:
    """A live camera. Time is seconds since it was opened."""

    def __init__(self, index: int = 0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise OSError("Could not open the webcam. Is another app using it?")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.opened = time.monotonic()

    def next(self) -> tuple[float, np.ndarray, np.ndarray, float] | None:
        for _ in range(4):                                   # drop frames that queued up while the last one was analysed
            self.cap.grab()
        ok, frame = self.cap.read()
        seconds = time.monotonic() - self.opened
        time.sleep(FLOW_GAP)
        ok2, later = self.cap.read()
        if not (ok and ok2):
            return None
        return seconds, frame, later, time.monotonic() - self.opened - seconds


def zone_records(statuses: dict[str, Status], ahead: dict[str, float] | None = None) -> list[dict]:
    """Each zone's status as plain numbers, for the timeline, the dashboard and the report. `ahead` is each
    zone's forecast density in 30 s, if there is one."""
    return [{"name": s.name, "people": round(s.people, 1), "density": round(s.density, 2),
             "inflow_per_min": round(s.inflow, 1), "level": s.level,
             "minutes_to_critical": None if s.minutes_to_critical is None else round(s.minutes_to_critical, 1),
             "in_30s": None if ahead is None else round(ahead[s.name], 2)}
            for s in statuses.values()]


class Analyser:
    """`thermal` reads a thermal camera (see thermal.py) instead of ordinary video. Zones come from a file, or
    are given directly in frame pixels."""

    def __init__(self, frame_size: tuple[int, int], zones_file: Path | None = None, thermal: bool = False,
                 zones: list[Zone] | None = None):
        self.model = ThermalModel() if thermal else DensityModel()
        self.thermal = thermal
        self.zones = zones if zones is not None else load_zones(zones_file, frame_size) if zones_file else []
        self.trends = {z.name: ZoneTrend(z.name, z.area_m2) for z in self.zones}
        self.steady = Steady()
        self.vmax = None
        self.outline_at = None                                # people per pixel the forecast is outlined at

    def set_colour_scale(self, frame: np.ndarray) -> None:
        """One colour scale for a whole run, from its first frame, so colours compare across frames."""
        grid = cv2.GaussianBlur(self.model.predict(frame).grid, (0, 0), sigmaX=1.5)
        self.vmax = max(float(np.percentile(grid, 99.5)) * 1.3, 1e-3)

    def annotate(self, seconds: float, frame: np.ndarray, later: np.ndarray, gap: float,
                 caption: str | None = None) -> tuple[np.ndarray, dict, dict[str, Status]]:
        """The annotated frame, that frame's readings as a dict, and each zone's status."""
        density = self.model.predict(frame)
        flow = self.steady.update(seconds, crowd_flow(frame, later, gap))
        people = people_per_cell(density, flow)
        ahead = forecast(flow, people)
        image = overlay.arrows(overlay.heatmap(frame, density, self.vmax), density, flow)
        if self.outline_at is None:                            # half the densest crowd at the start, kept for the run
            self.outline_at = max(0.5 * float(np.percentile(people, 99)) / flow.cell ** 2, OUTLINE_FLOOR / flow.cell ** 2)
        image = overlay.forecast(image, ahead, self.outline_at)
        record = {"t": round(seconds, 2), "people": round(density.count, 1), "zones": []}
        statuses = {}
        if self.zones:
            statuses = {name: self.trends[name].update(seconds, r) for name, r in measure(self.zones, density, flow).items()}
            image = overlay.zones(image, self.zones, statuses)
            record["zones"] = zone_records(statuses, zone_ahead(self.zones, people, ahead,
                                                                {n: s.density for n, s in statuses.items()}))
        caption = caption or f"{seconds:5.1f} s   about {density.count:.0f} people" + ("   (thermal)" if self.thermal else "")
        return overlay.label(image, caption), record, statuses
