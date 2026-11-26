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
from .flow import Steady, crowd_flow
from .risk import Status, ZoneTrend
from .zones import load_zones, measure

FLOW_GAP = 0.25                                # seconds between the two frames compared for movement


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


class Analyser:
    def __init__(self, frame_size: tuple[int, int], zones_file: Path | None = None):
        self.model = DensityModel()
        self.zones = load_zones(zones_file, frame_size) if zones_file else []
        self.trends = {z.name: ZoneTrend(z.name, z.area_m2) for z in self.zones}
        self.steady = Steady()
        self.vmax = None

    def set_colour_scale(self, frame: np.ndarray) -> None:
        """One colour scale for a whole run, from its first frame, so colours compare across frames."""
        grid = cv2.GaussianBlur(self.model.predict(frame).grid, (0, 0), sigmaX=1.5)
        self.vmax = max(float(np.percentile(grid, 99.5)) * 1.3, 1e-3)

    def annotate(self, seconds: float, frame: np.ndarray, later: np.ndarray, gap: float) -> tuple[np.ndarray, dict, dict[str, Status]]:
        """The annotated frame, that frame's readings as a dict, and each zone's status."""
        density = self.model.predict(frame)
        flow = self.steady.update(seconds, crowd_flow(frame, later, gap))
        image = overlay.arrows(overlay.heatmap(frame, density, self.vmax), density, flow)
        record = {"t": round(seconds, 2), "people": round(density.count, 1), "zones": []}
        statuses = {}
        if self.zones:
            statuses = {name: self.trends[name].update(seconds, r) for name, r in measure(self.zones, density, flow).items()}
            image = overlay.zones(image, self.zones, statuses)
            record["zones"] = [{"name": s.name, "people": round(s.people, 1), "density": round(s.density, 2),
                                "inflow_per_min": round(s.inflow, 1), "level": s.level,
                                "minutes_to_critical": None if s.minutes_to_critical is None else round(s.minutes_to_critical, 1)}
                               for s in statuses.values()]
        return overlay.label(image, f"{seconds:5.1f} s   about {density.count:.0f} people"), record, statuses
