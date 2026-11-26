"""Named areas of a camera view, each with its real floor area, so head counts become people per square metre.

A zones file is JSON:

    {"frame_size": [1920, 1080],
     "zones": [{"name": "Staircase", "area_m2": 30, "polygon": [[1300, 300], [1550, 260], ...]}]}

Polygons are in pixels of a frame of `frame_size`. Make one with scripts/draw_zones.py.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .density import Density
from .flow import Flow, ahead, people_per_cell

LOOK_AHEAD = 5.0                               # seconds people are moved along their arrows to measure inflow


@dataclass
class Zone:
    name: str
    area_m2: float
    polygon: np.ndarray                        # (n, 2) pixel points, in the video's own frame size

    def mask(self, shape: tuple[int, int], pixels_per_cell: float) -> np.ndarray:
        """(rows, cols) 0..1 share of each grid cell inside the zone, for a grid of `pixels_per_cell`-pixel cells."""
        scale = 8                                            # draw finer, then average, so edge cells count in part
        fine = np.zeros((shape[0] * scale, shape[1] * scale), np.uint8)
        cv2.fillPoly(fine, [np.round(self.polygon / pixels_per_cell * scale).astype(np.int32)], 1)
        return fine.reshape(shape[0], scale, shape[1], scale).mean(axis=(1, 3))

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Boolean array, True for each (x, y) point inside the zone. `points` has shape (..., 2)."""
        contour = self.polygon.reshape(-1, 1, 2).astype(np.float32)
        flat = points.reshape(-1, 2)
        inside = [cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0 for x, y in flat]
        return np.array(inside).reshape(points.shape[:-1])


@dataclass
class Reading:
    people: float
    density: float                             # people per square metre
    inflow: float                              # people per minute crossing into the zone; negative when emptying


def load_zones(path: Path, frame_size: tuple[int, int]) -> list[Zone]:
    """The zones in `path`, scaled to a video whose frames are `frame_size` (width, height)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sx, sy = frame_size[0] / data["frame_size"][0], frame_size[1] / data["frame_size"][1]
    return [Zone(z["name"], float(z["area_m2"]), np.array(z["polygon"], np.float32) * [sx, sy]) for z in data["zones"]]


def measure(zones: list[Zone], density: Density, flow: Flow) -> dict[str, Reading]:
    """People, density and inflow for every zone in one frame."""
    crowd, now, later = ahead(flow, people_per_cell(density, flow), LOOK_AHEAD)
    out = {}
    for zone in zones:
        people = float((density.grid * zone.mask(density.grid.shape, 8 / density.scale)).sum())
        arriving = float(crowd[zone.contains(later)].sum() - crowd[zone.contains(now)].sum())
        out[zone.name] = Reading(people, people / zone.area_m2, arriving / LOOK_AHEAD * 60)
    return out
