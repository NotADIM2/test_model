"""Which way the crowd is moving, from two frames a fraction of a second apart.

Dense optical flow (Farneback, in OpenCV) gives every pixel a motion vector. The camera's own movement (a
hand-held phone that pans, turns or zooms, a swaying pole) is removed first: corner points are tracked between
the two frames, the shift, turn and zoom that best explains most of them (RANSAC, so walking people are
outvoted) is taken as the camera's motion, and the first frame is moved to line up with the second. Only then
is dense flow measured, so it sees just the people, whose steps are small. The vectors are then averaged over
a grid of cells, one arrow per cell.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .density import STRIDE, Density

WORK_WIDTH = 640                               # frames are shrunk to this width first: flow is slow at full size
STILL_BELOW = 12.0                             # pixels per second; slower cells are people shuffling in place
STEADY_OVER = 4.0                              # seconds the motion is averaged over; real crowd motion persists, jitter does not


@dataclass
class Flow:
    vectors: np.ndarray                        # (rows, cols, 2): mean (dx, dy) per cell, in original pixels per second
    cell: int                                  # cell size in original pixels

    def centres(self) -> np.ndarray:
        """(rows, cols, 2) pixel centres of the cells, in the original frame."""
        rows, cols = self.vectors.shape[:2]
        ys, xs = np.mgrid[0:rows, 0:cols]
        return np.stack([(xs + 0.5) * self.cell, (ys + 0.5) * self.cell], axis=-1)


def camera_motion(before: np.ndarray, after: np.ndarray) -> np.ndarray | None:
    """The 2x3 matrix taking `before` to `after` for the camera's shift, turn and zoom, or None if unclear."""
    points = cv2.goodFeaturesToTrack(before, maxCorners=500, qualityLevel=0.01, minDistance=8)
    if points is None:
        return None
    moved, found, _ = cv2.calcOpticalFlowPyrLK(before, after, points, None)
    ok = found.reshape(-1) == 1
    if ok.sum() < 12:
        return None
    matrix, _ = cv2.estimateAffinePartial2D(points[ok], moved[ok], method=cv2.RANSAC, ransacReprojThreshold=1.5)
    return matrix


def crowd_flow(before_bgr: np.ndarray, after_bgr: np.ndarray, seconds: float, cell: int = 64) -> Flow:
    """Average crowd motion per `cell` x `cell` pixel square between two frames taken `seconds` apart."""
    h, w = before_bgr.shape[:2]
    k = WORK_WIDTH / w
    small = lambda f: cv2.cvtColor(cv2.resize(f, (WORK_WIDTH, round(h * k)), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    before, after = small(before_bgr), small(after_bgr)
    matrix = camera_motion(before, after)
    seen = np.ones(before.shape, np.float32)
    if matrix is not None:                                   # line the first frame up with the second
        size = (before.shape[1], before.shape[0])
        before = cv2.warpAffine(before, matrix, size, borderMode=cv2.BORDER_REPLICATE)
        seen = cv2.warpAffine(seen, matrix, size, borderValue=0)
    flow = cv2.calcOpticalFlowFarneback(before, after, None,
                                        pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    flow *= (seen > 0.99)[..., None]                         # no motion where the first frame had no picture
    flow /= k * seconds                                      # back to original pixels, per second
    rows, cols = h // cell, w // cell
    full = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)[:rows * cell, :cols * cell]
    vectors = full.reshape(rows, cell, cols, cell, 2).mean(axis=(1, 3))
    return Flow(vectors, cell)


def people_per_cell(density: Density, flow: Flow) -> np.ndarray:
    """(rows, cols) head count inside each flow cell."""
    rows, cols = flow.vectors.shape[:2]
    step = flow.cell * density.scale / STRIDE                # flow cell size in density-map cells
    out = np.zeros((rows, cols), np.float32)
    for r in range(rows):
        for c in range(cols):
            out[r, c] = density.grid[int(r * step):int((r + 1) * step), int(c * step):int((c + 1) * step)].sum()
    return out


class Steady:
    """Averages each cell's motion over the last few seconds of a video, frame after frame."""

    def __init__(self, seconds: float = STEADY_OVER):
        self.seconds, self.vectors, self.last = seconds, None, None

    def update(self, at: float, flow: Flow) -> Flow:
        if self.vectors is None or self.vectors.shape != flow.vectors.shape:
            self.vectors = flow.vectors.copy()
        else:
            self.vectors += (1 - np.exp(-(at - self.last) / self.seconds)) * (flow.vectors - self.vectors)
        self.last = at
        return Flow(self.vectors.copy(), flow.cell)


def ahead(flow: Flow, people: np.ndarray, seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where each cell's people are now and will be after `seconds`, if they keep moving as they are.

    Returns (people, now, later): people per cell and (x, y) positions, on a grid that reaches past the
    picture's edges. Comparing who is inside a zone now with who will be inside it then gives the zone's
    inflow, so a zone can be seen filling before its own density has risen. Cells slower than STILL_BELOW
    stay put. The crowd just outside the picture is taken to look like the crowd at its edge, so people walk
    in from off screen as well as out: a steady stream through the picture neither fills nor drains a zone.
    """
    moving = np.hypot(flow.vectors[..., 0], flow.vectors[..., 1]) >= STILL_BELOW
    vectors = flow.vectors * moving[..., None]
    pad = int(min(np.ceil(np.abs(vectors).max(initial=0) * seconds / flow.cell) + 1, 12))
    vectors = np.pad(vectors, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    people = np.pad(people, pad, mode="edge")
    ys, xs = np.mgrid[0:people.shape[0], 0:people.shape[1]]
    now = np.stack([(xs - pad + 0.5) * flow.cell, (ys - pad + 0.5) * flow.cell], axis=-1).astype(np.float32)
    return people, now, now + vectors * seconds
