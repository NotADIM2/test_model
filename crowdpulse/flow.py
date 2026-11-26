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

WORK_WIDTH = 640                               # frames are shrunk to this width first: flow is slow at full size


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
