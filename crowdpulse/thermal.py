"""Crowd density from a thermal camera.

A thermal camera sees body heat: people are warm blobs on cooler ground, in the dark, through haze or smoke, and
without showing anyone's face. The crowd model in density.py was trained on ordinary video, so it cannot read
these pictures. Instead this finds the warm pixels and divides their area by the area one person covers.

How much area is one person? It depends on the camera's height and lens, so unless it is given, it is measured
from the footage: the typical size of the separate warm blobs, while the crowd is still sparse enough for most
blobs to be one person each. The warmth that counts as a person is fixed at that moment, as a step above the
ground's temperature, so that a hot packed crowd later on is measured the same way as the first few people.

Expects "white-hot" footage (warmer is brighter), the default on most thermal cameras; colour palettes such as
ironbow also mostly get brighter with heat and work roughly. A packed crowd merges into one warm mass, so counts
come from area rather than blobs; past about 8 people per m2, seen from above, the warm area stops growing.
"""
import cv2
import numpy as np

from .density import STRIDE, Density

MIN_CONTRAST = 30                              # grey levels between ground and the warmest spots; less means nobody there
MIN_BLOB = 6                                   # warm blobs smaller than this many pixels are noise
CALIBRATE_FROM = 5                             # separate blobs needed to measure one person's size


class ThermalModel:
    def __init__(self, person_px: float | None = None, warmer_by: float | None = None):
        self.person_px = person_px             # warm pixels one person covers in the analysed frame
        self.warmer_by = warmer_by             # grey levels above the ground that count as a person

    @staticmethod
    def ground(gray: np.ndarray) -> float:
        """The ground's grey level: the most common one, since most of a camera's view is floor, not people or sky."""
        counts = cv2.GaussianBlur(np.bincount(gray.ravel(), minlength=256).astype(np.float32).reshape(1, -1), (0, 0), 2)
        return float(np.argmax(counts))

    def warm(self, gray: np.ndarray) -> np.ndarray:
        """Mask of pixels warm enough to be people."""
        ground, high = self.ground(gray), float(np.percentile(gray, 99.95))
        if high - ground < MIN_CONTRAST:
            return np.zeros(gray.shape, np.uint8)
        if self.warmer_by is not None:
            level = ground + self.warmer_by
        else:                                  # until calibrated: half-way between the ground and the warmest spots
            level = ground + 0.5 * (high - ground)
        mask = (gray > level).astype(np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    @staticmethod
    def singles(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(labels, the label numbers of blobs that look like one person, their areas). Blobs much bigger than
        the typical one are groups; tiny ones are noise."""
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        real = areas >= MIN_BLOB
        if not real.any():
            return labels, np.zeros(0, int), np.zeros(0, int)
        keep = real & (areas < 2.5 * np.median(areas[real]))
        return labels, np.flatnonzero(keep) + 1, areas[keep]

    def calibrate(self, gray: np.ndarray, mask: np.ndarray) -> None:
        """Measure one person's warm area from the separate blobs, if there are enough of them, and fix the
        warmth that counts as a person from then on."""
        labels, ids, _ = self.singles(mask)
        if len(ids) < CALIBRATE_FROM:
            return
        peak = float(np.median([gray[labels == i].max() for i in ids]))
        self.warmer_by = 0.5 * (peak - self.ground(gray))           # half-way up a lone person's warmth
        _, _, areas = self.singles(self.warm(gray))                  # their size at that warmth
        self.person_px = float(np.median(areas)) if len(areas) else None
        if self.person_px is None:
            self.warmer_by = None

    def predict(self, frame_bgr: np.ndarray, max_side: int = 1024) -> Density:
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1.0:
            frame_bgr = cv2.resize(frame_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        gray = cv2.GaussianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        mask = self.warm(gray)
        if self.person_px is None:
            self.calibrate(gray, mask)
            mask = self.warm(gray)
        rows, cols = gray.shape[0] // STRIDE, gray.shape[1] // STRIDE
        if self.person_px is None:                       # not measured yet: report nobody rather than guess
            return Density(np.zeros((rows, cols), np.float32), scale)
        warm_per_cell = cv2.resize(mask[:rows * STRIDE, :cols * STRIDE].astype(np.float32), (cols, rows),
                                   interpolation=cv2.INTER_AREA) * STRIDE * STRIDE
        return Density(warm_per_cell / self.person_px, scale)
