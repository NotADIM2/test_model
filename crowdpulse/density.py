"""Crowd density from one video frame, with DM-Count (MIT licence, UCF-QNRF weights) run through ONNX Runtime.

The model takes a 512x512 RGB image and returns a 64x64 density map: each cell holds the number of people in
that 8x8 pixel patch, so the sum of the map is the head count. Frames larger than 512 pixels are cut into
512x512 tiles, so people keep their real size in pixels instead of being squashed.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "dmcount" / "dmcount.onnx"
TILE = 512
STRIDE = 8                                     # each output cell covers 8x8 input pixels
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class Density:
    grid: np.ndarray                           # people per 8x8 patch of the analysed frame, shape (H/8, W/8)
    scale: float                               # analysed frame size / original frame size

    @property
    def count(self) -> float:
        return float(self.grid.sum())


class DensityModel:
    def __init__(self, path: Path = MODEL_PATH):
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing. Run: python scripts/get_model.py")
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def predict(self, frame_bgr: np.ndarray, max_side: int = 1024) -> Density:
        """Density map of a BGR frame. Frames longer than `max_side` pixels are shrunk first, to bound the work."""
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale < 1:
            frame_bgr = cv2.resize(frame_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = (rgb - MEAN) / STD
        h, w = image.shape[:2]
        ph, pw = -h % TILE, -w % TILE                # pad to whole tiles; zero is the mean colour after normalising
        image = np.pad(image, ((0, ph), (0, pw), (0, 0)))
        grid = np.zeros((image.shape[0] // STRIDE, image.shape[1] // STRIDE), np.float32)
        for y in range(0, image.shape[0], TILE):
            for x in range(0, image.shape[1], TILE):
                tile = image[y:y + TILE, x:x + TILE].transpose(2, 0, 1)[None]
                out = self.session.run(None, {self.input: np.ascontiguousarray(tile)})[0][0, 0]
                grid[y // STRIDE:(y + TILE) // STRIDE, x // STRIDE:(x + TILE) // STRIDE] = out
        return Density(grid[:h // STRIDE, :w // STRIDE], scale)
