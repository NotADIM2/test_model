"""Draw the density heatmap, the movement arrows and the head count onto a frame."""
import cv2
import numpy as np

from .density import Density
from .flow import Flow

ARROW_SECONDS = 1.0            # an arrow shows where that part of the crowd will be in this many seconds
MIN_SPEED = 12.0               # pixels per second; slower cells get no arrow
MIN_PEOPLE_PER_CELL = 0.4      # cells with fewer people than this get no arrow


def heatmap(frame: np.ndarray, density: Density, vmax: float | None = None) -> np.ndarray:
    """The frame with crowd density laid over it: clear where empty, through yellow to red where packed.

    `vmax` is the density shown as full red. Pass the same value for every frame of a video so colours compare.
    """
    grid = cv2.GaussianBlur(density.grid, (0, 0), sigmaX=1.5)
    vmax = vmax or max(float(np.percentile(grid, 99.5)), 1e-3)
    level = np.clip(cv2.resize(grid, (frame.shape[1], frame.shape[0])) / vmax, 0, 1)
    colours = cv2.applyColorMap((level * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    alpha = np.clip(level * 1.6, 0, 0.65)[..., None]          # empty floor stays visible
    return (frame * (1 - alpha) + colours * alpha).astype(np.uint8)


def people_per_cell(density: Density, flow: Flow) -> np.ndarray:
    """Head count inside each flow cell, so arrows are drawn only where there are people."""
    rows, cols = flow.vectors.shape[:2]
    cell_in_grid = flow.cell * density.scale / 8
    out = np.zeros((rows, cols), np.float32)
    for r in range(rows):
        for c in range(cols):
            y0, x0 = int(r * cell_in_grid), int(c * cell_in_grid)
            y1, x1 = int((r + 1) * cell_in_grid), int((c + 1) * cell_in_grid)
            out[r, c] = density.grid[y0:y1, x0:x1].sum()
    return out


def arrows(image: np.ndarray, density: Density, flow: Flow) -> np.ndarray:
    """White arrows showing which way the people in each cell are moving."""
    out = image.copy()
    people = people_per_cell(density, flow)
    for (cx, cy), (dx, dy), n in zip(flow.centres().reshape(-1, 2), flow.vectors.reshape(-1, 2), people.reshape(-1)):
        if n < MIN_PEOPLE_PER_CELL or np.hypot(dx, dy) < MIN_SPEED:
            continue
        step = np.array([dx, dy]) * ARROW_SECONDS
        step *= min(1.0, flow.cell * 0.9 / max(np.hypot(*step), 1e-6))      # never longer than a cell
        start = (int(cx - step[0] / 2), int(cy - step[1] / 2))
        end = (int(cx + step[0] / 2), int(cy + step[1] / 2))
        cv2.arrowedLine(out, start, end, (20, 20, 20), 5, cv2.LINE_AA, tipLength=0.35)
        cv2.arrowedLine(out, start, end, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)
    return out


def label(image: np.ndarray, text: str) -> np.ndarray:
    """A dark caption box in the top-left corner."""
    out = image.copy()
    scale = max(image.shape[1] / 1600, 0.6)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    pad = int(12 * scale)
    cv2.rectangle(out, (pad, pad), (pad * 3 + tw, pad * 3 + th + base), (25, 25, 25), -1)
    cv2.putText(out, text, (pad * 2, pad * 2 + th), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out
