"""Draw the density heatmap, the movement arrows and the head count onto a frame."""
import cv2
import numpy as np

from .density import Density
from .flow import STILL_BELOW, Flow, people_per_cell
from .forecast import Forecast
from .risk import Status
from .zones import Zone

ARROW_SECONDS = 1.0            # an arrow shows where that part of the crowd will be in this many seconds
MIN_PEOPLE_PER_CELL = 0.4      # cells with fewer people than this get no arrow


def heatmap(frame: np.ndarray, density: Density, vmax: float | None = None) -> np.ndarray:
    """The frame with crowd density laid over it: clear where empty, through yellow to red where packed.

    `vmax` is the density shown as full red. Pass the same value for every frame of a video so colours compare.
    """
    grid = cv2.GaussianBlur(density.grid, (0, 0), sigmaX=1.5)
    vmax = vmax or max(float(np.percentile(grid, 99.5)), 1e-3)
    level = np.clip(cv2.resize(grid, (frame.shape[1], frame.shape[0])) / vmax, 0, 1)
    colours = cv2.applyColorMap((level * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    alpha = np.clip(level * 1.6, 0, 0.65).astype(np.float32)  # empty floor stays visible
    return cv2.blendLinear(frame, colours, 1 - alpha, alpha)


def arrows(image: np.ndarray, density: Density, flow: Flow) -> np.ndarray:
    """White arrows showing which way the people in each cell are moving."""
    out = image.copy()
    people = people_per_cell(density, flow)
    for (cx, cy), (dx, dy), n in zip(flow.centres().reshape(-1, 2), flow.vectors.reshape(-1, 2), people.reshape(-1)):
        if n < MIN_PEOPLE_PER_CELL or np.hypot(dx, dy) < STILL_BELOW:
            continue
        step = np.array([dx, dy]) * ARROW_SECONDS
        step *= min(1.0, flow.cell * 0.9 / max(np.hypot(*step), 1e-6))      # never longer than a cell
        start = (int(cx - step[0] / 2), int(cy - step[1] / 2))
        end = (int(cx + step[0] / 2), int(cy + step[1] / 2))
        cv2.arrowedLine(out, start, end, (20, 20, 20), 5, cv2.LINE_AA, tipLength=0.35)
        cv2.arrowedLine(out, start, end, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.35)
    return out


DASH = 14                      # pixels per dash of the forecast outline


def _dashed(image: np.ndarray, points: np.ndarray, colour: tuple, thickness: int) -> None:
    """A dashed closed line through `points`."""
    travelled = 0.0
    for a, b in zip(points, np.roll(points, -1, axis=0)):
        length = float(np.hypot(*(b - a)))
        for start in np.arange(0.0, length, DASH):
            if int((travelled + start) // DASH) % 2 == 0:
                end = min(start + DASH, length)
                p, q = a + (b - a) * start / max(length, 1e-6), a + (b - a) * end / max(length, 1e-6)
                cv2.line(image, tuple(int(v) for v in p), tuple(int(v) for v in q), colour, thickness, cv2.LINE_AA)
        travelled += length


def forecast(image: np.ndarray, ahead: Forecast, level: float) -> np.ndarray:
    """A dashed outline around where the crowd will be dense in `ahead.seconds`: wherever its density reaches
    `level` people per pixel. Compared with the heatmap under it, it shows where people are about to collect."""
    out = image.copy()
    h, w = image.shape[:2]
    rows, cols = ahead.grid.shape
    quarter = np.zeros((h // 4, w // 4), np.float32)                     # work at a quarter size: smooth and quick
    patch = cv2.resize(ahead.grid / ahead.cell ** 2, (cols * ahead.cell // 4, rows * ahead.cell // 4),
                       interpolation=cv2.INTER_CUBIC)[:h // 4, :w // 4]
    quarter[:patch.shape[0], :patch.shape[1]] = patch
    quarter = cv2.GaussianBlur(quarter, (0, 0), ahead.cell / 8)
    contours, _ = cv2.findContours((quarter >= level).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= (ahead.cell / 4) ** 2]   # skip specks
    scale = max(w / 1600, 0.55)
    for contour in contours:
        points = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2).astype(np.float32) * 4
        _dashed(out, points, (25, 25, 25), max(4, int(5 * scale)))
        _dashed(out, points, (255, 255, 255), max(2, int(2.5 * scale)))
    if contours:
        biggest = max(contours, key=cv2.contourArea).reshape(-1, 2) * 4       # label it under its lowest point,
        x, y = biggest[np.argmax(biggest[:, 1])]                               # clear of zone tags at the top
        text = f"crowd in {ahead.seconds:.0f} s"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, 2)
        x, y = int(min(max(x - tw // 2, 4), w - tw - 12)), int(min(y + th + 14, h - base - 6))
        cv2.rectangle(out, (x - 5, y - th - 5), (x + tw + 5, y + base + 2), (25, 25, 25), -1)
        cv2.putText(out, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (255, 255, 255), 2, cv2.LINE_AA)
    return out


LEVEL_COLOURS = {"green": (80, 175, 60), "amber": (0, 170, 255), "red": (40, 40, 220)}     # BGR


def zones(image: np.ndarray, zone_list: list[Zone], statuses: dict[str, Status]) -> np.ndarray:
    """Each zone outlined in its warning colour, with a tag giving its density and countdown."""
    out = image.copy()
    scale = max(image.shape[1] / 1600, 0.55)
    for zone in zone_list:
        status = statuses[zone.name]
        colour = LEVEL_COLOURS[status.level]
        points = np.round(zone.polygon).astype(np.int32)
        cv2.polylines(out, [points], True, colour, max(2, int(3 * scale)), cv2.LINE_AA)
        lines = [f"{zone.name}  {status.density:.1f}/m2  {status.level.upper()}"]
        if status.inflow >= 1:
            lines.append(f"+{status.inflow:.0f} people/min")
        if status.minutes_to_critical is not None:
            lines.append("CRITICAL NOW" if status.minutes_to_critical == 0 else
                         "critical in <1 min" if status.minutes_to_critical < 1 else f"critical in {status.minutes_to_critical:.0f} min")
        x, y = points[:, 0].min() + 8, points[:, 1].min() + 8
        x = min(max(x, 8), image.shape[1] - int(330 * scale))
        y = min(max(y, int(70 * scale)), image.shape[0] - int(100 * scale))
        for i, text in enumerate(lines):
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, 2)
            top = y + i * int((th + base + 10))
            cv2.rectangle(out, (x, top), (x + tw + 14, top + th + base + 8), colour, -1)
            cv2.putText(out, text, (x + 7, top + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (255, 255, 255), 2, cv2.LINE_AA)
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
