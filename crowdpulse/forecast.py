"""Where the crowd will be in half a minute, if everyone keeps walking as they are now.

Each patch of crowd is moved along the movement arrows in short steps. Where it walks through other people it
takes on their movement; across empty ground it keeps its own pace. People who reach a part of the crowd that is
standing still stop there, so the crowd builds up where a queue or a crush is forming: you see people collecting
before they arrive.

Only people already in view are moved. Newcomers from off screen are not counted, and people who walk out of the
picture are dropped, so near the picture's edge the forecast is a floor, not a ceiling.
"""
from dataclasses import dataclass

import numpy as np

from .flow import STILL_BELOW, Flow
from .zones import Zone

HORIZON = 30.0                                 # seconds ahead
STEP = 1.0                                     # longest move, in seconds; shorter when people would skip a cell
OCCUPIED = 0.3                                 # people in a cell for its movement to count; emptier cells have none
PACKED = 9.0                                   # people per m2: nobody can be packed tighter, whatever the arrows say


@dataclass
class Forecast:
    grid: np.ndarray                           # people per flow cell, HORIZON seconds from now
    cell: int                                  # cell size in original pixels
    seconds: float


def forecast(flow: Flow, people: np.ndarray, seconds: float = HORIZON, step: float = STEP) -> Forecast:
    """Move each cell's `people` along `flow` for `seconds`, and count them up again per cell."""
    rows, cols = people.shape
    speed = np.hypot(flow.vectors[..., 0], flow.vectors[..., 1])
    vectors = flow.vectors * (speed >= STILL_BELOW)[..., None]      # shuffling on the spot is standing still
    occupied = people >= OCCUPIED
    ys, xs = np.nonzero(people > 1e-3)
    pos = np.stack([(xs + 0.5) * flow.cell, (ys + 0.5) * flow.cell], axis=1).astype(np.float32)
    vel = vectors[ys, xs].copy()
    mass = people[ys, xs].astype(np.float32)
    fastest = float(np.abs(vectors).max(initial=0.0))
    moves = int(np.ceil(seconds / min(step, 0.5 * flow.cell / fastest))) if fastest > 0 else 1
    dt = seconds / moves
    for _ in range(moves):
        col, row = np.floor(pos[:, 0] / flow.cell).astype(int), np.floor(pos[:, 1] / flow.cell).astype(int)
        inside = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)          # people who left the picture go
        pos, vel, mass, row, col = pos[inside], vel[inside], mass[inside], row[inside], col[inside]
        among = occupied[row, col]
        vel[among] = vectors[row[among], col[among]]                          # among others, move as they move
        pos += vel * dt
    col, row = np.floor(pos[:, 0] / flow.cell).astype(int), np.floor(pos[:, 1] / flow.cell).astype(int)
    inside = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
    grid = np.zeros((rows, cols), np.float32)
    np.add.at(grid, (row[inside], col[inside]), mass[inside])
    return Forecast(grid, flow.cell, seconds)


def zone_ahead(zones: list[Zone], people: np.ndarray, ahead: Forecast, density_now: dict[str, float]) -> dict[str, float]:
    """Each zone's density in people per m2 at the forecast's time.

    The change the forecast predicts is added to the zone's measured density now, so the coarse grid of the
    movement arrows does not blur the zone's edges.
    """
    out = {}
    for zone in zones:
        inside = zone.mask(ahead.grid.shape, ahead.cell)
        change = float(((ahead.grid - people) * inside).sum()) / zone.area_m2
        out[zone.name] = float(np.clip(density_now[zone.name] + change, 0.0, PACKED))
    return out
