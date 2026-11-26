"""A stadium-entry simulation, to show CrowdPulse's warnings on a crowd that really does become dangerous.

A fictional plaza in front of a stadium wall with four gates. People arrive from three approach roads, walk to
a gate and are let through at a ticket-check rate. When far more people arrive than the open gates can take,
the crowd in front of a gate packs tighter and tighter: the pattern behind entry crushes such as the one
outside Bengaluru's Chinnaswamy Stadium on 4 June 2025. The layout and numbers are illustrative, not a
reconstruction of that day.

Each person is a point. They walk towards their gate at up to 1.3 m/s, slow down as the crowd around them
packs (stopping at about 7 people per m²), and are pushed from denser towards emptier ground. Densities are
therefore real people per square metre, and they go through the same warning code as the camera.

What the operator can do, from the dashboard: open or close a gate, hold new arrivals at the outer cordon,
and send stewards to point people to the gate with the shortest wait instead of the nearest one.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .density import Density
from .overlay import heatmap
from .zones import Reading, Zone

WIDTH, DEPTH = 80.0, 50.0                      # plaza size in metres; the stadium wall runs along the top edge
GATES = (12.0, 30.0, 50.0, 68.0)               # gate centres along the wall, in metres from the left
GATE_RATE = 1.1                                # people a gate lets through per second: about 66 a minute,
                                               # so one gate cannot take the 250-a-minute surge but four can
ROADS = ((15.0, 5.0), (40.0, 6.0), (65.0, 5.0))  # approach roads at the bottom edge: (centre, half-width) in metres
WALK = 1.3                                     # free walking speed, m/s
JAM = 7.0                                      # people per m2 at which nobody can move
PUSH = 0.35                                    # how hard people are pushed from denser to emptier ground
ZONE_HALF_WIDTH, ZONE_DEPTH = 6.0, 8.0         # the zone in front of each gate: 12 m x 8 m
COMMITTED = 10.0                               # people this close to their gate are in its queue and stay
LOOK_AHEAD = 5.0                               # seconds, for inflow, as with the camera
STEP = 0.2                                     # simulation step, seconds
PX = 16                                        # pixels per metre in the picture
TOP = 48                                       # pixels above the plaza for the stadium wall and gate names


@dataclass
class Gate:
    name: str
    x: float
    open: bool
    tokens: float = 0.0                        # fractional people the gate can let through next


def arrival_rate(t: float) -> float:
    """People per minute arriving: a trickle, then a surge after a 'free entry' announcement at 2 minutes."""
    if t < 120:
        return 30.0
    if t < 300:
        return 30.0 + (t - 120) / 180 * 220     # ramps up to 250 a minute over 3 minutes
    if t < 900:
        return 250.0
    return max(30.0, 250.0 - (t - 900) / 300 * 220)


class Simulation:
    def __init__(self, open_gates: tuple[int, ...] = (3,), seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self.gates = [Gate(f"Gate {i + 1}", x, (i + 1) in open_gates) for i, x in enumerate(GATES)]
        self.pos = np.zeros((0, 2), np.float32)
        self.vel = np.zeros((0, 2), np.float32)
        self.target = np.zeros(0, np.int32)
        self.t = 0.0
        self.arrivals_due = 0.0
        self.holding, self.stewards = False, False
        self.held = 0                          # people stopped at the outer cordon
        self.arrived = self.entered = 0
        self.next_choice = 0.0
        self.zones = [Zone(g.name, 2 * ZONE_HALF_WIDTH * ZONE_DEPTH,
                           np.array([[g.x - ZONE_HALF_WIDTH, 0], [g.x + ZONE_HALF_WIDTH, 0],
                                     [g.x + ZONE_HALF_WIDTH, ZONE_DEPTH], [g.x - ZONE_HALF_WIDTH, ZONE_DEPTH]], np.float32))
                      for g in self.gates]

    # ---- what the operator can do

    def act(self, action: str, gate: int | None = None) -> None:
        if action in ("open_gate", "close_gate"):
            if gate is None or not 1 <= gate <= len(self.gates):
                raise ValueError("Unknown gate")
            self.gates[gate - 1].open = action == "open_gate"
        elif action == "hold_arrivals":
            self.holding = not self.holding
        elif action == "stewards":
            self.stewards = not self.stewards
        else:
            raise ValueError("Unknown action")
        self.next_choice = self.t                            # people react to a change straight away

    # ---- the crowd

    def density_grid(self, cell: float = 1.0) -> np.ndarray:
        """People per m2 on a grid of `cell`-metre squares, lightly smoothed."""
        rows, cols = int(DEPTH / cell), int(WIDTH / cell)
        grid, _, _ = np.histogram2d(self.pos[:, 1], self.pos[:, 0], bins=(rows, cols), range=((0, DEPTH), (0, WIDTH)))
        return cv2.GaussianBlur(grid.astype(np.float32) / cell ** 2, (0, 0), sigmaX=1.0)

    def choose_gates(self) -> None:
        open_x = np.array([g.x for g in self.gates if g.open])
        open_index = np.array([i for i, g in enumerate(self.gates) if g.open])
        if len(open_x) == 0 or len(self.pos) == 0:
            if len(open_x) == 0 and len(self.pos):                 # no gate open: people press towards the nearest one
                self.target = np.abs(self.pos[:, :1] - np.array(GATES)).argmin(axis=1).astype(np.int32)
            return
        distance = np.hypot(self.pos[:, :1] - open_x, self.pos[:, 1:2])
        cost = distance / WALK
        if self.stewards:                                          # stewards point people to the shortest wait
            waiting = np.array([np.sum(np.hypot(self.pos[:, 0] - x, self.pos[:, 1]) < 15) for x in open_x])
            cost = cost + waiting / GATE_RATE
        choice = open_index[cost.argmin(axis=1)].astype(np.int32)
        if len(self.target) == len(self.pos):                     # people already in a queue cannot turn round
            queued = np.hypot(self.pos[:, 0] - np.array(GATES)[self.target], self.pos[:, 1]) < COMMITTED
            still_open = np.array([g.open for g in self.gates])[self.target]
            choice = np.where(queued & still_open, self.target, choice)
        self.target = choice

    def spawn(self, dt: float) -> None:
        self.arrivals_due += arrival_rate(self.t) / 60 * dt
        new = int(self.arrivals_due)
        self.arrivals_due -= new
        if self.holding:
            self.held += new
            return
        if self.held:                                              # released people come through at 120 a minute
            release = min(self.held, int(self.rng.poisson(2.0 * dt)))
            self.held -= release
            new += release
        if new == 0:
            return
        road = self.rng.integers(0, len(ROADS), new)
        centre, half = np.array(ROADS)[road].T
        x = centre + self.rng.uniform(-1, 1, new) * half
        y = DEPTH - self.rng.uniform(0.2, 1.5, new)
        self.pos = np.vstack([self.pos, np.stack([x, y], axis=1).astype(np.float32)])
        self.vel = np.vstack([self.vel, np.zeros((new, 2), np.float32)])
        self.target = np.concatenate([self.target, np.zeros(new, np.int32)])
        self.arrived += new
        self.next_choice = self.t

    def step(self, dt: float = STEP) -> None:
        self.spawn(dt)
        if self.t >= self.next_choice:
            self.choose_gates()
            self.next_choice = self.t + 10.0
        if len(self.pos):
            grid = self.density_grid()
            gy, gx = np.gradient(grid)
            r = np.clip(self.pos[:, 1].astype(int), 0, grid.shape[0] - 1)
            c = np.clip(self.pos[:, 0].astype(int), 0, grid.shape[1] - 1)
            local = grid[r, c]
            goal = np.stack([np.array(GATES)[self.target], np.full(len(self.pos), 0.4)], axis=1) - self.pos
            direction = goal / np.maximum(np.linalg.norm(goal, axis=1, keepdims=True), 1e-6)
            speed = WALK * np.clip(1 - local / JAM, 0.03, 1)[:, None]
            push = -PUSH * np.stack([gx[r, c], gy[r, c]], axis=1)
            vel = direction * speed + push + self.rng.normal(0, 0.05, self.pos.shape)
            limit = np.maximum(np.linalg.norm(vel, axis=1, keepdims=True) / WALK, 1)
            self.vel = (vel / limit).astype(np.float32)
            self.pos = self.pos + self.vel * dt
            self.pos[:, 0] = np.clip(self.pos[:, 0], 0.2, WIDTH - 0.2)
            self.pos[:, 1] = np.clip(self.pos[:, 1], 0.3, DEPTH - 0.1)   # the wall stops everyone; only gates let people in
            self.admit(dt)
        self.t += dt

    def admit(self, dt: float) -> None:
        through = np.zeros(len(self.pos), bool)
        for i, gate in enumerate(self.gates):
            if not gate.open:
                gate.tokens = 0.0
                continue
            gate.tokens += GATE_RATE * dt
            at_gate = np.flatnonzero((self.target == i) & (np.hypot(self.pos[:, 0] - gate.x, self.pos[:, 1]) < 1.5))
            n = min(int(gate.tokens), len(at_gate))
            if n:
                closest = at_gate[np.argsort(np.hypot(self.pos[at_gate, 0] - gate.x, self.pos[at_gate, 1]))[:n]]
                through[closest] = True
                gate.tokens -= n
            gate.tokens = min(gate.tokens, 2.0)                 # an idle gate does not bank up capacity
        if through.any():
            self.pos, self.vel, self.target = self.pos[~through], self.vel[~through], self.target[~through]
            self.entered += int(through.sum())

    def advance(self, seconds: float) -> None:
        for _ in range(max(1, round(seconds / STEP))):
            self.step()

    # ---- what CrowdPulse sees

    def readings(self) -> dict[str, Reading]:
        later = self.pos + self.vel * LOOK_AHEAD
        out = {}
        for zone in self.zones:
            now_in = zone.contains(self.pos).sum() if len(self.pos) else 0
            later_in = zone.contains(later).sum() if len(self.pos) else 0
            out[zone.name] = Reading(float(now_in), float(now_in / zone.area_m2), float((later_in - now_in) / LOOK_AHEAD * 60))
        return out

    def pixel_zones(self) -> list[Zone]:
        return [Zone(z.name, z.area_m2, z.polygon * PX + [0, TOP]) for z in self.zones]

    def render(self, caption: str = "") -> np.ndarray:
        """A top-down picture: plaza, density heatmap, people, the wall and its gates, and `caption` bottom left."""
        h, w = TOP + int(DEPTH * PX), int(WIDTH * PX)
        image = np.full((h, w, 3), (52, 48, 44), np.uint8)
        cv2.rectangle(image, (0, 0), (w, TOP), (90, 86, 82), -1)
        cv2.putText(image, "Stadium wall", (8, TOP - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        for centre, half in ROADS:
            x0, x1 = int((centre - half) * PX), int((centre + half) * PX)
            cv2.rectangle(image, (x0, h - 10), (x1, h), (110, 104, 98), -1)
        cells = np.zeros((h // 8, w // 8), np.float32)               # people per 8x8 px cell, for the heatmap
        if len(self.pos):
            px = (self.pos * PX + [0, TOP]).astype(int)
            np.add.at(cells, (np.clip(px[:, 1] // 8, 0, cells.shape[0] - 1), np.clip(px[:, 0] // 8, 0, cells.shape[1] - 1)), 1)
        crush_per_cell = 6.0 * (8 / PX) ** 2                         # 6 people per m2 shows as full red
        image = heatmap(image, Density(cv2.GaussianBlur(cells, (0, 0), 1.2), 1.0), vmax=crush_per_cell)
        for x, y in (self.pos * PX + [0, TOP]).astype(int):
            cv2.circle(image, (int(x), int(y)), 2, (235, 235, 235), -1, cv2.LINE_AA)
        for gate in self.gates:
            x = int(gate.x * PX)
            colour = (90, 190, 70) if gate.open else (60, 60, 210)
            cv2.rectangle(image, (x - int(1.5 * PX), TOP - 6), (x + int(1.5 * PX), TOP + 2), colour, -1)
            label = f"{gate.name} {'OPEN' if gate.open else 'CLOSED'}"
            cv2.putText(image, label, (x - 60, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2, cv2.LINE_AA)
        if caption:
            cv2.putText(image, caption, (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA)
        return image

    def render_thermal(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """What a white-hot thermal camera over the plaza would show: each person a warm blob about 0.4 m across
        on cool paving, with sensor noise. Same size as render(), so the zones line up."""
        rng = rng or self.rng
        h, w = TOP + int(DEPTH * PX), int(WIDTH * PX)
        heat = np.zeros((h, w), np.float32)
        if len(self.pos):
            px = (self.pos * PX + [0, TOP]).astype(int)
            np.add.at(heat, (np.clip(px[:, 1], 0, h - 1), np.clip(px[:, 0], 0, w - 1)), 1.0)
        body = 0.2 * PX / 1.5                                  # blur so a lone person peaks well above the ground
        heat = cv2.GaussianBlur(heat, (0, 0), body) * (2 * np.pi * body ** 2) * 0.55
        ground = 0.22 + 0.04 * np.linspace(0, 1, h, dtype=np.float32)[:, None]   # paving a little warmer in the sun
        image = np.clip(ground + np.minimum(heat, 0.7) + rng.normal(0, 0.015, (h, w)).astype(np.float32), 0, 1)
        image[:TOP] = 0.12                                     # the stadium wall, cooler than the paving
        return cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def info(self) -> dict:
        return {"gates": [{"number": i + 1, "name": g.name, "open": g.open} for i, g in enumerate(self.gates)],
                "holding": self.holding, "stewards": self.stewards, "held": self.held,
                "arrived": self.arrived, "entered": self.entered, "on_plaza": int(len(self.pos)),
                "arrivals_per_min": round(arrival_rate(self.t))}
