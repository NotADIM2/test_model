"""The local dashboard: a small web server that only this laptop can reach, with the analysis running behind it.

It listens on 127.0.0.1, so no other device can connect. It also refuses requests whose Host header is not
its own address, and only accepts JSON for commands: together these stop a web page open in another tab from
quietly driving it (starting the webcam, say) through the browser.
"""
import json
import queue
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from . import overlay, report
from .alerts import AlertLog
from .engine import Analyser, VideoFile, Webcam, zone_records
from .risk import ZoneTrend
from .density import STRIDE
from .flow import Flow
from .forecast import forecast, zone_ahead
from .sim import PX, STEP, Simulation

ROOT = Path(__file__).resolve().parent.parent
WEB = Path(__file__).resolve().parent / "web"
VIDEO_DIRS = (ROOT / "data" / "videos", ROOT / "data" / "samples")
ZONE_DIR = ROOT / "data" / "zones"
VIDEO_TYPES = {".mp4", ".webm", ".avi", ".mov", ".mkv"}
STATIC = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8"), "/favicon.ico": ("favicon.ico", "image/x-icon")}
FRAME_WIDTH = 1280                             # annotated frames are sent at most this wide
SIM = "sim:stadium"
SIM_THERMAL = "sim:stadium-thermal"
SIMS = (SIM, SIM_THERMAL)
SIM_FRAME = 2.0                                # simulated seconds per dashboard frame
SIM_SPEED = 5.0                                # simulated seconds per real second
SIM_LENGTH = 20 * 60                           # the scenario runs for 20 simulated minutes
SIM_CELL = 32                                  # pixels per movement cell in the simulation's picture (2 m)
TIMELINE_MAX = 20_000                          # records kept for the report (about 5.5 hours at one a second)


def videos() -> dict[str, Path]:
    """Videos the dashboard may open, by file name: data/videos (yours) first, then data/samples."""
    found = {}
    for folder in VIDEO_DIRS:
        for path in sorted(folder.glob("*")) if folder.exists() else []:
            if path.suffix.lower() in VIDEO_TYPES and path.name not in found:
                found[path.name] = path
    return found


def zone_files() -> dict[str, Path]:
    return {p.name: p for p in sorted(ZONE_DIR.glob("*.json"))} if ZONE_DIR.exists() else {}


def sources() -> dict:
    zones = zone_files()
    items = [{"id": SIM, "label": "Simulation: stadium entry surge", "zones": None},
             {"id": SIM_THERMAL, "label": "Simulation: same, seen by a thermal camera", "zones": None},
             {"id": "webcam", "label": "Laptop webcam (live)", "zones": None}]
    for name, path in videos().items():
        items.append({"id": name, "label": name, "zones": f"{path.stem}.json" if f"{path.stem}.json" in zones else None})
    return {"sources": items, "zones": list(zones)}


class Monitor:
    """Runs the analysis on a background thread and keeps the latest results for the dashboard."""

    def __init__(self):
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.stopping = threading.Event()
        self.reset()

    def reset(self) -> None:
        self.running, self.finished, self.error = False, False, None
        self.source = self.zones = None
        self.frame, self.frame_id, self.record = None, 0, None
        self.history: deque[dict] = deque(maxlen=900)          # what the dashboard chart shows
        self.timeline: deque[dict] = deque(maxlen=TIMELINE_MAX)  # the whole run, for the report
        self.alerts: deque[dict] = deque(maxlen=100)
        self.all_alerts: list[dict] = []
        self.done: list[dict] = []                               # operator actions in the simulation
        self.worst = (-1.0, None, None)                          # (how bad, jpeg, record) of the worst frame so far
        self.alert_id = 0
        self.seconds_per_frame = None
        self.sim_info = None
        self.actions: queue.Queue = queue.Queue()

    def start(self, source: str, zones: str | None, every: float, thermal: bool = False) -> None:
        if source not in (*SIMS, "webcam") and source not in videos():
            raise ValueError("Unknown video")
        if zones is not None and zones not in zone_files():
            raise ValueError("Unknown zones file")
        if not 0.25 <= every <= 10:
            raise ValueError("Analyse between every 0.25 s and every 10 s")
        self.stop()
        with self.lock:
            self.reset()
            self.running, self.source, self.zones = True, source, zones
        self.stopping.clear()
        self.thread = threading.Thread(target=self._run, args=(source, zones, every, thermal), daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stopping.set()
        if self.thread is not None:
            self.thread.join(timeout=15)
        with self.lock:
            self.running = False

    def act(self, action: str, gate: int | None) -> None:
        """Queue an operator action for the running simulation; it applies before the next step."""
        if self.source not in SIMS or not self.running:
            raise ValueError("Actions work only while the simulation is running")
        if action not in ("open_gate", "close_gate", "hold_arrivals", "stewards"):
            raise ValueError("Unknown action")
        self.actions.put((action, gate))

    def publish(self, image, record: dict, alerts: list, started: float, sim_info: dict | None = None) -> None:
        if image.shape[1] > FRAME_WIDTH:
            image = cv2.resize(image, (FRAME_WIDTH, round(image.shape[0] * FRAME_WIDTH / image.shape[1])),
                               interpolation=cv2.INTER_AREA)
        jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()
        badness = max((z["density"] for z in record["zones"]), default=record["people"] / 1e6)
        with self.lock:
            self.frame, self.frame_id, self.record = jpeg, self.frame_id + 1, record
            self.history.append(record)
            self.timeline.append(record)
            if badness > self.worst[0]:
                self.worst = (badness, jpeg, record)
            for alert in alerts:
                self.alert_id += 1
                self.alerts.append({"id": self.alert_id, **alert.to_dict()})
                if len(self.all_alerts) < TIMELINE_MAX:
                    self.all_alerts.append(alert.to_dict())
            self.seconds_per_frame = round(time.monotonic() - started, 2)
            self.sim_info = sim_info

    def _run(self, source: str, zones: str | None, every: float, thermal: bool) -> None:
        try:
            if source in SIMS:
                self._run_simulation(thermal=source == SIM_THERMAL)
                return
            feed = Webcam() if source == "webcam" else VideoFile(videos()[source], every)
            analyser = Analyser(feed.size, zone_files()[zones] if zones else None, thermal=thermal)
            log = AlertLog()
            first = True
            while not self.stopping.is_set():
                started = time.monotonic()
                step = feed.next()
                if step is None:
                    with self.lock:
                        self.finished = True
                    break
                if first:
                    analyser.set_colour_scale(step[1])
                    first = False
                image, record, statuses = analyser.annotate(*step)
                self.publish(image, record, log.update(step[0], statuses), started)
        except Exception as exc:                             # shown on the dashboard instead of dying silently
            traceback.print_exc()
            with self.lock:
                self.error = str(exc)
        finally:
            with self.lock:
                self.running = False

    def _run_simulation(self, thermal: bool = False) -> None:
        """The stadium-entry scenario: a step of simulated time per frame, at SIM_SPEED times real time.

        Normally the zones' readings come straight from the simulated people. With `thermal`, the simulation is
        drawn as a thermal camera would see it, and CrowdPulse has to count the people from those pixels alone.
        """
        sim = Simulation()
        zones = sim.pixel_zones()
        trends = {z.name: ZoneTrend(z.name, z.area_m2) for z in sim.zones}
        log = AlertLog()
        analyser = None
        if thermal:
            frame = sim.render_thermal()
            analyser = Analyser((frame.shape[1], frame.shape[0]), thermal=True, zones=zones)
            analyser.vmax = 6.0 * (STRIDE / PX) ** 2              # full red at 6 people per m2
        while not self.stopping.is_set() and sim.t < SIM_LENGTH:
            started = time.monotonic()
            while not self.actions.empty():
                action, gate = self.actions.get()
                try:
                    sim.act(action, gate)
                except ValueError:
                    continue
                if action in ("open_gate", "close_gate"):
                    what = f"{'Opened' if action == 'open_gate' else 'Closed'} Gate {gate}"
                elif action == "hold_arrivals":
                    what = "Held new arrivals at the outer cordon" if sim.holding else "Released the people held at the cordon"
                else:
                    what = "Sent stewards to point people to the gate with the shortest wait" if sim.stewards else "Stood the stewards down"
                with self.lock:
                    self.done.append({"t": round(sim.t, 1), "what": what})
            if analyser:                                          # two thermal frames a step apart, for movement
                sim.advance(SIM_FRAME - STEP)
                frame = sim.render_thermal()
                sim.step()
                info = sim.info()
                minutes, seconds = divmod(int(sim.t), 60)
                image, record, statuses = analyser.annotate(sim.t, frame, sim.render_thermal(), STEP,
                                                            caption=f"{minutes:02d}:{seconds:02d} thermal view")
                record["t"] = round(sim.t, 1)
                cv2.putText(image, f"Counted from heat: {record['people']:.0f}   Really on the plaza: {info['on_plaza']}",
                            (12, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                sim.advance(SIM_FRAME)
                statuses = {name: trends[name].update(sim.t, r) for name, r in sim.readings().items()}
                info = sim.info()
                minutes, seconds = divmod(int(sim.t), 60)
                caption = f"{minutes:02d}:{seconds:02d} simulated   {info['on_plaza']} people on the plaza"
                people, velocity = sim.crowd_grid(SIM_CELL)
                ahead = forecast(Flow(velocity, SIM_CELL), people)
                in_30s = zone_ahead(zones, people, ahead, {n: s.density for n, s in statuses.items()})
                image = overlay.forecast(sim.render(caption), ahead, 3.0 / PX ** 2)    # outline at 3 people per m2
                image = overlay.zones(image, zones, statuses)
                record = {"t": round(sim.t, 1), "people": float(info["on_plaza"]), "zones": zone_records(statuses, in_30s)}
            self.publish(image, record, log.update(sim.t, statuses), started, info)
            self.stopping.wait(max(0.0, SIM_FRAME / SIM_SPEED - (time.monotonic() - started)))
        with self.lock:
            self.finished = sim.t >= SIM_LENGTH

    def state(self, since_alert: int = 0) -> dict:
        with self.lock:
            return {"running": self.running, "finished": self.finished, "error": self.error, "sim": self.sim_info,
                    "source": self.source, "zones": self.zones, "frame_id": self.frame_id,
                    "record": self.record, "seconds_per_frame": self.seconds_per_frame,
                    "history": [{"t": r["t"], "people": r["people"],
                                 "zones": {z["name"]: z["density"] for z in r["zones"]}} for r in self.history],
                    "alerts": [a for a in self.alerts if a["id"] > since_alert]}

    def report(self) -> str | None:
        with self.lock:
            if not self.timeline:
                return None
            if self.source == SIM:
                source = "Simulation: stadium entry surge"
            elif self.source == SIM_THERMAL:
                source = "Simulation: stadium entry surge, counted from a thermal camera view"
            elif self.source == "webcam":
                source = "Laptop webcam (live)"
            else:
                source = self.source + (f", zones {self.zones}" if self.zones else "")
            return report.build(source, list(self.timeline), list(self.all_alerts), list(self.done), self.worst[1], self.worst[2])

    def latest_frame(self) -> bytes | None:
        with self.lock:
            return self.frame


def make_handler(monitor: Monitor, port: int):
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):                         # keep the terminal quiet
            pass

        def send(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, data, status: int = 200) -> None:
            self.send(status, json.dumps(data).encode("utf-8"), "application/json")

        def host_ok(self) -> bool:
            if self.headers.get("Host") in allowed_hosts:
                return True
            self.send_json({"error": "This dashboard only answers to its own address"}, 403)
            return False

        def do_GET(self):
            if not self.host_ok():
                return
            path, _, query = self.path.partition("?")
            if path in STATIC:
                name, kind = STATIC[path]
                self.send(200, (WEB / name).read_bytes(), kind)
            elif path == "/api/sources":
                self.send_json(sources())
            elif path == "/api/state":
                since = dict(p.split("=", 1) for p in query.split("&") if "=" in p).get("since", "0")
                self.send_json(monitor.state(int(since) if since.isdigit() else 0))
            elif path == "/api/report":
                page = monitor.report()
                if page is None:
                    self.send_json({"error": "Nothing has been analysed yet"}, 404)
                    return
                body = page.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="crowdpulse-report-{datetime.now():%Y%m%d-%H%M}.html"')
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif path == "/frame.jpg":
                frame = monitor.latest_frame()
                if frame is None:
                    self.send_json({"error": "No frame yet"}, 404)
                else:
                    self.send(200, frame, "image/jpeg")
            else:
                self.send_json({"error": "Not found"}, 404)

        def do_POST(self):
            if not self.host_ok():
                return
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                self.send_json({"error": "Send JSON"}, 415)
                return
            try:
                length = min(int(self.headers.get("Content-Length", 0)), 10_000)
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/start":
                    monitor.start(str(body["source"]), body.get("zones") or None, float(body.get("every", 1.0)),
                                  bool(body.get("thermal", False)))
                    self.send_json({"ok": True})
                elif self.path == "/api/stop":
                    monitor.stop()
                    self.send_json({"ok": True})
                elif self.path == "/api/action":
                    gate = body.get("gate")
                    monitor.act(str(body["action"]), int(gate) if gate is not None else None)
                    self.send_json({"ok": True})
                else:
                    self.send_json({"error": "Not found"}, 404)
            except (KeyError, ValueError, TypeError) as exc:
                self.send_json({"error": str(exc)}, 400)

    return Handler


class LocalServer(ThreadingHTTPServer):
    # On Windows, address reuse lets a second program bind the same port and take some of the requests; refuse it
    allow_reuse_address = sys.platform != "win32"
    daemon_threads = True


def serve(port: int) -> ThreadingHTTPServer:
    monitor = Monitor()
    server = LocalServer(("127.0.0.1", port), make_handler(monitor, port))
    server.monitor = monitor
    return server
