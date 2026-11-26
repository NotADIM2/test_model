"""The local dashboard: a small web server that only this laptop can reach, with the analysis running behind it.

It listens on 127.0.0.1, so no other device can connect. It also refuses requests whose Host header is not
its own address, and only accepts JSON for commands: together these stop a web page open in another tab from
quietly driving it (starting the webcam, say) through the browser.
"""
import json
import threading
import time
import traceback
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

from .alerts import AlertLog
from .engine import Analyser, VideoFile, Webcam

ROOT = Path(__file__).resolve().parent.parent
WEB = Path(__file__).resolve().parent / "web"
VIDEO_DIRS = (ROOT / "data" / "videos", ROOT / "data" / "samples")
ZONE_DIR = ROOT / "data" / "zones"
VIDEO_TYPES = {".mp4", ".webm", ".avi", ".mov", ".mkv"}
STATIC = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8")}
FRAME_WIDTH = 1280                             # annotated frames are sent at most this wide


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
    items = [{"id": "webcam", "label": "Laptop webcam (live)", "zones": None}]
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
        self.history: deque[dict] = deque(maxlen=900)
        self.alerts: deque[dict] = deque(maxlen=100)
        self.alert_id = 0
        self.seconds_per_frame = None

    def start(self, source: str, zones: str | None, every: float) -> None:
        if source != "webcam" and source not in videos():
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
        self.thread = threading.Thread(target=self._run, args=(source, zones, every), daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stopping.set()
        if self.thread is not None:
            self.thread.join(timeout=15)
        with self.lock:
            self.running = False

    def _run(self, source: str, zones: str | None, every: float) -> None:
        try:
            feed = Webcam() if source == "webcam" else VideoFile(videos()[source], every)
            analyser = Analyser(feed.size, zone_files()[zones] if zones else None)
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
                if image.shape[1] > FRAME_WIDTH:
                    image = cv2.resize(image, (FRAME_WIDTH, round(image.shape[0] * FRAME_WIDTH / image.shape[1])),
                                       interpolation=cv2.INTER_AREA)
                jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()
                new_alerts = log.update(step[0], statuses)
                with self.lock:
                    self.frame, self.frame_id, self.record = jpeg, self.frame_id + 1, record
                    self.history.append(record)
                    for alert in new_alerts:
                        self.alert_id += 1
                        self.alerts.append({"id": self.alert_id, **alert.to_dict()})
                    self.seconds_per_frame = round(time.monotonic() - started, 2)
        except Exception as exc:                             # shown on the dashboard instead of dying silently
            traceback.print_exc()
            with self.lock:
                self.error = str(exc)
        finally:
            with self.lock:
                self.running = False

    def state(self, since_alert: int = 0) -> dict:
        with self.lock:
            return {"running": self.running, "finished": self.finished, "error": self.error,
                    "source": self.source, "zones": self.zones, "frame_id": self.frame_id,
                    "record": self.record, "seconds_per_frame": self.seconds_per_frame,
                    "history": [{"t": r["t"], "people": r["people"],
                                 "zones": {z["name"]: z["density"] for z in r["zones"]}} for r in self.history],
                    "alerts": [a for a in self.alerts if a["id"] > since_alert]}

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
                    monitor.start(str(body["source"]), body.get("zones") or None, float(body.get("every", 1.0)))
                    self.send_json({"ok": True})
                elif self.path == "/api/stop":
                    monitor.stop()
                    self.send_json({"ok": True})
                else:
                    self.send_json({"error": "Not found"}, 404)
            except (KeyError, ValueError, TypeError) as exc:
                self.send_json({"error": str(exc)}, 400)

    return Handler


def serve(port: int) -> ThreadingHTTPServer:
    monitor = Monitor()
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(monitor, port))
    server.monitor = monitor
    return server
