"""Make a demo video of CrowdPulse, entirely on this laptop: out/crowdpulse-demo.mp4 (about 2 minutes).

    python scripts/make_demo.py

Every picture in it comes from CrowdPulse itself: the station footage is analysed frame by frame, and the
stadium scenario is the dashboard's simulation, played twice from the moment of the red alert (once with
nobody acting, once with the control room acting). Captions use the numbers of this run, not fixed text.
Takes a few minutes. The incident report scene needs Microsoft Edge, and is skipped without it.
"""
import copy
import functools
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from crowdpulse import overlay, report                                       # noqa: E402
from crowdpulse.alerts import AlertLog                                       # noqa: E402
from crowdpulse.density import STRIDE                                        # noqa: E402
from crowdpulse.engine import Analyser, VideoFile, zone_records              # noqa: E402
from crowdpulse.flow import Flow                                             # noqa: E402
from crowdpulse.forecast import forecast, zone_ahead                         # noqa: E402
from crowdpulse.risk import ZoneTrend                                        # noqa: E402
from crowdpulse.sim import PX, Simulation                                    # noqa: E402

OUT = ROOT / "out" / "crowdpulse-demo.mp4"
STATION = ROOT / "data" / "samples" / "saint-lazare.webm"
STATION_ZONES = ROOT / "data" / "zones" / "saint-lazare.json"
W, H, FPS = 1920, 1080, 24
DRAW_FPS = 12                                  # simulation frames drawn per second of video; each is shown twice
ACT_AT = 320.0                                 # simulated seconds: the control room acts at 5:20, after the red alert
SIM_CELL = 32

FONTS = Path("C:/Windows/Fonts")
INK, INK_2, GROUND, PANEL, BORDER = (17, 24, 39), (91, 100, 117), (244, 245, 247), (255, 255, 255), (226, 229, 234)
LEVEL = {"green": ((22, 131, 63), (230, 244, 234)), "amber": ((180, 83, 9), (254, 243, 199)),
         "red": ((198, 40, 40), (253, 232, 232))}
ACCENT = (29, 78, 216)


@functools.lru_cache(maxsize=None)
def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    name = {"regular": "segoeui.ttf", "semibold": "seguisb.ttf", "bold": "segoeuib.ttf"}[weight]
    try:
        return ImageFont.truetype(str(FONTS / name), size)
    except OSError:
        return ImageFont.load_default(size)


def clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"


def wrap(draw: ImageDraw.ImageDraw, text: str, face, width: int) -> list[str]:
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            trial = f"{line} {word}".strip()
            if draw.textlength(trial, font=face) <= width or not line:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


class Writer:
    """H.264 through the ffmpeg that comes with imageio-ffmpeg: small, and plays everywhere. Without it,
    OpenCV's MPEG-4 (bigger, and some browsers will not play it)."""

    def __init__(self, path: Path):
        path.parent.mkdir(exist_ok=True)
        self.pipe = self.video = None
        try:
            import imageio_ffmpeg
            self.pipe = subprocess.Popen(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                 "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "22",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
        except ImportError:
            self.video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        self.frames = 0

    def add(self, frame_bgr: np.ndarray, seconds: float | None = None) -> None:
        data = frame_bgr.tobytes()
        for _ in range(1 if seconds is None else max(1, round(seconds * FPS))):
            if self.pipe:
                self.pipe.stdin.write(data)
            else:
                self.video.write(frame_bgr)
            self.frames += 1

    def close(self) -> None:
        if self.pipe:
            self.pipe.stdin.close()
            self.pipe.wait()
        else:
            self.video.release()


# ---- drawing a frame: header, the view, the zone panel and a caption

def compose(view_bgr: np.ndarray | None, caption: str, scene: str, status: str = "", level: str | None = None,
            zones: list[dict] | None = None, alerts: list | None = None) -> np.ndarray:
    page = Image.new("RGB", (W, H), GROUND)
    draw = ImageDraw.Draw(page)
    draw.rectangle((0, 0, W, 84), fill=PANEL)
    draw.line((0, 84, W, 84), fill=BORDER, width=2)
    draw.ellipse((32, 22, 72, 62), fill=LEVEL["red"][1])
    draw.ellipse((44, 34, 60, 50), fill=LEVEL["red"][0])
    draw.text((88, 14), "CrowdPulse", font=font(30, "semibold"), fill=INK)
    draw.text((90, 52), "Crowd-crush early warning · running on this laptop only", font=font(17), fill=INK_2)
    face = font(20, "semibold")
    draw.text((W - 32 - draw.textlength(scene, font=face), 30), scene, font=face, fill=INK_2)

    top = 100
    if status:
        x = 32
        if level:
            pill = level.upper()
            pw = draw.textlength(pill, font=font(16, "bold")) + 24
            draw.rounded_rectangle((x, top, x + pw, top + 30), 15, fill=LEVEL[level][1])
            draw.text((x + 12, top + 4), pill, font=font(16, "bold"), fill=LEVEL[level][0])
            x += pw + 14
        draw.text((x, top + 2), status, font=font(19), fill=INK_2)
        top += 44

    caption_top = H - 150
    side = zones is not None
    area = (32, top, (W - 32 - 560 - 24) if side else W - 32, caption_top - 16)
    if view_bgr is not None:
        scale = min((area[2] - area[0]) / view_bgr.shape[1], (area[3] - area[1]) / view_bgr.shape[0])
        size = (round(view_bgr.shape[1] * scale), round(view_bgr.shape[0] * scale))
        view = Image.fromarray(cv2.cvtColor(cv2.resize(view_bgr, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
        x = area[0] + ((area[2] - area[0]) - view.width) // 2
        page.paste(view, (x, area[1]))
    if side:
        panel(draw, W - 32 - 560, top, 560, caption_top - 16 - top, zones, alerts or [])

    draw.rectangle((0, caption_top, W, H), fill=(17, 24, 39))
    face = font(30, "semibold")
    lines = wrap(draw, caption, face, W - 120)[:3]
    y = caption_top + (150 - 40 * len(lines)) // 2
    for line in lines:
        draw.text((60, y), line, font=face, fill=(255, 255, 255))
        y += 40
    return cv2.cvtColor(np.asarray(page), cv2.COLOR_RGB2BGR)


def panel(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int, zones: list[dict], alerts: list) -> None:
    draw.rounded_rectangle((x, y, x + w, y + h), 12, fill=PANEL, outline=BORDER, width=2)
    draw.text((x + 20, y + 14), "Zones", font=font(22, "semibold"), fill=INK)
    cy = y + 54
    for z in zones:
        ink, bg = LEVEL[z["level"]]
        card_bg = PANEL if z["level"] == "green" else bg
        draw.rounded_rectangle((x + 16, cy, x + w - 16, cy + 104), 10, fill=card_bg, outline=BORDER, width=2)
        draw.text((x + 32, cy + 8), z["name"], font=font(19, "semibold"), fill=INK)
        pill = z["level"].upper()
        pw = draw.textlength(pill, font=font(14, "bold")) + 20
        draw.rounded_rectangle((x + w - 32 - pw, cy + 10, x + w - 32, cy + 34), 12, fill=bg)
        draw.text((x + w - 32 - pw + 10, cy + 13), pill, font=font(14, "bold"), fill=ink)
        big = f"{z['density']:.1f}"
        draw.text((x + 32, cy + 30), big, font=font(34, "bold"), fill=INK)
        draw.text((x + 40 + draw.textlength(big, font=font(34, "bold")), cy + 46), "people/m²", font=font(16), fill=INK_2)
        bits = [f"{z['people']:.0f} people"]
        if z["inflow_per_min"] >= 1:
            bits.append(f"+{z['inflow_per_min']:.0f}/min walking in")
        if z.get("in_30s") is not None:
            bits.append(f"in 30 s: {z['in_30s']:.1f}/m²")
        draw.text((x + 32, cy + 78), " · ".join(bits), font=font(16), fill=INK_2)
        m = z["minutes_to_critical"]
        if m is not None:
            words = "crush-risk now" if m == 0 else "critical < 1 min" if m < 1 else f"critical in ~{m:.0f} min"
            draw.text((x + w - 32 - draw.textlength(words, font=font(17, "bold")), cy + 46), words, font=font(17, "bold"), fill=ink)
        cy += 116
    if not alerts:
        return
    draw.text((x + 20, cy + 6), "Latest alerts", font=font(22, "semibold"), fill=INK)
    cy += 44
    for alert in reversed(alerts[-3:]):
        ink, bg = LEVEL[alert.level]
        lines = wrap(draw, alert.message, font(15), w - 64)[:2] + wrap(draw, alert.action, font(15, "semibold"), w - 64)[:2]
        box = 30 + 20 * len(lines)
        if cy + box > y + h - 10:
            break
        draw.rounded_rectangle((x + 16, cy, x + w - 16, cy + box), 8, fill=bg)
        draw.text((x + 30, cy + 6), f"{clock(alert.t)} · {alert.zone} · {alert.level.upper()}", font=font(14, "semibold"), fill=ink)
        for i, line in enumerate(lines):
            draw.text((x + 30, cy + 26 + 20 * i), line, font=font(15, "semibold" if i >= 2 else "regular"), fill=INK)
        cy += box + 10


def card(title: str, lines: list[str], scene: str) -> np.ndarray:
    page = Image.new("RGB", (W, H), (17, 24, 39))
    draw = ImageDraw.Draw(page)
    draw.ellipse((W // 2 - 44, 230, W // 2 + 44, 318), fill=LEVEL["red"][1])
    draw.ellipse((W // 2 - 16, 258, W // 2 + 16, 290), fill=LEVEL["red"][0])
    face = font(68, "bold")
    draw.text(((W - draw.textlength(title, font=face)) // 2, 350), title, font=face, fill=(255, 255, 255))
    y = 470
    for line in lines:
        face = font(32)
        for part in wrap(draw, line, face, 1500):
            draw.text(((W - draw.textlength(part, font=face)) // 2, y), part, font=face, fill=(200, 205, 215))
            y += 48
        y += 18
    face = font(20)
    draw.text(((W - draw.textlength(scene, font=face)) // 2, H - 80), scene, font=face, fill=(140, 146, 160))
    return cv2.cvtColor(np.asarray(page), cv2.COLOR_RGB2BGR)


# ---- the stadium scenario

class Run:
    """The simulation plus everything CrowdPulse keeps about it, so it can be copied at a branch point."""

    def __init__(self):
        self.sim = Simulation()
        self.zones = self.sim.pixel_zones()
        self.trends = {z.name: ZoneTrend(z.name, z.area_m2) for z in self.sim.zones}
        self.log = AlertLog()
        self.alerts, self.history, self.actions = [], [], []
        self.worst = (-1.0, None, None)

    def step(self, seconds: float) -> tuple[np.ndarray, dict, list]:
        self.sim.advance(seconds)
        statuses = {n: self.trends[n].update(self.sim.t, r) for n, r in self.sim.readings().items()}
        people, velocity = self.sim.crowd_grid(SIM_CELL)
        ahead = forecast(Flow(velocity, SIM_CELL), people)
        in_30s = zone_ahead(self.zones, people, ahead, {n: s.density for n, s in statuses.items()})
        info = self.sim.info()
        image = overlay.forecast(self.sim.render(f"{clock(self.sim.t)} simulated   {info['on_plaza']} people on the plaza"),
                                 ahead, 3.0 / PX ** 2)
        image = overlay.zones(image, self.zones, statuses)
        record = {"t": round(self.sim.t, 1), "people": float(info["on_plaza"]), "zones": zone_records(statuses, in_30s)}
        new = self.log.update(self.sim.t, statuses)
        self.alerts += new
        self.history.append(record)
        top = max(z["density"] for z in record["zones"])
        if top > self.worst[0]:
            self.worst = (top, cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes(), record)
        return image, record, new

    def act(self, action: str, gate: int | None, what: str) -> None:
        self.sim.act(action, gate)
        self.actions.append({"t": round(self.sim.t, 1), "what": what})


def worst_level(record: dict) -> str:
    levels = [z["level"] for z in record["zones"]]
    return "red" if "red" in levels else "amber" if "amber" in levels else "green"


def play(writer: Writer, run: Run, until: float, step: float, scene: str, caption_for, hold_on_alert: float = 3.5) -> None:
    """Play `run` until simulated time `until`, `step` simulated seconds per video frame."""
    while run.sim.t < until - 1e-6:
        image, record, new = run.step(step)
        caption = caption_for(run, record, new)
        status = f"{clock(run.sim.t)} simulated · {record['people']:.0f} people on the plaza · shown at {step * DRAW_FPS:.0f}× speed"
        frame = compose(image, caption, scene, status, worst_level(record), record["zones"], run.alerts)
        writer.add(frame, hold_on_alert if any(a.level != "green" for a in new) else 1 / DRAW_FPS)


def first(run: Run, zone: str, level: str):
    return next((a for a in run.alerts if a.zone == zone and a.level == level), None)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)                               # progress shows up straight away
    writer = Writer(OUT)
    print("Title")
    writer.add(card("CrowdPulse", [
        "On 4 June 2025, 11 people died in a crowd crush outside Chinnaswamy Stadium in Bengaluru.",
        "CrowdPulse watches ordinary CCTV and warns minutes before a crowd becomes dangerous: "
        "where people are packing in, how fast, and what to do about it.",
        "Everything runs on one laptop. No video leaves it."], "Demo · made on this laptop from CrowdPulse itself"), 7)

    # 1. real footage
    if STATION.exists() and STATION_ZONES.exists():
        print("Station footage (a couple of minutes: every frame is analysed)")
        video = VideoFile(STATION, every=0.25, start=48, end=62)
        analyser = Analyser(video.size, STATION_ZONES)
        caption = ("Real footage, Saint-Lazare station, Paris. A crowd-counting AI turns each frame into people per "
                   "square metre (the heatmap); arrows show which way people move; each zone gets a warning level. "
                   "Dashed outlines: where the crowd will be in 30 seconds.")
        first_frame = True
        while (step := video.next()) is not None:
            if first_frame:
                analyser.set_colour_scale(step[1])
                first_frame = False
            image, record, _ = analyser.annotate(*step)
            status = f"Recorded video · {record['t']:.1f} s · about {record['people']:.0f} people in view"
            writer.add(compose(image, caption, "1 · Real CCTV footage", status, worst_level(record) if record["zones"] else None,
                               record["zones"]), 0.25)
    else:
        print("Station footage missing (see data/samples/CREDITS.md); skipping that scene")

    # 2. the scenario, up to the moment the control room acts
    print("Simulation: the build-up")
    scene = "2 · Stadium entry surge (simulated)"
    writer.add(card("What if a gate jams?", [
        "None of the sample videos ever gets dangerous, so here is a simulated plaza in front of a stadium with four gates.",
        "Only Gate 3 is open. After an announcement, 250 people a minute arrive: far more than one gate lets through.",
        "Illustrative numbers, not a reconstruction of any real event."], scene), 6)
    base = Run()

    def build_up(run, record, new):
        amber, red = first(run, "Gate 3", "amber"), first(run, "Gate 3", "red")
        if red:
            return (f"{clock(red.t)} RED. At this rate Gate 3 reaches crush density within minutes. The alert says "
                    "what to do: stop people entering, open every exit, send stewards.")
        if amber:
            return (f"{clock(amber.t)} AMBER, while Gate 3 holds only {density_at(run, 'Gate 3', amber.t):.1f} "
                    "people per m². CrowdPulse sees people walking in faster than the gate lets them through, and "
                    "counts down to crush density.")
        if run.sim.t < 120:
            return "Quiet at first: 30 people a minute, one open gate copes."
        return "The surge starts. Watch the dashed outline: where the crowd will be in 30 seconds."
    play(writer, base, 200, 0.8, scene, build_up)
    play(writer, base, ACT_AT, 0.4, scene, build_up)

    # 3a. nobody acts
    print("Simulation: if nobody acts")
    idle = copy.deepcopy(base)
    scene_a = "3 · If nobody acts"

    def no_action(run, record, new):
        gate3 = next(z for z in record["zones"] if z["name"] == "Gate 3")
        return (f"If nobody acts: Gate 3 keeps packing. Now {gate3['density']:.1f} people per m². Above 5, people can "
                "no longer move on their own; this is how crushes happen.")
    play(writer, idle, 600, 2.0, scene_a, no_action, hold_on_alert=0)
    peak_idle = max(z["density"] for r in idle.history for z in r["zones"] if z["name"] == "Gate 3")
    writer.add(compose(**_last_frame(idle), scene=scene_a,
                       caption=f"Without action, Gate 3 peaks at {peak_idle:.1f} people per m²: crush conditions."), 4)

    # 3b. the control room acts on the warning
    print("Simulation: the control room acts")
    scene_b = "4 · Acting on the warning"
    acted = base
    for gate in (1, 2, 4):
        acted.act("open_gate", gate, f"Opened Gate {gate}")
    acted.act("stewards", None, "Sent stewards to point people to the gate with the shortest wait")

    def recovery(run, record, new):
        back = first(run, "Gate 3", "green")
        if back:
            peak = max(z["density"] for r in run.history for z in r["zones"] if z["name"] == "Gate 3")
            return (f"{clock(back.t)} Gate 3 is back to normal. With a few minutes' warning, its peak stayed at "
                    f"{peak:.1f} people per m² instead of {peak_idle:.1f}.")
        return (f"{clock(ACT_AT)} The control room acts: Gates 1, 2 and 4 open, and stewards send people to the "
                "shortest queue. The crowd spreads out across all four gates.")
    play(writer, acted, 480, 0.4, scene_b, recovery)
    writer.add(compose(**_last_frame(acted), scene=scene_b, caption=recovery(acted, acted.history[-1], [])), 3)

    # 4. thermal
    print("Thermal view")
    scene_t = "5 · Thermal camera (prototype)"
    sim = Simulation()
    first_view = sim.render_thermal()
    thermal = Analyser((first_view.shape[1], first_view.shape[0]), thermal=True, zones=sim.pixel_zones())
    thermal.vmax = 6.0 * (STRIDE / PX) ** 2
    log, alerts = AlertLog(), []
    sim.advance(200)
    while sim.t < 350:
        sim.advance(1.8)
        frame = sim.render_thermal()
        sim.step()
        image, record, statuses = thermal.annotate(sim.t, frame, sim.render_thermal(), 0.2,
                                                   caption=f"{clock(sim.t)} thermal view")
        alerts += log.update(sim.t, statuses)
        red = next((a for a in alerts if a.zone == "Gate 3" and a.level == "red"), None)
        caption = (f"The same scene through a thermal camera: heat, not faces, and it works in the dark. Counted from "
                   f"body heat: {record['people']:.0f} people (really {len(sim.pos)})."
                   + (f" Gate 3 still turns red at {clock(red.t)}." if red else ""))
        if sim.t >= 240:
            writer.add(compose(image, caption, scene_t, f"{clock(sim.t)} simulated · thermal view",
                               worst_level(record), record["zones"], alerts), 0.125)

    # 5. the incident report
    page = report.build("Simulation: stadium entry surge", acted.history, [a.to_dict() for a in acted.alerts],
                        acted.actions, acted.worst[1], acted.worst[2])
    shot = _screenshot(page)
    if shot is not None:
        print("Incident report")
        scene_r = "6 · Incident report"
        view_h = H - 150 - 16 - 100
        width = W - 64
        scaled = cv2.resize(shot, (width, round(shot.shape[0] * width / shot.shape[1])), interpolation=cv2.INTER_AREA)
        caption = ("Save report: one HTML file with the timeline, every alert and what was done about it. "
                   "Works offline, prints cleanly, for organisers and police.")
        travel = max(scaled.shape[0] - view_h, 0)
        frames = 9 * FPS
        for i in range(frames):
            y = round(travel * min(1.0, max(0.0, (i / frames - 0.15) / 0.7)))
            writer.add(compose(scaled[y:y + view_h], caption, scene_r))
    else:
        print("Microsoft Edge not found; skipping the report scene")

    writer.add(card("Run it", [
        "Double-click CrowdPulse.bat. It sets itself up the first time and opens the dashboard.",
        "Choose a camera, a video or the simulation, and press Start. Zones are drawn by clicking on a frame.",
        "It runs only on this laptop: the dashboard answers to this computer alone, and no video is uploaded."],
        "CrowdPulse"), 7)
    writer.close()
    print(f"Saved {OUT} ({writer.frames / FPS:.0f} s)")


def _last_frame(run: Run) -> dict:
    """compose() arguments for the latest view of `run` (drawn from a copy, so `run` itself does not move on)."""
    image, record, _ = copy.deepcopy(run).step(0.2)
    return {"view_bgr": image, "status": f"{clock(run.sim.t)} simulated · {record['people']:.0f} people on the plaza",
            "level": worst_level(record), "zones": record["zones"], "alerts": run.alerts}


def density_at(run: Run, zone: str, t: float) -> float:
    """`zone`'s density in the record at simulated time `t`."""
    record = min(run.history, key=lambda r: abs(r["t"] - t))
    return next(z["density"] for z in record["zones"] if z["name"] == zone)


def _screenshot(page: str) -> np.ndarray | None:
    """The report rendered by Microsoft Edge, as a tall image; None without Edge."""
    edges = [Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
             Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe")]
    edge = next((e for e in edges if e.exists()), None)
    if edge is None:
        return None
    work = ROOT / "out" / "demo-report"
    work.mkdir(parents=True, exist_ok=True)
    html, png = work / "report.html", work / "report.png"
    html.write_text(page, encoding="utf-8")
    subprocess.run([str(edge), "--headless=new", "--disable-gpu", f"--user-data-dir={work / 'edge'}",
                    "--window-size=1100,2700", "--hide-scrollbars", f"--screenshot={png}", html.as_uri()],
                   capture_output=True, timeout=120)
    return cv2.imread(str(png)) if png.exists() else None


if __name__ == "__main__":
    main()
