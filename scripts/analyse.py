"""Draw crowd density, movement and zone warnings onto a video.

    python scripts/analyse.py data/samples/saint-lazare.webm --at 60
        one annotated frame, 60 seconds in, saved as out/<video>-60s.png
    python scripts/analyse.py data/samples/saint-lazare.webm --start 40 --end 70 --every 0.5
        an annotated clip, one analysed frame every 0.5 s of video, saved as out/<video>-40-70s.mp4

Add --zones data/zones/saint-lazare.json to outline each zone in its warning colour. With a clip, each
frame's zone readings are also written to out/<video>-40-70s.jsonl, one JSON line per analysed frame.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crowdpulse import overlay
from crowdpulse.density import DensityModel
from crowdpulse.flow import Steady, crowd_flow
from crowdpulse.risk import ZoneTrend
from crowdpulse.zones import load_zones, measure

FLOW_GAP = 0.25                                # seconds between the two frames compared for movement
OUT = Path(__file__).resolve().parent.parent / "out"


def frame_pair(cap: cv2.VideoCapture, seconds: float) -> tuple[np.ndarray, np.ndarray, float]:
    """The frame at `seconds`, the frame about FLOW_GAP later, and the exact time between them.

    Seeking in compressed video can land a frame or two off, so it seeks once and reads forward from there.
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
    ok, frame = cap.read()
    gap = max(1, round(FLOW_GAP * fps))
    for _ in range(gap):
        ok, later = cap.read() if ok else (False, None)
    if not ok:
        sys.exit(f"Could not read frames at {seconds:.1f} s. Is the video that long?")
    return frame, later, gap / fps


class Analyser:
    def __init__(self, cap: cv2.VideoCapture, zones_file: Path | None):
        self.cap, self.model = cap, DensityModel()
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.zones = load_zones(zones_file, size) if zones_file else []
        self.trends = {z.name: ZoneTrend(z.name, z.area_m2) for z in self.zones}
        self.steady = Steady()
        self.vmax = None

    def annotate(self, seconds: float) -> tuple[np.ndarray, dict]:
        """The annotated frame at `seconds`, and that frame's readings as a dict."""
        frame, later, gap = frame_pair(self.cap, seconds)
        density = self.model.predict(frame)
        flow = self.steady.update(seconds, crowd_flow(frame, later, gap))
        image = overlay.arrows(overlay.heatmap(frame, density, self.vmax), density, flow)
        record = {"t": round(seconds, 2), "people": round(density.count, 1), "zones": []}
        if self.zones:
            statuses = {name: self.trends[name].update(seconds, r) for name, r in measure(self.zones, density, flow).items()}
            image = overlay.zones(image, self.zones, statuses)
            record["zones"] = [{"name": s.name, "people": round(s.people, 1), "density": round(s.density, 2),
                                "inflow_per_min": round(s.inflow, 1), "level": s.level,
                                "minutes_to_critical": None if s.minutes_to_critical is None else round(s.minutes_to_critical, 1)}
                               for s in statuses.values()]
        return overlay.label(image, f"{seconds:5.1f} s   about {density.count:.0f} people"), record

    def set_colour_scale(self, seconds: float) -> None:
        """One colour scale for a whole clip, from its first frame, so colours compare across frames."""
        grid = cv2.GaussianBlur(self.model.predict(frame_pair(self.cap, seconds)[0]).grid, (0, 0), sigmaX=1.5)
        self.vmax = max(float(np.percentile(grid, 99.5)) * 1.3, 1e-3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--zones", type=Path, help="zones JSON file (see crowdpulse/zones.py)")
    parser.add_argument("--at", type=float, help="annotate one frame at this many seconds")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--every", type=float, default=0.5, help="seconds of video between analysed frames")
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"Could not open {args.video}")
    OUT.mkdir(exist_ok=True)
    analyser = Analyser(cap, args.zones)

    if args.at is not None:
        image, record = analyser.annotate(args.at)
        path = OUT / f"{args.video.stem}-{args.at:g}s.png"
        cv2.imwrite(str(path), image)
        print(f"About {record['people']:.0f} people. Saved {path}")
        for z in record["zones"]:
            print(f"  {z['name']}: {z['density']:.1f} people/m2, {z['level']}")
        return

    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    end = min(args.end or duration, duration - FLOW_GAP - 0.05)
    times = np.arange(args.start, end, args.every)
    analyser.set_colour_scale(args.start)
    name = f"{args.video.stem}-{args.start:g}-{end:.0f}s"
    video_path, timeline_path = OUT / f"{name}.mp4", OUT / f"{name}.jsonl"
    writer = None
    with open(timeline_path, "w", encoding="utf-8") as timeline:
        for i, t in enumerate(times):
            image, record = analyser.annotate(float(t))
            timeline.write(json.dumps(record) + "\n")
            if writer is None:
                h, w = image.shape[:2]
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 1 / args.every, (w, h))
            writer.write(image)
            levels = " ".join(f"{z['name']}={z['level']}" for z in record["zones"])
            print(f"\r{i + 1}/{len(times)}  {t:5.1f} s  about {record['people']:.0f} people  {levels}   ", end="", flush=True)
    writer.release()
    print(f"\nSaved {video_path} and {timeline_path}")


if __name__ == "__main__":
    main()
