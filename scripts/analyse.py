"""Draw crowd density and movement onto a video.

    python scripts/analyse.py data/samples/saint-lazare.webm --at 60
        one annotated frame, 60 seconds in, saved as out/<video>-60s.png
    python scripts/analyse.py data/samples/saint-lazare.webm --start 40 --end 70 --every 0.5
        an annotated clip, one analysed frame every 0.5 s of video, saved as out/<video>-40-70s.mp4
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crowdpulse.density import DensityModel
from crowdpulse.flow import crowd_flow
from crowdpulse.overlay import arrows, heatmap, label

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


def annotate(model: DensityModel, cap: cv2.VideoCapture, seconds: float, vmax: float | None) -> tuple[np.ndarray, float]:
    frame, later, gap = frame_pair(cap, seconds)
    density = model.predict(frame)
    image = arrows(heatmap(frame, density, vmax), density, crowd_flow(frame, later, gap))
    return label(image, f"{seconds:5.1f} s   about {density.count:.0f} people"), density.count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--at", type=float, help="annotate one frame at this many seconds")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--every", type=float, default=0.5, help="seconds of video between analysed frames")
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"Could not open {args.video}")
    OUT.mkdir(exist_ok=True)
    model = DensityModel()

    if args.at is not None:
        image, count = annotate(model, cap, args.at, None)
        path = OUT / f"{args.video.stem}-{args.at:g}s.png"
        cv2.imwrite(str(path), image)
        print(f"About {count:.0f} people. Saved {path}")
        return

    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    end = min(args.end or duration, duration - FLOW_GAP - 0.05)
    times = np.arange(args.start, end, args.every)
    vmax = None
    path = OUT / f"{args.video.stem}-{args.start:g}-{end:g}s.mp4"
    writer = None
    for i, t in enumerate(times):
        if vmax is None:                       # one colour scale for the whole clip, set from its first frame
            grid = cv2.GaussianBlur(model.predict(frame_pair(cap, t)[0]).grid, (0, 0), sigmaX=1.5)
            vmax = max(float(np.percentile(grid, 99.5)) * 1.3, 1e-3)
        image, count = annotate(model, cap, float(t), vmax)
        if writer is None:
            h, w = image.shape[:2]
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1 / args.every, (w, h))
        writer.write(image)
        print(f"\r{i + 1}/{len(times)}  {t:5.1f} s  about {count:.0f} people", end="", flush=True)
    writer.release()
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
