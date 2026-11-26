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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crowdpulse.engine import Analyser, VideoFile

OUT = Path(__file__).resolve().parent.parent / "out"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--zones", type=Path, help="zones JSON file (see crowdpulse/zones.py)")
    parser.add_argument("--at", type=float, help="annotate one frame at this many seconds")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float)
    parser.add_argument("--every", type=float, default=0.5, help="seconds of video between analysed frames")
    parser.add_argument("--thermal", action="store_true", help="the video is from a thermal camera (white-hot)")
    args = parser.parse_args()

    try:
        video = VideoFile(args.video, args.every, args.start, args.end)
    except OSError as exc:
        sys.exit(str(exc))
    OUT.mkdir(exist_ok=True)
    analyser = Analyser(video.size, args.zones, thermal=args.thermal)

    if args.at is not None:
        pair = video.pair_at(args.at)
        if pair is None:
            sys.exit(f"Could not read frames at {args.at:.1f} s. Is the video that long?")
        image, record, _ = analyser.annotate(args.at, *pair)
        path = OUT / f"{args.video.stem}-{args.at:g}s.png"
        cv2.imwrite(str(path), image)
        print(f"About {record['people']:.0f} people. Saved {path}")
        for z in record["zones"]:
            print(f"  {z['name']}: {z['density']:.1f} people/m2, {z['level']}")
        return

    first = video.pair_at(args.start)
    if first is None:
        sys.exit(f"Could not read frames at {args.start:.1f} s.")
    analyser.set_colour_scale(first[0])
    name = f"{args.video.stem}-{args.start:g}-{video.end:.0f}s"
    video_path, timeline_path = OUT / f"{name}.mp4", OUT / f"{name}.jsonl"
    total = int((video.end - args.start) / args.every + 0.999)
    writer = None
    with open(timeline_path, "w", encoding="utf-8") as timeline:
        while (step := video.next()) is not None:
            image, record, _ = analyser.annotate(*step)
            timeline.write(json.dumps(record) + "\n")
            if writer is None:
                h, w = image.shape[:2]
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 1 / args.every, (w, h))
            writer.write(image)
            levels = " ".join(f"{z['name']}={z['level']}" for z in record["zones"])
            done = round((step[0] - args.start) / args.every) + 1
            print(f"\r{done}/{total}  {step[0]:5.1f} s  about {record['people']:.0f} people  {levels}   ", end="", flush=True)
    if writer is not None:
        writer.release()
    print(f"\nSaved {video_path} and {timeline_path}")


if __name__ == "__main__":
    main()
