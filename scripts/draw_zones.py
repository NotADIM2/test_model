"""Draw zones on a frame of a video by clicking, and save them as a zones file.

    python scripts/draw_zones.py data/samples/canteen.mp4 --at 5

In the window:
    left click    add a corner
    right click   remove the last corner
    Enter         finish this zone; then type its name and floor area in the terminal
    z             delete the last finished zone
    s             save and quit
    Esc           quit without saving

The file goes to data/zones/<video>.json unless --out says otherwise. Measure floor areas on site if you can:
pace out the length and width and multiply. A rough area still works; the warnings scale with it.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WINDOW = "CrowdPulse zones - click corners, Enter to finish a zone, s to save"
MAX_WIDTH = 1400                               # the frame is shown smaller if it is wider than this


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--at", type=float, default=1.0, help="seconds into the video for the frame to draw on")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_MSEC, args.at * 1000)
    ok, frame = cap.read()
    if not ok:
        sys.exit(f"Could not read a frame of {args.video} at {args.at} s")
    h, w = frame.shape[:2]
    show = min(1.0, MAX_WIDTH / w)
    out_path = args.out or ROOT / "data" / "zones" / f"{args.video.stem}.json"

    zones, corners = [], []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            corners.append([round(x / show), round(y / show)])
        elif event == cv2.EVENT_RBUTTONDOWN and corners:
            corners.pop()

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        view = frame.copy()
        for zone in zones:
            points = np.array(zone["polygon"], np.int32)
            cv2.polylines(view, [points], True, (80, 175, 60), 3, cv2.LINE_AA)
            cv2.putText(view, f"{zone['name']} ({zone['area_m2']:g} m2)", tuple(points.min(axis=0) + [8, 30]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 175, 60), 2, cv2.LINE_AA)
        if corners:
            cv2.polylines(view, [np.array(corners, np.int32)], False, (0, 200, 255), 2, cv2.LINE_AA)
            for x, y in corners:
                cv2.circle(view, (x, y), 6, (0, 200, 255), -1)
        cv2.imshow(WINDOW, cv2.resize(view, None, fx=show, fy=show) if show < 1 else view)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10):                                  # Enter
            if len(corners) < 3:
                print("A zone needs at least 3 corners.")
                continue
            name = input("Zone name (for example 'Dosa counter'): ").strip() or f"Zone {len(zones) + 1}"
            while True:
                try:
                    area = float(input(f"Floor area of {name} in square metres: "))
                    if area > 0:
                        break
                except ValueError:
                    pass
                print("Type a number above 0, for example 25.")
            zones.append({"name": name, "area_m2": area, "polygon": list(corners)})
            corners.clear()
            print(f"Added {name}. Draw the next zone, or press s to save.")
        elif key == ord("z") and zones:
            print(f"Deleted {zones.pop()['name']}.")
        elif key == ord("s"):
            if not zones:
                print("No zones yet. Click corners and press Enter first.")
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"frame_size": [w, h], "zones": zones}, indent=2), encoding="utf-8")
            print(f"Saved {len(zones)} zones to {out_path}")
            break
        elif key == 27:                                      # Esc
            print("Quit without saving.")
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed without saving.")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
