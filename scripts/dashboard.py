"""Open the CrowdPulse dashboard on this laptop. The easy way is to double-click CrowdPulse.bat.

    python scripts/dashboard.py            then open http://127.0.0.1:8765 in your browser
    python scripts/dashboard.py --open     opens the browser for you
    python scripts/dashboard.py --check    only checks the setup and says what to fix

Only this laptop can reach it. Put your own videos in data/videos/ and their zones in data/zones/ with the
same name (canteen.mp4 and canteen.json) and the dashboard pairs them up. Stop it with Ctrl+C.
"""
import argparse
import importlib.util
import sys
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODEL = ROOT / "models" / "dmcount" / "dmcount.onnx"
PACKAGES = {"cv2": "opencv-python", "numpy": "numpy", "onnxruntime": "onnxruntime"}


def preflight() -> tuple[list[str], list[str]]:
    """(problems that stop CrowdPulse starting, problems that only switch part of it off), each with its fix."""
    blocking, partial = [], []
    if sys.version_info < (3, 10):
        blocking.append(f"CrowdPulse needs Python 3.10 or newer; this is {sys.version.split()[0]}. "
                        "Install Python 3.12 from https://www.python.org/downloads/")
    missing = [pip for module, pip in PACKAGES.items() if importlib.util.find_spec(module) is None]
    if missing:
        blocking.append(f"Python packages missing: {', '.join(missing)}. Fix: double-click CrowdPulse.bat, or run "
                        r"'.venv\Scripts\python -m pip install -r requirements.txt'")
    if not MODEL.exists():
        partial.append("The crowd-counting model is not downloaded, so videos and the webcam will not work "
                       r"(the simulations will). Fix: '.venv\Scripts\python scripts\get_model.py' with the internet on.")
    return blocking, partial


def already_running(port: int) -> bool:
    """Is a CrowdPulse dashboard already answering on this port?"""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/sources", timeout=2) as response:
            return response.status == 200 and b"sources" in response.read(4096)
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the dashboard in your browser")
    parser.add_argument("--check", action="store_true", help="only check the setup")
    args = parser.parse_args()
    url = f"http://127.0.0.1:{args.port}"

    blocking, partial = preflight()
    for problem in blocking + partial:
        print(f"! {problem}\n")
    if blocking:
        return 1
    if args.check:
        print("Setup OK." if not partial else "CrowdPulse can start, with the limits above.")
        return 0

    if already_running(args.port):
        print(f"CrowdPulse is already running at {url}; opening it.")
        webbrowser.open(url)
        return 0
    from crowdpulse.server import serve                  # after the checks: it needs the packages above
    try:
        server = serve(args.port)
    except OSError:
        print(f"Port {args.port} is in use by another program. Start CrowdPulse on another port, for example:\n"
              rf"    .venv\Scripts\python scripts\dashboard.py --open --port {args.port + 1}")
        return 1
    print(f"CrowdPulse dashboard: {url}\nOnly this laptop can reach it. Keep this window open; close it or press Ctrl+C to stop.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.monitor.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
