"""Open the CrowdPulse dashboard on this laptop.

    python scripts/dashboard.py            then open http://127.0.0.1:8765 in your browser
    python scripts/dashboard.py --open     opens the browser for you

Only this laptop can reach it. Put your own videos in data/videos/ and their zones in data/zones/ with the
same name (canteen.mp4 and canteen.json) and the dashboard pairs them up. Stop it with Ctrl+C.
"""
import argparse
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crowdpulse.server import serve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the dashboard in your browser")
    args = parser.parse_args()
    server = serve(args.port)
    url = f"http://127.0.0.1:{args.port}"
    print(f"CrowdPulse dashboard: {url}   (only this laptop can reach it; Ctrl+C to stop)")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.monitor.stop()
        server.server_close()


if __name__ == "__main__":
    main()
