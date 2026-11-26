"""Download the DM-Count crowd model (MIT licence) and check it is exactly the published file.

    python scripts/get_model.py
"""
import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "mlboydaisuke/DM-Count-Crowd-OpenVINO"   # ONNX export of litert-community/DM-Count-Crowd-LiteRT
SHA256 = "3e5a6c65a35f68f3115b23483f07cd7f0c1e7ad3c87db39638f1184852d96197"
DEST = Path(__file__).resolve().parent.parent / "models" / "dmcount"


def main() -> None:
    path = Path(hf_hub_download(REPO, "dmcount.onnx", local_dir=DEST))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        path.unlink()
        raise SystemExit(f"Checksum mismatch ({digest}); the download was deleted. Try again.")
    print(f"Model ready: {path}")


if __name__ == "__main__":
    main()
