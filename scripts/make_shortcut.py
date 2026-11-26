"""Put a CrowdPulse shortcut on the Windows desktop: double-click it to start the dashboard.

    python scripts/make_shortcut.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def quoted(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"              # a PowerShell string literal


def main() -> None:
    if sys.platform != "win32":
        raise SystemExit("Desktop shortcuts are for Windows; elsewhere run: python scripts/dashboard.py --open")
    script = "; ".join([
        "$shell = New-Object -ComObject WScript.Shell",
        "$link = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Desktop')) 'CrowdPulse.lnk'))",
        f"$link.TargetPath = {quoted(ROOT / 'CrowdPulse.bat')}",
        f"$link.WorkingDirectory = {quoted(ROOT)}",
        f"$link.IconLocation = {quoted(ROOT / 'crowdpulse' / 'web' / 'favicon.ico')}",
        "$link.Description = 'CrowdPulse crowd-crush early warning, running on this laptop only'",
        "$link.Save()",
        "$link.FullName",
    ])
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"Could not make the shortcut: {result.stderr.strip()}")
    print(f"Shortcut made: {result.stdout.strip()}")


if __name__ == "__main__":
    main()
