@echo off
rem Double-click to start CrowdPulse. It runs only on this laptop.
rem The first time, it sets itself up: a few minutes, and it needs the internet once.
setlocal
cd /d "%~dp0"
title CrowdPulse

if exist ".venv\crowdpulse-ready" goto start

echo Setting up CrowdPulse for the first time. This takes a few minutes and needs the internet once.
echo.
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv 2>nul || py -3 -m venv .venv 2>nul || python -m venv .venv 2>nul
)
if not exist ".venv\Scripts\python.exe" (
    echo Could not find Python. Install Python 3.12 from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" in the installer, then double-click CrowdPulse again.
    pause
    exit /b 1
)
echo Installing the Python packages CrowdPulse needs...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 (
    echo.
    echo Installing the packages failed. Check the internet connection and double-click CrowdPulse again.
    pause
    exit /b 1
)
echo Downloading the crowd-counting model...
".venv\Scripts\python.exe" scripts\get_model.py
if errorlevel 1 (
    echo.
    echo The model did not download. The simulations still work; for videos and the webcam, double-click
    echo CrowdPulse again later with the internet on.
) else (
    echo ready> ".venv\crowdpulse-ready"
)
echo.

:start
".venv\Scripts\python.exe" scripts\dashboard.py --open %*
if errorlevel 1 pause
