@echo off
rem Double-click to open FaceGroup. The Python environment goes in .venv and data in data\, both in this folder.
setlocal
cd /d "%~dp0"
python -m uv --version >nul 2>&1 || python -m pip install --user uv
python -m uv run facegroup ui %*
if errorlevel 1 pause
