@echo off
REM Quick pipeline status. Safe to run while pipeline is running.

cd /d "%~dp0"
uv run --no-sync python -m pipeline.status
pause
