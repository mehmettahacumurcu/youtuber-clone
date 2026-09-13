@echo off
REM Detached overnight driver for the local home-IP audio download.
REM Launched via WMI Win32_Process.Create so it survives the Claude Code session closing.
REM Resumable: re-running only targets never-seen videos, so a restart continues cleanly.
cd /d E:\youtuber-clone
set UV_LINK_MODE=copy
"C:\Users\Developer\.local\bin\uv.exe" run python scripts\download_pending.py --config config.yaml > logs\download_pending.out.log 2>&1
