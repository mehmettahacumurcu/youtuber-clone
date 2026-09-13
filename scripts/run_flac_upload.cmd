@echo off
REM Detached: FLAC-encode the not-yet-uploaded WAVs and upload them to Drive (~2x faster,
REM lossless). Survives the Claude Code session closing. Resumable — just re-run.
cd /d E:\youtuber-clone
"E:\youtuber-clone\.venv\Scripts\python.exe" scripts\flac_upload_remaining.py > logs\flac_upload.log 2>&1
