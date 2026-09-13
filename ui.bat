@echo off
REM Launch the youtuber-clone Gradio review UI.
REM Double-click this file or run from any cwd.

cd /d "%~dp0"
echo Starting youtuber-clone review UI at http://127.0.0.1:7860 ...
echo (Ctrl+C in this window to stop.)
echo.
REM --no-sync: skip uv's env reconciliation. Required when another pipeline
REM process is running and holds locks on shared DLLs (e.g. tokenizers.pyd).
uv run --no-sync python -m ui.review_app
echo.
echo UI process exited.
pause
