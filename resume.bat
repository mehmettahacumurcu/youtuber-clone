@echo off
REM Resume the youtuber-clone pipeline (stages 1-4) after a shutdown / crash.
REM Every stage is idempotent — already-completed videos are skipped.
REM Safe to run repeatedly.

cd /d "%~dp0"
echo ================================================
echo youtuber-clone pipeline RESUME
echo ================================================
echo.

REM Clean up any zero-byte JSON files (partial writes from a killed process).
echo Cleaning up any partial output files...
uv run --no-sync python -c "from pathlib import Path; n = 0; [(_ for _ in [p.unlink()]).__next__() for d in ['data/vad','data/diarize','data/identify','data/transcribe'] for p in Path(d).glob('*.json') if p.exists() and p.stat().st_size == 0]; print('  (done)')"
echo.

echo === Stage 1: Download (up to 35 newest videos) ===
uv run --no-sync python -m pipeline.download --limit 35
if errorlevel 1 (
    echo Download stage failed. Check log above.
    pause
    exit /b 1
)
echo.

echo === Stage 2: VAD ===
uv run --no-sync python -m pipeline.vad --all
if errorlevel 1 (
    echo VAD stage failed. Check log above.
    pause
    exit /b 1
)
echo.

echo === Stage 3: Diarize ===
uv run --no-sync python -m pipeline.diarize --all
if errorlevel 1 (
    echo Diarize stage failed. Check log above.
    pause
    exit /b 1
)
echo.

echo === Stage 4: Identify ===
uv run --no-sync python -m pipeline.identify --all
if errorlevel 1 (
    echo Identify stage failed. Check log above.
    pause
    exit /b 1
)
echo.

echo ================================================
echo Stages 1-4 complete.
echo Next: open ui.bat to review re-mined ref candidates,
echo       OR ping Claude to proceed to Stage 5 (transcribe).
echo ================================================
pause
