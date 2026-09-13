@echo off
title YOUTUBER STUDIO
cd /d "%~dp0"
echo.
echo  YOUTUBER STUDIO baslatiliyor...
echo.

REM ---- python (TTS env): local .venv-tts, yoksa dev repo'nunki
set "PY=%~dp0.venv-tts\Scripts\python.exe"
if not exist "%PY%" set "PY=E:\youtuber-clone\.venv-tts\Scripts\python.exe"
if not exist "%PY%" (
  echo [HATA] Python ortami bulunamadi: ne yerel .venv-tts ne E:\youtuber-clone\.venv-tts var.
  pause
  exit /b 1
)

REM ---- RAG env python'u studio'ya bildir (retrieve server bununla acilir)
if not defined SPEAKER_RAG_PY (
  if exist "%~dp0.venv\Scripts\python.exe" (
    set "SPEAKER_RAG_PY=%~dp0.venv\Scripts\python.exe"
  ) else (
    set "SPEAKER_RAG_PY=E:\youtuber-clone\.venv\Scripts\python.exe"
  )
)

REM ---- Ollama ayakta mi? Degilse baslat
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:11434/api/version -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
  where ollama >nul 2>nul || (
    echo [HATA] Ollama bulunamadi. Once ollama.com uzerinden kurun ve tekrar deneyin.
    pause
    exit /b 1
  )
  echo  Ollama baslatiliyor...
  start "" /min ollama serve
  timeout /t 4 /nobreak >nul
)

REM ---- UI hazir olunca tarayiciyi ac (arka planda bekler)
start "" /min powershell -NoProfile -Command "for($i=0;$i -lt 90;$i++){ try { Invoke-WebRequest -UseBasicParsing http://127.0.0.1:7861 -TimeoutSec 2 | Out-Null; Start-Process 'http://127.0.0.1:7861'; break } catch { Start-Sleep -Seconds 2 } }"

echo  Studio aciliyor -- tarayici otomatik acilacak (http://127.0.0.1:7861).
echo  RAG READY yesil rozetini bekle (ilk aciliste ~30 sn). Kapatmak icin bu pencerede Ctrl+C.
echo.
"%PY%" "%~dp0ui\speaker_studio.py"
echo.
echo  Studio kapandi.
pause
