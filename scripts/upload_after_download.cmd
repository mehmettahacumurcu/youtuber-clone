@echo off
REM Detached watcher: when the local download is finished AND rclone is configured,
REM upload all audio to Google Drive. Launched via WMI so it survives Claude Code closing.
REM
REM It waits for two conditions, then uploads once:
REM   1) an rclone remote named "gdrive:" exists (you create it with `rclone config`)
REM   2) the downloader has printed its "DONE:" line into logs\download_pending.out.log
REM rclone copy is resumable/idempotent (re-running skips already-uploaded files), so an
REM interrupted upload just continues on the next run.
cd /d E:\youtuber-clone
set RCLONE="C:\Users\Developer\.local\bin\rclone.exe"
set LOG=logs\upload_after.log
echo === UPLOAD WATCHER START %date% %time% === >> %LOG%

:waitremote
%RCLONE% listremotes | findstr /B /C:"gdrive:" >nul 2>&1
if errorlevel 1 (
  echo waiting for rclone "gdrive:" remote ^(run: rclone config^)... %date% %time% >> %LOG%
  timeout /t 120 /nobreak >nul
  goto waitremote
)
echo gdrive: remote found. %date% %time% >> %LOG%

:waitdownload
findstr /C:"DONE:" logs\download_pending.out.log >nul 2>&1
if errorlevel 1 (
  timeout /t 60 /nobreak >nul
  goto waitdownload
)
echo download complete. starting upload %date% %time% >> %LOG%

%RCLONE% copy data\audio gdrive:youtuber_audio --transfers 4 --checkers 8 --drive-chunk-size 64M --fast-list --log-file logs\rclone_upload.log --log-level INFO >> %LOG% 2>&1
echo === UPLOAD FINISHED rc=%errorlevel% %date% %time% === >> %LOG%
