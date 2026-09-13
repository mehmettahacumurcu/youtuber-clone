@echo off
title YOUTUBER STUDIO (PAYLASIMLI)
REM Gecici, sifreli bir *.gradio.live linki ile baslatir (uzaktan gosterim icin).
REM Normal yerel kullanim icin START_SPEAKER.bat kullan.

set /p SPEAKER_PASS=Link sifresi belirle (kullanici adi 'speaker' olacak):
if "%SPEAKER_PASS%"=="" (
  echo [HATA] Sifre bos olamaz. Sifresiz paylasma.
  pause
  exit /b 1
)
set "SPEAKER_SHARE=1"
set "SPEAKER_AUTH=speaker:%SPEAKER_PASS%"
echo.
echo  Paylasim linki asagida "Running on public URL: https://....gradio.live" satirinda
echo  gorunecek (~72 saat gecerli; bu pencereyi kapatinca link OLUR).
echo  Karsi tarafa link + kullanici adi 'speaker' + belirledigin sifreyi ver.
echo.
call "%~dp0START_SPEAKER.bat"
