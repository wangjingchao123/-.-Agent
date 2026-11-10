@echo off
setlocal
set EDGE=C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe
set URL=http://localhost:8000/
if not "%~1"=="" set URL=%~1
set OUT=%TEMP%\sa-dump.html
echo [headless] opening %URL%
"%EDGE%" --headless=new --disable-gpu --no-first-run --no-default-browser-check --dump-dom --virtual-time-budget=8000 "%URL%" > "%OUT%" 2>&1
echo [headless] dump saved to %OUT%