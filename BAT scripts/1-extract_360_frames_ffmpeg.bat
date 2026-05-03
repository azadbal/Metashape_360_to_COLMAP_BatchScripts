@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM User-configurable frames per second.
set "FPS=2"
REM Set to jpg or png.
set "IMAGE_FORMAT=jpg"
REM Lower is higher quality for JPG output. Ignored for PNG.
set "JPG_QUALITY=3"

set "ROOT=%~dp0"
set "OUT=%ROOT%images"

if not exist "%OUT%\" mkdir "%OUT%"

for /r "%ROOT%" %%F in (*.mp4 *.mov *.avi *.mkv *.wmv *.webm *.m4v) do (
  set "FILE=%%~fF"
  set "DIR=%%~dpF"
  set "BASE=%%~nF"
  set "REL=!DIR:%ROOT%=!"
  if "!REL!"=="" set "REL=."
  set "OUTDIR=%OUT%\!REL!\!BASE!"
  if not exist "!OUTDIR!\" mkdir "!OUTDIR!"
  echo Extracting "%%~nxF" to "!OUTDIR!"
  if /I "%IMAGE_FORMAT%"=="jpg" (
    ffmpeg -hide_banner -loglevel error -i "%%~fF" -vf fps=%FPS% -q:v %JPG_QUALITY% "!OUTDIR!\frame_%%06d.%IMAGE_FORMAT%"
  ) else (
    ffmpeg -hide_banner -loglevel error -i "%%~fF" -vf fps=%FPS% "!OUTDIR!\frame_%%06d.%IMAGE_FORMAT%"
  )
  if errorlevel 1 (
    echo ERROR: ffmpeg failed on "%%~fF"
    exit /b 1
  )
)

echo Done.
exit /b 0
