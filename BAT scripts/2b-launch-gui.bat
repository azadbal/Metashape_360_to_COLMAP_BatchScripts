@echo off
setlocal EnableExtensions

if not defined PYTHON_EXE set "PYTHON_EXE=python"

set "BATCH_DIR=%~dp0"
for %%I in ("%BATCH_DIR%.") do set "REPO_ROOT=%%~fI"
set "SCRIPT_DIR=%REPO_ROOT%\Metashape_360_to_COLMAP_plane"
set "GUI_SCRIPT=%SCRIPT_DIR%\metashape_360_gui.py"

if not exist "%GUI_SCRIPT%" (
  echo GUI script not found:
  echo   "%GUI_SCRIPT%"
  exit /b 1
)

where "%PYTHON_EXE%" >nul 2>nul
if errorlevel 1 (
  where py >nul 2>nul
  if errorlevel 1 (
    echo Python launcher not found. Set PYTHON_EXE or install Python.
    exit /b 1
  )
  set "PYTHON_EXE=py"
)

if "%DRY_RUN%"=="1" (
  echo Would run:
  echo   "%PYTHON_EXE%" "%GUI_SCRIPT%"
  exit /b 0
)

pushd "%SCRIPT_DIR%"
start "Metashape 360 GUI" "%PYTHON_EXE%" "%GUI_SCRIPT%"
set "RUN_EXIT=%ERRORLEVEL%"
popd

exit /b %RUN_EXIT%
