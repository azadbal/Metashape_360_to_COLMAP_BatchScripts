@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Drag and drop a dataset folder or its "images" subfolder onto this batch.
REM Expected layout:
REM   <dataset-root>\images\...
REM   <dataset-root>\*.xml
REM   <dataset-root>\*.ply    (optional)
REM
REM Important:
REM   This script calls metashape_360_to_colmap.py, which requires Metashape
REM   camera export XML. XMP files are not accepted by that Python script.

REM User-tunable defaults. Existing environment variables win.
if not defined PYTHON_EXE set "PYTHON_EXE=python"
if not defined REPO_ROOT set "REPO_ROOT=C:\Dev\3DGS\MetaShape360-to-colmap-kotohibi"
if not defined CROP_SIZE set "CROP_SIZE=1920"
if not defined FOV_DEG set "FOV_DEG=90"
if not defined MAX_IMAGES set "MAX_IMAGES=20000"
if not defined NUM_WORKERS set "NUM_WORKERS=20"
if not defined YAW_OFFSET set "YAW_OFFSET=0"
if not defined OUTPUT_FORMAT set "OUTPUT_FORMAT=auto"
if not defined OUTPUT_DIR_NAME set "OUTPUT_DIR_NAME=colmap"
if not defined ENABLE_FLIP_VERTICAL set "ENABLE_FLIP_VERTICAL=1"
if not defined ENABLE_ROTATE_Z180 set "ENABLE_ROTATE_Z180=1"
if not defined APPLY_COMPONENT_TRANSFORM_FOR_PLY set "APPLY_COMPONENT_TRANSFORM_FOR_PLY=0"
if not defined SKIP_DIRECTIONS set "SKIP_DIRECTIONS="
if not defined RANGE_IMAGES set "RANGE_IMAGES="
if not defined ENABLE_MASKS set "ENABLE_MASKS=0"
if not defined YOLO_CLASSES set "YOLO_CLASSES=0"
if not defined YOLO_CONF set "YOLO_CONF=0.25"
if not defined YOLO_MODEL set "YOLO_MODEL=yolo11n-seg.pt"
if not defined INVERT_MASK set "INVERT_MASK=0"
if not defined ENABLE_MASK_OVEREXPOSURE set "ENABLE_MASK_OVEREXPOSURE=0"
if not defined OVEREXPOSURE_THRESHOLD set "OVEREXPOSURE_THRESHOLD=250"
if not defined OVEREXPOSURE_DILATE set "OVEREXPOSURE_DILATE=5"
if not defined QUIET set "QUIET=0"

if "%~1"=="" (
  echo Drag and drop a dataset folder onto this batch file.
  echo.
  echo Expected:
  echo   dataset-root\images\...
  echo   dataset-root\cameras.xml
  echo   dataset-root\pointcloud.ply  ^(optional^)
  exit /b 1
)

set "INPUT=%~1"
if not exist "%INPUT%" (
  echo Input path not found:
  echo   "%INPUT%"
  exit /b 1
)

set "ROOT=%INPUT%"
if /i "%~nx1"=="images" set "ROOT=%~dp1"
for %%I in ("%ROOT%") do set "ROOT=%%~fI"

set "IMAGES_DIR=%ROOT%\images"
if not exist "%IMAGES_DIR%" set "IMAGES_DIR=%INPUT%"
for %%I in ("%IMAGES_DIR%") do set "IMAGES_DIR=%%~fI"

if not exist "%IMAGES_DIR%" (
  echo Images folder not found:
  echo   "%IMAGES_DIR%"
  exit /b 1
)

set "XML="
set "PLY="
set "XMP="

for %%F in ("%ROOT%\*.xml") do (
  if not defined XML set "XML=%%~fF"
)
for %%F in ("%ROOT%\*.ply") do (
  if not defined PLY set "PLY=%%~fF"
)
for %%F in ("%ROOT%\*.xmp") do (
  if not defined XMP set "XMP=%%~fF"
)

if not defined XML (
  for %%F in ("%IMAGES_DIR%\*.xml") do (
    if not defined XML set "XML=%%~fF"
  )
)
if not defined PLY (
  for %%F in ("%IMAGES_DIR%\*.ply") do (
    if not defined PLY set "PLY=%%~fF"
  )
)
if not defined XMP (
  for %%F in ("%IMAGES_DIR%\*.xmp") do (
    if not defined XMP set "XMP=%%~fF"
  )
)

if not defined XML (
  echo No Metashape XML file was found in:
  echo   "%ROOT%"
  echo   "%IMAGES_DIR%"
  if defined XMP (
    echo.
    echo Found XMP:
    echo   "%XMP%"
    echo.
    echo This converter does not take XMP. It requires a Metashape camera export XML.
  )
  exit /b 1
)

set "OUTPUT=%ROOT%\%OUTPUT_DIR_NAME%"
if not exist "%OUTPUT%" mkdir "%OUTPUT%"

set "TEMP_IMAGES=%OUTPUT%\_flat_images"
if exist "%TEMP_IMAGES%" rmdir /s /q "%TEMP_IMAGES%"
mkdir "%TEMP_IMAGES%"

powershell -NoProfile -Command ^
  "$src = $env:IMAGES_DIR;" ^
  "$dst = $env:TEMP_IMAGES;" ^
  "$exts = '*.jpg','*.jpeg','*.png','*.tif','*.tiff','*.webp';" ^
  "$files = Get-ChildItem -Path $src -Recurse -File -Include $exts;" ^
  "foreach ($file in $files) { Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $dst $file.Name) -Force };" ^
  "if ($files.Count -eq 0) { exit 2 }"

if errorlevel 2 (
  echo No supported image files were found under:
  echo   "%IMAGES_DIR%"
  if exist "%TEMP_IMAGES%" rmdir /s /q "%TEMP_IMAGES%"
  exit /b 1
)

if errorlevel 1 (
  echo Failed while collecting images from:
  echo   "%IMAGES_DIR%"
  if exist "%TEMP_IMAGES%" rmdir /s /q "%TEMP_IMAGES%"
  exit /b 1
)

set "SCRIPT_DIR=%REPO_ROOT%\Metashape_360_to_COLMAP_plane"
set "SCRIPT_PATH=%SCRIPT_DIR%\metashape_360_to_colmap.py"

if not exist "%SCRIPT_PATH%" (
  echo Converter script not found:
  echo   "%SCRIPT_PATH%"
  if exist "%TEMP_IMAGES%" rmdir /s /q "%TEMP_IMAGES%"
  exit /b 1
)

where "%PYTHON_EXE%" >nul 2>nul
if errorlevel 1 (
  where py >nul 2>nul
  if errorlevel 1 (
    echo Python launcher not found. Set PYTHON_EXE or install Python.
    if exist "%TEMP_IMAGES%" rmdir /s /q "%TEMP_IMAGES%"
    exit /b 1
  )
  set "PYTHON_EXE=py"
)

echo Running converter with:
echo   input folder: "%ROOT%"
echo   images:       "%IMAGES_DIR%"
echo   xml:          "%XML%"
if defined PLY (
  echo   ply:          "%PLY%"
) else (
  echo   ply:          [not provided]
)
echo   output:       "%OUTPUT%"
echo.

pushd "%SCRIPT_DIR%"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$argList = @($env:SCRIPT_PATH, '--images', $env:TEMP_IMAGES, '--xml', $env:XML, '--output', $env:OUTPUT, '--crop-size', $env:CROP_SIZE, '--fov-deg', $env:FOV_DEG, '--max-images', $env:MAX_IMAGES, '--num-workers', $env:NUM_WORKERS, '--yaw-offset', $env:YAW_OFFSET, '--output-format', $env:OUTPUT_FORMAT);" ^
  "if ($env:PLY) { $argList += @('--ply', $env:PLY) };" ^
  "if ($env:SKIP_DIRECTIONS) { $argList += @('--skip-directions', $env:SKIP_DIRECTIONS) };" ^
  "if ($env:RANGE_IMAGES) { $argList += @('--range-images', $env:RANGE_IMAGES) };" ^
  "if ($env:ENABLE_FLIP_VERTICAL -eq '1') { $argList += '--flip-vertical' } else { $argList += '--no-flip-vertical' };" ^
  "if ($env:ENABLE_ROTATE_Z180 -eq '1') { $argList += '--rotate-z180' } else { $argList += '--no-rotate-z180' };" ^
  "if ($env:APPLY_COMPONENT_TRANSFORM_FOR_PLY -eq '1') { $argList += '--apply-component-transform-for-ply' };" ^
  "if ($env:ENABLE_MASKS -eq '1') { $argList += @('--generate-masks', '--yolo-classes', $env:YOLO_CLASSES, '--yolo-conf', $env:YOLO_CONF, '--yolo-model', $env:YOLO_MODEL) };" ^
  "if ($env:INVERT_MASK -eq '1') { $argList += '--invert-mask' };" ^
  "if ($env:ENABLE_MASK_OVEREXPOSURE -eq '1') { $argList += @('--mask-overexposure', '--overexposure-threshold', $env:OVEREXPOSURE_THRESHOLD, '--overexposure-dilate', $env:OVEREXPOSURE_DILATE) };" ^
  "if ($env:QUIET -eq '1') { $argList += '--quiet' };" ^
  "& $env:PYTHON_EXE @argList"
set "RUN_EXIT=%ERRORLEVEL%"
popd

if exist "%TEMP_IMAGES%" rmdir /s /q "%TEMP_IMAGES%"

if not "%RUN_EXIT%"=="0" (
  echo.
  echo Conversion failed with exit code %RUN_EXIT%.
  exit /b %RUN_EXIT%
)

echo.
echo Done. Output written to:
echo   "%OUTPUT%"
exit /b 0
