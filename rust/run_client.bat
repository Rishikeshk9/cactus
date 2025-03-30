@echo off
setlocal

REM Check if CUDA is available
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: NVIDIA GPU not detected. The client will run in CPU-only mode.
    pause
)

REM Get command line arguments
set SERVER_URL=%1
set PUBLIC_IP=%2
set PORT=%3

REM Use default values if not provided
if "%SERVER_URL%"=="" set SERVER_URL=http://3.110.255.211:8001
if "%PUBLIC_IP%"=="" set PUBLIC_IP=3.110.255.211
if "%PORT%"=="" set PORT=8002

REM Add Python and CUDA paths to PATH
set PATH=%~dp0target\release\python;%PATH%
if defined CUDA_PATH set PATH=%CUDA_PATH%\bin;%PATH%

REM Set Python environment variables
set PYTHONHOME=%~dp0target\release\python
set PYTHONPATH=%~dp0target\release\python\site-packages

echo Starting GPU client...
echo Server URL: %SERVER_URL%
echo Public IP: %PUBLIC_IP%
echo Port: %PORT%
echo.

REM Run the client
target\release\cactus.exe client --server-url %SERVER_URL% --public-ip %PUBLIC_IP% --port %PORT%

pause 