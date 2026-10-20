@echo off
setlocal EnableDelayedExpansion

title LS-Trainer - Python Environment Setup

echo ========================================================
echo   LS-Trainer: Python Environment Setup
echo ========================================================
echo.

set "APP_DIR=%~dp0"
set "VENV_DIR=%APP_DIR%.venv"
set "PYTHON_CMD="

REM Check if user supplied -y / --yes flag
set "AUTO_YES="
if /i "%~1"=="/y" set "AUTO_YES=1"
if /i "%~1"=="-y" set "AUTO_YES=1"
if /i "%~1"=="--yes" set "AUTO_YES=1"

REM 1. Check for system Python on PATH
echo [*] Checking for Python on PATH...
python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_CMD=python"
    goto :PYTHON_DETECTED
)

py -3.12 --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_CMD=py -3.12"
    goto :PYTHON_DETECTED
)

py --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_CMD=py"
    goto :PYTHON_DETECTED
)

REM Also check common installation directories if not found on PATH
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :PYTHON_DETECTED
)
if exist "%ProgramFiles%\Python312\python.exe" (
    set "PYTHON_CMD=%ProgramFiles%\Python312\python.exe"
    goto :PYTHON_DETECTED
)
if exist "C:\Python312\python.exe" (
    set "PYTHON_CMD=C:\Python312\python.exe"
    goto :PYTHON_DETECTED
)

REM 2. If Python isn't found, ask to install 3.12 and install on Yes
echo [!] Python was not found on your system PATH.
echo.

set "USER_CHOICE=No"
if defined AUTO_YES (
    set "USER_CHOICE=Yes"
    goto :DO_INSTALL
)

REM Prompt via Windows MessageBox for GUI/installer visibility
for /f "usebackq delims=" %%A in (`powershell -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show('Python was not found on your system PATH. Would you like to download and install Python 3.12 now?', 'LS-Trainer Setup - Python Required', 4, 32)" 2^>nul`) do (
    set "USER_CHOICE=%%A"
)

REM CLI fallback if PowerShell GUI prompt was empty
if "%USER_CHOICE%"=="" (
    set /p "CLI_PROMPT=Would you like to install Python 3.12 now? (Y/N): "
    if /i "!CLI_PROMPT!"=="Y" set "USER_CHOICE=Yes"
    if /i "!CLI_PROMPT!"=="YES" set "USER_CHOICE=Yes"
)

if /i not "%USER_CHOICE%"=="Yes" (
    echo ========================================================
    echo  ERROR: Python 3.12 is required to run LS-Trainer.
    echo  Setup cancelled by user.
    echo.
    echo  Please install Python 3.12 from https://www.python.org
    echo  and re-run setup-venv.bat.
    echo ========================================================
    pause
    exit /b 1
)

:DO_INSTALL
echo [*] Installing Python 3.12...
set "INSTALL_DONE=0"

REM Try winget
where winget >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [*] Installing Python 3.12 using Windows Package Manager (winget)...
    winget install --id Python.Python.3.12 -e --accept-package-agreements --accept-source-agreements
    if %ERRORLEVEL% EQU 0 set "INSTALL_DONE=1"
)

REM Fallback to downloading official python-3.12 installer
if "%INSTALL_DONE%"=="0" (
    echo [*] Downloading Python 3.12 installer from python.org...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe' -OutFile '%TEMP%\python-3.12.8-amd64.exe'"
    if exist "%TEMP%\python-3.12.8-amd64.exe" (
        echo [*] Running Python 3.12 installer in passive mode...
        start /wait "" "%TEMP%\python-3.12.8-amd64.exe" /passive PrependPath=1 Include_pip=1
        del "%TEMP%\python-3.12.8-amd64.exe" >nul 2>&1
        set "INSTALL_DONE=1"
    )
)

REM Re-detect Python binary
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :PYTHON_DETECTED
)
if exist "%ProgramFiles%\Python312\python.exe" (
    set "PYTHON_CMD=%ProgramFiles%\Python312\python.exe"
    goto :PYTHON_DETECTED
)
if exist "C:\Python312\python.exe" (
    set "PYTHON_CMD=C:\Python312\python.exe"
    goto :PYTHON_DETECTED
)
python --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_CMD=python"
    goto :PYTHON_DETECTED
)

echo ========================================================
echo  ERROR: Python 3.12 installation finished, but the Python
echo  executable could not be detected.
echo.
echo  Please verify that Python 3.12 is installed and on PATH,
echo  then re-run setup-venv.bat.
echo ========================================================
pause
exit /b 1

:PYTHON_DETECTED
echo [*] Python detected: "%PYTHON_CMD%"
"%PYTHON_CMD%" --version

REM 3. Creates a .venv folder next to the installed app.exe (using %~dp0)
echo.
echo [*] Target directory: "%APP_DIR%"
echo [*] Creating virtual environment at "%VENV_DIR%"...
"%PYTHON_CMD%" -m venv "%VENV_DIR%"
if %ERRORLEVEL% NEQ 0 (
    echo ========================================================
    echo  ERROR: Failed to create virtual environment.
    echo  Error Code: %ERRORLEVEL%
    echo ========================================================
    pause
    exit /b %ERRORLEVEL%
)

set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo ========================================================
    echo  ERROR: Virtual environment Python not found at:
    echo  "%VENV_PYTHON%"
    echo ========================================================
    pause
    exit /b 1
)

REM 4. Upgrades pip and installs ultralytics into that venv
echo.
echo [*] Upgrading pip in virtual environment...
"%VENV_PYTHON%" -m pip install --upgrade pip
if %ERRORLEVEL% NEQ 0 (
    echo [!] Warning: pip upgrade returned non-zero exit code. Continuing...
)

echo.
echo [*] Installing ultralytics...
"%VENV_PYTHON%" -m pip install ultralytics
if %ERRORLEVEL% NEQ 0 (
    echo ========================================================
    echo  ERROR: Failed to install ultralytics into the virtual environment.
    echo  Error Code: %ERRORLEVEL%
    echo ========================================================
    pause
    exit /b %ERRORLEVEL%
)

REM 5. Check GPU / CUDA availability
echo.
echo [*] Checking GPU / CUDA availability...
"%VENV_PYTHON%" -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

for /f "usebackq delims=" %%G in (`"%VENV_PYTHON%" -c "import torch; print(torch.cuda.is_available())" 2^>nul`) do (
    set "CUDA_AVAILABLE=%%G"
)

if /i "%CUDA_AVAILABLE%"=="True" (
    echo [*] GPU acceleration is available for training.
    goto :GPU_CHECK_DONE
)

echo.
echo --------------------------------------------------------
echo  NOTE: CUDA is not available on this machine.
echo  LS-Trainer will fall back to CPU for training.
echo  (If you have an NVIDIA GPU, ensure NVIDIA drivers and
echo   CUDA are installed if you wish to use GPU acceleration.)
echo --------------------------------------------------------

:GPU_CHECK_DONE

REM 6. Shows clear success/error messages and pauses so the user can see the result
echo.
echo ========================================================
echo  SUCCESS: Python environment successfully configured!
echo.
echo  Virtual Environment: "%VENV_DIR%"
echo  Packages Installed:  ultralytics
echo  Ready for LS-Trainer!
echo ========================================================
echo.
pause
exit /b 0
