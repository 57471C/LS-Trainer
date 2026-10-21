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

REM Parse flags
set "AUTO_YES="
set "UNATTENDED="
for %%A in (%*) do (
    if /i "%%~A"=="/y" set "AUTO_YES=1"
    if /i "%%~A"=="-y" set "AUTO_YES=1"
    if /i "%%~A"=="--yes" set "AUTO_YES=1"
    if /i "%%~A"=="--unattended" (
        set "AUTO_YES=1"
        set "UNATTENDED=1"
    )
)

echo [*] Target directory: "%APP_DIR%"
echo [*] Checking for existing virtual environment...

if exist "%VENV_DIR%\Scripts\python.exe" (
    "%VENV_DIR%\Scripts\python.exe" -c "import ultralytics" >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo [*] Virtual environment and ultralytics already installed and ready at:
        echo     "%VENV_DIR%"
        set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
        goto :GPU_CHECK
    )
)

REM 1. Check for system Python
echo [*] Detecting system Python...
call :DETECT_PYTHON
if defined PYTHON_CMD goto :PYTHON_DETECTED

REM 2. If Python isn't found, ask to install 3.12 (or auto-install if unattended/yes)
echo [!] Python was not found on your system.
echo.

set "USER_CHOICE=No"
if defined AUTO_YES (
    set "USER_CHOICE=Yes"
    goto :DO_INSTALL
)

REM Prompt via Windows MessageBox for GUI/installer visibility
for /f "usebackq delims=" %%A in (`powershell -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.MessageBox]::Show('Python was not found on your system. Would you like to download and install Python 3.12 now?', 'LS-Trainer Setup - Python Required', 4, 32)" 2^>nul`) do (
    set "USER_CHOICE=%%A"
)

REM CLI fallback if PowerShell GUI prompt was empty
if "!USER_CHOICE!"=="" (
    set /p "CLI_PROMPT=Would you like to install Python 3.12 now? (Y/N): "
    if /i "!CLI_PROMPT!"=="Y" set "USER_CHOICE=Yes"
    if /i "!CLI_PROMPT!"=="YES" set "USER_CHOICE=Yes"
)

if /i not "!USER_CHOICE!"=="Yes" (
    echo ========================================================
    echo  ERROR: Python 3.12 is required to run LS-Trainer.
    echo  Setup cancelled by user.
    echo.
    echo  Please install Python 3.12 from https://www.python.org
    echo  and re-run setup-venv.bat.
    echo ========================================================
    if not defined UNATTENDED pause
    exit /b 1
)

:DO_INSTALL
echo [*] Installing Python 3.12...
set "INSTALL_DONE=0"

REM Try winget
where winget >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    echo [*] Installing Python 3.12 using Windows Package Manager (winget)...
    winget install --id Python.Python.3.12 -e --accept-package-agreements --accept-source-agreements --disable-interactivity
    if !ERRORLEVEL! EQU 0 set "INSTALL_DONE=1"
)

REM Fallback to downloading official python-3.12 installer
if "!INSTALL_DONE!"=="0" (
    echo [*] Downloading Python 3.12 installer from python.org...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe' -OutFile '%TEMP%\python-3.12.8-amd64.exe'"
    if exist "%TEMP%\python-3.12.8-amd64.exe" (
        echo [*] Running Python 3.12 installer in passive mode...
        start /wait "" "%TEMP%\python-3.12.8-amd64.exe" /passive PrependPath=1 Include_pip=1 Include_launcher=1 InstallLauncherAllUsers=1
        del "%TEMP%\python-3.12.8-amd64.exe" >nul 2>&1
        set "INSTALL_DONE=1"
    )
)

REM Re-detect Python binary after install
call :DETECT_PYTHON
if defined PYTHON_CMD goto :PYTHON_DETECTED

echo ========================================================
echo  ERROR: Python 3.12 installation finished, but the Python
echo  executable could not be detected.
echo.
echo  Please verify that Python 3.12 is installed and on PATH,
echo  then re-run setup-venv.bat.
echo ========================================================
if not defined UNATTENDED pause
exit /b 1

:PYTHON_DETECTED
echo [*] Python detected: "%PYTHON_CMD%"
"%PYTHON_CMD%" --version

REM 3. Creates a .venv folder next to the installed app.exe (using %~dp0)
echo.
echo [*] Creating virtual environment at "%VENV_DIR%"...
if exist "%VENV_DIR%" (
    if not exist "%VENV_DIR%\Scripts\python.exe" (
        echo [*] Cleaning up incomplete virtual environment directory...
        rmdir /s /q "%VENV_DIR%" 2>nul
    )
)

"%PYTHON_CMD%" -m venv "%VENV_DIR%"
if !ERRORLEVEL! NEQ 0 (
    echo ========================================================
    echo  ERROR: Failed to create virtual environment.
    echo  Error Code: !ERRORLEVEL!
    echo ========================================================
    if not defined UNATTENDED pause
    exit /b !ERRORLEVEL!
)

set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo ========================================================
    echo  ERROR: Virtual environment Python not found at:
    echo  "%VENV_PYTHON%"
    echo ========================================================
    if not defined UNATTENDED pause
    exit /b 1
)

REM 4. Upgrades pip and installs ultralytics into that venv
echo.
echo [*] Upgrading pip in virtual environment...
"%VENV_PYTHON%" -m pip install --upgrade pip
if !ERRORLEVEL! NEQ 0 (
    echo [!] Warning: pip upgrade returned non-zero exit code. Continuing...
)

echo.
echo [*] Installing ultralytics...
"%VENV_PYTHON%" -m pip install ultralytics
if !ERRORLEVEL! NEQ 0 (
    echo ========================================================
    echo  ERROR: Failed to install ultralytics into the virtual environment.
    echo  Error Code: !ERRORLEVEL!
    echo ========================================================
    if not defined UNATTENDED pause
    exit /b !ERRORLEVEL!
)

:GPU_CHECK
REM 5. Check GPU / CUDA availability
echo.
echo [*] Checking GPU / CUDA availability...
"%VENV_PYTHON%" -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

for /f "usebackq delims=" %%G in (`"%VENV_PYTHON%" -c "import torch; print(torch.cuda.is_available())" 2^>nul`) do (
    set "CUDA_AVAILABLE=%%G"
)

if /i "!CUDA_AVAILABLE!"=="True" (
    echo [*] GPU acceleration is available for training.
    goto :COMPLETED
)

echo.
echo --------------------------------------------------------
echo  NOTE: CUDA is not available on this machine.
echo  LS-Trainer will fall back to CPU for training.
echo  (If you have an NVIDIA GPU, ensure NVIDIA drivers and
echo   CUDA are installed if you wish to use GPU acceleration.)
echo --------------------------------------------------------

:COMPLETED
echo.
echo ========================================================
echo  SUCCESS: Python environment successfully configured!
echo.
echo  Virtual Environment: "%VENV_DIR%"
echo  Packages Installed:  ultralytics
echo  Ready for LS-Trainer!
echo ========================================================
echo.
if not defined UNATTENDED pause
exit /b 0

REM ========================================================
REM SUBROUTINES
REM ========================================================
:DETECT_PYTHON
set "PYTHON_CMD="

REM A. Check py launcher specifically for working python executable
for /f "usebackq delims=" %%V in (`py -3.12 -c "import sys; print(sys.executable)" 2^>nul`) do (
    if exist "%%V" (
        set "PYTHON_CMD=%%V"
        exit /b 0
    )
)
for /f "usebackq delims=" %%V in (`py -3.11 -c "import sys; print(sys.executable)" 2^>nul`) do (
    if exist "%%V" (
        set "PYTHON_CMD=%%V"
        exit /b 0
    )
)
for /f "usebackq delims=" %%V in (`py -3.10 -c "import sys; print(sys.executable)" 2^>nul`) do (
    if exist "%%V" (
        set "PYTHON_CMD=%%V"
        exit /b 0
    )
)
for /f "usebackq delims=" %%V in (`py -3 -c "import sys; print(sys.executable)" 2^>nul`) do (
    if exist "%%V" (
        set "PYTHON_CMD=%%V"
        exit /b 0
    )
)

REM B. Check 64-bit Program Files (handles 32-bit NSIS WOW64 redirection)
if defined ProgramW6432 (
    if exist "%ProgramW6432%\Python312\python.exe" (
        set "PYTHON_CMD=%ProgramW6432%\Python312\python.exe"
        exit /b 0
    )
    if exist "%ProgramW6432%\Python311\python.exe" (
        set "PYTHON_CMD=%ProgramW6432%\Python311\python.exe"
        exit /b 0
    )
    if exist "%ProgramW6432%\Python310\python.exe" (
        set "PYTHON_CMD=%ProgramW6432%\Python310\python.exe"
        exit /b 0
    )
)

REM C. Check ProgramFiles
if exist "%ProgramFiles%\Python312\python.exe" (
    set "PYTHON_CMD=%ProgramFiles%\Python312\python.exe"
    exit /b 0
)
if exist "%ProgramFiles%\Python311\python.exe" (
    set "PYTHON_CMD=%ProgramFiles%\Python311\python.exe"
    exit /b 0
)
if exist "%ProgramFiles%\Python310\python.exe" (
    set "PYTHON_CMD=%ProgramFiles%\Python310\python.exe"
    exit /b 0
)

REM D. Check LocalAppData
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    exit /b 0
)
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    exit /b 0
)
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    exit /b 0
)

REM E. Check root paths
if exist "C:\Python312\python.exe" (
    set "PYTHON_CMD=C:\Python312\python.exe"
    exit /b 0
)
if exist "C:\Python311\python.exe" (
    set "PYTHON_CMD=C:\Python311\python.exe"
    exit /b 0
)
if exist "C:\Python310\python.exe" (
    set "PYTHON_CMD=C:\Python310\python.exe"
    exit /b 0
)

REM F. Query registry
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "(Get-ItemProperty -Path 'HKLM:\SOFTWARE\Python\PythonCore\3.12\InstallPath', 'HKCU:\SOFTWARE\Python\PythonCore\3.12\InstallPath', 'HKLM:\SOFTWARE\WOW6432Node\Python\PythonCore\3.12\InstallPath', 'HKLM:\SOFTWARE\Python\PythonCore\*\InstallPath', 'HKCU:\SOFTWARE\Python\PythonCore\*\InstallPath' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty ExecutablePath -ErrorAction SilentlyContinue | Where-Object { Test-Path $_ } | Select-Object -First 1)" 2^>nul`) do (
    if exist "%%P" (
        set "PYTHON_CMD=%%P"
        exit /b 0
    )
)

REM G. Refresh PATH from registry and check python on PATH
for /f "usebackq delims=" %%A in (`powershell -NoProfile -Command "[System.Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path', 'User')"` ) do (
    set "PATH=%%A;!PATH!"
)

python --version >nul 2>&1
if !ERRORLEVEL! EQU 0 (
    set "PYTHON_CMD=python"
    exit /b 0
)

exit /b 1
