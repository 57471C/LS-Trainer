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
    "%VENV_DIR%\Scripts\python.exe" -c "import importlib.util as u, sys; sys.exit(0 if all(u.find_spec(m) for m in ('ultralytics','rfdetr','vernier')) else 1)" >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        echo [*] Virtual environment, ultralytics, and rfdetr already installed and ready at:
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
    echo [*] Installing Python 3.12 using Windows Package Manager winget...
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

REM 4. Upgrades pip, installs matching PyTorch GPU/CPU build first, then installs ultralytics & rfdetr
echo.
echo [*] Upgrading pip in virtual environment...
"%VENV_PYTHON%" -m pip install --upgrade pip
if !ERRORLEVEL! NEQ 0 (
    echo [!] Warning: pip upgrade returned non-zero exit code. Continuing...
)

REM Install matching PyTorch (Intel Arc XPU -> NVIDIA CUDA -> CPU) BEFORE ultralytics/rfdetr
REM so pip never pulls CUDA or generic PyPI torch packages onto an Intel Arc or CPU machine.
call :ENSURE_PYTORCH_GPU

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

echo.
echo [*] Installing rfdetr and COCO/ONNX training dependencies...
"%VENV_PYTHON%" -m pip install rfdetr pytorch_lightning "torchmetrics[detection]<1.9.0,>=1.8.2" vernier faster-coco-eval hotcoco pycocotools scipy peft onnx onnxruntime
if !ERRORLEVEL! NEQ 0 (
    echo [!] Warning: rfdetr installation returned non-zero exit code. Continuing...
)

:GPU_CHECK
call :ENSURE_PYTORCH_GPU
goto :COMPLETED

:ENSURE_PYTORCH_GPU
REM 5. Auto-detect GPU & driver and install matching PyTorch wheel.
REM
REM Detection priority:
REM   1. Intel Arc / XPU GPU (Win32_VideoController matching "Intel" + "Arc", or torch.xpu)
REM      -> Installs Intel Arc XPU build from https://download.pytorch.org/whl/xpu (no CUDA packages)
REM   2. NVIDIA GPU (nvidia-smi or Win32_VideoController matching "NVIDIA")
REM      -> Installs CUDA 12.8 (Blackwell / driver >= 560) or CUDA 12.4 build
REM   3. CPU Fallback (neither Intel Arc nor NVIDIA detected)
REM      -> Installs CPU-only build from https://download.pytorch.org/whl/cpu
echo.
echo [*] Detecting GPU (Intel Arc XPU / NVIDIA CUDA / CPU fallback)...

set "GPU_DETECT_OUT=%TEMP%\ls_trainer_gpu_detect.txt"
set "GPU_VERIFY_OUT=%TEMP%\ls_trainer_gpu_verify.txt"
del "%GPU_DETECT_OUT%" >nul 2>&1
del "%GPU_VERIFY_OUT%" >nul 2>&1

"%VENV_PYTHON%" -c "import json,re,shutil,subprocess; arc=''; nv=''; drv=''; maj=0; p=['powershell','-NoProfile','-Command','Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress']; raw=(subprocess.run(p,capture_output=True,text=True).stdout.strip() if shutil.which('powershell') else ''); items=json.loads(raw) if raw else []; items=[items] if isinstance(items,dict) else items; arc=next((str(i.get('Name','')).strip() for i in items if 'intel' in str(i.get('Name','')).lower() and ('arc' in str(i.get('Name','')).lower() or 'xpu' in str(i.get('Name','')).lower())), ''); nv_item=next((i for i in items if 'nvidia' in str(i.get('Name','')).lower()), None); nv=str(nv_item.get('Name','')).strip() if nv_item else ''; rdrv=re.sub(r'\D','',str(nv_item.get('DriverVersion',''))) if nv_item else ''; drv=(rdrv[-5:-2]+'.'+rdrv[-2:]) if len(rdrv)>=5 else ''; smi=(subprocess.run(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],capture_output=True,text=True) if (not arc and shutil.which('nvidia-smi')) else None); smi_line=smi.stdout.splitlines()[0] if (smi and smi.returncode==0 and smi.stdout.strip()) else ''; smi_parts=[x.strip() for x in smi_line.split(',')] if smi_line else []; nv=smi_parts[0] if smi_parts and smi_parts[0] else nv; drv=smi_parts[1] if len(smi_parts)>1 and smi_parts[1] else drv; maj=int(drv.split('.')[0]) if (drv and drv.split('.')[0].isdigit()) else 0; low=nv.lower(); bw=any(k in low for k in ('rtx 50','rtx50','blackwell','gb10','gb20','b100','b200')); print(f'xpu|https://download.pytorch.org/whl/xpu|Intel Arc XPU|{arc}' if arc else ((f'cu128|https://download.pytorch.org/whl/cu128|CUDA 12.8|{nv}' if (bw or maj>=560) else f'cu124|https://download.pytorch.org/whl/cu124|CUDA 12.4|{nv}') if nv else 'cpu|https://download.pytorch.org/whl/cpu|CPU-only|No discrete GPU'))" > "%GPU_DETECT_OUT%" 2>nul

set "TARGET_WHEEL=cpu"
set "TARGET_URL=https://download.pytorch.org/whl/cpu"
set "CUDA_LABEL=CPU-only"
set "GPU_DISPLAY=CPU"

if exist "%GPU_DETECT_OUT%" (
    for /f "usebackq tokens=1,2,3,4 delims=|" %%A in ("%GPU_DETECT_OUT%") do (
        set "TARGET_WHEEL=%%A"
        set "TARGET_URL=%%B"
        set "CUDA_LABEL=%%C"
        set "GPU_DISPLAY=%%D"
    )
    del "%GPU_DETECT_OUT%" >nul 2>&1
)

echo [*] Detected Device: !GPU_DISPLAY! ^(!CUDA_LABEL! - !TARGET_WHEEL!^)

"%VENV_PYTHON%" -c "import sys; tw='!TARGET_WHEEL!'; ok=False; import torch; ver=getattr(torch,'__version__',''); xpu=bool(hasattr(torch,'xpu') and torch.xpu.is_available()); cuda=bool(torch.cuda.is_available()); t=torch.zeros(1,device='xpu')+1 if (tw=='xpu' and xpu) else (torch.zeros(1,device='cuda')+1 if (tw.startswith('cu') and cuda) else None); (torch.xpu.synchronize() if (tw=='xpu' and xpu) else (torch.cuda.synchronize() if (tw.startswith('cu') and cuda) else None)); ok=(xpu and '+xpu' in ver) if tw=='xpu' else ((cuda and ('+'+tw) in ver) if tw.startswith('cu') else (bool(ver) and '+cu' not in ver and '+xpu' not in ver)); print(ok)" > "%GPU_VERIFY_OUT%" 2>nul

set "WHEEL_READY=False"
if exist "%GPU_VERIFY_OUT%" (
    for /f "usebackq delims=" %%G in ("%GPU_VERIFY_OUT%") do (
        set "WHEEL_READY=%%G"
    )
    del "%GPU_VERIFY_OUT%" >nul 2>&1
)

if /i "!TARGET_WHEEL!"=="cpu" (
    echo [*] Neither Intel Arc nor NVIDIA GPU detected. Verifying CPU-only PyTorch wheel...
    if /i not "!WHEEL_READY!"=="True" (
        echo [*] Installing CPU-only PyTorch from !TARGET_URL!...
        "%VENV_PYTHON%" -m pip uninstall -y torch torchvision torchaudio
        "%VENV_PYTHON%" -m pip install torch torchvision torchaudio --index-url !TARGET_URL!
    )
    echo cpu> "%VENV_DIR%\.ls_trainer_wheel"
    goto :CUDA_STILL_CPU
)

if /i "!WHEEL_READY!"=="True" (
    echo [*] Matching PyTorch wheel ^(!TARGET_WHEEL!^) is already installed and verified.
    echo !TARGET_WHEEL!> "%VENV_DIR%\.ls_trainer_wheel"
    "%VENV_PYTHON%" -c "import torch; xpu=bool(hasattr(torch,'xpu') and torch.xpu.is_available()); cuda=bool(torch.cuda.is_available()); print('Intel Arc XPU available:', xpu); print('NVIDIA CUDA available:  ', cuda); print('PyTorch version:        ', torch.__version__); print('Active training device: ', torch.xpu.get_device_name(0) if xpu else (torch.cuda.get_device_name(0) if cuda else 'CPU only'))"
    exit /b 0
)

echo [*] Installing PyTorch for !GPU_DISPLAY! ^(!CUDA_LABEL!^) from !TARGET_URL!...
"%VENV_PYTHON%" -m pip uninstall -y torch torchvision torchaudio
"%VENV_PYTHON%" -m pip install torch torchvision torchaudio --index-url !TARGET_URL!
if !ERRORLEVEL! NEQ 0 (
    echo [!] Warning: !CUDA_LABEL! wheels failed to install. Reinstalling fallback CPU PyTorch...
    "%VENV_PYTHON%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

echo !TARGET_WHEEL!> "%VENV_DIR%\.ls_trainer_wheel"

echo.
echo [*] Re-checking GPU ^(Intel Arc XPU / NVIDIA CUDA^) availability...
"%VENV_PYTHON%" -c "import torch; xpu=bool(hasattr(torch,'xpu') and torch.xpu.is_available()); cuda=bool(torch.cuda.is_available()); print('Intel Arc XPU available:', xpu); print('NVIDIA CUDA available:  ', cuda); print('PyTorch version:        ', torch.__version__); print('Active training device: ', torch.xpu.get_device_name(0) if xpu else (torch.cuda.get_device_name(0) if cuda else 'CPU only'))"

"%VENV_PYTHON%" -c "import torch; xpu=bool(hasattr(torch,'xpu') and torch.xpu.is_available()); cuda=bool(torch.cuda.is_available()); print(torch.xpu.get_device_name(0) if xpu else (torch.cuda.get_device_name(0) if cuda else 'CPU_ONLY'))" > "%GPU_VERIFY_OUT%" 2>nul
set "ACTIVE_GPU_DEV=CPU_ONLY"
if exist "%GPU_VERIFY_OUT%" (
    for /f "usebackq delims=" %%D in ("%GPU_VERIFY_OUT%") do (
        set "ACTIVE_GPU_DEV=%%D"
    )
    del "%GPU_VERIFY_OUT%" >nul 2>&1
)

if /i not "!ACTIVE_GPU_DEV!"=="CPU_ONLY" (
    echo [*] GPU Device: !ACTIVE_GPU_DEV!
    echo [*] GPU acceleration is available for training.
    exit /b 0
)

:CUDA_STILL_CPU
echo.
echo --------------------------------------------------------
echo  NOTE: GPU acceleration ^(Intel Arc XPU / NVIDIA CUDA^) is
echo  not active on this machine. LS-Trainer will fall back
echo  to CPU for training.
echo --------------------------------------------------------
exit /b 0

:COMPLETED
echo.
echo ========================================================
echo  SUCCESS: Python environment successfully configured!
echo.
echo  Virtual Environment: "%VENV_DIR%"
echo  Packages Installed:  ultralytics, rfdetr
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
