@echo off
setlocal
cd /d "%~dp0"
set HF_HUB_DISABLE_XET=1
set PYTHONUTF8=1
if exist ".venv\Scripts\python.exe" goto run
where uv >nul 2>&1
if errorlevel 1 goto pythonlauncher
uv venv --python 3.11 .venv
if errorlevel 1 goto failed
uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt
if errorlevel 1 goto failed
goto run
:pythonlauncher
py -3.11 -m venv .venv
if errorlevel 1 goto missing
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
:run
rem Fast check (find_spec does not import the heavy modules).
".venv\Scripts\python.exe" -c "import importlib.util as u,sys; sys.exit(any(u.find_spec(m) is None for m in 'alkana numpy scipy soundfile sounddevice fugashi unidic_lite faster_whisper tkinter tkinterdnd2'.split()))"
if errorlevel 1 goto repair
".venv\Scripts\python.exe" -c "import importlib.util as u,sys; sys.exit(u.find_spec('torch') is None or u.find_spec('transformers.models.qwen3_asr') is None or u.find_spec('nagisa') is None)"
if not errorlevel 1 goto launch
where nvidia-smi >nul 2>&1
if errorlevel 1 goto qwencpu
".venv\Scripts\python.exe" setup_qwen.py gpu
goto launch
:qwencpu
".venv\Scripts\python.exe" setup_qwen.py cpu
:launch
rem Launch the GUI without a console window; this window closes immediately.
start "" ".venv\Scripts\pythonw.exe" app.py
exit /b 0
:repair
where uv >nul 2>&1
if errorlevel 1 goto repairpip
uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt
if errorlevel 1 goto failed
goto run
:repairpip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
goto run
:missing
echo Please install Python 3.11 with Tcl/Tk, or install uv, then run start.bat again.
pause
exit /b 1
:failed
echo An error occurred. Please send the error text shown above.
pause
exit /b 1
