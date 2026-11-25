@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" goto missing
".venv\Scripts\python.exe" setup_qwen.py cpu
if errorlevel 1 goto failed
echo Setup completed. Run start.bat.
pause
exit /b 0
:missing
echo Run start.bat once to create the environment, close the app, then run this file again.
pause
exit /b 1
:failed
echo Setup failed. Please send the error text above. Whisper remains available.
pause
exit /b 1
