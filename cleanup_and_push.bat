@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo        CLEANUP REPO VA PUSH LEN NHANH REFACTOR-CLEAN
echo ========================================================

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0cleanup_and_push.ps1"

pause
