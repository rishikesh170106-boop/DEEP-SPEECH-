@echo off
title FairVoice-Lite Telugu ASR
cd /d "%~dp0"
set PYTHONPATH=src;.
set PYTHONUNBUFFERED=1
set GRADIO_ANALYTICS_ENABLED=False
set HF_HUB_DISABLE_TELEMETRY=1
set TRANSFORMERS_NO_ADVISORY_WARNINGS=1

echo =====================================================================
echo  FAIRVOICE-LITE: TELUGU ASR (PHASE 6 KENLM + GRADIO APP)
echo  Authors: Rishikesh (24BDS030) ^| Arjun (24BDS039) ^| Vineel (24BDS010)
echo =====================================================================
echo [INFO] Starting web server on http://127.0.0.1:8080 ...
echo [INFO] Keep this terminal window OPEN while testing.
echo =====================================================================

.\.venv\Scripts\python.exe -u app\app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application stopped unexpectedly with code %ERRORLEVEL%.
    pause
)
