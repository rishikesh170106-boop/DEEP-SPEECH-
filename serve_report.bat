@echo off
title FairVoice-Lite Mobile Report Server
cd /d "%~dp0"
echo ===================================================
echo   FAIRVOICE-LITE: MOBILE REPORT WEB SERVER
echo ===================================================
echo.
echo Serving Project Report on your Local Wi-Fi network:
echo.
echo   -> On your Computer: http://localhost:8000
echo   -> On your Mobile:   http://10.0.11.92:8000
echo.
echo Make sure your mobile phone is connected to the same Wi-Fi!
echo Press Ctrl+C in this terminal window to stop the server.
echo ===================================================
echo.
.\.venv\Scripts\python.exe -m http.server 8000 --directory reports
pause

