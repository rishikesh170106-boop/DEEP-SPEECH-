# FairVoice-Lite Telugu ASR Launcher
$host.UI.RawUI.WindowTitle = "FairVoice-Lite Telugu ASR"
Set-Location -Path $PSScriptRoot
$env:PYTHONPATH = "src;."
$env:PYTHONUNBUFFERED = "1"
$env:GRADIO_ANALYTICS_ENABLED = "False"

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " FAIRVOICE-LITE: TELUGU ASR (PHASE 6 KENLM + GRADIO APP)" -ForegroundColor Yellow
Write-Host " Authors: Rishikesh (24BDS030) | Arjun (24BDS039) | Vineel (24BDS010)" -ForegroundColor White
Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " Starting web server on http://127.0.0.1:8080 ..." -ForegroundColor Green
Write-Host " Keep this PowerShell terminal OPEN while testing." -ForegroundColor Gray
Write-Host "=====================================================================" -ForegroundColor Cyan

& "$PSScriptRoot\.venv\Scripts\python.exe" -u "app\app.py"
