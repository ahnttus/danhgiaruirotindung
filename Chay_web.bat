@echo off
title CreditRisk AI - Enterprise Fintech Console
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
)
python -m streamlit run app.py

pause
