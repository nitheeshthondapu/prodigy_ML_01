@echo off
echo =======================================================
echo Activating Virtual Environment and Starting Web Application...
echo =======================================================
call .venv\Scripts\activate
python app.py
echo =======================================================
pause
