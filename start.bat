@echo off
title Ramayan Lyric Finder
cd /d "%~dp0"
call venv_ctc\Scripts\activate.bat
python app.py
pause