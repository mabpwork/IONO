@echo off
rem ionolum: starts the program from the folder of this file (it can be installed in any folder).
rem Python: the one written in python_path.txt (next to this file: one line with the full path of
rem pythonw.exe / python.exe) or, without that file, pythonw.exe of the PATH.
cd /d "%~dp0"
set "PY="
if exist "%~dp0python_path.txt" set /p PY=<"%~dp0python_path.txt"
if not defined PY set "PY=pythonw"
start "" "%PY%" "%~dp0ionolum.py" %*
