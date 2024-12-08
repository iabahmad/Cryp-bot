@echo off

REM Specify the full path to the conda executable
set CONDA_PATH=C:\Users\Ali\anaconda3\condabin\conda.bat

REM Check if Anaconda is installed and conda is accessible
if not exist "%CONDA_PATH%" (
    echo Anaconda is not installed or the path is incorrect. Please check the installation or update the path in the script.
    pause
    exit /b 1
)

REM Navigate to the directory where the script is located
cd /d "%~dp0"

REM Check if the "neurog" environment exists
call "%CONDA_PATH%" env list | findstr "neurog" >nul 2>&1
if %errorlevel% equ 0 (
    echo Activating "neurog" environment...
    call "%CONDA_PATH%" activate neurog
) else (
    echo "neurog" environment not found. Using base environment...
)

REM Check if PYTHON_HOME is set, otherwise assume 'python' is in PATH
if defined PYTHON_HOME (
    set PYTHON_EXECUTABLE=%PYTHON_HOME%\python.exe
) else (
    set PYTHON_EXECUTABLE=python
)

REM Check if Python is accessible
%PYTHON_EXECUTABLE% --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python was not found; please ensure it is installed and added to your PATH.
    pause
    exit /b 1
)

REM Run the Python script using the relative path
python "..\simulation.py"

REM Deactivate the environment if it was activated
call "%CONDA_PATH%" deactivate

pause