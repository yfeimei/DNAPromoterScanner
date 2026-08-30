@echo off
REM ===================================================================
REM  DNA Promoter Scanner -- one-time setup for Windows.
REM
REM  Double-click this file (or run it from Command Prompt/PowerShell).
REM  It creates a private virtual environment in .venv, installs the
REM  four required packages into it, and trains both models so the
REM  website has something to load.
REM
REM  Run it once. After that, use run_app.bat to start the website.
REM ===================================================================
setlocal EnableExtensions

REM Work from the folder this script lives in, whatever the current
REM directory happens to be (important when double-clicked from Explorer).
cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"

echo ==================================================
echo  DNA Promoter Scanner - Windows setup
echo ==================================================
echo.

if exist "%VENV_PY%" goto install_packages

REM Prefer "py", the official Windows Python launcher. Plain "python3" is
REM avoided on purpose: on Windows it often resolves to an MSYS2 or
REM Microsoft Store Python that builds a Unix-style venv/bin layout, and
REM then none of the venv\Scripts paths below exist.
where py >nul 2>nul
if errorlevel 1 goto try_python

echo Creating the virtual environment in .venv (using "py -3")...
py -3 -m venv .venv
goto check_venv

:try_python
where python >nul 2>nul
if errorlevel 1 goto no_python

echo Creating the virtual environment in .venv (using "python")...
python -m venv .venv
goto check_venv

:no_python
echo ERROR: No Python installation was found.
echo.
echo Install Python 3.10 or newer (free) from:
echo     https://www.python.org/downloads/windows/
echo In the installer, tick "Add python.exe to PATH", then run this
echo script again.
goto fail

:check_venv
if not exist "%VENV_PY%" goto venv_failed
goto install_packages

:venv_failed
echo ERROR: The virtual environment was not created correctly
echo        ("%VENV_PY%" is missing).
echo.
echo If you have MSYS2/Cygwin Python first on your PATH, install the
echo standard Windows Python from python.org and try again.
goto fail

:install_packages
echo.
echo Installing required packages (numpy, scikit-learn, matplotlib, streamlit)...
echo This can take a few minutes the first time.
echo.
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install -r requirements.txt
if errorlevel 1 goto install_failed

echo.
echo Training the two models (creates data\model.pkl and data\tata_model.pkl)...
echo.
"%VENV_PY%" src\train_model.py
if errorlevel 1 goto train_failed
echo.
"%VENV_PY%" src\train_tata_model.py
if errorlevel 1 goto train_failed

echo.
echo ==================================================
echo  Setup finished successfully.
echo  Now double-click run_app.bat to open the website.
echo ==================================================
goto done

:install_failed
echo.
echo ERROR: Installing the packages failed. Check your internet
echo        connection and the messages above.
goto fail

:train_failed
echo.
echo ERROR: Model training failed. See the messages above.
goto fail

:fail
echo.
pause
exit /b 1

:done
echo.
pause
exit /b 0
