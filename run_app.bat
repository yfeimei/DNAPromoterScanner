@echo off
REM ===================================================================
REM  DNA Promoter Scanner -- start the website on Windows.
REM
REM  Double-click this file. It opens the Streamlit app in your browser
REM  at http://localhost:8501. Close the window (or press Ctrl+C) to
REM  stop the site.
REM
REM  Run setup.bat once first.
REM ===================================================================
setlocal EnableExtensions

cd /d "%~dp0"

set "VENV_PY=.venv\Scripts\python.exe"

if not exist "%VENV_PY%" goto no_venv
if not exist "data\model.pkl" goto no_model
if not exist "data\tata_model.pkl" goto no_model

echo ==================================================
echo  Starting the DNA Promoter Scanner website...
echo.
echo  A browser tab should open automatically. If it
echo  does not, go to:  http://localhost:8501
echo.
echo  Press Ctrl+C in this window to stop the site.
echo ==================================================
echo.

REM "python -m streamlit" rather than "streamlit" so this works even if the
REM venv's Scripts folder is not on PATH.
"%VENV_PY%" -m streamlit run app\app.py
goto done

:no_venv
echo ERROR: The virtual environment is missing (%VENV_PY% not found).
echo.
echo Run setup.bat first (double-click it), then try again.
goto fail

:no_model
echo ERROR: A trained model file is missing from the data folder.
echo.
echo Run setup.bat first, or train the models manually:
echo     .venv\Scripts\python src\train_model.py
echo     .venv\Scripts\python src\train_tata_model.py
goto fail

:fail
echo.
pause
exit /b 1

:done
exit /b 0
