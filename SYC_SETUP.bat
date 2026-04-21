@echo off
chcp 65001 >nul
set "VENV_NAME=MYVENV"

echo.
echo =====================================
echo  🛸 PYTHON VENV SETUP
echo =====================================

if exist "%VENV_NAME%\Scripts\activate.bat" goto :Activate

echo.
echo 🌍 [1/7] python -m venv %VENV_NAME%
python -m venv %VENV_NAME%

echo.
echo 🚀 [2/7] %VENV_NAME%\Scripts\activate
call %VENV_NAME%\Scripts\activate

echo.
echo 🔧 [3/7] python -m pip install --upgrade pip
python -m pip install --upgrade pip

echo.
echo 🐼 [4/7] pip install pandas
pip install pandas

echo.
echo 🎁 [5/7] pip list
pip list

echo.
echo 🐍 [6/7] python --version
python --version

echo.
echo 📍 [7/7] where python
where python
goto :Done

:Activate
echo.
echo 📡  Existing %VENV_NAME% found
echo.
echo 🚀  Activating %VENV_NAME%
call %VENV_NAME%\Scripts\activate
goto :Done

:Done
echo.
echo 🦕  All Done! 🦖
echo.