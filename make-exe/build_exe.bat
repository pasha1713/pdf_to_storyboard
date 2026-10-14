@echo off
chcp 65001 >nul
setlocal

echo ============================================================
echo  Build pdf_to_storyboard.exe  (ONEFILE)
echo ============================================================
echo.
echo  NOTE: this script does NOT install or upgrade any packages.
echo        It uses your current Python + site-packages as-is.
echo.

python --version
if errorlevel 1 goto :no_python

echo.
echo [CHECK] pyinstaller ...
python -m PyInstaller --version >nul 2>&1
if errorlevel 1 goto :no_pyinstaller

goto :build

:no_python
echo [ERROR] python not found in PATH
pause
exit /b 1

:no_pyinstaller
echo [WARN] pyinstaller module not found for this python.
set /p ANS=Install pyinstaller now Y/N: 
if /i not "%ANS%"=="Y" goto :abort
python -m pip install "pyinstaller==5.13.2"
if errorlevel 1 goto :error
goto :build

:abort
echo Aborted by user.
pause
exit /b 1

:build
echo.
echo [1/3] Cleaning previous build ...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo.
echo [2/3] Running PyInstaller ...
python -m PyInstaller --clean --noconfirm pdf_to_storyboard.spec
if errorlevel 1 goto :error

echo.
echo [3/3] Done.
echo ============================================================
echo  BUILD SUCCESS
echo ============================================================
echo  Output (ONE FILE):  dist\pdf_to_storyboard.exe
echo.
echo  Just copy this single .exe wherever you need. No _internal
echo  folder, no dlls next to it.
echo.
echo  First launch may take 5-15 seconds (unpacks to %%TEMP%%).
echo.
pause
exit /b 0

:error
echo.
echo ============================================================
echo  BUILD FAILED
echo ============================================================
pause
exit /b 1
