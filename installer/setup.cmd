@echo off
setlocal enabledelayedexpansion

REM ========== PENGATURAN AWAL ==========
set PYTHON_INSTALLER=python-3.9.12-amd64.exe
set BUILD_TOOLS_INSTALLER=vs_BuildTools.exe

echo -------------------------------------
echo  [1/3] Instalasi Microsoft C++ Build Tools
echo -------------------------------------

if exist %BUILD_TOOLS_INSTALLER% (
    echo Menjalankan installer Build Tools...
    %BUILD_TOOLS_INSTALLER% --passive --wait --norestart ^
    --installPath "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools" ^
    --add Microsoft.VisualStudio.Workload.VCTools ^
    --includeRecommended
    echo Build Tools selesai diinstal.
) else (
    echo ERROR: %BUILD_TOOLS_INSTALLER% tidak ditemukan!
    goto end
)

echo.
echo -------------------------------------
echo  [2/3] Instalasi Python
echo -------------------------------------

python --version >nul 2>nul
if errorlevel 1 (
    echo Python belum terinstal. Menjalankan installer GUI...

    if exist %PYTHON_INSTALLER% (
        %PYTHON_INSTALLER% /passive InstallAllUsers=1 PrependPath=1 Include_test=0
        echo Silakan selesaikan instalasi Python melalui GUI.
        echo Setelah instalasi selesai, tekan tombol apa saja untuk lanjut...
        pause >nul
    ) else (
        echo ERROR: %PYTHON_INSTALLER% tidak ditemukan!
        goto end
    )
) else (
    echo Python sudah terinstal.
)

echo.
echo -------------------------------------
echo  [3/3] Instalasi Python Requirements (di CMD baru)
echo -------------------------------------

echo Mengecek pip...
python -m ensurepip --default-pip

echo Meng-upgrade pip...
python -m pip install --upgrade pip

if exist requirements.txt (
    echo Menginstal package dari requirements.txt...
    python -m pip install -r requirements.txt
    echo Instalasi requirements selesai.
) else (
    echo ERROR: File requirements.txt tidak ditemukan!
    goto end
)

:end
echo.
echo -------------------------------------
echo  Semua instalasi selesai. Tekan tombol apa saja untuk keluar...
pause >nul
exit