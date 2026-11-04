@echo off
rem SPDX-License-Identifier: GPL-2.0-only
rem Run from a CUDA-compatible x64 MSVC Developer Command Prompt.
where nvcc >nul 2>nul
if errorlevel 1 (
    echo CUDA Toolkit nvcc is required on PATH.
    exit /b 1
)
where cl >nul 2>nul
if errorlevel 1 (
    echo Open a CUDA-compatible x64 MSVC Developer Command Prompt first.
    exit /b 1
)
if not exist "%~dp0build" mkdir "%~dp0build"
nvcc -O3 -std=c++17 -arch=sm_89 -lcublas -lcublasLt "%~dp0matmul-tune.cu" -o "%~dp0build\matmul-tune.exe"
exit /b %errorlevel%
