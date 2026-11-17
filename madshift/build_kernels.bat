@echo off
rem SPDX-License-Identifier: Apache-2.0
rem Copyright 2026 Matthew Priester
rem Builds madshift_kernels.dll (the fast trellis search loaded by madshift\kernels.py) with nvcc and MSVC.
rem Needs Visual Studio with the C++ tools for this machine and the CUDA Toolkit (CUDA_PATH set).
rem GPU architecture: set CUDA_ARCH (for example sm_89, sm_120, sm_121); default: the GPU of this machine (native).
for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`) do set "VSDIR=%%i"
set "HOSTARCH=x64"
reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v PROCESSOR_ARCHITECTURE | findstr /i ARM64 >nul && set "HOSTARCH=arm64"
call "%VSDIR%\VC\Auxiliary\Build\vcvarsall.bat" %HOSTARCH% >nul
if errorlevel 1 exit /b 1
if not defined CUDA_PATH (
  echo CUDA_PATH is not set
  exit /b 1
)
if not defined CUDA_ARCH set "CUDA_ARCH=native"
cd /d "%~dp0"
"%CUDA_PATH%\bin\nvcc.exe" -O3 -shared -o madshift_kernels.dll kernels.cu -Xcompiler /MD -arch=%CUDA_ARCH%
