@echo off
rem FNMX: build the patched llama-quantize (CPU only) from llama.cpp 5aeac37 + mxfp4_wsearch.patch
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=C:\Users\tsaip\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%
cd /d C:\ClaudeWorkspace\tools\llamacpp_mxq
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DBUILD_SHARED_LIBS=OFF || exit /b 1
cmake --build build --target llama-quantize -j 16 || exit /b 1
